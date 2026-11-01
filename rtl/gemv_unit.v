`timescale 1ns/1ps
// One GEMV unit: y[r] = sum_k W[r][k] * x[k] for its slice of rows, on one HP port.
//
// Command flow:
//   1. read x (K int8 bytes, K/8 beats) into the activation buffer
//   2. stream W row-major and accumulate one row at a time
//   3. write one int32 per row to y (two rows per 64-bit beat)
//
// mode 0 (ternary): 32 weights per beat, signed 2-bit (00=0, 01=+1, 11=-1)
// mode 1 (int8):     8 weights per beat
//
// Datapath: one beat per clock, six pipeline stages, no internal stalls.
// The input FIFO is only popped while the output FIFO has room for everything
// that could still come out of the pipeline.
module gemv_unit (
    input  wire        clk,
    input  wire        rstn,

    input  wire        start,
    input  wire        mode,
    input  wire [15:0] k,        // multiple of 32, <= 8192
    input  wire [15:0] rows,     // even, >= 2
    input  wire [31:0] x_addr,
    input  wire [31:0] w_addr,
    input  wire [31:0] y_addr,
    output reg         done,     // sticky until next start
    output wire        busy,
    output wire        err,

    // AXI3 master (HP port)
    output wire [31:0] araddr,
    output wire [3:0]  arlen,
    output wire        arvalid,
    input  wire        arready,
    input  wire [63:0] rdata,
    input  wire        rvalid,
    input  wire        rlast,
    input  wire [1:0]  rresp,
    output wire        rready,
    output wire [31:0] awaddr,
    output wire [3:0]  awlen,
    output wire        awvalid,
    input  wire        awready,
    output wire [63:0] wdata,
    output wire        wlast,
    output wire        wvalid,
    input  wire        wready,
    input  wire [1:0]  bresp,
    input  wire        bvalid,
    output wire        bready
);
    // Local synchronous reset: one register per unit keeps the reset net short.
    (* max_fanout = 32 *) reg rst_n;
    always @(posedge clk) rst_n <= rstn;

    localparam S_IDLE = 3'd0, S_ACT = 3'd1, S_WREQ = 3'd2, S_RUN = 3'd3, S_FLUSH = 3'd4;

    reg [2:0]  state;
    reg        mode_q;
    reg [15:0] rows_q;
    reg [12:0] row_beats;      // beats per weight row
    reg [10:0] act_beats;      // K/8
    reg [31:0] w_addr_q;

    // ---------------- read / write masters ----------------
    reg        rd_cmd;
    reg [31:0] rd_addr;
    reg [27:0] rd_beats;
    wire       rd_idle, rd_err;
    reg        fifo_rd;
    wire [63:0] fifo_dout;
    wire       fifo_empty;

    axi3_rd u_rd (
        .clk(clk), .rstn(rst_n),
        .cmd_valid(rd_cmd), .cmd_addr(rd_addr), .cmd_beats(rd_beats),
        .idle(rd_idle), .err(rd_err),
        .araddr(araddr), .arlen(arlen), .arvalid(arvalid), .arready(arready),
        .rdata(rdata), .rvalid(rvalid), .rlast(rlast), .rresp(rresp), .rready(rready),
        .fifo_rd(fifo_rd), .fifo_dout(fifo_dout), .fifo_empty(fifo_empty)
    );

    reg        wr_cmd;
    wire       wr_idle, wr_err;
    reg        out_push;
    reg [63:0] out_word;
    wire [5:0] out_cnt;

    axi3_wr u_wr (
        .clk(clk), .rstn(rst_n),
        .cmd_valid(wr_cmd), .cmd_addr(y_addr), .cmd_beats({13'd0, rows[15:1]}),
        .idle(wr_idle), .err(wr_err),
        .push(out_push), .din(out_word), .fifo_count(out_cnt),
        .awaddr(awaddr), .awlen(awlen), .awvalid(awvalid), .awready(awready),
        .wdata(wdata), .wlast(wlast), .wvalid(wvalid), .wready(wready),
        .bresp(bresp), .bvalid(bvalid), .bready(bready)
    );

    assign err  = rd_err | wr_err;
    assign busy = (state != S_IDLE);

    // ---------------- activation buffer: 4 banks x 256 x 64 bit ----------------
    reg        act_we;
    reg [9:0]  act_widx;       // 8-byte group index being written
    reg [7:0]  act_raddr;
    reg        act_re;
    (* ram_style = "block" *) reg [63:0] bank0 [0:255];
    (* ram_style = "block" *) reg [63:0] bank1 [0:255];
    (* ram_style = "block" *) reg [63:0] bank2 [0:255];
    (* ram_style = "block" *) reg [63:0] bank3 [0:255];
    reg [63:0] a0, a1, a2, a3;

    always @(posedge clk) begin
        if (act_we) begin
            case (act_widx[1:0])
                2'd0: bank0[act_widx[9:2]] <= fifo_dout;
                2'd1: bank1[act_widx[9:2]] <= fifo_dout;
                2'd2: bank2[act_widx[9:2]] <= fifo_dout;
                2'd3: bank3[act_widx[9:2]] <= fifo_dout;
            endcase
        end
        if (act_re) begin
            a0 <= bank0[act_raddr];
            a1 <= bank1[act_raddr];
            a2 <= bank2[act_raddr];
            a3 <= bank3[act_raddr];
        end
    end

    // ---------------- stage A: pop + counters ----------------
    reg [12:0] col;
    reg [15:0] row;
    reg        popped_all;
    reg [10:0] act_cnt;        // groups popped during S_ACT

    reg        vb, last_b;
    reg [1:0]  sel_b;

    // Room check: at most 4 rows can be in flight in stages B..E plus the pair register.
    wire out_room = (out_cnt <= 6'd24);
    wire run_pop  = (state == S_RUN) && !popped_all && !fifo_empty && out_room;
    wire act_pop  = (state == S_ACT) && (act_cnt != act_beats) && !fifo_empty;

    always @(*) begin
        fifo_rd   = run_pop | act_pop;
        act_re    = run_pop;
        act_raddr = mode_q ? col[9:2] : col[7:0];
    end

    // ---------------- stage R: register BRAM / FIFO outputs ----------------
    reg         vr, last_r;
    reg [63:0]  w_r;
    reg [255:0] act_word;
    reg [63:0]  act_sel;
    always @(posedge clk) begin
        w_r      <= fifo_dout;
        act_word <= {a3, a2, a1, a0};
        case (sel_b)
            2'd0: act_sel <= a0;
            2'd1: act_sel <= a1;
            2'd2: act_sel <= a2;
            default: act_sel <= a3;
        endcase
    end

    // ---------------- stage B: products ----------------
    reg               vc, last_c;
    reg signed [8:0]  tp [0:31];
    (* use_dsp = "yes" *) reg signed [15:0] ip [0:7];
    integer i, j, m;

    always @(posedge clk) begin
        for (i = 0; i < 32; i = i + 1) begin
            case (w_r[2*i +: 2])
                2'b01:   tp[i] <=  $signed({act_word[8*i+7], act_word[8*i +: 8]});
                2'b11:   tp[i] <= -$signed({act_word[8*i+7], act_word[8*i +: 8]});
                default: tp[i] <= 9'sd0;
            endcase
        end
        for (i = 0; i < 8; i = i + 1)
            ip[i] <= $signed(act_sel[8*i +: 8]) * $signed(w_r[8*i +: 8]);
    end

    // ---------------- stage C: partial sums ----------------
    reg               vd, last_d;
    reg signed [11:0] ts [0:3];
    reg signed [17:0] is [0:1];
    always @(posedge clk) begin
        for (j = 0; j < 4; j = j + 1)
            ts[j] <= tp[8*j] + tp[8*j+1] + tp[8*j+2] + tp[8*j+3]
                   + tp[8*j+4] + tp[8*j+5] + tp[8*j+6] + tp[8*j+7];
        for (m = 0; m < 2; m = m + 1)
            is[m] <= ip[4*m] + ip[4*m+1] + ip[4*m+2] + ip[4*m+3];
    end

    // ---------------- stage D: beat total ----------------
    reg               ve, last_e;
    reg signed [31:0] beat_sum;
    always @(posedge clk) begin
        if (mode_q)
            beat_sum <= is[0] + is[1];
        else
            beat_sum <= ts[0] + ts[1] + ts[2] + ts[3];
    end

    // ---------------- stage E: accumulate, emit rows in pairs ----------------
    reg signed [31:0] acc;
    reg               have_lo;
    reg [31:0]        lo;
    wire signed [31:0] acc_next = acc + beat_sum;

    wire pipe_empty = !vb && !vr && !vc && !vd && !ve;

    always @(posedge clk) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            done       <= 1'b0;
            mode_q     <= 1'b0;
            rows_q     <= 0;
            row_beats  <= 0;
            act_beats  <= 0;
            w_addr_q   <= 0;
            rd_cmd     <= 1'b0;
            rd_addr    <= 0;
            rd_beats   <= 0;
            wr_cmd     <= 1'b0;
            act_we     <= 1'b0;
            act_widx   <= 0;
            act_cnt    <= 0;
            col        <= 0;
            row        <= 0;
            popped_all <= 1'b0;
            vb <= 0; vr <= 0; vc <= 0; vd <= 0; ve <= 0;
            last_b <= 0; last_r <= 0; last_c <= 0; last_d <= 0; last_e <= 0;
            sel_b      <= 0;
            acc        <= 0;
            have_lo    <= 1'b0;
            lo         <= 0;
            out_push   <= 1'b0;
            out_word   <= 0;
        end else begin
            rd_cmd   <= 1'b0;
            wr_cmd   <= 1'b0;
            out_push <= 1'b0;

            // activation write happens the cycle after the pop (FIFO latency)
            act_we <= act_pop;
            if (act_we) act_widx <= act_widx + 1'b1;

            case (state)
                S_IDLE: if (start) begin
                    done      <= 1'b0;
                    mode_q    <= mode;
                    rows_q    <= rows;
                    row_beats <= mode ? k[15:3] : {2'b00, k[15:5]};
                    act_beats <= k[13:3];
                    w_addr_q  <= w_addr;
                    rd_addr   <= x_addr;
                    rd_beats  <= {17'd0, k[13:3]};
                    rd_cmd    <= 1'b1;
                    wr_cmd    <= 1'b1;
                    act_cnt   <= 0;
                    act_widx  <= 0;
                    col       <= 0;
                    row       <= 0;
                    popped_all <= 1'b0;
                    acc       <= 0;
                    have_lo   <= 1'b0;
                    state     <= S_ACT;
                end
                S_ACT: begin
                    if (act_pop) act_cnt <= act_cnt + 1'b1;
                    // all groups popped and the last write has landed
                    if (act_cnt == act_beats && !act_we && rd_idle && !rd_cmd) begin
                        rd_addr  <= w_addr_q;
                        rd_beats <= rows_q * row_beats;
                        rd_cmd   <= 1'b1;
                        state    <= S_WREQ;
                    end
                end
                S_WREQ: state <= S_RUN;
                S_RUN: begin
                    if (popped_all && pipe_empty && !have_lo)
                        state <= S_FLUSH;
                end
                S_FLUSH: if (wr_idle && !wr_cmd) begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase

            // stage A
            vb <= run_pop;
            if (run_pop) begin
                sel_b  <= col[1:0];
                last_b <= (col == row_beats - 1'b1);
                if (col == row_beats - 1'b1) begin
                    col <= 0;
                    row <= row + 1'b1;
                    if (row == rows_q - 1'b1) popped_all <= 1'b1;
                end else begin
                    col <= col + 1'b1;
                end
            end
            // stages B..D valid/last
            vr <= vb; last_r <= last_b;
            vc <= vr; last_c <= last_r;
            vd <= vc; last_d <= last_c;
            ve <= vd; last_e <= last_d;

            // stage E
            if (ve) begin
                if (last_e) begin
                    acc <= 0;
                    if (have_lo) begin
                        out_word <= {acc_next, lo};
                        out_push <= 1'b1;
                        have_lo  <= 1'b0;
                    end else begin
                        lo      <= acc_next;
                        have_lo <= 1'b1;
                    end
                end else begin
                    acc <= acc_next;
                end
            end
        end
    end
endmodule
