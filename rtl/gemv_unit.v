`timescale 1ns/1ps
// One GEMV unit: y[r] = sum_k W[r][k] * x[k] for its slice of rows, on one HP port.
//
// Command flow:
//   1. read x (x_beats x 8 int8 bytes) into the activation buffer
//   2. stream W row-major (row_beats beats per row) and accumulate one row at a time
//   3. write one int32 per row to y (two rows per 64-bit beat)
//
// mode 0 (ternary, "T5"): 40 weights per beat, 5 per byte in base 3 (rtl/t5_decode.v),
//                         x_beats = 5 * row_beats (K padded to a multiple of 40)
// mode 1 (int8):          8 weights per beat, row_beats = x_beats
//
// nb2 (T5 only): two activation vectors (x_addr, x_addr + 8 * x_beats) share every
// weight beat: two 40-lane trees, two accumulators, and one 64-bit output word per
// row {y_b, y_a}. With nb2 = 0 the output packs two rows per word as before. This is
// what verifies a speculative draft token in the same pass over the weights.
//
// Activation buffer: 5 banks x 256 x 64 bit. x beat j lives in bank j % 5 at address
// j / 5, so one address read gives the 40 activations of a T5 beat, and an int8 beat
// reads one bank.
//
// Datapath: one beat per clock, eight pipeline stages, no internal stalls. The input
// FIFO is only popped while the output FIFO has room for everything that could
// still come out of the pipeline.
//
// progress: write bursts acknowledged for the current command. Every burst but the
// last is 16 beats = 32 rows (16 rows with nb2), so those rows of this unit are in DDR.
module gemv_unit (
    input  wire        clk,
    input  wire        rstn,

    input  wire        start,
    input  wire        mode,
    input  wire        nb2,       // two activation vectors (T5 mode)
    input  wire [10:0] x_beats,   // K / 8 per vector, <= 1280
    input  wire [12:0] row_beats, // beats per weight row, >= 1
    input  wire [15:0] rows,      // even, >= 2
    input  wire [31:0] x_addr,
    input  wire [31:0] w_addr,
    input  wire [31:0] y_addr,
    output reg         done,      // sticky until next start
    output wire        busy,
    output wire        err,
    output wire [15:0] progress,

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
    reg        nb2_q;
    reg [15:0] rows_q;
    reg [12:0] row_beats_q;
    reg [10:0] act_beats;      // x beats per vector
    reg [11:0] act_total;      // x beats to load (1 or 2 vectors)
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
        .cmd_valid(wr_cmd), .cmd_addr(y_addr), .cmd_beats(nb2 ? {12'd0, rows} : {13'd0, rows[15:1]}),
        .idle(wr_idle), .err(wr_err), .bursts_done(progress),
        .push(out_push), .din(out_word), .fifo_count(out_cnt),
        .awaddr(awaddr), .awlen(awlen), .awvalid(awvalid), .awready(awready),
        .wdata(wdata), .wlast(wlast), .wvalid(wvalid), .wready(wready),
        .bresp(bresp), .bvalid(bvalid), .bready(bready)
    );

    assign err  = rd_err | wr_err;
    assign busy = (state != S_IDLE);

    // ---------------- activation buffer: 2 sets x 5 banks x 256 x 64 bit ----------------
    reg        act_we;
    reg        wset;           // vector being stored (0: a, 1: b)
    reg [10:0] wcnt;           // beats stored into the current set
    reg [2:0]  wbank;          // bank of the next x beat to store
    reg [7:0]  waddr;
    reg [7:0]  act_raddr;
    reg        act_re;
    (* ram_style = "block" *) reg [63:0] bank0 [0:255];
    (* ram_style = "block" *) reg [63:0] bank1 [0:255];
    (* ram_style = "block" *) reg [63:0] bank2 [0:255];
    (* ram_style = "block" *) reg [63:0] bank3 [0:255];
    (* ram_style = "block" *) reg [63:0] bank4 [0:255];
    (* ram_style = "block" *) reg [63:0] bnkb0 [0:255];
    (* ram_style = "block" *) reg [63:0] bnkb1 [0:255];
    (* ram_style = "block" *) reg [63:0] bnkb2 [0:255];
    (* ram_style = "block" *) reg [63:0] bnkb3 [0:255];
    (* ram_style = "block" *) reg [63:0] bnkb4 [0:255];
    reg [63:0] a0, a1, a2, a3, a4;
    reg [63:0] b0, b1, b2, b3, b4;

    always @(posedge clk) begin
        if (act_we && !wset) begin
            case (wbank)
                3'd0: bank0[waddr] <= fifo_dout;
                3'd1: bank1[waddr] <= fifo_dout;
                3'd2: bank2[waddr] <= fifo_dout;
                3'd3: bank3[waddr] <= fifo_dout;
                default: bank4[waddr] <= fifo_dout;
            endcase
        end
        if (act_we && wset) begin
            case (wbank)
                3'd0: bnkb0[waddr] <= fifo_dout;
                3'd1: bnkb1[waddr] <= fifo_dout;
                3'd2: bnkb2[waddr] <= fifo_dout;
                3'd3: bnkb3[waddr] <= fifo_dout;
                default: bnkb4[waddr] <= fifo_dout;
            endcase
        end
        if (act_re) begin
            a0 <= bank0[act_raddr];
            a1 <= bank1[act_raddr];
            a2 <= bank2[act_raddr];
            a3 <= bank3[act_raddr];
            a4 <= bank4[act_raddr];
        end
        // second vector: only read (and toggle the second tree) in two-vector mode
        if (act_re && nb2_q) begin
            b0 <= bnkb0[act_raddr];
            b1 <= bnkb1[act_raddr];
            b2 <= bnkb2[act_raddr];
            b3 <= bnkb3[act_raddr];
            b4 <= bnkb4[act_raddr];
        end
    end

    // ---------------- stage A: pop + counters ----------------
    reg [12:0] col;
    reg [15:0] row;
    reg        popped_all;
    reg [11:0] act_cnt;        // x beats popped during S_ACT
    reg [2:0]  ibank;          // int8: bank of the current beat
    reg [7:0]  iaddr;          // int8: address of the current beat

    reg        vb, last_b;
    reg [2:0]  sel_b;

    // Room check: at most a few rows can be in flight in stages B..E plus the pair register.
    wire out_room = (out_cnt <= 6'd24);
    wire run_pop  = (state == S_RUN) && !popped_all && !fifo_empty && out_room;
    wire act_pop  = (state == S_ACT) && (act_cnt != act_total) && !fifo_empty;
    wire row_end  = (col == row_beats_q - 1'b1);

    always @(*) begin
        fifo_rd   = run_pop | act_pop;
        act_re    = run_pop;
        act_raddr = mode_q ? iaddr : col[7:0];
    end

    // ---------------- stage R: register BRAM / FIFO outputs ----------------
    reg         vr, last_r;
    reg [2:0]   sel_r;
    reg [63:0]  w_r;
    reg [319:0] act_r, actb_r;
    always @(posedge clk) begin
        w_r    <= fifo_dout;
        act_r  <= {a4, a3, a2, a1, a0};
        actb_r <= {b4, b3, b2, b1, b0};
        sel_r  <= sel_b;
    end

    // ---------------- stage T: T5 decode, int8 operand select ----------------
    wire [79:0] codes;
    genvar gb;
    generate
        for (gb = 0; gb < 8; gb = gb + 1) begin : g_dec
            t5_decode u_dec (.b(w_r[8*gb +: 8]), .c(codes[10*gb +: 10]));
        end
    endgenerate

    reg         vt, last_t;
    reg [79:0]  codes_t;
    reg [319:0] act_t, actb_t;
    reg [63:0]  w_t;
    reg [63:0]  act_sel;
    always @(posedge clk) begin
        codes_t <= codes;
        act_t   <= act_r;
        actb_t  <= nb2_q ? actb_r : 320'd0;   // idle second tree sees constant zeros
        w_t     <= w_r;
        if (!mode_q)
            act_sel <= 64'd0;            // int8 multipliers idle in T5 mode (power)
        else case (sel_r)
            3'd0: act_sel <= act_r[63:0];
            3'd1: act_sel <= act_r[127:64];
            3'd2: act_sel <= act_r[191:128];
            3'd3: act_sel <= act_r[255:192];
            default: act_sel <= act_r[319:256];
        endcase
    end

    // ---------------- stage B: products ----------------
    reg               vc, last_c;
    reg signed [8:0]  tp [0:39];
    reg signed [8:0]  tq [0:39];       // second vector
    (* use_dsp = "yes" *) reg signed [15:0] ip [0:7];
    integer i, j, m;

    always @(posedge clk) begin
        for (i = 0; i < 40; i = i + 1) begin
            case (codes_t[2*i +: 2])
                2'b01:   tp[i] <=  $signed({act_t[8*i+7], act_t[8*i +: 8]});
                2'b11:   tp[i] <= -$signed({act_t[8*i+7], act_t[8*i +: 8]});
                default: tp[i] <= 9'sd0;
            endcase
            case (codes_t[2*i +: 2])
                2'b01:   tq[i] <=  $signed({actb_t[8*i+7], actb_t[8*i +: 8]});
                2'b11:   tq[i] <= -$signed({actb_t[8*i+7], actb_t[8*i +: 8]});
                default: tq[i] <= 9'sd0;
            endcase
        end
        for (i = 0; i < 8; i = i + 1)
            ip[i] <= $signed(act_sel[8*i +: 8]) * $signed(w_t[8*i +: 8]);
    end

    // ---------------- stage C1: sums of 4 (ternary) / 2 (int8) ----------------
    reg               vq, last_q;
    reg signed [10:0] t4 [0:9];
    reg signed [10:0] q4 [0:9];
    reg signed [16:0] i2 [0:3];
    always @(posedge clk) begin
        for (j = 0; j < 10; j = j + 1)
            t4[j] <= tp[4*j] + tp[4*j+1] + tp[4*j+2] + tp[4*j+3];
        for (j = 0; j < 10; j = j + 1)
            q4[j] <= tq[4*j] + tq[4*j+1] + tq[4*j+2] + tq[4*j+3];
        for (m = 0; m < 4; m = m + 1)
            i2[m] <= ip[2*m] + ip[2*m+1];
    end

    // ---------------- stage C2: partial sums ----------------
    reg               vd, last_d;
    reg signed [11:0] ts [0:4];
    reg signed [11:0] qs [0:4];
    reg signed [17:0] is [0:1];
    always @(posedge clk) begin
        for (j = 0; j < 5; j = j + 1)
            ts[j] <= t4[2*j] + t4[2*j+1];
        for (j = 0; j < 5; j = j + 1)
            qs[j] <= q4[2*j] + q4[2*j+1];
        for (m = 0; m < 2; m = m + 1)
            is[m] <= i2[2*m] + i2[2*m+1];
    end

    // ---------------- stage D: beat total ----------------
    reg               ve, last_e;
    reg signed [31:0] beat_sum, beat_sum_b;
    always @(posedge clk) begin
        if (mode_q)
            beat_sum <= is[0] + is[1];
        else
            beat_sum <= ts[0] + ts[1] + ts[2] + ts[3] + ts[4];
        beat_sum_b <= qs[0] + qs[1] + qs[2] + qs[3] + qs[4];
    end

    // ---------------- stage E: accumulate, emit rows in pairs ----------------
    reg signed [31:0] acc, accb;
    reg               have_lo;
    reg [31:0]        lo;
    wire signed [31:0] acc_next  = acc + beat_sum;
    wire signed [31:0] accb_next = accb + beat_sum_b;

    wire pipe_empty = !vb && !vr && !vt && !vc && !vq && !vd && !ve;

    always @(posedge clk) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            done       <= 1'b0;
            mode_q     <= 1'b0;
            nb2_q      <= 1'b0;
            act_total  <= 0;
            wset       <= 1'b0;
            wcnt       <= 0;
            accb       <= 0;
            rows_q     <= 0;
            row_beats_q <= 0;
            act_beats  <= 0;
            w_addr_q   <= 0;
            rd_cmd     <= 1'b0;
            rd_addr    <= 0;
            rd_beats   <= 0;
            wr_cmd     <= 1'b0;
            act_we     <= 1'b0;
            wbank      <= 0;
            waddr      <= 0;
            act_cnt    <= 0;
            col        <= 0;
            row        <= 0;
            ibank      <= 0;
            iaddr      <= 0;
            popped_all <= 1'b0;
            vb <= 0; vr <= 0; vt <= 0; vc <= 0; vq <= 0; vd <= 0; ve <= 0;
            last_b <= 0; last_r <= 0; last_t <= 0; last_c <= 0; last_q <= 0; last_d <= 0; last_e <= 0;
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

            // activation write happens the cycle after the pop (FIFO latency);
            // after x_beats beats the second vector goes to the second bank set
            act_we <= act_pop;
            if (act_we) begin
                if (wcnt == act_beats - 1'b1) begin
                    wset  <= 1'b1;
                    wcnt  <= 0;
                    wbank <= 0;
                    waddr <= 0;
                end else begin
                    wcnt <= wcnt + 1'b1;
                    if (wbank == 3'd4) begin
                        wbank <= 0;
                        waddr <= waddr + 1'b1;
                    end else begin
                        wbank <= wbank + 1'b1;
                    end
                end
            end

            case (state)
                S_IDLE: if (start) begin
                    done        <= 1'b0;
                    mode_q      <= mode;
                    nb2_q       <= nb2;
                    act_total   <= nb2 ? {x_beats, 1'b0} : {1'b0, x_beats};
                    wset        <= 1'b0;
                    wcnt        <= 0;
                    accb        <= 0;
                    rows_q      <= rows;
                    row_beats_q <= row_beats;
                    act_beats   <= x_beats;
                    w_addr_q    <= w_addr;
                    rd_addr     <= x_addr;
                    rd_beats    <= nb2 ? {16'd0, x_beats, 1'b0} : {17'd0, x_beats};
                    rd_cmd      <= 1'b1;
                    wr_cmd      <= 1'b1;
                    act_cnt     <= 0;
                    wbank       <= 0;
                    waddr       <= 0;
                    col         <= 0;
                    row         <= 0;
                    ibank       <= 0;
                    iaddr       <= 0;
                    popped_all  <= 1'b0;
                    acc         <= 0;
                    have_lo     <= 1'b0;
                    state       <= S_ACT;
                end
                S_ACT: begin
                    if (act_pop) act_cnt <= act_cnt + 1'b1;
                    // all x beats popped and the last write has landed
                    if (act_cnt == act_total && !act_we && rd_idle && !rd_cmd) begin
                        rd_addr  <= w_addr_q;
                        rd_beats <= rows_q * row_beats_q;
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
                sel_b  <= ibank;
                last_b <= row_end;
                if (row_end) begin
                    col   <= 0;
                    ibank <= 0;
                    iaddr <= 0;
                    row   <= row + 1'b1;
                    if (row == rows_q - 1'b1) popped_all <= 1'b1;
                end else begin
                    col <= col + 1'b1;
                    if (ibank == 3'd4) begin
                        ibank <= 0;
                        iaddr <= iaddr + 1'b1;
                    end else begin
                        ibank <= ibank + 1'b1;
                    end
                end
            end
            // stages R..D valid/last
            vr <= vb; last_r <= last_b;
            vt <= vr; last_t <= last_r;
            vc <= vt; last_c <= last_t;
            vq <= vc; last_q <= last_c;
            vd <= vq; last_d <= last_q;
            ve <= vd; last_e <= last_d;

            // stage E
            if (ve) begin
                if (last_e) begin
                    acc  <= 0;
                    accb <= 0;
                    if (nb2_q) begin
                        out_word <= {accb_next, acc_next};
                        out_push <= 1'b1;
                    end else if (have_lo) begin
                        out_word <= {acc_next, lo};
                        out_push <= 1'b1;
                        have_lo  <= 1'b0;
                    end else begin
                        lo      <= acc_next;
                        have_lo <= 1'b1;
                    end
                end else begin
                    acc  <= acc_next;
                    accb <= accb_next;
                end
            end
        end
    end
endmodule
