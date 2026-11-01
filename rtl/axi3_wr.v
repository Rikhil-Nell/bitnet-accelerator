`timescale 1ns/1ps
// AXI3 burst write master with a small input FIFO.
//
// A command writes `cmd_beats` 64-bit words to `cmd_addr` (128-byte aligned)
// as bursts of up to 16 beats. A burst starts only once all of its data is in
// the FIFO, so WVALID never drops mid-burst. `idle` goes high again after the
// last write response. `bursts_done` counts write responses for the current command
// (every burst but the last is 16 beats, so 16 * bursts_done beats are in memory).
module axi3_wr #(
    parameter FIFO_AW = 5    // 32 x 64-bit, distributed RAM
)(
    input  wire        clk,
    input  wire        rstn,

    input  wire        cmd_valid,
    input  wire [31:0] cmd_addr,
    input  wire [27:0] cmd_beats,
    output wire        idle,
    output reg         err,
    output reg  [15:0] bursts_done,

    input  wire        push,
    input  wire [63:0] din,
    output wire [FIFO_AW:0] fifo_count,

    output reg  [31:0] awaddr,
    output reg  [3:0]  awlen,
    output reg         awvalid,
    input  wire        awready,
    output wire [63:0] wdata,
    output wire        wlast,
    output wire        wvalid,
    input  wire        wready,
    input  wire [1:0]  bresp,
    input  wire        bvalid,
    output wire        bready
);
    localparam DEPTH = 1 << FIFO_AW;

    reg [27:0] remaining;
    reg [31:0] next_addr;
    reg [4:0]  w_left;        // beats left in the current burst
    reg [7:0]  b_pending;     // bursts awaiting a write response
    reg [FIFO_AW:0] cnt;
    reg [FIFO_AW-1:0] wp, rp;
    reg [63:0] mem [0:DEPTH-1];

    // Registered min(16, remaining): lags by a cycle, which is safe because after
    // every burst start AWVALID / w_left keep start_burst low for several cycles.
    reg [4:0]  blen;
    reg        rem_nz;
    wire       start_burst = !awvalid && (w_left == 0) && rem_nz && (cnt >= blen);
    wire       wfire = wvalid && wready;

    assign fifo_count = cnt;
    assign wdata  = mem[rp];
    assign wvalid = (w_left != 0);
    assign wlast  = (w_left == 5'd1);
    assign bready = 1'b1;
    assign idle   = (remaining == 0) && (w_left == 0) && !awvalid && (b_pending == 0);

    always @(posedge clk) begin
        if (push) mem[wp] <= din;
    end

    always @(posedge clk) begin
        if (!rstn) begin
            remaining <= 0;
            next_addr <= 0;
            w_left    <= 0;
            b_pending <= 0;
            cnt       <= 0;
            wp        <= 0;
            rp        <= 0;
            awvalid   <= 1'b0;
            awaddr    <= 0;
            awlen     <= 0;
            err       <= 1'b0;
            blen      <= 0;
            rem_nz    <= 1'b0;
            bursts_done <= 0;
        end else begin
            blen   <= (remaining >= 28'd16) ? 5'd16 : remaining[4:0];
            rem_nz <= (remaining != 0);
            if (cmd_valid && idle) begin
                remaining <= cmd_beats;
                next_addr <= cmd_addr;
                err       <= 1'b0;
                bursts_done <= 0;
            end else if (start_burst) begin
                awvalid   <= 1'b1;
                awaddr    <= next_addr;
                awlen     <= blen - 1'b1;
                w_left    <= blen;
                remaining <= remaining - blen;
                next_addr <= next_addr + {blen, 3'b000};
            end
            if (awvalid && awready)
                awvalid <= 1'b0;

            if (wfire) begin
                w_left <= w_left - 1'b1;
                rp     <= rp + 1'b1;
            end
            if (push) wp <= wp + 1'b1;
            cnt <= cnt + (push ? 1'b1 : 1'b0) - (wfire ? 1'b1 : 1'b0);

            b_pending <= b_pending + ((start_burst && !(cmd_valid && idle)) ? 8'd1 : 8'd0)
                                   - (bvalid ? 8'd1 : 8'd0);
            if (bvalid && bresp != 2'b00) err <= 1'b1;
            if (bvalid && !(cmd_valid && idle)) bursts_done <= bursts_done + 1'b1;
        end
    end
endmodule
