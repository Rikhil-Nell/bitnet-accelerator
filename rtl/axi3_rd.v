`timescale 1ns/1ps
// AXI3 burst read master feeding a BRAM FIFO.
//
// A command reads `cmd_beats` 64-bit words starting at `cmd_addr` (128-byte
// aligned) as 16-beat bursts, so no burst crosses a 4 KiB boundary. Bursts are
// only issued when the FIFO is guaranteed to have room for every beat in
// flight, so RREADY is tied high and the HP port never stalls on us.
//
// FIFO read side is a plain synchronous FIFO: assert fifo_rd while !fifo_empty
// and fifo_dout is valid on the following cycle.
module axi3_rd #(
    parameter FIFO_AW = 9,   // 512 x 64-bit = one RAMB36
    parameter MAX_OUT = 8    // outstanding bursts
)(
    input  wire        clk,
    input  wire        rstn,

    input  wire        cmd_valid,
    input  wire [31:0] cmd_addr,
    input  wire [27:0] cmd_beats,
    output wire        idle,
    output reg         err,

    output reg  [31:0] araddr,
    output reg  [3:0]  arlen,
    output reg         arvalid,
    input  wire        arready,
    input  wire [63:0] rdata,
    input  wire        rvalid,
    input  wire        rlast,
    input  wire [1:0]  rresp,
    output wire        rready,

    input  wire        fifo_rd,
    output reg  [63:0] fifo_dout,
    output wire        fifo_empty
);
    localparam DEPTH = 1 << FIFO_AW;

    reg [27:0] remaining;      // beats not yet requested
    reg [31:0] next_addr;
    reg [3:0]  outstanding;    // bursts requested but not fully received
    reg [FIFO_AW:0] fifo_cnt;  // words stored
    reg [FIFO_AW:0] inflight;  // words requested but not yet stored

    wire       push = rvalid;  // rready is always 1
    wire       pop  = fifo_rd && !fifo_empty;
    wire [4:0] blen = (remaining >= 28'd16) ? 5'd16 : remaining[4:0];

    wire can_issue = !arvalid && (remaining != 0) && (outstanding < MAX_OUT) &&
                     ({1'b0, fifo_cnt} + {1'b0, inflight} + blen <= DEPTH);

    assign rready     = 1'b1;
    assign fifo_empty = (fifo_cnt == 0);
    assign idle       = (remaining == 0) && (outstanding == 0) && !arvalid;

    // FIFO storage (BRAM)
    (* ram_style = "block" *) reg [63:0] mem [0:DEPTH-1];
    reg [FIFO_AW-1:0] wp, rp;
    always @(posedge clk) begin
        if (push) mem[wp] <= rdata;
        if (pop)  fifo_dout <= mem[rp];
    end

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            remaining   <= 0;
            next_addr   <= 0;
            outstanding <= 0;
            fifo_cnt    <= 0;
            inflight    <= 0;
            arvalid     <= 1'b0;
            araddr      <= 0;
            arlen       <= 0;
            wp          <= 0;
            rp          <= 0;
            err         <= 1'b0;
        end else begin
            if (cmd_valid && idle) begin
                remaining <= cmd_beats;
                next_addr <= cmd_addr;
                err       <= 1'b0;
            end else if (can_issue) begin
                arvalid   <= 1'b1;
                araddr    <= next_addr;
                arlen     <= blen - 1'b1;
                remaining <= remaining - blen;
                next_addr <= next_addr + {blen, 3'b000};
            end
            if (arvalid && arready)
                arvalid <= 1'b0;

            outstanding <= outstanding + (can_issue && !(cmd_valid && idle) ? 4'd1 : 4'd0)
                                       - (rvalid && rlast ? 4'd1 : 4'd0);
            inflight    <= inflight + (can_issue && !(cmd_valid && idle) ? blen : 5'd0)
                                    - (push ? 1'b1 : 1'b0);
            fifo_cnt    <= fifo_cnt + (push ? 1'b1 : 1'b0) - (pop ? 1'b1 : 1'b0);

            if (push) wp <= wp + 1'b1;
            if (pop)  rp <= rp + 1'b1;
            if (rvalid && rresp != 2'b00) err <= 1'b1;
        end
    end
endmodule
