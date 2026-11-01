`timescale 1ns/1ps
// Testbench for gemv_top: AXI-Lite CSR driver + four AXI3 memory models with
// random stalls, checked bit-exactly against vectors from gen_vectors.py.
module tb_gemv_top;
    localparam MEM_WORDS = (4 << 20) / 8;
    localparam NU = 4;

    reg clk = 0, rstn = 0;
    always #3.333 clk = ~clk;   // ~150 MHz

    reg [63:0] mem [0:MEM_WORDS-1];
    reg [63:0] exp_mem [0:MEM_WORDS-1];

    // stall probability in percent for every handshake the memory controls
    int unsigned stall_pct = 25;

    // ---------------- AXI-Lite ----------------
    reg  [11:0] awaddr = 0, araddr = 0;
    reg         awvalid = 0, wvalid = 0, bready = 0, arvalid = 0, rready = 0;
    reg  [31:0] wdata = 0;
    wire        awready, wready, bvalid, arready, rvalid;
    wire [1:0]  bresp, rresp;
    wire [31:0] rdata;

    // ---------------- AXI3 ports ----------------
    wire [31:0] m_araddr [NU];  wire [3:0] m_arlen [NU];  wire m_arvalid [NU];  reg m_arready [NU];
    reg  [63:0] m_rdata  [NU];  reg  m_rvalid [NU];  reg m_rlast [NU];  wire m_rready [NU];
    wire [31:0] m_awaddr [NU];  wire [3:0] m_awlen [NU];  wire m_awvalid [NU];  reg m_awready [NU];
    wire [63:0] m_wdata  [NU];  wire m_wlast [NU];  wire m_wvalid [NU];  reg m_wready [NU];
    reg         m_bvalid [NU];  wire m_bready [NU];

    gemv_top dut (
        .aclk(clk), .aresetn(rstn),
        .s_axi_awaddr(awaddr), .s_axi_awprot(3'b0), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arprot(3'b0), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready),
`define PORT(u) \
        .m``u``_axi_araddr(m_araddr[u]), .m``u``_axi_arlen(m_arlen[u]), .m``u``_axi_arsize(), .m``u``_axi_arburst(), \
        .m``u``_axi_arlock(), .m``u``_axi_arcache(), .m``u``_axi_arprot(), .m``u``_axi_arqos(), .m``u``_axi_arid(), \
        .m``u``_axi_arvalid(m_arvalid[u]), .m``u``_axi_arready(m_arready[u]), \
        .m``u``_axi_rdata(m_rdata[u]), .m``u``_axi_rresp(2'b00), .m``u``_axi_rlast(m_rlast[u]), .m``u``_axi_rid(6'd0), \
        .m``u``_axi_rvalid(m_rvalid[u]), .m``u``_axi_rready(m_rready[u]), \
        .m``u``_axi_awaddr(m_awaddr[u]), .m``u``_axi_awlen(m_awlen[u]), .m``u``_axi_awsize(), .m``u``_axi_awburst(), \
        .m``u``_axi_awlock(), .m``u``_axi_awcache(), .m``u``_axi_awprot(), .m``u``_axi_awqos(), .m``u``_axi_awid(), \
        .m``u``_axi_awvalid(m_awvalid[u]), .m``u``_axi_awready(m_awready[u]), \
        .m``u``_axi_wdata(m_wdata[u]), .m``u``_axi_wstrb(), .m``u``_axi_wlast(m_wlast[u]), .m``u``_axi_wid(), \
        .m``u``_axi_wvalid(m_wvalid[u]), .m``u``_axi_wready(m_wready[u]), \
        .m``u``_axi_bresp(2'b00), .m``u``_axi_bid(6'd0), .m``u``_axi_bvalid(m_bvalid[u]), .m``u``_axi_bready(m_bready[u])
        `PORT(0), `PORT(1), `PORT(2), `PORT(3)
    );

    // ---------------- memory model, one process set per port ----------------
    function automatic bit stall();
        return ($urandom_range(99) < stall_pct);
    endfunction

    for (genvar u = 0; u < NU; u++) begin : port
        typedef struct { bit [31:0] addr; int beats; } burst_t;
        burst_t rq[$], wq[$];
        int b_owed = 0;
        int max_rq = 0;

        // read address
        always @(posedge clk) begin
            m_arready[u] <= rstn && !stall();
            if (m_arvalid[u] && m_arready[u]) begin
                rq.push_back('{m_araddr[u], m_arlen[u] + 1});
                if (rq.size() > max_rq) max_rq = rq.size();
            end
        end
        // read data
        int rbeat = 0;
        always @(posedge clk) begin
            if (!rstn) begin
                m_rvalid[u] <= 0; m_rlast[u] <= 0;
            end else begin
                if (m_rvalid[u] && m_rready[u]) begin
                    rbeat++;
                    if (m_rlast[u]) begin rq.pop_front(); rbeat = 0; end
                end
                m_rvalid[u] <= 0; m_rlast[u] <= 0;
                if (rq.size() > 0 && !stall()) begin
                    // don't re-present the beat that was just accepted on the last cycle
                    m_rdata[u]  <= mem[(rq[0].addr >> 3) + rbeat];
                    m_rlast[u]  <= (rbeat == rq[0].beats - 1);
                    m_rvalid[u] <= 1;
                end
            end
        end
        // write address / data / response
        int wbeat = 0;
        always @(posedge clk) begin
            if (m_awvalid[u] && m_awready[u]) wq.push_back('{m_awaddr[u], m_awlen[u] + 1});
            if (m_wvalid[u] && m_wready[u]) begin
                if (wq.size() == 0) begin
                    $display("FAIL: port %0d W beat with no address", u);
                    $finish;
                end
                mem[(wq[0].addr >> 3) + wbeat] <= m_wdata[u];
                if (m_wlast[u] != (wbeat == wq[0].beats - 1)) begin
                    $display("FAIL: port %0d WLAST mismatch at beat %0d", u, wbeat);
                    $finish;
                end
                wbeat++;
                if (m_wlast[u]) begin wq.pop_front(); wbeat = 0; b_owed++; end
            end
            if (m_bvalid[u] && m_bready[u]) begin m_bvalid[u] <= 0; b_owed--; end
            else if (!m_bvalid[u] && b_owed > 0 && !stall()) m_bvalid[u] <= 1;
            if (!rstn) m_bvalid[u] <= 0;
            // ready signals for the next cycle, from the queue state after this edge
            m_awready[u] <= rstn && !stall();
            m_wready[u]  <= rstn && (wq.size() > 0) && !stall();
        end
        // 4 KiB boundary check
        always @(posedge clk) begin
            if (m_arvalid[u] && m_arready[u] &&
                (m_araddr[u][31:12] != (m_araddr[u] + (m_arlen[u] + 1) * 8 - 1) >> 12)) begin
                $display("FAIL: port %0d read burst crosses 4 KiB at %h", u, m_araddr[u]);
                $finish;
            end
        end
    end

    // ---------------- AXI-Lite tasks ----------------
    // Signals sampled right after a posedge hold the value of the cycle that just
    // ended, so a handshake seen here is the one that completed on this edge.
    task automatic csr_write(input [11:0] a, input [31:0] d);
        @(posedge clk);
        awaddr <= a; wdata <= d; awvalid <= 1; wvalid <= 1; bready <= 1;
        forever begin
            @(posedge clk);
            if (awvalid && awready && wready) begin awvalid <= 0; wvalid <= 0; end
            if (bvalid && bready) break;
        end
        bready <= 0;
    endtask

    task automatic csr_read(input [11:0] a, output [31:0] d);
        @(posedge clk);
        araddr <= a; arvalid <= 1; rready <= 1;
        forever begin
            @(posedge clk);
            if (arvalid && arready) arvalid <= 0;
            if (rvalid && rready) begin d = rdata; break; end
        end
        rready <= 0;
    endtask

    // ---------------- test sequence ----------------
    int fd, mode, k, n, ncase = 0, errors = 0;
    int unsigned xa, wa[NU], ya[NU];
    reg [31:0] st, cyc, ver;
    string vdir;

    initial begin
        // xsim.bat on Windows mangles "+ARG=value", so settings come from files
        vdir = ".";
        fd = $fopen("stall.txt", "r");
        if (fd) begin void'($fscanf(fd, "%d", stall_pct)); $fclose(fd); end
        $readmemh({vdir, "/mem.hex"}, mem);
        $readmemh({vdir, "/exp.hex"}, exp_mem);
        repeat (10) @(posedge clk);
        rstn = 1;
        repeat (5) @(posedge clk);

        csr_read(12'h1C, ver);
        if (ver !== 32'h47454D31) begin $display("FAIL: VERSION %h", ver); $finish; end

        fd = $fopen({vdir, "/cases.txt"}, "r");
        while ($fscanf(fd, "%d %d %d %d %d %d %d %d %d %d %d %d\n", mode, k, n, xa,
                       wa[0], wa[1], wa[2], wa[3], ya[0], ya[1], ya[2], ya[3]) == 12) begin
            csr_write(12'h08, mode);
            csr_write(12'h0C, k);
            csr_write(12'h10, n / NU);
            csr_write(12'h14, xa);
            for (int u = 0; u < NU; u++) begin
                csr_write(12'h20 + 8*u, wa[u]);
                csr_write(12'h24 + 8*u, ya[u]);
            end
            csr_write(12'h00, 1);
            do csr_read(12'h04, st); while (!st[0]);
            csr_read(12'h18, cyc);
            if (st[2]) begin $display("FAIL: case %0d AXI error flag, STATUS=%h", ncase, st); errors++; end
            for (int u = 0; u < NU; u++)
                for (int i = 0; i < n / NU / 2; i++) begin
                    int unsigned a = (ya[u] >> 3) + i;
                    if (mem[a] !== exp_mem[a]) begin
                        if (errors < 10)
                            $display("FAIL: case %0d unit %0d rows %0d,%0d got %h exp %h",
                                     ncase, u, 2*i, 2*i+1, mem[a], exp_mem[a]);
                        errors++;
                    end
                end
            $display("case %0d: mode=%0d K=%0d N=%0d cycles=%0d (%0.2f beats/cycle/unit)", ncase, mode, k, n, cyc,
                     real'((mode ? k/8 : k/32) * (n/NU) + k/8) / cyc);
            ncase++;
        end
        $fclose(fd);
        if (errors == 0 && ncase > 0) $display("TEST PASSED (%0d cases, stall %0d%%)", ncase, stall_pct);
        else $display("TEST FAILED (%0d errors)", errors);
        $finish;
    end

    initial begin
        #50ms;
        $display("FAIL: timeout");
        $finish;
    end
endmodule
