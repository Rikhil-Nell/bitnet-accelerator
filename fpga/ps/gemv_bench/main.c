/*
 * gemv_bench: bring-up test for the PL GEMV engine (rtl/gemv_top.v).
 *
 *   1. correctness: random ternary / int8 matrices at the model's shapes,
 *      every output checked against the A9 computing the same product
 *   2. bandwidth:   one 64 MB ternary matrix streamed through all 4 HP ports
 *   3. token:       the 121 engine commands of one BitNet decode step
 *                   (30 x {qkv, o, gate_up, down} + int8 LM head), timed
 *
 * Output on UART1 (115200 8N1). Formats: docs/ARCHITECTURE.md.
 */
#include <stdint.h>
#include <string.h>
#include "xil_io.h"
#include "xil_cache.h"
#include "xil_printf.h"

#define GEMV_BASE   0x43C00000u
#define R_CTRL      0x00u
#define R_STATUS    0x04u
#define R_MODE      0x08u
#define R_K         0x0Cu
#define R_ROWS      0x10u
#define R_X         0x14u
#define R_CYCLES    0x18u
#define R_VERSION   0x1Cu
#define R_W(u)      (0x20u + 8u * (u))
#define R_Y(u)      (0x24u + 8u * (u))
#define NU          4
/* Actual FCLK0: 1000 MHz IO PLL / 7 (150 was requested; PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ). */
#define PL_KHZ      142857u
#define BUF_BASE    0x10000000u   /* test buffers live above the program */
#define BUF_END     0x3F000000u

static inline void     csr_w(uint32_t off, uint32_t v) { Xil_Out32(GEMV_BASE + off, v); }
static inline uint32_t csr_r(uint32_t off)             { return Xil_In32(GEMV_BASE + off); }

/* ---------------- helpers ---------------- */
static uint32_t rng = 0x9E3779B9u;
static inline uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static uint32_t heap = BUF_BASE;
static uint32_t alloc(uint32_t bytes) {
    uint32_t a = (heap + 4095u) & ~4095u;
    heap = a + bytes;
    if (heap > BUF_END) { xil_printf("ERROR: out of test memory\r\n"); for (;;); }
    return a;
}

/* ARM global timer (CPU clock / 2), read directly: the 2026 BSP's XTime path reads 0. */
#define GT_BASE 0xF8F00200u
#define GT_HZ   333333343ull
static uint64_t now(void) {
    if (!(Xil_In32(GT_BASE + 8) & 1u)) Xil_Out32(GT_BASE + 8, 1u);
    uint32_t hi, lo;
    do { hi = Xil_In32(GT_BASE + 4); lo = Xil_In32(GT_BASE); } while (hi != Xil_In32(GT_BASE + 4));
    return ((uint64_t)hi << 32) | lo;
}
static uint32_t us_since(uint64_t t0) { return (uint32_t)((now() - t0) * 1000000ull / GT_HZ); }
static uint32_t pl_us(uint32_t cycles) { return (uint32_t)((uint64_t)cycles * 1000u / PL_KHZ); }

typedef struct {
    int mode, k, n;               /* n = total rows, split over NU units */
    uint32_t x, w[NU], y[NU];
} cmd_t;

/* Lay out a matrix: x, then NU weight slices, then NU output slices. */
static void layout(cmd_t *c, int mode, int k, int n, uint32_t share_x) {
    int r = n / NU;
    uint32_t row_bytes = mode ? (uint32_t)k : (uint32_t)k / 4u;
    c->mode = mode; c->k = k; c->n = n;
    c->x = share_x ? share_x : alloc((uint32_t)k);
    for (int u = 0; u < NU; u++) c->w[u] = alloc(row_bytes * (uint32_t)r);
    for (int u = 0; u < NU; u++) c->y[u] = alloc(4u * (uint32_t)r);
}

/* Program and run one command; returns PL cycles, or 0 on error. */
static uint32_t run(const cmd_t *c) {
    int r = c->n / NU;
    Xil_DCacheFlushRange(c->x, (uint32_t)c->k);
    csr_w(R_MODE, (uint32_t)c->mode);
    csr_w(R_K, (uint32_t)c->k);
    csr_w(R_ROWS, (uint32_t)r);
    csr_w(R_X, c->x);
    for (int u = 0; u < NU; u++) { csr_w(R_W(u), c->w[u]); csr_w(R_Y(u), c->y[u]); }
    csr_w(R_CTRL, 1u);
    uint32_t st, spins = 0;
    while (!((st = csr_r(R_STATUS)) & 1u)) {
        if (++spins > 200000000u) { xil_printf("ERROR: timeout, STATUS=%08x\r\n", st); return 0; }
    }
    if (st & 4u) { xil_printf("ERROR: AXI error, STATUS=%08x\r\n", st); return 0; }
    for (int u = 0; u < NU; u++) Xil_DCacheInvalidateRange(c->y[u], 4u * (uint32_t)r);
    return csr_r(R_CYCLES);
}

static void fill_random(uint32_t addr, uint32_t bytes, int ternary) {
    uint64_t *p = (uint64_t *)addr;
    for (uint32_t i = 0; i < bytes / 8u; i++) {
        uint64_t w = ((uint64_t)rnd() << 32) | rnd();
        if (ternary) {   /* code 10 is unused: turn it into 00 */
            uint64_t hi = w & 0xAAAAAAAAAAAAAAAAull;
            uint64_t lo_up = (w << 1) & 0xAAAAAAAAAAAAAAAAull;
            w &= ~(hi & ~lo_up);
        }
        p[i] = w;
    }
    Xil_DCacheFlushRange(addr, bytes);
}

static void fill_x(uint32_t addr, int k) {
    int8_t *x = (int8_t *)addr;
    for (int i = 0; i < k; i++) x[i] = (int8_t)rnd();
    x[0] = -128; x[k - 1] = 127;
}

/* A9 reference for one row; returns the dot product. */
static int32_t ref_row(const cmd_t *c, int u, int row) {
    const int8_t *x = (const int8_t *)c->x;
    int32_t acc = 0;
    if (c->mode == 0) {
        const uint64_t *w = (const uint64_t *)(c->w[u] + (uint32_t)row * (uint32_t)c->k / 4u);
        for (int b = 0; b < c->k / 32; b++) {
            uint64_t word = w[b];
            for (int j = 0; j < 32; j++) {
                uint32_t code = (uint32_t)(word >> (2 * j)) & 3u;
                if (code == 1u) acc += x[32 * b + j];
                else if (code == 3u) acc -= x[32 * b + j];
            }
        }
    } else {
        const int8_t *w = (const int8_t *)(c->w[u] + (uint32_t)row * (uint32_t)c->k);
        for (int i = 0; i < c->k; i++) acc += (int32_t)w[i] * x[i];
    }
    return acc;
}

static int check(const cmd_t *c, int max_rows) {
    int r = c->n / NU, errs = 0;
    for (int u = 0; u < NU; u++) {
        const int32_t *y = (const int32_t *)c->y[u];
        int lim = (max_rows && max_rows < r) ? max_rows : r;
        for (int i = 0; i < lim; i++) {
            int32_t e = ref_row(c, u, i);
            if (y[i] != e) {
                if (errs < 5) xil_printf("  MISMATCH unit %d row %d: got %d exp %d\r\n", u, i, y[i], e);
                errs++;
            }
        }
    }
    return errs;
}

static uint32_t bytes_streamed(const cmd_t *c) {
    uint32_t row_bytes = c->mode ? (uint32_t)c->k : (uint32_t)c->k / 4u;
    return row_bytes * (uint32_t)c->n + (uint32_t)NU * (uint32_t)c->k + 4u * (uint32_t)c->n;
}

/* MB/s from bytes and PL cycles (integer math, xil_printf has no %f). */
static uint32_t mbps(uint32_t bytes, uint32_t cycles) {
    return cycles ? (uint32_t)((uint64_t)bytes * PL_KHZ / 1000u / cycles) : 0;
}

/* ---------------- tests ---------------- */
static int correctness(void) {
    static const int shapes[][3] = {   /* mode, K, N */
        {0, 2560, 3840},   /* qkv */
        {0, 2560, 2560},   /* o */
        {0, 6912, 2560},   /* down */
        {0, 2560, 13824},  /* gate_up */
        {1, 2560, 1024},   /* LM head slice, int8 */
    };
    int fails = 0;
    xil_printf("\r\n[1] correctness vs A9 reference\r\n");
    for (unsigned s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        uint32_t mark = heap;
        cmd_t c;
        layout(&c, shapes[s][0], shapes[s][1], shapes[s][2], 0);
        fill_x(c.x, c.k);
        uint32_t wbytes = (c.mode ? (uint32_t)c.k : (uint32_t)c.k / 4u) * (uint32_t)(c.n / NU);
        for (int u = 0; u < NU; u++) fill_random(c.w[u], wbytes, c.mode == 0);
        for (int u = 0; u < NU; u++) {   /* poison outputs */
            memset((void *)c.y[u], 0xA5, 4u * (uint32_t)(c.n / NU));
            Xil_DCacheFlushRange(c.y[u], 4u * (uint32_t)(c.n / NU));
        }
        uint32_t cyc = run(&c);
        uint64_t t0 = now();
        int errs = cyc ? check(&c, 0) : -1;
        uint32_t ref_us = us_since(t0);
        uint32_t macs_m = (uint32_t)((uint64_t)c.k * (uint32_t)c.n / 1000000u);
        xil_printf("  %s K=%4d N=%5d: %s  PL %5u us (%4u MB/s)  A9 reference %7u us for %u M MACs\r\n",
                   c.mode ? "int8   " : "ternary", c.k, c.n, errs == 0 ? "PASS" : "FAIL",
                   pl_us(cyc), mbps(bytes_streamed(&c), cyc), ref_us, macs_m);
        if (errs) fails++;
        heap = mark;
    }
    return fails;
}

/* Stream one ternary matrix of `rows_per_unit` x 2560 per unit. When the four unit
   slices start a multiple of 32 KB apart they can land in the same DDR bank. */
static void bw_run(uint32_t rows_per_unit, const char *note) {
    uint32_t mark = heap;
    cmd_t c;
    layout(&c, 0, 2560, 4 * (int)rows_per_unit, 0);
    fill_x(c.x, c.k);
    /* contiguous slices, like the packed image (layout() 4K-aligns each slice) */
    uint32_t slice = rows_per_unit * 640u;
    for (int u = 0; u < NU; u++) c.w[u] = c.w[0] + (uint32_t)u * slice;
    uint64_t t0 = now();
    uint32_t cyc = run(&c);
    uint32_t wall = us_since(t0);
    xil_printf("  rows/unit %5u (slice mod 32K = %5u): %u MB/s PL, wall %u us %s\r\n", rows_per_unit,
               slice % 32768u, mbps(bytes_streamed(&c), cyc), wall, note);
    heap = mark;
}

static void bandwidth(void) {
    xil_printf("\r\n[2] bandwidth vs unit-slice offset (64 MB ternary, 4 HP ports)\r\n");
    bw_run(25600, "(offsets 0 mod 32K)");
    bw_run(25664, "(8K apart)");
    bw_run(25632, "(20K apart)");
    bw_run(25616, "(10K apart)");
    bw_run(25728, "(16K apart)");
}

static void token(void) {
    xil_printf("\r\n[3] one decode step: 30 x {qkv, o, gate_up, down} + int8 LM head\r\n");
    uint32_t mark = heap;
    cmd_t qkv, o, gu, down, head;
    uint32_t x2560 = alloc(2560), x6912 = alloc(6912);
    fill_x(x2560, 2560);
    fill_x(x6912, 6912);
    layout(&qkv, 0, 2560, 3840, x2560);
    layout(&o, 0, 2560, 2560, x2560);
    layout(&gu, 0, 2560, 13824, x2560);
    layout(&down, 0, 6912, 2560, x6912);
    layout(&head, 1, 2560, 128256, x2560);   /* 328 MB, LM head */
    xil_printf("  buffers up to 0x%08x\r\n", heap);

    uint32_t cyc_layer[4] = {0}, wall_layers = 0, bytes_layers = 0;
    cmd_t *per_layer[4] = {&qkv, &o, &gu, &down};
    uint64_t t0 = now();
    for (int l = 0; l < 30; l++) {
        for (int i = 0; i < 4; i++) {
            uint32_t cyc = run(per_layer[i]);
            if (!cyc) { heap = mark; return; }
            cyc_layer[i] += cyc;
            bytes_layers += bytes_streamed(per_layer[i]);
        }
    }
    wall_layers = us_since(t0);
    t0 = now();
    uint32_t cyc_head = run(&head);
    uint32_t wall_head = us_since(t0);
    /* split: 4 x 32016 rows (slices 8 KB apart mod 32 KB) + 4 x 48 rows */
    cmd_t h1 = head, h2 = head;
    h1.n = 128064; h2.n = 192;
    for (int u = 0; u < NU; u++) {
        h1.w[u] = head.w[0] + (uint32_t)u * 32016u * 2560u;
        h2.w[u] = head.w[0] + 128064u * 2560u + (uint32_t)u * 48u * 2560u;
    }
    t0 = now();
    uint32_t cyc_split = run(&h1);
    cyc_split += run(&h2);
    uint32_t wall_split = us_since(t0);
    xil_printf("  LM head one command : %u us wall (%u us PL)\r\n", wall_head, pl_us(cyc_head));
    xil_printf("  LM head split rows  : %u us wall (%u us PL)\r\n", wall_split, pl_us(cyc_split));
    if (wall_split < wall_head) wall_head = wall_split;

    static const char *names[4] = {"qkv    ", "o      ", "gate_up", "down   "};
    for (int i = 0; i < 4; i++)
        xil_printf("  %s x30: %6u us\r\n", names[i], pl_us(cyc_layer[i]));
    xil_printf("  30 layers: %u us wall, %u MB/s\r\n", wall_layers,
               (uint32_t)((uint64_t)bytes_layers / (wall_layers ? wall_layers : 1)));
    uint32_t total = wall_layers + wall_head;
    xil_printf("  => matrix work per token: %u ms -> ceiling %u.%02u tok/s (before attention/norms)\r\n",
               total / 1000u, 1000000u / total, (100000000u / total) % 100u);
    heap = mark;
}

int main(void) {
    xil_printf("\r\n==== gemv_bench ====\r\n");
    uint32_t ver = csr_r(R_VERSION);
    xil_printf("engine version %08x %s\r\n", ver, ver == 0x47454D31u ? "(GEM1)" : "(UNEXPECTED)");
    if (ver != 0x47454D31u) { xil_printf("BENCH FAIL\r\n"); return 1; }
    int fails = correctness();
    bandwidth();
    token();
    xil_printf("\r\n%s\r\n", fails ? "BENCH FAIL" : "BENCH PASS");
    return 0;
}
