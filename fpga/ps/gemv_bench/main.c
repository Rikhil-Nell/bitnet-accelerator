/*
 * gemv_bench: bring-up test for the PL GEMV engine v2 (rtl/gemv_top.v, "GEM2").
 *
 *   1. correctness: random T5 ternary / int8 matrices at the model's shapes,
 *      every output checked against the A9 computing the same product
 *   2. progress:    rows reported done while the engine runs are already in DDR
 *   3. bandwidth:   one T5 matrix streamed through all 4 HP ports
 *   4. token:       the engine commands of one BitNet decode step
 *                   (30 x {qkv, o, gate_up, down} + T5 coarse head), timed
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
#define R_XBEATS    0x0Cu
#define R_RBEATS    0x50u
#define R_PROG(u)   (0x40u + 4u * (u))
#define R_ROWS      0x10u
#define R_X         0x14u
#define R_CYCLES    0x18u
#define R_VERSION   0x1Cu
#define R_W(u)      (0x20u + 8u * (u))
#define R_Y(u)      (0x24u + 8u * (u))
#define NU          4
/* Actual FCLK0: 1000 MHz IO PLL / 7 (150 was requested; PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ). */
static uint32_t PL_KHZ = 142857u;   /* follows set_fclk() */

/* Engine clock for the bench: 1000 MHz / BENCH_FCLK_DIV. 12 (83 MHz) is safe on USB power;
   7 (142.9 MHz, full bandwidth) needs the 5 V barrel supply. BENCH_POWER_SWEEP=1 adds a
   clock sweep that shows where a weak supply browns the board out (it will reset). */
#ifndef BENCH_FCLK_DIV
#define BENCH_FCLK_DIV 12
#endif
#ifndef BENCH_POWER_SWEEP
#define BENCH_POWER_SWEEP 0
#endif
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

static int t5_k(int k) { return (k + 39) / 40 * 40; }
static uint32_t row_bytes_of(int mode, int k) { return mode ? (uint32_t)k : (uint32_t)t5_k(k) / 5u; }
static uint32_t x_bytes_of(int mode, int k) { return mode ? (uint32_t)k : (uint32_t)t5_k(k); }

/* Lay out a matrix: x, then NU weight slices, then NU output slices. */
static void layout(cmd_t *c, int mode, int k, int n, uint32_t share_x) {
    int r = n / NU;
    uint32_t row_bytes = row_bytes_of(mode, k);
    c->mode = mode; c->k = k; c->n = n;
    c->x = share_x ? share_x : alloc(x_bytes_of(mode, k));
    for (int u = 0; u < NU; u++) c->w[u] = alloc(row_bytes * (uint32_t)r);
    for (int u = 0; u < NU; u++) c->y[u] = alloc(4u * (uint32_t)r);
}

/* Program and start one command. */
static void launch(const cmd_t *c) {
    int r = c->n / NU;
    uint32_t xb = x_bytes_of(c->mode, c->k);
    Xil_DCacheFlushRange(c->x, xb);
    csr_w(R_MODE, (uint32_t)c->mode);
    csr_w(R_XBEATS, xb / 8u);
    csr_w(R_RBEATS, row_bytes_of(c->mode, c->k) / 8u);
    csr_w(R_ROWS, (uint32_t)r);
    csr_w(R_X, c->x);
    for (int u = 0; u < NU; u++) { csr_w(R_W(u), c->w[u]); csr_w(R_Y(u), c->y[u]); }
    csr_w(R_CTRL, 1u);
}

/* Program and run one command; returns PL cycles, or 0 on error. */
static uint32_t run(const cmd_t *c) {
    int r = c->n / NU;
    launch(c);
    uint32_t st, spins = 0;
    while (!((st = csr_r(R_STATUS)) & 1u)) {
        if (++spins > 200000000u) { xil_printf("ERROR: timeout, STATUS=%08x\r\n", st); return 0; }
    }
    if (st & 4u) { xil_printf("ERROR: AXI error, STATUS=%08x\r\n", st); return 0; }
    for (int u = 0; u < NU; u++) Xil_DCacheInvalidateRange(c->y[u], 4u * (uint32_t)r);
    return csr_r(R_CYCLES);
}

/* random weights: T5 bytes are base-3 digit groups in [0, 242] */
static void fill_random(uint32_t addr, uint32_t bytes, int ternary) {
    uint8_t *p = (uint8_t *)addr;
    for (uint32_t i = 0; i < bytes; i++) {
        uint32_t v = rnd();
        p[i] = ternary ? (uint8_t)(v % 243u) : (uint8_t)v;
    }
    Xil_DCacheFlushRange(addr, bytes);
}

/* x[0..k) random with extremes; zero padding up to the T5 width */
static void fill_x(uint32_t addr, int k) {
    int8_t *x = (int8_t *)addr;
    for (int i = 0; i < k; i++) x[i] = (int8_t)rnd();
    for (int i = k; i < t5_k(k); i++) x[i] = 0;
    x[0] = -128; x[k - 1] = 127;
}

/* A9 reference for one row; returns the dot product. */
static int32_t ref_row(const cmd_t *c, int u, int row) {
    const int8_t *x = (const int8_t *)c->x;
    int32_t acc = 0;
    if (c->mode == 0) {   /* T5: byte b holds weights 5b..5b+4, base 3, digit = w + 1 */
        const uint8_t *w = (const uint8_t *)(c->w[u] + (uint32_t)row * row_bytes_of(0, c->k));
        for (int b = 0; b < t5_k(c->k) / 5; b++) {
            uint32_t v = w[b];
            for (int j = 0; j < 5; j++) {
                int t = (int)(v % 3u) - 1;
                v /= 3u;
                acc += t * x[5 * b + j];
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
    return row_bytes_of(c->mode, c->k) * (uint32_t)c->n + (uint32_t)NU * x_bytes_of(c->mode, c->k)
           + 4u * (uint32_t)c->n;
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
        uint32_t wbytes = row_bytes_of(c.mode, c.k) * (uint32_t)(c.n / NU);
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
                   c.mode ? "int8   " : "T5     ", c.k, c.n, errs == 0 ? "PASS" : "FAIL",
                   pl_us(cyc), mbps(bytes_streamed(&c), cyc), ref_us, macs_m);
        if (errs) fails++;
        heap = mark;
    }
    return fails;
}

/* While a large command runs, every row that PROGRESS reports as done must already
   hold its final value in DDR (the firmware overlaps its work on that guarantee). */
/* GEM3: two activation vectors per weight pass (NB = 1). Output word per row =
   {y_b, y_a}; both checked against the A9. Also times NB=2 vs NB=1 on the same matrix. */
static int batch2(void) {
    xil_printf("\r\n[2b] two activation vectors per pass (speculative verification)\r\n");
    static const int shapes[][2] = {{2560, 3840}, {6912, 2560}, {2560, 13824}};
    int fails = 0;
    for (unsigned s = 0; s < 3; s++) {
        uint32_t mark = heap;
        cmd_t c;
        layout(&c, 0, shapes[s][0], shapes[s][1], 0);
        int kp = t5_k(c.k), r = c.n / NU;
        uint32_t x2 = alloc(2u * (uint32_t)kp);       /* x_a then x_b, each kp bytes */
        fill_x(x2, c.k);
        fill_x(x2 + (uint32_t)kp, c.k);
        c.x = x2;
        for (int u = 0; u < NU; u++) {
            fill_random(c.w[u], row_bytes_of(0, c.k) * (uint32_t)r, 1);
            c.y[u] = alloc(8u * (uint32_t)r);
            memset((void *)c.y[u], 0xA5, 8u * (uint32_t)r);
            Xil_DCacheFlushRange(c.y[u], 8u * (uint32_t)r);
        }
        Xil_DCacheFlushRange(x2, 2u * (uint32_t)kp);
        csr_w(0x54u, 1u);
        uint32_t cyc2 = run(&c);
        csr_w(0x54u, 0u);
        int errs = cyc2 ? 0 : 1;
        for (int u = 0; u < NU && cyc2; u++) {
            Xil_DCacheInvalidateRange(c.y[u], 8u * (uint32_t)r);
            const int32_t *y = (const int32_t *)c.y[u];
            for (int i = 0; i < r; i += 3) {
                c.x = x2;
                int32_t ea = ref_row(&c, u, i);
                c.x = x2 + (uint32_t)kp;
                int32_t eb = ref_row(&c, u, i);
                c.x = x2;
                if (y[2 * i] != ea || y[2 * i + 1] != eb) {
                    if (errs < 5) xil_printf("  MISMATCH unit %d row %d: got %d,%d exp %d,%d\r\n", u, i, y[2 * i], y[2 * i + 1], ea, eb);
                    errs++;
                }
            }
        }
        for (int u = 0; u < NU; u++) c.y[u] = alloc(4u * (uint32_t)r);
        uint32_t cyc1 = run(&c);
        xil_printf("  T5 K=%4d N=%5d x2: %s  PL %5u us for two vectors, %5u us for one\r\n", c.k, c.n,
                   errs ? "FAIL" : "PASS", pl_us(cyc2), pl_us(cyc1));
        fails += errs != 0;
        heap = mark;
    }
    return fails;
}

static int progress(void) {
    xil_printf("\r\n[2] progress counters during a run (gate_up shape)\r\n");
    uint32_t mark = heap;
    cmd_t c;
    layout(&c, 0, 2560, 13824, 0);
    fill_x(c.x, c.k);
    for (int u = 0; u < NU; u++) fill_random(c.w[u], row_bytes_of(0, c.k) * (uint32_t)(c.n / NU), 1);
    int r = c.n / NU, errs = 0, samples = 0;
    for (int u = 0; u < NU; u++) {
        memset((void *)c.y[u], 0xA5, 4u * (uint32_t)r);
        Xil_DCacheFlushRange(c.y[u], 4u * (uint32_t)r);
    }
    int checked[NU] = {0};
    launch(&c);
    for (;;) {
        int done = csr_r(R_STATUS) & 1u;
        for (int u = 0; u < NU; u++) {
            uint32_t rows = csr_r(R_PROG(u)) * 32u;
            if (rows > (uint32_t)r) rows = (uint32_t)r;
            if ((int)rows > checked[u]) {
                Xil_DCacheInvalidateRange(c.y[u] + 4u * (uint32_t)checked[u], 4u * (rows - (uint32_t)checked[u]));
                const int32_t *y = (const int32_t *)c.y[u];
                for (int i = checked[u]; i < (int)rows; i += 7) {   /* spot-check while running */
                    if (y[i] != ref_row(&c, u, i)) {
                        if (errs < 5) xil_printf("  EARLY READ WRONG unit %d row %d\r\n", u, i);
                        errs++;
                    }
                }
                checked[u] = (int)rows;
                samples++;
            }
        }
        if (done) break;
    }
    xil_printf("  %d progress updates observed, final rows/unit %d: %s\r\n", samples, checked[0],
               errs == 0 && checked[0] == r ? "PASS" : "FAIL");
    heap = mark;
    return errs == 0 && checked[0] == r ? 0 : 1;
}

/* Stream one T5 matrix of `rows_per_unit` x 2560 per unit. When the four unit
   slices start a multiple of 32 KB apart they can land in the same DDR bank. */
static void bw_run(uint32_t rows_per_unit, const char *note) {
    uint32_t mark = heap;
    cmd_t c;
    layout(&c, 0, 2560, 4 * (int)rows_per_unit, 0);
    fill_x(c.x, c.k);
    /* contiguous slices (layout() 4K-aligns each slice) */
    uint32_t slice = rows_per_unit * 512u;
    for (int u = 0; u < NU; u++) c.w[u] = c.w[0] + (uint32_t)u * slice;
    uint64_t t0 = now();
    uint32_t cyc = run(&c);
    uint32_t wall = us_since(t0);
    xil_printf("  rows/unit %5u (slice mod 32K = %5u): %u MB/s PL, wall %u us %s\r\n", rows_per_unit,
               slice % 32768u, mbps(bytes_streamed(&c), cyc), wall, note);
    heap = mark;
}

static void bandwidth(void) {
    xil_printf("\r\n[3] bandwidth vs unit-slice offset (52 MB T5, 4 HP ports)\r\n");
    bw_run(25600, "(offsets 0 mod 32K)");
    bw_run(25616, "(8K apart)");
    bw_run(25640, "(20K apart)");
    bw_run(25620, "(10K apart)");
    bw_run(25632, "(16K apart)");
}

static void token(void) {
    xil_printf("\r\n[4] one decode step: 30 x {qkv, o, gate_up, down} + T5 coarse head\r\n");
    uint32_t mark = heap;
    cmd_t qkv, o, gu, down, head;
    uint32_t x2560 = alloc(2560), x6912 = alloc(6920);
    fill_x(x2560, 2560);
    fill_x(x6912, 6912);
    layout(&qkv, 0, 2560, 3840, x2560);
    layout(&o, 0, 2560, 2560, x2560);
    layout(&gu, 0, 2560, 13824, x2560);
    layout(&down, 0, 6912, 2560, x6912);
    layout(&head, 0, 2560, 128256, x2560);   /* 66 MB, T5 coarse head */
    /* unit slices as in the v2 image: stride = round_up(slice, 32K) + 8K */
    cmd_t *all[5] = {&qkv, &o, &gu, &down, &head};
    for (int i = 0; i < 5; i++) {
        uint32_t sl = row_bytes_of(0, all[i]->k) * (uint32_t)(all[i]->n / NU);
        uint32_t stride = (sl + 32767u) / 32768u * 32768u + 8192u;
        uint32_t base = alloc(stride * NU);
        for (int u = 0; u < NU; u++) all[i]->w[u] = base + (uint32_t)u * stride;
    }
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
    xil_printf("  coarse head (T5)    : %u us wall (%u us PL)\r\n", wall_head, pl_us(cyc_head));

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

/* FCLK0 = 1000 MHz IO PLL / div. Lower clock = lower bandwidth and power. */
static void set_fclk(uint32_t div) {
    Xil_Out32(0xF8000008u, 0xDF0Du);
    uint32_t v = Xil_In32(0xF8000170u);
    Xil_Out32(0xF8000170u, (v & ~(0x3Fu << 8)) | (div << 8));
    Xil_Out32(0xF8000004u, 0x767Bu);
    PL_KHZ = 1000000u / div;
}

#if BENCH_POWER_SWEEP
/* Power sweep: stream a 52 MB T5 matrix repeatedly at rising clocks. On a weak
   supply the board browns out at some step; the last line printed says where. */
static void power_sweep(void) {
    static const uint32_t divs[] = {12, 10, 9, 8, 7};
    xil_printf("\r\n[0] power sweep: 20 x 52 MB per clock step\r\n");
    for (unsigned i = 0; i < 5; i++) {
        set_fclk(divs[i]);
        xil_printf("  FCLK %u kHz ... ", PL_KHZ);
        uint32_t mark = heap;
        cmd_t c;
        layout(&c, 0, 2560, 4 * 25616, 0);
        fill_x(c.x, c.k);
        uint32_t slice = 25616u * 512u, best = 0;
        for (int u = 0; u < NU; u++) c.w[u] = c.w[0] + (uint32_t)u * slice;
        for (int r = 0; r < 20; r++) {
            uint32_t cyc = run(&c);
            uint32_t m = mbps(bytes_streamed(&c), cyc);
            if (m > best) best = m;
        }
        xil_printf("survived, %u MB/s\r\n", best);
        heap = mark;
    }
}
#endif

int main(void) {
    xil_printf("\r\n==== gemv_bench ====\r\n");
#if BENCH_POWER_SWEEP
    power_sweep();
#endif
    set_fclk(BENCH_FCLK_DIV);
    xil_printf("engine clock %u kHz\r\n", PL_KHZ);
    uint32_t ver = csr_r(R_VERSION);
    int v3 = ver == 0x47454D33u;
    xil_printf("engine version %08x %s\r\n", ver, ver == 0x47454D32u ? "(GEM2)" : v3 ? "(GEM3)" : "(UNEXPECTED)");
    if (ver != 0x47454D32u && !v3) { xil_printf("BENCH FAIL\r\n"); return 1; }
    int fails = correctness();
    fails += progress();
    if (v3) fails += batch2();
    bandwidth();
    token();
    xil_printf("\r\n%s\r\n", fails ? "BENCH FAIL" : "BENCH PASS");
    return 0;
}
