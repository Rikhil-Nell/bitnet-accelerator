/*
 * BitLinear server: ternary matmuls on the PL accelerator.
 *
 * Weights in DDR tile format (tools/bitnet/pack_weights.py).
 * UART1 @ ~921600 8N1, little-endian:
 *   'P' -> OK u32 version
 *   'S' -> OK u32 self-test
 *   'C' u32 addr u32 nbytes -> OK u32 checksum
 *   'G' u8 M u16 K u16 T {u32 addr u16 N}*M  i8 x[T][K] u32 xsum
 *       -> OK u32 ticks  i32 y[T][sum N] u32 ysum
 *   When all M matrices share the same N and K, one tile walk feeds every matrix (ffn gate+up).
 */
#include <stdint.h>
#include <string.h>
#include "xil_io.h"
#include "xparameters.h"
#include "xuartps_hw.h"

#define BITNET_BASE 0x43C00000U
#define UART_BASE   XPAR_XUARTPS_0_BASEADDR
#define VERSION     3U

#define BAUD_CD   18U
#define BAUD_BDIV 5U
#define GT_BASE   0xF8F00200U

#define MAX_K     6912
#define MAX_T     64
#define MAX_M     4
#define MAX_OUT   (MAX_T * 6912 * 2)

#define MMIO_CTRL   0x00U
#define MMIO_STATUS 0x04U
#define MMIO_ACT    0x10U
#define MMIO_WGT    0x14U
#define MMIO_OUTIDX 0x18U
#define MMIO_OUTDAT 0x1CU

static int8_t  g_x[MAX_T * MAX_K];
static int32_t g_y[MAX_OUT];

static inline void mmio_write(uint32_t off, uint32_t v) { Xil_Out32(BITNET_BASE + off, v); }
static inline uint32_t mmio_read(uint32_t off) { return Xil_In32(BITNET_BASE + off); }

static void uart_setup(void) {
    Xil_Out32(UART_BASE + XUARTPS_CR_OFFSET, XUARTPS_CR_TX_DIS | XUARTPS_CR_RX_DIS);
    Xil_Out32(UART_BASE + XUARTPS_BAUDGEN_OFFSET, BAUD_CD);
    Xil_Out32(UART_BASE + XUARTPS_BAUDDIV_OFFSET, BAUD_BDIV);
    Xil_Out32(UART_BASE + XUARTPS_CR_OFFSET, XUARTPS_CR_TXRST | XUARTPS_CR_RXRST);
    Xil_Out32(UART_BASE + XUARTPS_CR_OFFSET, XUARTPS_CR_TX_EN | XUARTPS_CR_RX_EN);
}

static uint8_t rx8(void) { return XUartPs_RecvByte(UART_BASE); }
static void tx8(uint8_t b) { XUartPs_SendByte(UART_BASE, b); }
static uint32_t rx16(void) { uint32_t v = rx8(); return v | ((uint32_t)rx8() << 8); }
static uint32_t rx32(void) { uint32_t v = rx16(); return v | (rx16() << 16); }
static void tx32(uint32_t v) { for (int i = 0; i < 4; i++) tx8((uint8_t)(v >> (8 * i))); }
static void tx_ok(void) { tx8('O'); tx8('K'); }
static void tx_err(uint32_t code) { tx8('E'); tx8('R'); tx32(code); }

static void timer_start(void) {
    Xil_Out32(GT_BASE + 0x8, 0);
    Xil_Out32(GT_BASE + 0x0, 0);
    Xil_Out32(GT_BASE + 0x4, 0);
    Xil_Out32(GT_BASE + 0x8, 1);
}
static uint32_t timer_low(void) { return Xil_In32(GT_BASE + 0x0); }

static int run_tile(void) {
    mmio_write(MMIO_CTRL, 1U);
    /* ~80 PL cycles @ 50 MHz; status poll is the hot loop — cap spins, no 100k timeout. */
    for (int i = 0; i < 512; i++) {
        if (mmio_read(MMIO_STATUS) & 0x1U) return 0;
    }
    return -1;
}

static void load_act_tile(const int8_t *xt) {
    for (int j = 0; j < 8; j++)
        mmio_write(MMIO_ACT, (uint32_t)(uint8_t)xt[j] | ((uint32_t)j << 8));
}

static void load_wgt_tile(const uint16_t *tile) {
    for (int r = 0; r < 8; r++)
        mmio_write(MMIO_WGT, (uint32_t)tile[r] | ((uint32_t)r << 16));
}

static void accum_outputs(int32_t *yt) {
    for (int r = 0; r < 8; r++) {
        mmio_write(MMIO_OUTIDX, (uint32_t)r);
        yt[r] += (int32_t)mmio_read(MMIO_OUTDAT);
    }
}

static int gemm_one(uint32_t addr, int n, int k, int t_count, const int8_t *x, int32_t *y, int y_stride) {
    const uint16_t *w = (const uint16_t *)(uintptr_t)addr;
    const int ng = n / 8, nb = k / 8;
    for (int t = 0; t < t_count; t++)
        memset(&y[t * y_stride], 0, (size_t)n * sizeof(int32_t));

    for (int cb = 0; cb < nb; cb++) {
        for (int rg = 0; rg < ng; rg++) {
            const uint16_t *tile = w + ((uint32_t)cb * ng + rg) * 8;
            load_wgt_tile(tile);
            for (int t = 0; t < t_count; t++) {
                load_act_tile(&x[t * k + cb * 8]);
                if (run_tile() != 0) return -1;
                accum_outputs(&y[t * y_stride + rg * 8]);
            }
        }
    }
    return 0;
}

/* Same N,K for every matrix: load activations once per tile, run each weight tile. */
static int gemm_fused_same(int m, const uint32_t *addr, int n, int k, int t_count, const int8_t *x,
                           int32_t *y, int y_stride) {
    const int ng = n / 8, nb = k / 8;
    for (int t = 0; t < t_count; t++)
        memset(&y[t * y_stride], 0, (size_t)(m * n) * sizeof(int32_t));

    for (int cb = 0; cb < nb; cb++) {
        for (int rg = 0; rg < ng; rg++) {
            for (int t = 0; t < t_count; t++) {
                load_act_tile(&x[t * k + cb * 8]);
                for (int mi = 0; mi < m; mi++) {
                    const uint16_t *w = (const uint16_t *)(uintptr_t)addr[mi];
                    const uint16_t *tile = w + ((uint32_t)cb * ng + rg) * 8;
                    load_wgt_tile(tile);
                    if (run_tile() != 0) return -1;
                    accum_outputs(&y[t * y_stride + mi * n + rg * 8]);
                }
            }
        }
    }
    return 0;
}

static int all_same_shape(int m, const uint32_t *n) {
    for (int i = 1; i < m; i++)
        if (n[i] != n[0]) return 0;
    return 1;
}

static uint32_t self_test(void) {
    static const int8_t act[8] = {5, -3, 7, 2, 4, -2, 6, 1};
    static const uint16_t wgt[8] = {0x2461, 0x9214, 0x4185, 0x1852, 0x4918, 0x1249, 0x6126, 0x8584};
    static const int32_t exp[8] = {-8, 5, 5, 12, 17, 12, -16, -4};
    load_act_tile(act);
    load_wgt_tile(wgt);
    if (run_tile() != 0) return 0;
    for (int i = 0; i < 8; i++) {
        mmio_write(MMIO_OUTIDX, (uint32_t)i);
        if ((int32_t)mmio_read(MMIO_OUTDAT) != exp[i]) return 0;
    }
    return 1;
}

static void cmd_gemm(void) {
    uint32_t m = rx8(), k = rx16(), t = rx16();
    uint32_t addr[MAX_M], n[MAX_M], total_n = 0;
    for (uint32_t i = 0; i < m && i < MAX_M; i++) {
        addr[i] = rx32();
        n[i] = rx16();
        total_n += n[i];
    }
    uint32_t bad = (m == 0 || m > MAX_M) ? 1 : (k == 0 || k > MAX_K || (k % 8)) ? 2
                 : (t == 0 || t > MAX_T) ? 3 : (t * total_n > MAX_OUT) ? 4 : 0;
    if (bad) { tx_err(bad); return; }
    for (uint32_t i = 0; i < m; i++) if (n[i] % 8) { tx_err(5); return; }

    uint32_t sum = 0;
    for (uint32_t i = 0; i < t * k; i++) { g_x[i] = (int8_t)rx8(); sum += (uint8_t)g_x[i]; }
    if (rx32() != sum) { tx_err(6); return; }

    timer_start();
    int rc;
    if (m > 1 && all_same_shape((int)m, n)) {
        rc = gemm_fused_same((int)m, addr, (int)n[0], (int)k, (int)t, g_x, g_y, (int)total_n);
    } else {
        uint32_t col = 0;
        rc = 0;
        for (uint32_t i = 0; i < m && rc == 0; i++) {
            rc = gemm_one(addr[i], (int)n[i], (int)k, (int)t, g_x, &g_y[col], (int)total_n);
            col += n[i];
        }
    }
    if (rc != 0) { tx_err(7); return; }
    uint32_t ticks = timer_low();

    tx_ok();
    tx32(ticks);
    uint32_t ysum = 0;
    for (uint32_t i = 0; i < t * total_n; i++) {
        uint32_t v = (uint32_t)g_y[i];
        tx32(v);
        ysum += v;
    }
    tx32(ysum);
}

int main(void) {
    uart_setup();
    for (;;) {
        if (rx8() != 0xB1) continue;
        if (rx8() != 0x7A) continue;
        uint8_t cmd = rx8();
        if (cmd == 'P') {
            tx_ok();
            tx32(VERSION);
        } else if (cmd == 'S') {
            tx_ok();
            tx32(self_test());
        } else if (cmd == 'C') {
            uint32_t a = rx32(), nbytes = rx32(), s = 0;
            const uint32_t *p = (const uint32_t *)(uintptr_t)a;
            for (uint32_t i = 0; i < nbytes / 4; i++) s += p[i];
            tx_ok();
            tx32(s);
        } else if (cmd == 'G') {
            cmd_gemm();
        } else {
            tx_err(0xFF);
        }
    }
}
