/* Driver for the PL GEMV engine v2 (rtl/gemv_top.v, "GEM2"). Register map:
 *   0x00 CTRL  (write 1: start)          0x04 STATUS (bit0 done, bit2 error)
 *   0x08 MODE  (0 = T5 ternary, 1 = int8) 0x0C X_BEATS (K_padded / 8)
 *   0x10 ROWS  (per unit)                0x14 X address
 *   0x18 CYCLES                          0x1C VERSION ("GEM2")
 *   0x20 + 8u  W address of unit u       0x24 + 8u  Y address of unit u
 *   0x40 + 4u  PROGRESS of unit u (write bursts acknowledged: 32 rows each)
 *   0x50 ROW_BEATS (beats per weight row)
 */
#ifndef GEMV_H
#define GEMV_H

#include <stdint.h>
#include "xil_io.h"

#define GEMV_BASE    0x43C00000u
#define GEMV_VERSION 0x47454D32u   /* "GEM2" */
#define GEMV_NU      4
#define GEMV_KHZ     142857u   /* actual FCLK0: 1000 MHz / 7 */

enum { GEMV_T5 = 0, GEMV_INT8 = 1 };

static inline void     gemv_w(uint32_t off, uint32_t v) { Xil_Out32(GEMV_BASE + off, v); }
static inline uint32_t gemv_r(uint32_t off)             { return Xil_In32(GEMV_BASE + off); }

/* K padded to whole T5 beats (40 weights) */
static inline int t5_k(int k) { return (k + 39) / 40 * 40; }

/*
 * y[n] = W[n x k] . x[k] as exact int32. W is split into 4 unit slices of n/4 rows,
 * slice u at w + u * ustride (engine format: T5 rows of t5_k(k)/5 bytes, or int8
 * rows of k bytes). x: int8, zero-padded to t5_k(k) for T5, 128-byte aligned, already
 * visible in DDR. y: int32 [n], unit u writes rows [u*n/4, (u+1)*n/4).
 */
static inline void gemv_start(int mode, const void *w, uint32_t ustride, const int8_t *x,
                              int32_t *y, int n, int k) {
    uint32_t rows = (uint32_t)n / GEMV_NU;
    uint32_t kp = mode == GEMV_T5 ? (uint32_t)t5_k(k) : (uint32_t)k;
    gemv_w(0x08, (uint32_t)mode);
    gemv_w(0x0C, kp / 8u);
    gemv_w(0x50, mode == GEMV_T5 ? kp / 40u : kp / 8u);
    gemv_w(0x10, rows);
    gemv_w(0x14, (uint32_t)(uintptr_t)x);
    for (uint32_t u = 0; u < GEMV_NU; u++) {
        gemv_w(0x20 + 8u * u, (uint32_t)(uintptr_t)w + u * ustride);
        gemv_w(0x24 + 8u * u, (uint32_t)(uintptr_t)y + u * rows * 4u);
    }
    gemv_w(0x00, 1u);
}

static inline int gemv_done(void) { return gemv_r(0x04) & 1u; }

/* rows of unit u already written to DDR (all of them once the run is done) */
static inline uint32_t gemv_rows_done(int u, uint32_t rows_per_unit) {
    uint32_t r = gemv_r(0x40 + 4u * (uint32_t)u) * 32u;
    return r < rows_per_unit ? r : rows_per_unit;
}

static inline uint32_t gemv_wait(void) {
    uint32_t st;
    while (!((st = gemv_r(0x04)) & 1u)) { }
    return (st & 4u) ? 0u : gemv_r(0x18);
}

#endif
