/* Driver for the PL GEMV engine (rtl/gemv_top.v). Register map: docs/ARCHITECTURE.md. */
#ifndef GEMV_H
#define GEMV_H

#include <stdint.h>
#include "xil_io.h"
#include "xil_cache.h"

#define GEMV_BASE    0x43C00000u
#define GEMV_VERSION 0x47454D31u   /* "GEM1" */
#define GEMV_NU      4
#define GEMV_KHZ     142857u   /* actual FCLK0: 1000 MHz / 7 */

enum { GEMV_TERNARY = 0, GEMV_INT8 = 1 };

static inline void     gemv_w(uint32_t off, uint32_t v) { Xil_Out32(GEMV_BASE + off, v); }
static inline uint32_t gemv_r(uint32_t off)             { return Xil_In32(GEMV_BASE + off); }

/*
 * y[n] = W[n x k] . x[k] as exact int32, W row-major in engine format at `w`
 * (contiguous; unit u takes rows [u*n/4, (u+1)*n/4)). x is int8 [k] at a
 * 128-byte aligned address; y is int32 [n], 128-byte aligned.
 *
 * gemv_start: x must already be flushed from the cache (gemv_run does it).
 * gemv_wait:  polls for completion; returns PL cycles, or 0 on an engine error.
 *             The caller invalidates y before reading it.
 */
static inline void gemv_start(int mode, const void *w, const int8_t *x, int32_t *y, int n, int k) {
    uint32_t rows = (uint32_t)n / GEMV_NU;
    uint32_t row_bytes = mode == GEMV_INT8 ? (uint32_t)k : (uint32_t)k / 4u;
    gemv_w(0x08, (uint32_t)mode);
    gemv_w(0x0C, (uint32_t)k);
    gemv_w(0x10, rows);
    gemv_w(0x14, (uint32_t)(uintptr_t)x);
    for (uint32_t u = 0; u < GEMV_NU; u++) {
        gemv_w(0x20 + 8u * u, (uint32_t)(uintptr_t)w + u * rows * row_bytes);
        gemv_w(0x24 + 8u * u, (uint32_t)(uintptr_t)y + u * rows * 4u);
    }
    gemv_w(0x00, 1u);
}

static inline int gemv_done(void) { return gemv_r(0x04) & 1u; }

static inline uint32_t gemv_wait(void) {
    uint32_t st;
    while (!((st = gemv_r(0x04)) & 1u)) { }
    return (st & 4u) ? 0u : gemv_r(0x18);
}

static inline uint32_t gemv_run(int mode, const void *w, const int8_t *x, int32_t *y, int n, int k) {
    Xil_DCacheFlushRange((INTPTR)x, (unsigned)k);
    gemv_start(mode, w, x, y, n, k);
    uint32_t c = gemv_wait();
    Xil_DCacheInvalidateRange((INTPTR)y, (unsigned)n * 4u);
    return c;
}

#endif
