/* NEON kernels for the A9 side of the forward pass, and cheap cache maintenance
 * for the engine's x / y buffers.
 *
 * Every function here has a scalar twin in tools/bitnet/engine.py; results match
 * it to float32 rounding (sums are re-associated into 16 lanes).
 */
#ifndef KERNELS_H
#define KERNELS_H

#include <arm_neon.h>
#include <math.h>
#include <stdint.h>

/* ---------------- cache maintenance (L1 only) ----------------
 * The 1 MB section holding x and y is mapped inner write-back / outer non-cacheable
 * (attribute 0x14DE6: TEX=100, C=0, B=1), so it never lives in the PL310 L2.
 * Keeping it coherent with the engine then needs only L1 line operations, which
 * are coprocessor instructions instead of uncached writes to the L2 controller. */
#define WORK_SECTION_ATTR 0x14DE6u
#define CACHE_LINE 32u

static inline void l1_clean(const void *p, uint32_t n) {
    uintptr_t a = (uintptr_t)p & ~(uintptr_t)(CACHE_LINE - 1), end = (uintptr_t)p + n;
    for (; a < end; a += CACHE_LINE) __asm__ volatile("mcr p15, 0, %0, c7, c10, 1" ::"r"(a) : "memory");
    __asm__ volatile("dsb" ::: "memory");
}
static inline void l1_invalidate(const void *p, uint32_t n) {
    uintptr_t a = (uintptr_t)p & ~(uintptr_t)(CACHE_LINE - 1), end = (uintptr_t)p + n;
    for (; a < end; a += CACHE_LINE) __asm__ volatile("mcr p15, 0, %0, c7, c6, 1" ::"r"(a) : "memory");
    __asm__ volatile("dsb" ::: "memory");
}

/* software prefetch: DDR latency is ~100+ ns per line, the loops below stream */
#define PF(p) __builtin_prefetch((const void *)(p))

/* ---------------- reductions ---------------- */
static inline float hsum4(float32x4_t v) {
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(s, s), 0);
}
static inline float hmax4(float32x4_t v) {
    float32x2_t s = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpmax_f32(s, s), 0);
}

/* n % 16 == 0 for every call site (2560, 6912) */
static inline float sum_sq(const float *x, int n) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    for (int i = 0; i < n; i += 16) {
        PF(x + i + 128);
        float32x4_t v0 = vld1q_f32(x + i), v1 = vld1q_f32(x + i + 4);
        float32x4_t v2 = vld1q_f32(x + i + 8), v3 = vld1q_f32(x + i + 12);
        a0 = vmlaq_f32(a0, v0, v0); a1 = vmlaq_f32(a1, v1, v1);
        a2 = vmlaq_f32(a2, v2, v2); a3 = vmlaq_f32(a3, v3, v3);
    }
    return hsum4(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
}

static inline float abs_max(const float *x, int n) {
    float32x4_t m0 = vdupq_n_f32(0), m1 = m0, m2 = m0, m3 = m0;
    for (int i = 0; i < n; i += 16) {
        PF(x + i + 128);
        m0 = vmaxq_f32(m0, vabsq_f32(vld1q_f32(x + i)));
        m1 = vmaxq_f32(m1, vabsq_f32(vld1q_f32(x + i + 4)));
        m2 = vmaxq_f32(m2, vabsq_f32(vld1q_f32(x + i + 8)));
        m3 = vmaxq_f32(m3, vabsq_f32(vld1q_f32(x + i + 12)));
    }
    return hmax4(vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3)));
}

/* out = x * (1/sqrt(mean(x^2) + eps)) * w */
static inline void k_rms_norm(float *out, const float *x, const float *w, int n, float eps) {
    float r = 1.0f / sqrtf(sum_sq(x, n) / (float)n + eps);
    for (int i = 0; i < n; i += 8) {
        PF(w + i + 128);
        PF(x + i + 128);
        float32x4_t v0 = vmulq_n_f32(vld1q_f32(x + i), r), v1 = vmulq_n_f32(vld1q_f32(x + i + 4), r);
        vst1q_f32(out + i, vmulq_f32(v0, vld1q_f32(w + i)));
        vst1q_f32(out + i + 4, vmulq_f32(v1, vld1q_f32(w + i + 4)));
    }
}

/* Per-token absmax int8 (bitnet.cpp quantize_row_i8_s): q = roundf(v * 127/amax).
 * roundf(x) == trunc(x + copysign(0.49999997f, x)) for |x| <= 127.5.
 * Returns the scale 127/amax (0 if v is all zero). */
static inline float k_quantize(int8_t *q, const float *v, int n) {
    float amax = abs_max(v, n);
    float s = amax > 0.0f ? 127.0f / amax : 0.0f;
    const uint32x4_t sign = vdupq_n_u32(0x80000000u);
    const float32x4_t half = vdupq_n_f32(0.49999997f);
    for (int i = 0; i < n; i += 16) {
        int32x4_t r[4];
        for (int j = 0; j < 4; j++) {
            float32x4_t x = vmulq_n_f32(vld1q_f32(v + i + 4 * j), s);
            float32x4_t h = vreinterpretq_f32_u32(
                vorrq_u32(vandq_u32(vreinterpretq_u32_f32(x), sign), vreinterpretq_u32_f32(half)));
            r[j] = vcvtq_s32_f32(vaddq_f32(x, h));
        }
        int16x8_t lo = vcombine_s16(vqmovn_s32(r[0]), vqmovn_s32(r[1]));
        int16x8_t hi = vcombine_s16(vqmovn_s32(r[2]), vqmovn_s32(r[3]));
        vst1q_s8(q + i, vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi)));
    }
    return s;
}

/* out[i] = (float)y[i] * f   (n % 8 == 0) */
static inline void k_dequant(float *out, const int32_t *y, int n, float f) {
    for (int i = 0; i < n; i += 8) {
        PF(y + i + 128);
        vst1q_f32(out + i, vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(y + i)), f));
        vst1q_f32(out + i + 4, vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(y + i + 4)), f));
    }
}

/* x[i] += (float)y[i] * f   (residual add fused with dequant) */
static inline void k_dequant_add(float *x, const int32_t *y, int n, float f) {
    for (int i = 0; i < n; i += 8) {
        PF(y + i + 128);
        PF(x + i + 128);
        vst1q_f32(x + i, vmlaq_n_f32(vld1q_f32(x + i), vcvtq_f32_s32(vld1q_s32(y + i)), f));
        vst1q_f32(x + i + 4, vmlaq_n_f32(vld1q_f32(x + i + 4), vcvtq_f32_s32(vld1q_s32(y + i + 4)), f));
    }
}

/* a[i] = relu(g[i] * fg)^2 * (u[i] * fu), straight from the engine's int32 outputs */
static inline void k_relu2_gate(float *a, const int32_t *g, const int32_t *u, int n, float fg, float fu) {
    const float32x4_t z = vdupq_n_f32(0);
    for (int i = 0; i < n; i += 4) {
        if (!(i & 7)) { PF(g + i + 128); PF(u + i + 128); }
        float32x4_t gv = vmaxq_f32(vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(g + i)), fg), z);
        float32x4_t uv = vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(u + i)), fu);
        vst1q_f32(a + i, vmulq_f32(vmulq_f32(gv, gv), uv));
    }
}

/* NEOX RoPE on nheads x 128: pairs (i, i+64) */
static inline void k_rope(float *v, int nheads, const float *cs, const float *sn) {
    for (int h = 0; h < nheads; h++) {
        float *x = v + h * 128;
        for (int i = 0; i < 64; i += 4) {
            float32x4_t x1 = vld1q_f32(x + i), x2 = vld1q_f32(x + i + 64);
            float32x4_t c = vld1q_f32(cs + i), s = vld1q_f32(sn + i);
            vst1q_f32(x + i, vmlsq_f32(vmulq_f32(x1, c), x2, s));
            vst1q_f32(x + i + 64, vmlaq_f32(vmulq_f32(x2, c), x1, s));
        }
    }
}

/* dot of two 128-float vectors */
static inline float dot128(const float *a, const float *b) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    for (int i = 0; i < 128; i += 16) {
        s0 = vmlaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vmlaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        s2 = vmlaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        s3 = vmlaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    return hsum4(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
}

/* o[0..127] += p * v[0..127] */
static inline void axpy128(float *o, float p, const float *v) {
    for (int i = 0; i < 128; i += 8) {
        vst1q_f32(o + i, vmlaq_n_f32(vld1q_f32(o + i), vld1q_f32(v + i), p));
        vst1q_f32(o + i + 4, vmlaq_n_f32(vld1q_f32(o + i + 4), vld1q_f32(v + i + 4), p));
    }
}

/* argmax over r of y[r] * scale[r] */
static inline int k_argmax_scaled(const int32_t *y, const float *scale, int n, float *best_v) {
    float32x4_t bv = vdupq_n_f32(-INFINITY);
    uint32x4_t bi = vdupq_n_u32(0), idx = {0, 1, 2, 3};
    const uint32x4_t four = vdupq_n_u32(4);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vmulq_f32(vcvtq_f32_s32(vld1q_s32(y + i)), vld1q_f32(scale + i));
        uint32x4_t gt = vcgtq_f32(v, bv);
        bv = vbslq_f32(gt, v, bv);
        bi = vbslq_u32(gt, idx, bi);
        idx = vaddq_u32(idx, four);
    }
    float lv[4];
    uint32_t li[4];
    vst1q_f32(lv, bv);
    vst1q_u32(li, bi);
    int best = (int)li[0];
    float b = lv[0];
    for (int j = 1; j < 4; j++)   /* ties -> lowest index, like the scalar loop */
        if (lv[j] > b || (lv[j] == b && (int)li[j] < best)) { b = lv[j]; best = (int)li[j]; }
    for (; i < n; i++) {
        float v = (float)y[i] * scale[i];
        if (v > b) { b = v; best = i; }
    }
    if (best_v) *best_v = b;
    return best;
}

#endif
