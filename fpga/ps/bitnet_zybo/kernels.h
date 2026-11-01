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

/* The elementwise kernels below work on 16 elements per step with every load issued
 * before any arithmetic and every store after it, and `restrict` pointers. The A9's
 * NEON pipe is in-order: written as load->convert->multiply->store per vector, each
 * step waits out the full latency chain (~12 cycles per 4 elements, measured). */
#define R_ __restrict

/* out[i] = (float)y[i] * f   (n % 16 == 0) */
static inline void k_dequant(float *R_ out, const int32_t *R_ y, int n, float f) {
    for (int i = 0; i < n; i += 16) {
        PF(y + i + 128);
        int32x4_t y0 = vld1q_s32(y + i), y1 = vld1q_s32(y + i + 4), y2 = vld1q_s32(y + i + 8), y3 = vld1q_s32(y + i + 12);
        float32x4_t a0 = vcvtq_f32_s32(y0), a1 = vcvtq_f32_s32(y1), a2 = vcvtq_f32_s32(y2), a3 = vcvtq_f32_s32(y3);
        a0 = vmulq_n_f32(a0, f); a1 = vmulq_n_f32(a1, f); a2 = vmulq_n_f32(a2, f); a3 = vmulq_n_f32(a3, f);
        vst1q_f32(out + i, a0); vst1q_f32(out + i + 4, a1); vst1q_f32(out + i + 8, a2); vst1q_f32(out + i + 12, a3);
    }
}

/* ---------------- fused epilogues for the streamed (overlapped) forward pass ----------------
 * Each processes one chunk of engine output rows as soon as they are in DDR and
 * accumulates the statistics the next quantization needs, so after the engine is
 * done only k_quantize_s() is left. n % 16 == 0 except where noted.
 *
 * quantize(rms_norm(x) * w) == quantize(x * w) exactly in real arithmetic: the
 * 1/rms factor is a positive scalar and cancels in the absmax scaling. It only
 * enters the dequantization scale, as a scalar. */
typedef struct { float32x4_t ss, amax; } stats_t;
static inline void stats_init(stats_t *st) { st->ss = vdupq_n_f32(0); st->amax = vdupq_n_f32(0); }
static inline float stats_ss(const stats_t *st) { return hsum4(st->ss); }
static inline float stats_amax(const stats_t *st) { return hmax4(st->amax); }

/* x += y * f; u = x * w; ss += x^2; amax = max |u|   (residual add + next norm) */
static inline void k_resid_norm(float *R_ x, float *R_ u, const int32_t *R_ y, const float *R_ w,
                                int n, float f, stats_t *st) {
    float32x4_t ss = st->ss, am = st->amax;
    for (int i = 0; i < n; i += 8) {
        PF(y + i + 128);
        PF(w + i + 128);
        int32x4_t y0 = vld1q_s32(y + i), y1 = vld1q_s32(y + i + 4);
        float32x4_t x0 = vld1q_f32(x + i), x1 = vld1q_f32(x + i + 4);
        float32x4_t w0 = vld1q_f32(w + i), w1 = vld1q_f32(w + i + 4);
        float32x4_t a0 = vcvtq_f32_s32(y0), a1 = vcvtq_f32_s32(y1);
        x0 = vmlaq_n_f32(x0, a0, f); x1 = vmlaq_n_f32(x1, a1, f);
        float32x4_t u0 = vmulq_f32(x0, w0), u1 = vmulq_f32(x1, w1);
        ss = vmlaq_f32(ss, x0, x0); ss = vmlaq_f32(ss, x1, x1);
        am = vmaxq_f32(am, vabsq_f32(u0)); am = vmaxq_f32(am, vabsq_f32(u1));
        vst1q_f32(x + i, x0); vst1q_f32(x + i + 4, x1);
        vst1q_f32(u + i, u0); vst1q_f32(u + i + 4, u1);
    }
    st->ss = ss; st->amax = am;
}

/* a = relu(g * fg)^2 * (up * fu); u = a * w; ss += a^2; amax = max |u|   (FFN + sub-norm) */
static inline void k_gate_norm(float *R_ u, const int32_t *R_ g, const int32_t *R_ up, const float *R_ w,
                               int n, float fg, float fu, stats_t *st) {
    const float32x4_t z = vdupq_n_f32(0);
    float32x4_t ss = st->ss, am = st->amax;
    for (int i = 0; i < n; i += 8) {
        PF(g + i + 128);
        PF(up + i + 128);
        PF(w + i + 128);
        int32x4_t g0 = vld1q_s32(g + i), g1 = vld1q_s32(g + i + 4), p0 = vld1q_s32(up + i), p1 = vld1q_s32(up + i + 4);
        float32x4_t w0 = vld1q_f32(w + i), w1 = vld1q_f32(w + i + 4);
        float32x4_t gf0 = vcvtq_f32_s32(g0), gf1 = vcvtq_f32_s32(g1), uf0 = vcvtq_f32_s32(p0), uf1 = vcvtq_f32_s32(p1);
        gf0 = vmaxq_f32(vmulq_n_f32(gf0, fg), z); gf1 = vmaxq_f32(vmulq_n_f32(gf1, fg), z);
        uf0 = vmulq_n_f32(uf0, fu); uf1 = vmulq_n_f32(uf1, fu);
        float32x4_t a0 = vmulq_f32(vmulq_f32(gf0, gf0), uf0), a1 = vmulq_f32(vmulq_f32(gf1, gf1), uf1);
        float32x4_t v0 = vmulq_f32(a0, w0), v1 = vmulq_f32(a1, w1);
        ss = vmlaq_f32(ss, a0, a0); ss = vmlaq_f32(ss, a1, a1);
        am = vmaxq_f32(am, vabsq_f32(v0)); am = vmaxq_f32(am, vabsq_f32(v1));
        vst1q_f32(u + i, v0); vst1q_f32(u + i + 4, v1);
    }
    st->ss = ss; st->amax = am;
}

/* u = x * w; ss += x^2; amax = max |u|   (a norm whose input is already in memory) */
static inline void k_norm_stats(float *R_ u, const float *R_ x, const float *R_ w, int n, stats_t *st) {
    float32x4_t ss = st->ss, am = st->amax;
    for (int i = 0; i < n; i += 8) {
        PF(w + i + 128);
        float32x4_t x0 = vld1q_f32(x + i), x1 = vld1q_f32(x + i + 4);
        float32x4_t w0 = vld1q_f32(w + i), w1 = vld1q_f32(w + i + 4);
        float32x4_t u0 = vmulq_f32(x0, w0), u1 = vmulq_f32(x1, w1);
        ss = vmlaq_f32(ss, x0, x0); ss = vmlaq_f32(ss, x1, x1);
        am = vmaxq_f32(am, vabsq_f32(u0)); am = vmaxq_f32(am, vabsq_f32(u1));
        vst1q_f32(u + i, u0); vst1q_f32(u + i + 4, u1);
    }
    st->ss = ss; st->amax = am;
}

/* q = round(u * s) with roundf semantics (bitnet.cpp quantize_row_i8_s):
 * roundf(x) == trunc(x + copysign(0.49999997f, x)) for |x| <= 127.5 */
static inline void k_quantize_s(int8_t *R_ q, const float *R_ v, int n, float s) {
    const uint32x4_t sign = vdupq_n_u32(0x80000000u);
    const uint32x4_t half = vreinterpretq_u32_f32(vdupq_n_f32(0.49999997f));
    for (int i = 0; i < n; i += 16) {
        float32x4_t x0 = vld1q_f32(v + i), x1 = vld1q_f32(v + i + 4), x2 = vld1q_f32(v + i + 8), x3 = vld1q_f32(v + i + 12);
        x0 = vmulq_n_f32(x0, s); x1 = vmulq_n_f32(x1, s); x2 = vmulq_n_f32(x2, s); x3 = vmulq_n_f32(x3, s);
        float32x4_t h0 = vreinterpretq_f32_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_f32(x0), sign), half));
        float32x4_t h1 = vreinterpretq_f32_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_f32(x1), sign), half));
        float32x4_t h2 = vreinterpretq_f32_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_f32(x2), sign), half));
        float32x4_t h3 = vreinterpretq_f32_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_f32(x3), sign), half));
        int32x4_t r0 = vcvtq_s32_f32(vaddq_f32(x0, h0)), r1 = vcvtq_s32_f32(vaddq_f32(x1, h1));
        int32x4_t r2 = vcvtq_s32_f32(vaddq_f32(x2, h2)), r3 = vcvtq_s32_f32(vaddq_f32(x3, h3));
        int16x8_t lo = vcombine_s16(vqmovn_s32(r0), vqmovn_s32(r1));
        int16x8_t hi = vcombine_s16(vqmovn_s32(r2), vqmovn_s32(r3));
        vst1q_s8(q + i, vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi)));
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

/* exp(x) for x <= 0 (cephes / neon_mathfun polynomial, ~1 ulp); x < -87 -> 0 */
static inline float32x4_t exp_q(float32x4_t x) {
    const float32x4_t lo = vdupq_n_f32(-87.0f);
    uint32x4_t tiny = vcltq_f32(x, lo);
    x = vmaxq_f32(x, lo);
    float32x4_t fx = vmlaq_n_f32(vdupq_n_f32(0.5f), x, 1.44269504088896341f);
    float32x4_t t = vcvtq_f32_s32(vcvtq_s32_f32(fx));          /* trunc */
    t = vsubq_f32(t, vreinterpretq_f32_u32(vandq_u32(vcgtq_f32(t, fx), vreinterpretq_u32_f32(vdupq_n_f32(1.0f)))));
    fx = t;                                                     /* floor */
    x = vmlsq_n_f32(x, fx, 0.693359375f);
    x = vmlsq_n_f32(x, fx, -2.12194440e-4f);
    float32x4_t y = vdupq_n_f32(1.9875691500e-4f);
    y = vmlaq_f32(vdupq_n_f32(1.3981999507e-3f), y, x);
    y = vmlaq_f32(vdupq_n_f32(8.3334519073e-3f), y, x);
    y = vmlaq_f32(vdupq_n_f32(4.1665795894e-2f), y, x);
    y = vmlaq_f32(vdupq_n_f32(1.6666665459e-1f), y, x);
    y = vmlaq_f32(vdupq_n_f32(5.0000001201e-1f), y, x);
    y = vmlaq_f32(vaddq_f32(x, vdupq_n_f32(1.0f)), y, vmulq_f32(x, x));
    int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(127)), 23);
    y = vmulq_f32(y, vreinterpretq_f32_s32(e));
    return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(y), tiny));
}

/* horizontal sums of four vectors -> one vector {sum a, sum b, sum c, sum d}, all NEON */
static inline float32x4_t hsum4x4(float32x4_t a, float32x4_t b, float32x4_t c, float32x4_t d) {
    float32x2_t ab = vpadd_f32(vpadd_f32(vget_low_f32(a), vget_high_f32(a)),
                               vpadd_f32(vget_low_f32(b), vget_high_f32(b)));
    float32x2_t cd = vpadd_f32(vpadd_f32(vget_low_f32(c), vget_high_f32(c)),
                               vpadd_f32(vget_low_f32(d), vget_high_f32(d)));
    return vcombine_f32(ab, cd);
}

/* GQA attention for one KV head and its 4 query heads, all scores kept 4-wide:
 *   s[t] = {q_j . k_t} / sqrt(128), p = softmax_t(s) per head, o_j = sum_t p_j[t] v_t
 * k, v: fp16 rows of 128, stride 128 (contiguous per KV head). sc: scratch, T + 1 vectors. */
static inline void attn_group(float *R_ o, const float *R_ q, const __fp16 *R_ k, const __fp16 *R_ v,
                              int T, float32x4_t *R_ sc) {
    const float inv_sqrt = 0.08838834764831845f;   /* 1/sqrt(128) */
    float32x4_t mx = vdupq_n_f32(-INFINITY);
    int t = 0;
    /* two key rows per step: each q load feeds 2 x 4 accumulators */
    for (; t + 2 <= T; t += 2) {
        const __fp16 *k0 = k + (size_t)t * 128, *k1 = k0 + 128;
        PF(k0 + 512); PF(k0 + 528); PF(k0 + 544); PF(k0 + 560);
        PF(k0 + 576); PF(k0 + 592); PF(k0 + 608); PF(k0 + 624);
        float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
        for (int c = 0; c < 128; c += 4) {
            float32x4_t kf = vcvt_f32_f16(vld1_f16(k0 + c)), kg = vcvt_f32_f16(vld1_f16(k1 + c));
            float32x4_t q0 = vld1q_f32(q + c), q1 = vld1q_f32(q + 128 + c);
            float32x4_t q2 = vld1q_f32(q + 256 + c), q3 = vld1q_f32(q + 384 + c);
            a0 = vmlaq_f32(a0, q0, kf); b0 = vmlaq_f32(b0, q0, kg);
            a1 = vmlaq_f32(a1, q1, kf); b1 = vmlaq_f32(b1, q1, kg);
            a2 = vmlaq_f32(a2, q2, kf); b2 = vmlaq_f32(b2, q2, kg);
            a3 = vmlaq_f32(a3, q3, kf); b3 = vmlaq_f32(b3, q3, kg);
        }
        float32x4_t s0 = vmulq_n_f32(hsum4x4(a0, a1, a2, a3), inv_sqrt);
        float32x4_t s1 = vmulq_n_f32(hsum4x4(b0, b1, b2, b3), inv_sqrt);
        sc[t] = s0; sc[t + 1] = s1;
        mx = vmaxq_f32(mx, vmaxq_f32(s0, s1));
    }
    for (; t < T; t++) {
        const __fp16 *k0 = k + (size_t)t * 128;
        float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
        for (int c = 0; c < 128; c += 4) {
            float32x4_t kf = vcvt_f32_f16(vld1_f16(k0 + c));
            a0 = vmlaq_f32(a0, vld1q_f32(q + c), kf);
            a1 = vmlaq_f32(a1, vld1q_f32(q + 128 + c), kf);
            a2 = vmlaq_f32(a2, vld1q_f32(q + 256 + c), kf);
            a3 = vmlaq_f32(a3, vld1q_f32(q + 384 + c), kf);
        }
        float32x4_t s0 = vmulq_n_f32(hsum4x4(a0, a1, a2, a3), inv_sqrt);
        sc[t] = s0;
        mx = vmaxq_f32(mx, s0);
    }
    /* softmax, 4 heads at once */
    float32x4_t sum = vdupq_n_f32(0);
    for (t = 0; t < T; t++) {
        float32x4_t e = exp_q(vsubq_f32(sc[t], mx));
        sc[t] = e;
        sum = vaddq_f32(sum, e);
    }
    float32x4_t inv = vrecpeq_f32(sum);
    inv = vmulq_f32(inv, vrecpsq_f32(sum, inv));
    inv = vmulq_f32(inv, vrecpsq_f32(sum, inv));
    for (t = 0; t < T; t++) sc[t] = vmulq_f32(sc[t], inv);
    /* o_j = sum_t p_j[t] v_t: 8 output floats x 4 heads in registers per pass over t */
    for (int c = 0; c < 128; c += 8) {
        float32x4_t o00 = vdupq_n_f32(0), o01 = o00, o10 = o00, o11 = o00;
        float32x4_t o20 = o00, o21 = o00, o30 = o00, o31 = o00;
        const __fp16 *vt = v + c;
        for (t = 0; t < T; t++, vt += 128) {
            float32x4_t v0 = vcvt_f32_f16(vld1_f16(vt)), v1 = vcvt_f32_f16(vld1_f16(vt + 4));
            float32x4_t p = sc[t];
            float32x2_t plo = vget_low_f32(p), phi = vget_high_f32(p);
            o00 = vmlaq_lane_f32(o00, v0, plo, 0); o01 = vmlaq_lane_f32(o01, v1, plo, 0);
            o10 = vmlaq_lane_f32(o10, v0, plo, 1); o11 = vmlaq_lane_f32(o11, v1, plo, 1);
            o20 = vmlaq_lane_f32(o20, v0, phi, 0); o21 = vmlaq_lane_f32(o21, v1, phi, 0);
            o30 = vmlaq_lane_f32(o30, v0, phi, 1); o31 = vmlaq_lane_f32(o31, v1, phi, 1);
        }
        vst1q_f32(o + c, o00); vst1q_f32(o + c + 4, o01);
        vst1q_f32(o + 128 + c, o10); vst1q_f32(o + 128 + c + 4, o11);
        vst1q_f32(o + 256 + c, o20); vst1q_f32(o + 256 + c + 4, o21);
        vst1q_f32(o + 384 + c, o30); vst1q_f32(o + 384 + c + 4, o31);
    }
}

/* float32 -> fp16 (round to nearest), n % 4 == 0 */
static inline void k_to_f16(__fp16 *dst, const float *src, int n) {
    for (int i = 0; i < n; i += 4) vst1_f16(dst + i, vcvt_f16_f32(vld1q_f32(src + i)));
}

#endif
