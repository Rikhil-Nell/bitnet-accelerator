/*
 * bitnet_zybo: BitNet b1.58 2B-4T inference on the Zybo Z7-10.
 *
 * Boot: load BITNET.BIN (tools/bitnet/pack_board.py) from the SD card into DDR,
 * verify it, then serve a line protocol on UART1 (115200 8N1):
 *
 *   G <max_new> <id> <id> ...   append ids to the context and generate
 *                               -> "T <id> <ms>" per token, then "E <n_prompt> <prefill_ms> <n_gen> <decode_ms>"
 *   R                           reset the context (new conversation)
 *   V                           toggle per-layer debug output for the next tokens
 *   I                           info
 * Lines starting with '#' are logs. Every ternary matmul and the int8 LM head run
 * on the PL engine; everything else runs here in float32, following
 * tools/bitnet/engine.py (the golden model).
 */
#include <math.h>
#include <stdint.h>
#include <string.h>
/* ffconf.h only picks up the library's xilffs_config.h (LFN, exFAT, read-only) under
   SDT. Without it the app sees a different FATFS layout than libxilffs was built
   with, and f_mount() writes past the end of our FATFS object. */
#ifndef SDT
#define SDT
#endif
#include "ff.h"
#include "diskio.h"
_Static_assert(FF_FS_EXFAT == 1 && FF_USE_LFN != 0 && FF_FS_READONLY == 1 && FF_MULTI_PARTITION == 1,
               "FatFs config differs from the xilffs library build");
#include "xil_cache.h"
#include "xil_printf.h"
#include "xil_io.h"
#include "xil_mmu.h"
#include "gemv.h"
#include "kernels.h"
#include "ngram_table.h"

/* ---------------- memory map (DDR) ---------------- */
#define WORK_ADDR   0x01000000u   /* engine x / y buffers */
#define KV_ADDR     0x02000000u   /* KV cache, float32 */
#define IMG_ADDR    0x08000000u   /* BITNET.BIN */
#define IMG_MAX     0x37000000u

#define MAX_CTX     512
#define N_EMBD      2560
#define N_FF        6912
#define N_HEAD      20
#define N_KV_HEAD   5
#define HEAD_DIM    128
#define KV_DIM      (N_KV_HEAD * HEAD_DIM)
#define QKV_N       (N_EMBD + 2 * KV_DIM)
#define N_LAYER_MAX 30
#define TOK_EOT     128009
#define TOK_EOS     128001

/* ---------------- image header (tools/bitnet/board_image.py) ---------------- */
typedef struct {
    char     magic[4];
    uint32_t version, header_bytes, n_layer, n_embd, n_ff, n_head, n_head_kv, vocab;
    float    rope_theta, rms_eps;
    uint32_t total_bytes, checksum, head_off, head_scale_off, output_norm_off, layer_table_off;
    /* v2: each matrix is 4 unit slices at off + u * ustride (DDR bank skew) */
    uint32_t ustride_qkv, ustride_o, ustride_gate_up, ustride_down;
    uint32_t coarse_off, coarse_scale_off, ustride_coarse, k_pad_ff;
} __attribute__((packed)) bn_header_t;

typedef struct {
    uint32_t qkv_off, o_off, gate_up_off, down_off;
    uint32_t attn_norm_off, attn_sub_norm_off, ffn_norm_off, ffn_sub_norm_off;
    float    s_q, s_k, s_v, s_o, s_gate, s_up, s_down;
} __attribute__((packed)) bn_layer_t;

static const bn_header_t *H;
static const bn_layer_t  *LT;
#define IMG(off) ((const void *)(uintptr_t)(IMG_ADDR + (off)))

/* ---------------- buffers ---------------- */
#define XQ_ADDR     0x01F00000u   /* engine x: own section, normal non-cacheable */
static int8_t  *const xq    = (int8_t *)(XQ_ADDR);                   /* <= 6912 B, engine reads it */
static int32_t *const ybuf  = (int32_t *)(WORK_ADDR + 0x4000);       /* <= 128256 x 4 B */
/* KV cache in fp16 (as bitnet.cpp / llama.cpp), [layer][kv_head][pos][128] */
static __fp16  *const kv_k  = (__fp16 *)(KV_ADDR);
static __fp16  *const kv_v  = (__fp16 *)(KV_ADDR + N_LAYER_MAX * MAX_CTX * KV_DIM * 2);
#define KVI(l, g, t) ((((size_t)(l) * N_KV_HEAD + (g)) * MAX_CTX + (t)) * HEAD_DIM)
#define KV_BYTES (2u * N_LAYER_MAX * MAX_CTX * KV_DIM * 2u)

#define A16 __attribute__((aligned(16)))
static float xres[N_EMBD] A16;      /* residual stream */
static float hbuf[N_FF] A16;        /* normed input / FFN activations */
static float gbuf[N_FF] A16;
static float qbuf[N_EMBD] A16;
static float obuf[N_EMBD] A16;
static float kbuf[KV_DIM] A16, vbuf[KV_DIM] A16;
static float rope_cos[HEAD_DIM / 2] A16, rope_sin[HEAD_DIM / 2] A16;
static double inv_freq[HEAD_DIM / 2];
static int8_t xq_c[N_FF + 40] A16;  /* cached engine input; tail stays 0 (T5 K padding) */

/* Publish the quantized input to the engine: copy into the uncached section (writes
   stream through the write buffer, no line fills, no cache maintenance), then a
   barrier so the data is in DDR before the engine is started. */
static inline void put_x(int k) {
    memcpy(xq, xq_c, (size_t)k);
    __asm__ volatile("dsb" ::: "memory");
}

static int n_past = 0;
static int verbose = 0;
static uint32_t pl_cycles = 0;

/* ---------------- small helpers ---------------- */
/* ARM global timer (CPU clock / 2), read directly: the 2026 BSP's XTime path reads 0. */
#define GT_BASE 0xF8F00200u
#define GT_HZ   333333343ull
static uint64_t now(void) {
    if (!(Xil_In32(GT_BASE + 8) & 1u)) Xil_Out32(GT_BASE + 8, 1u);
    uint32_t hi, lo;
    do { hi = Xil_In32(GT_BASE + 4); lo = Xil_In32(GT_BASE); } while (hi != Xil_In32(GT_BASE + 4));
    return ((uint64_t)hi << 32) | lo;
}
static uint32_t ms_since(uint64_t t0) { return (uint32_t)((now() - t0) * 1000ull / GT_HZ); }

/* ---------------- profiler: PT(c) charges the time since the last mark to c ---------------- */
enum { P_EMB, P_NORM, P_QUANT, P_FLUSH, P_PL, P_INVAL, P_DEQ, P_ROPE, P_ATTN, P_ELEM, P_HEAD, P_N };
static const char *const prof_name[P_N] = {
    "embed", "rmsnorm", "quantize", "flush_x", "PL_wait", "inval_y", "dequant",
    "rope", "attention", "elementwise", "head_argmax"};
static uint64_t prof[P_N], pt_mark;
#define PT_MARK() (pt_mark = now())
#define PT(c) do { uint64_t _t = now(); prof[c] += _t - pt_mark; pt_mark = _t; } while (0)

static void prof_print(const char *what, int ntok) {
    if (ntok <= 0) return;
    uint64_t tot = 0;
    for (int c = 0; c < P_N; c++) tot += prof[c];
    xil_printf("# profile %s, us per token:", what);
    for (int c = 0; c < P_N; c++)
        xil_printf(" %s=%u", prof_name[c], (uint32_t)(prof[c] * 1000000ull / GT_HZ / (uint64_t)ntok));
    xil_printf(" total=%u\r\n", (uint32_t)(tot * 1000000ull / GT_HZ / (uint64_t)ntok));
}
static void prof_reset(void) { for (int c = 0; c < P_N; c++) prof[c] = 0; }

static void print_fixed(const char *tag, float v) {   /* xil_printf has no %f */
    int32_t i = (int32_t)(v * 10000.0f);
    xil_printf("%s%s%d.%04d", tag, i < 0 ? "-" : "", (i < 0 ? -i : i) / 10000, (i < 0 ? -i : i) % 10000);
}

static void rms_norm(float *out, const float *x, const float *w, int n) {
    k_rms_norm(out, x, w, n, H->rms_eps);
}

static inline float dq(float w_scale, float s) { return s > 0.0f ? w_scale / s : 0.0f; }

/* Quantize v to int8, run W (ternary, n x k) on the engine. The int32 result is in
   ybuf; returns the activation scale s (real value = y * w_scale / s). */
static float engine_ternary(const void *w, uint32_t ustride, const float *v, int n, int k) {
    float s = k_quantize(xq_c, v, k);
    PT(P_QUANT);
    put_x(t5_k(k));
    PT(P_FLUSH);
    gemv_start(GEMV_T5, w, ustride, xq, ybuf, n, k);
    uint32_t cyc = gemv_wait();
    PT(P_PL);
    l1_invalidate(ybuf, (uint32_t)n * 4u);
    PT(P_INVAL);
    if (!cyc) xil_printf("# ERROR engine\r\n");
    pl_cycles += cyc;
    return s;
}

/* GQA attention for one layer: 4 query heads share each KV head, so each K/V row
   is loaded once per group. */
static int attn_v3 = 1;

/* ---------------- CPU1: attention worker ----------------
 * CPU1 (started by xsdb at cpu1_entry, cpu1.S) waits for jobs here. Memory is
 * coherent between the cores (SMP bit, shareable DDR), so a job is plain stores +
 * barriers; WFE/SEV avoid busy-polling the bus. */
#define CPU1_MAGIC 0xC0DE0001u
static struct {
    volatile uint32_t seq, done, alive;
    volatile int l, pos, g0, g1;
    float *volatile q, *volatile o;
} __attribute__((aligned(32))) job;
static int dual_core = 1;

void cpu1_main(void) {
    static float32x4_t sc4b[MAX_CTX + 1];
    uint32_t seen = job.seq;
    job.alive = CPU1_MAGIC;
    __asm__ volatile("dsb\n\tsev" ::: "memory");
    for (;;) {
        while (job.seq == seen) __asm__ volatile("wfe" ::: "memory");
        seen = job.seq;
        __asm__ volatile("dmb" ::: "memory");
        const int l = job.l, T = job.pos + 1;
        float *q = job.q, *o = job.o;
        for (int g = job.g0; g < job.g1; g++)
            attn_group(o + g * 4 * HEAD_DIM, q + g * 4 * HEAD_DIM, kv_k + KVI(l, g, 0),
                       kv_v + KVI(l, g, 0), T, sc4b);
        __asm__ volatile("dmb" ::: "memory");
        job.done = seen;
        __asm__ volatile("dsb\n\tsev" ::: "memory");
    }
}

static void attention(int l, int pos);

/* attention of the query in q (position pos, keys 0..pos) into o */
static void attention_into(int l, int pos, float *q, float *o) {
    static float32x4_t sc4[MAX_CTX + 1];
    int g_split = N_KV_HEAD;
    uint32_t s = 0;
    if (dual_core && job.alive == CPU1_MAGIC) {   /* CPU1 takes KV heads 3..4 */
        g_split = 3;
        job.l = l; job.pos = pos; job.g0 = g_split; job.g1 = N_KV_HEAD; job.q = q; job.o = o;
        __asm__ volatile("dmb" ::: "memory");
        s = job.seq + 1u;
        job.seq = s;
        __asm__ volatile("dsb\n\tsev" ::: "memory");
    }
    for (int g = 0; g < g_split; g++)
        attn_group(o + g * 4 * HEAD_DIM, q + g * 4 * HEAD_DIM, kv_k + KVI(l, g, 0),
                   kv_v + KVI(l, g, 0), pos + 1, sc4);
    if (g_split < N_KV_HEAD) {
        while (job.done != s) { }
        __asm__ volatile("dmb" ::: "memory");
    }
}

static void attention(int l, int pos) {
    if (attn_v3) { attention_into(l, pos, qbuf, obuf); return; }
    static float sc[4][MAX_CTX + 4] A16;
    const float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    const int G = N_HEAD / N_KV_HEAD, T = pos + 1;
    for (int g = 0; g < N_KV_HEAD; g++) {
        const __fp16 *kb = kv_k + KVI(l, g, 0);
        const __fp16 *vb = kv_v + KVI(l, g, 0);
        const float *q = qbuf + g * G * HEAD_DIM;
        for (int t = 0; t < T; t++) {
            const __fp16 *kt = kb + (size_t)t * HEAD_DIM;
            PF(kt + 512); PF(kt + 528); PF(kt + 544); PF(kt + 560);
            PF(kt + 576); PF(kt + 592); PF(kt + 608); PF(kt + 624);
            float d[4];
            qk4_f16(q, kt, d);
            sc[0][t] = d[0] * inv_sqrt; sc[1][t] = d[1] * inv_sqrt;
            sc[2][t] = d[2] * inv_sqrt; sc[3][t] = d[3] * inv_sqrt;
        }
        for (int j = 0; j < G; j++) {
            float *s = sc[j];
            float mx = -INFINITY;
            for (int t = 0; t < T; t++) if (s[t] > mx) mx = s[t];
            for (int t = T; t < ((T + 3) & ~3); t++) s[t] = -INFINITY;
            float32x4_t sum4 = vdupq_n_f32(0);
            for (int t = 0; t < T; t += 4) {
                float32x4_t e = exp_q(vsubq_f32(vld1q_f32(s + t), vdupq_n_f32(mx)));
                vst1q_f32(s + t, e);
                sum4 = vaddq_f32(sum4, e);
            }
            float inv = 1.0f / hsum4(sum4);
            for (int t = 0; t < T; t += 4) vst1q_f32(s + t, vmulq_n_f32(vld1q_f32(s + t), inv));
        }
        float *o = obuf + g * G * HEAD_DIM;
        pv2_f16(o, o + HEAD_DIM, sc[0], sc[1], vb, HEAD_DIM, T);
        pv2_f16(o + 2 * HEAD_DIM, o + 3 * HEAD_DIM, sc[2], sc[3], vb, HEAD_DIM, T);
    }
}

/* ---------------- coarse LM head ----------------
 * The int8 head is 328 MB per token. A ternary copy (66 MB in T5, absmean threshold,
 * least-squares row scale; built by pack_board.py) ranks all 128K rows on the engine;
 * the top CH_TOPK are then rescored exactly with the int8 rows. The exact winner was
 * always within the coarse top 16 over 250 decode steps
 * (tools/bitnet/eval_coarse_head.py), so 64 is a wide margin. */
#define CH_TOPK 64

/* exact int8 dot of one head row with xq */
static int32_t dot_i8(const int8_t *w, const int8_t *x, int n) {
    int32x4_t acc = vdupq_n_s32(0);
    for (int i = 0; i < n; i += 16) {
        __builtin_prefetch(w + i + 256);
        int8x16_t a = vld1q_s8(w + i), b = vld1q_s8(x + i);
        int16x8_t p = vmull_s8(vget_low_s8(a), vget_low_s8(b));
        p = vmlal_s8(p, vget_high_s8(a), vget_high_s8(b));
        acc = vpadalq_s16(acc, p);
    }
    return vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1) + vgetq_lane_s32(acc, 2) + vgetq_lane_s32(acc, 3);
}

/* Top-K of y[r] * cs[r] (unordered), returns K. */
static int top_k(const int32_t *y, const float *cs, int n, int *idx, float *val) {
    int k = 0, mi = 0;
    float thr = -INFINITY;
    for (int i = 0; i < n; i += 4) {
        __builtin_prefetch(y + i + 64);
        __builtin_prefetch(cs + i + 64);
        float32x4_t v = vmulq_f32(vcvtq_f32_s32(vld1q_s32(y + i)), vld1q_f32(cs + i));
        uint32x4_t gt = vcgtq_f32(v, vdupq_n_f32(thr));
        uint32x2_t m2 = vorr_u32(vget_low_u32(gt), vget_high_u32(gt));
        if (!(vget_lane_u32(m2, 0) | vget_lane_u32(m2, 1))) continue;
        float lv[4];
        vst1q_f32(lv, v);
        for (int j = 0; j < 4; j++) {
            if (!(lv[j] > thr)) continue;
            if (k < CH_TOPK) {
                idx[k] = i + j; val[k] = lv[j]; k++;
                if (k == CH_TOPK) {   /* threshold = current minimum */
                    mi = 0;
                    for (int t = 1; t < k; t++) if (val[t] < val[mi]) mi = t;
                    thr = val[mi];
                }
            } else {
                idx[mi] = i + j; val[mi] = lv[j];
                mi = 0;
                for (int t = 1; t < k; t++) if (val[t] < val[mi]) mi = t;
                thr = val[mi];
            }
        }
    }
    return k;
}

static int head_coarse(void) {
    static int cand[CH_TOPK];
    static float cval[CH_TOPK];
    const int E = N_EMBD, V = (int)H->vocab;
    const int8_t *head = (const int8_t *)IMG(H->head_off);
    const float *head_scale = (const float *)IMG(H->head_scale_off);
    put_x(t5_k(E));
    PT(P_FLUSH);
    gemv_start(GEMV_T5, IMG(H->coarse_off), H->ustride_coarse, xq, ybuf, V, E);
    uint32_t cyc = gemv_wait();
    PT(P_PL);
    l1_invalidate(ybuf, (uint32_t)V * 4u);
    PT(P_INVAL);
    if (!cyc) xil_printf("# ERROR engine (coarse head)\r\n");
    pl_cycles += cyc;
    int k = top_k(ybuf, (const float *)IMG(H->coarse_scale_off), V, cand, cval);
    int best = -1;
    float bv = -INFINITY;
    for (int j = 0; j < k; j++) {
        int r = cand[j];
        float v = (float)dot_i8(head + (size_t)r * E, xq_c, E) * head_scale[r];
        if (v > bv || (v == bv && r < best)) { bv = v; best = r; }
    }
    PT(P_HEAD);
    return best;
}

static int use_coarse_head = 1;

/* Cache policy experiments: bit0 = KV cache L1-only, bit1 = image + coarse head L1-only.
   L1-only regions stream from DDR without evicting the hot working buffers from L2. */
static int mem_mode = 0;
static void set_region(uintptr_t base, uintptr_t bytes, int l1_only) {
    for (uintptr_t a = base & ~(uintptr_t)0xFFFFF; a < base + bytes; a += 0x100000u)
        Xil_SetTlbAttributes(a, l1_only ? WORK_SECTION_ATTR : NORM_WB_CACHE);
}
static void set_mem_mode(int m) {
    set_region(KV_ADDR, KV_BYTES, m & 1);
    set_region(IMG_ADDR, 0x40000000u - IMG_ADDR, (m >> 1) & 1);
    Xil_DCacheFlush();
    mem_mode = m;
}

/* Kernel micro-benchmark on hot (cached) data: ps per element. */
volatile float bench_sink;
static void bench_kernels(void) {
    const int n = N_EMBD, R = 200;
    const float *w = (const float *)IMG(LT[0].attn_norm_off);
    for (int i = 0; i < n; i++) gbuf[i] = (float)((i * 37) % 101 - 50) * 0.01f;
    uint64_t t0;
#define BENCH(name, stmt) do { stmt; t0 = now(); for (int r_ = 0; r_ < R; r_++) { stmt; } \
        xil_printf("# bench %s: %u ps/elem\r\n", name, (uint32_t)((now() - t0) * 1000000000000ull / GT_HZ / ((uint64_t)R * n))); } while (0)
    BENCH("sum_sq", bench_sink += sum_sq(gbuf, n));
    BENCH("rms_norm", k_rms_norm(hbuf, gbuf, w, n, 1e-5f));
    BENCH("abs_max", bench_sink += abs_max(gbuf, n));
    BENCH("quantize", bench_sink += k_quantize(xq_c, gbuf, n));
    BENCH("put_x", put_x(n));
    BENCH("dequant", k_dequant(hbuf, ybuf, n, 0.5f));
    BENCH("dequant_add", k_dequant_add(xres, ybuf, n, 0.5f));
    BENCH("relu2_gate", k_relu2_gate(hbuf, ybuf, ybuf + n, n, 0.5f, 0.25f));
    BENCH("l1_inval(y)", l1_invalidate(ybuf, (uint32_t)n * 4u));
#undef BENCH
}

/* ---------------- forward pass for one token ---------------- */
static int forward(int token, int want_logits) {
    const int E = N_EMBD, F = N_FF, pos = n_past;
    const int8_t *head = (const int8_t *)IMG(H->head_off);
    const float *head_scale = (const float *)IMG(H->head_scale_off);

    PT_MARK();
    /* embedding = int8 LM head row (tied weights) */
    for (int i = 0; i < E; i++) xres[i] = (float)head[(uint32_t)token * E + i] * head_scale[token];

    for (int i = 0; i < HEAD_DIM / 2; i++) {
        double a = (double)pos * inv_freq[i];
        rope_cos[i] = (float)cos(a);
        rope_sin[i] = (float)sin(a);
    }
    PT(P_EMB);

    for (uint32_t l = 0; l < H->n_layer; l++) {
        const bn_layer_t *L = &LT[l];
        /* new K/V rows go to [layer][kv head][pos] */
        float s;

        /* attention */
        rms_norm(hbuf, xres, (const float *)IMG(L->attn_norm_off), E);
        PT(P_NORM);
        s = engine_ternary(IMG(L->qkv_off), H->ustride_qkv, hbuf, QKV_N, E);
        k_dequant(qbuf, ybuf, E, dq(L->s_q, s));
        k_dequant(kbuf, ybuf + E, KV_DIM, dq(L->s_k, s));
        k_dequant(vbuf, ybuf + E + KV_DIM, KV_DIM, dq(L->s_v, s));
        PT(P_DEQ);
        k_rope(qbuf, N_HEAD, rope_cos, rope_sin);
        k_rope(kbuf, N_KV_HEAD, rope_cos, rope_sin);
        for (int g = 0; g < N_KV_HEAD; g++) {
            k_to_f16(kv_k + KVI(l, g, pos), kbuf + g * HEAD_DIM, HEAD_DIM);
            k_to_f16(kv_v + KVI(l, g, pos), vbuf + g * HEAD_DIM, HEAD_DIM);
        }
        PT(P_ROPE);
        attention((int)l, pos);
        PT(P_ATTN);
        rms_norm(hbuf, obuf, (const float *)IMG(L->attn_sub_norm_off), E);
        PT(P_NORM);
        s = engine_ternary(IMG(L->o_off), H->ustride_o, hbuf, E, E);
        k_dequant_add(xres, ybuf, E, dq(L->s_o, s));
        PT(P_DEQ);

        /* feed-forward: relu(gate)^2 * up */
        rms_norm(hbuf, xres, (const float *)IMG(L->ffn_norm_off), E);
        PT(P_NORM);
        s = engine_ternary(IMG(L->gate_up_off), H->ustride_gate_up, hbuf, 2 * F, E);
        k_relu2_gate(gbuf, ybuf, ybuf + F, F, dq(L->s_gate, s), dq(L->s_up, s));
        PT(P_ELEM);
        rms_norm(hbuf, gbuf, (const float *)IMG(L->ffn_sub_norm_off), F);
        PT(P_NORM);
        s = engine_ternary(IMG(L->down_off), H->ustride_down, hbuf, E, F);
        k_dequant_add(xres, ybuf, E, dq(L->s_down, s));
        PT(P_DEQ);

        if (verbose) {
            double sa = 0.0;
            for (int i = 0; i < E; i++) sa += fabs(xres[i]);
            xil_printf("# L %d pos %d", (int)l, pos);
            print_fixed(" x0=", xres[0]);
            print_fixed(" x1=", xres[1]);
            print_fixed(" mean|x|=", (float)(sa / E));
            xil_printf("\r\n");
            PT_MARK();
        }
    }
    n_past++;
    if (!want_logits) return -1;

    /* LM head: int8 engine, argmax of acc * row_scale (the 1/s factor is common) */
    PT_MARK();
    rms_norm(hbuf, xres, (const float *)IMG(H->output_norm_off), E);
    PT(P_NORM);
    k_quantize(xq_c, hbuf, E);
    PT(P_QUANT);
    if (use_coarse_head) return head_coarse();
    /* Split the vocab so the four unit slices start 8 KB apart (mod 32 KB): if they
       start a multiple of 32 KB apart they hit the same DDR bank and bandwidth halves
       (measured 2.25 vs 4.05 GB/s). 4 x 32016 rows, then the remaining rows. */
    const int n1 = 4 * 32016;
    put_x(E);
    PT(P_FLUSH);
    gemv_start(GEMV_INT8, head, (uint32_t)(n1 / 4) * E, xq, ybuf, n1, E);
    uint32_t cyc = gemv_wait();
    gemv_start(GEMV_INT8, head + (uint32_t)n1 * E, (uint32_t)(((int)H->vocab - n1) / 4) * E, xq,
               ybuf + n1, (int)H->vocab - n1, E);
    uint32_t cyc2 = gemv_wait();
    PT(P_PL);
    l1_invalidate(ybuf, H->vocab * 4u);
    PT(P_INVAL);
    if (!cyc || !cyc2) xil_printf("# ERROR engine (head)\r\n");
    pl_cycles += cyc + cyc2;
    int best = k_argmax_scaled(ybuf, head_scale, (int)H->vocab, 0);
    PT(P_HEAD);
    return best;
}

/* ---------------- streamed forward pass (engine v2 progress counters) ----------------
 * The engine reports, per unit, how many output rows are already in DDR. While it
 * streams the rest of the weights, the A9 runs the epilogue on the finished rows:
 * dequantize, residual add, ReLU^2 gate, and the statistics of the next RMSNorm
 * (sum of squares, and absmax of x * norm_weight). When the engine finishes, only one
 * quantize pass is left before the next matrix can start. */
#define OV_CHUNK 64          /* rows: run an epilogue once this many new rows are ready */

static uint64_t ov_epi;      /* A9 time spent in epilogues during the current stream */

/* charge a finished stream: wall time minus epilogue -> PL wait, epilogue -> P_DEQ */
static void ov_account(void) {
    uint64_t t = now();
    uint32_t st = gemv_r(0x04);
    if (st & 4u) xil_printf("# ERROR engine (status %08x)\r\n", st);
    pl_cycles += gemv_r(0x18);
    prof[P_PL] += (t - pt_mark) - ov_epi;
    prof[P_DEQ] += ov_epi;
    pt_mark = t;
    ov_epi = 0;
}

typedef void (*rows_fn)(int r0, int r1);   /* global output rows [r0, r1), in L1-coherent ybuf */

/* Stream an n-row engine command already started; fn gets rows as they land. */
static void stream_rows(int n, rows_fn fn) {
    const int rpu = n / GEMV_NU;
    int done[GEMV_NU] = {0, 0, 0, 0};
    for (;;) {
        int fin = gemv_done(), left = 0;
        for (int u = 0; u < GEMV_NU; u++) {
            int r = fin ? rpu : (int)gemv_rows_done(u, (uint32_t)rpu);
            if (r > done[u] && (fin || r - done[u] >= OV_CHUNK)) {
                int g0 = u * rpu + done[u], g1 = u * rpu + r;
                uint64_t t = now();
                l1_invalidate(ybuf + g0, (uint32_t)(g1 - g0) * 4u);
                fn(g0, g1);
                ov_epi += now() - t;
                done[u] = r;
            }
            left |= done[u] < rpu;
        }
        if (!left) break;
    }
    (void)gemv_wait();   /* the unit may still be closing its last burst */
    ov_account();
}

/* ---- per-op epilogue context ---- */
static struct {
    float fq, fk, fv, f;          /* dequant factors */
    float fg, fu;
    const float *w;               /* norm weight applied in the epilogue */
    stats_t st;                   /* statistics for the next quantization */
} ov;

static void epi_qkv(int r0, int r1) {
    const int E = N_EMBD;
    int a = r0 < E ? r0 : E, b = r1 < E ? r1 : E;
    if (b > a) k_dequant(qbuf + a, ybuf + a, b - a, ov.fq);
    a = r0 > E ? r0 : E; b = r1 < E + KV_DIM ? r1 : E + KV_DIM;
    if (b > a) k_dequant(kbuf + (a - E), ybuf + a, b - a, ov.fk);
    a = r0 > E + KV_DIM ? r0 : E + KV_DIM; b = r1;
    if (b > a) k_dequant(vbuf + (a - E - KV_DIM), ybuf + a, b - a, ov.fv);
}

/* o / down: residual add, then the next norm's stats (u = x * w into hbuf) */
static void epi_resid(int r0, int r1) {
    k_resid_norm(xres + r0, hbuf + r0, ybuf + r0, ov.w + r0, r1 - r0, ov.f, &ov.st);
}

/* gate_up: gate rows [0, F) live in units 0-1, up rows [F, 2F) in units 2-3, so unit u
   and unit u+2 produce matching gate/up rows at the same pace. */
static void stream_gate_up(void) {
    const int F = N_FF, rpu = 2 * F / GEMV_NU;   /* 3456 */
    int done[2] = {0, 0};
    for (;;) {
        int fin = gemv_done(), left = 0;
        for (int p = 0; p < 2; p++) {
            int r = rpu;
            if (!fin) {
                int ra = (int)gemv_rows_done(p, (uint32_t)rpu), rb = (int)gemv_rows_done(p + 2, (uint32_t)rpu);
                r = ra < rb ? ra : rb;
            }
            if (r > done[p] && (fin || r - done[p] >= OV_CHUNK)) {
                int i0 = p * rpu + done[p], i1 = p * rpu + r;
                uint64_t t = now();
                l1_invalidate(ybuf + i0, (uint32_t)(i1 - i0) * 4u);
                l1_invalidate(ybuf + F + i0, (uint32_t)(i1 - i0) * 4u);
                k_gate_norm(gbuf + i0, ybuf + i0, ybuf + F + i0, ov.w + i0, i1 - i0, ov.fg, ov.fu, &ov.st);
                ov_epi += now() - t;
                done[p] = r;
            }
            left |= done[p] < rpu;
        }
        if (!left) break;
    }
    (void)gemv_wait();
    ov_account();
}

/* Quantize u (= x * w, whose x has sum of squares ss over n) for the engine and start
   it. Returns the true activation scale s (real value of the input = q / s). */
static float ov_launch(const void *w, uint32_t ustride, const float *u, const stats_t *st, int n_rows,
                       int k, int n_norm) {
    float ss = stats_ss(st), amax = stats_amax(st);
    float r = 1.0f / sqrtf(ss / (float)n_norm + H->rms_eps);   /* rms_norm factor */
    float qs = amax > 0.0f ? 127.0f / amax : 0.0f;
    k_quantize_s(xq_c, u, k, qs);
    PT(P_QUANT);
    put_x(t5_k(k));
    PT(P_FLUSH);
    gemv_start(GEMV_T5, w, ustride, xq, ybuf, n_rows, k);
    return qs / r;   /* s of quantize(rms_norm(x) * w) */
}

/* ---- incremental top-K over coarse scores ---- */
static struct { int k, mi; float thr; int idx[CH_TOPK]; float val[CH_TOPK]; } tk;

static void topk_reset(void) { tk.k = 0; tk.mi = 0; tk.thr = -INFINITY; }

static void topk_add(int r0, int r1) {
    const int32_t *y = ybuf;
    const float *cs = (const float *)IMG(H->coarse_scale_off);
    for (int i = r0; i < r1; i += 4) {
        PF(cs + i + 64);
        float32x4_t v = vmulq_f32(vcvtq_f32_s32(vld1q_s32(y + i)), vld1q_f32(cs + i));
        uint32x4_t gt = vcgtq_f32(v, vdupq_n_f32(tk.thr));
        uint32x2_t m2 = vorr_u32(vget_low_u32(gt), vget_high_u32(gt));
        if (!(vget_lane_u32(m2, 0) | vget_lane_u32(m2, 1))) continue;
        float lv[4];
        vst1q_f32(lv, v);
        for (int j = 0; j < 4; j++) {
            if (!(lv[j] > tk.thr)) continue;
            int slot = tk.k < CH_TOPK ? tk.k++ : tk.mi;
            tk.idx[slot] = i + j;
            tk.val[slot] = lv[j];
            if (tk.k == CH_TOPK) {
                tk.mi = 0;
                for (int t = 1; t < tk.k; t++) if (tk.val[t] < tk.val[tk.mi]) tk.mi = t;
                tk.thr = tk.val[tk.mi];
            }
        }
    }
}

static int forward_ov(int token, int want_logits) {
    const int E = N_EMBD, F = N_FF, pos = n_past;
    const int8_t *head = (const int8_t *)IMG(H->head_off);
    const float *head_scale = (const float *)IMG(H->head_scale_off);

    PT_MARK();
    for (int i = 0; i < E; i++) xres[i] = (float)head[(uint32_t)token * E + i] * head_scale[token];
    for (int i = 0; i < HEAD_DIM / 2; i++) {
        double a = (double)pos * inv_freq[i];
        rope_cos[i] = (float)cos(a);
        rope_sin[i] = (float)sin(a);
    }
    stats_init(&ov.st);
    k_norm_stats(hbuf, xres, (const float *)IMG(LT[0].attn_norm_off), E, &ov.st);
    PT(P_EMB);

    for (uint32_t l = 0; l < H->n_layer; l++) {
        const bn_layer_t *L = &LT[l];
        stats_t st_in = ov.st;
        float s;

        /* qkv: input hbuf = x * attn_norm, stats from the previous epilogue */
        s = ov_launch(IMG(L->qkv_off), H->ustride_qkv, hbuf, &st_in, QKV_N, E, E);
        ov.fq = dq(L->s_q, s); ov.fk = dq(L->s_k, s); ov.fv = dq(L->s_v, s);
        stream_rows(QKV_N, epi_qkv);
        k_rope(qbuf, N_HEAD, rope_cos, rope_sin);
        k_rope(kbuf, N_KV_HEAD, rope_cos, rope_sin);
        for (int g = 0; g < N_KV_HEAD; g++) {
            k_to_f16(kv_k + KVI(l, g, pos), kbuf + g * HEAD_DIM, HEAD_DIM);
            k_to_f16(kv_v + KVI(l, g, pos), vbuf + g * HEAD_DIM, HEAD_DIM);
        }
        PT(P_ROPE);
        attention((int)l, pos);
        PT(P_ATTN);

        /* o: input = attention output * attn_sub_norm; epilogue = residual + ffn_norm stats */
        stats_init(&ov.st);
        k_norm_stats(hbuf, obuf, (const float *)IMG(L->attn_sub_norm_off), E, &ov.st);
        PT(P_NORM);
        st_in = ov.st;
        s = ov_launch(IMG(L->o_off), H->ustride_o, hbuf, &st_in, E, E, E);
        ov.f = dq(L->s_o, s);
        ov.w = (const float *)IMG(L->ffn_norm_off);
        stats_init(&ov.st);
        stream_rows(E, epi_resid);

        /* gate_up: input hbuf = x * ffn_norm; epilogue = ReLU^2 gate + ffn_sub_norm stats */
        st_in = ov.st;
        s = ov_launch(IMG(L->gate_up_off), H->ustride_gate_up, hbuf, &st_in, 2 * F, E, E);
        ov.fg = dq(L->s_gate, s); ov.fu = dq(L->s_up, s);
        ov.w = (const float *)IMG(L->ffn_sub_norm_off);
        stats_init(&ov.st);
        stream_gate_up();

        /* down: input gbuf = a * ffn_sub_norm; epilogue = residual + next norm's stats */
        st_in = ov.st;
        s = ov_launch(IMG(L->down_off), H->ustride_down, gbuf, &st_in, E, F, F);
        ov.f = dq(L->s_down, s);
        ov.w = (const float *)IMG(l + 1 < H->n_layer ? LT[l + 1].attn_norm_off : H->output_norm_off);
        stats_init(&ov.st);
        stream_rows(E, epi_resid);
    }
    n_past++;
    if (!want_logits) return -1;

    /* LM head: hbuf = x * output_norm. Coarse T5 ranking streamed into a top-K,
       then exact int8 rescoring of the candidates. */
    {
        float amax = stats_amax(&ov.st);
        k_quantize_s(xq_c, hbuf, E, amax > 0.0f ? 127.0f / amax : 0.0f);
        PT(P_QUANT);
        put_x(t5_k(E));
        PT(P_FLUSH);
        gemv_start(GEMV_T5, IMG(H->coarse_off), H->ustride_coarse, xq, ybuf, (int)H->vocab, E);
        topk_reset();
        stream_rows((int)H->vocab, topk_add);
    }
    int best = -1;
    float bv = -INFINITY;
    for (int j = 0; j < tk.k; j++) {
        int r = tk.idx[j];
        float v = (float)dot_i8(head + (size_t)r * E, xq_c, E) * head_scale[r];
        if (v > bv || (v == bv && r < best)) { bv = v; best = r; }
    }
    PT(P_HEAD);
    return best;
}

static int use_overlap = 1;

/* ---------------- batched streamed pass: 1 or 2 positions per pass over the weights ----------------
 * Engine v3 (GEM3, NB register) multiplies every weight beat with two activation
 * vectors and writes one {y_a, y_b} word per row. Speculative decoding feeds the
 * current token and one guessed next token as the two vectors: both positions come
 * out of one pass, and the guess is kept only if the model agrees with it. */
#define NBMAX 2
static float xres_b[N_EMBD] A16, hbuf_b[N_FF] A16, gbuf_b[N_FF] A16, qbuf_b[N_EMBD] A16, obuf_b[N_EMBD] A16;
static float kbuf_b[KV_DIM] A16, vbuf_b[KV_DIM] A16;
static float rope_cos_b[HEAD_DIM / 2] A16, rope_sin_b[HEAD_DIM / 2] A16;
static int8_t xq_c_b[N_FF + 40] A16;
static int32_t ysplit[NBMAX][128256] A16;      /* de-interleaved engine output (vocab rows) */

static float *const XR[NBMAX] = {xres, xres_b}, *const HB[NBMAX] = {hbuf, hbuf_b};
static float *const GB[NBMAX] = {gbuf, gbuf_b}, *const QB[NBMAX] = {qbuf, qbuf_b};
static float *const OB[NBMAX] = {obuf, obuf_b}, *const KB[NBMAX] = {kbuf, kbuf_b};
static float *const VB[NBMAX] = {vbuf, vbuf_b};
static float *const RC[NBMAX] = {rope_cos, rope_cos_b}, *const RS[NBMAX] = {rope_sin, rope_sin_b};
static int8_t *const XQ[NBMAX] = {xq_c, xq_c_b};

static int nbc;                                  /* vectors in the current pass */
static const int32_t *Y[NBMAX];                  /* per-vector engine output, row-indexed */
static struct { float f[NBMAX], fq[NBMAX], fk[NBMAX], fv[NBMAX], fg[NBMAX], fu[NBMAX];
                const float *w; stats_t st[NBMAX]; } nv;

/* {a, b} pairs -> two arrays, rows [r0, r1) */
static void split2(int r0, int r1) {
    const int32_t *s = ybuf + 2 * r0;
    int32_t *a = ysplit[0] + r0, *b = ysplit[1] + r0;
    for (int i = 0; i < r1 - r0; i += 4) {
        int32x4x2_t v = vld2q_s32(s + 2 * i);
        vst1q_s32(a + i, v.val[0]);
        vst1q_s32(b + i, v.val[1]);
    }
}

/* rows [g0, g1) of the engine output are in DDR: make them visible, split if nb = 2 */
static void rows_ready(int g0, int g1) {
    if (nbc == 1) {
        l1_invalidate(ybuf + g0, (uint32_t)(g1 - g0) * 4u);
    } else {
        l1_invalidate(ybuf + 2 * g0, (uint32_t)(g1 - g0) * 8u);
        split2(g0, g1);
    }
}

static void stream_rows_nb(int n, rows_fn fn) {
    const int rpu = n / GEMV_NU, rpb = nbc == 2 ? 16 : 32;
    int done[GEMV_NU] = {0, 0, 0, 0};
    for (;;) {
        int fin = gemv_done(), left = 0;
        for (int u = 0; u < GEMV_NU; u++) {
            int r = rpu;
            if (!fin) {
                int b = (int)gemv_r(0x40 + 4u * (uint32_t)u) * rpb;
                r = b < rpu ? b : rpu;
            }
            if (r > done[u] && (fin || r - done[u] >= OV_CHUNK)) {
                int g0 = u * rpu + done[u], g1 = u * rpu + r;
                uint64_t t = now();
                rows_ready(g0, g1);
                fn(g0, g1);
                ov_epi += now() - t;
                done[u] = r;
            }
            left |= done[u] < rpu;
        }
        if (!left) break;
    }
    (void)gemv_wait();
    ov_account();
}

static void nb_epi_qkv(int r0, int r1) {
    const int E = N_EMBD;
    for (int v = 0; v < nbc; v++) {
        int a = r0 < E ? r0 : E, b = r1 < E ? r1 : E;
        if (b > a) k_dequant(QB[v] + a, Y[v] + a, b - a, nv.fq[v]);
        a = r0 > E ? r0 : E; b = r1 < E + KV_DIM ? r1 : E + KV_DIM;
        if (b > a) k_dequant(KB[v] + (a - E), Y[v] + a, b - a, nv.fk[v]);
        a = r0 > E + KV_DIM ? r0 : E + KV_DIM; b = r1;
        if (b > a) k_dequant(VB[v] + (a - E - KV_DIM), Y[v] + a, b - a, nv.fv[v]);
    }
}

static void nb_epi_resid(int r0, int r1) {
    for (int v = 0; v < nbc; v++)
        k_resid_norm(XR[v] + r0, HB[v] + r0, Y[v] + r0, nv.w + r0, r1 - r0, nv.f[v], &nv.st[v]);
}

/* gate rows live in units 0-1 and up rows in units 2-3 (matching pace) */
static void stream_gate_up_nb(void) {
    const int F = N_FF, rpu = 2 * F / GEMV_NU, rpb = nbc == 2 ? 16 : 32;
    int done[2] = {0, 0};
    for (;;) {
        int fin = gemv_done(), left = 0;
        for (int p = 0; p < 2; p++) {
            int r = rpu;
            if (!fin) {
                int ra = (int)gemv_r(0x40 + 4u * (uint32_t)p) * rpb;
                int rb = (int)gemv_r(0x40 + 4u * (uint32_t)(p + 2)) * rpb;
                r = ra < rb ? ra : rb;
                if (r > rpu) r = rpu;
            }
            if (r > done[p] && (fin || r - done[p] >= OV_CHUNK)) {
                int i0 = p * rpu + done[p], i1 = p * rpu + r;
                uint64_t t = now();
                rows_ready(i0, i1);
                rows_ready(F + i0, F + i1);
                for (int v = 0; v < nbc; v++)
                    k_gate_norm(GB[v] + i0, Y[v] + i0, Y[v] + F + i0, nv.w + i0, i1 - i0, nv.fg[v], nv.fu[v], &nv.st[v]);
                ov_epi += now() - t;
                done[p] = r;
            }
            left |= done[p] < rpu;
        }
        if (!left) break;
    }
    (void)gemv_wait();
    ov_account();
}

/* quantize each vector's u = x * w (stats st[v]) into its x slot; start the engine.
   s_out[v] = activation scale of quantize(rms_norm(x) * w). */
static void nb_launch(const void *w, uint32_t ustride, float *const *u, const stats_t *st, int n_rows,
                      int k, int n_norm, float *s_out) {
    const int kp = t5_k(k);
    for (int v = 0; v < nbc; v++) {
        float ss = stats_ss(&st[v]), amax = stats_amax(&st[v]);
        float r = 1.0f / sqrtf(ss / (float)n_norm + H->rms_eps);
        float qs = amax > 0.0f ? 127.0f / amax : 0.0f;
        k_quantize_s(XQ[v], u[v], k, qs);
        s_out[v] = qs / r;
    }
    PT(P_QUANT);
    for (int v = 0; v < nbc; v++) memcpy(xq + v * kp, XQ[v], (size_t)kp);
    __asm__ volatile("dsb" ::: "memory");
    PT(P_FLUSH);
    gemv_w(0x54, nbc == 2);
    gemv_start(GEMV_T5, w, ustride, xq, ybuf, n_rows, k);
}

static struct { int k, mi; float thr; int idx[CH_TOPK]; float val[CH_TOPK]; } tkv[NBMAX];

/* top-5 rescored candidates per position (for the web demo, 'C' command) */
static int trace_cands;
static struct { int id[5]; float logit[5]; } cand5[NBMAX];

static void topk_add_v(int v, int r0, int r1) {
    const int32_t *y = Y[v];
    const float *cs = (const float *)IMG(H->coarse_scale_off);
    for (int i = r0; i < r1; i += 4) {
        float32x4_t s = vmulq_f32(vcvtq_f32_s32(vld1q_s32(y + i)), vld1q_f32(cs + i));
        uint32x4_t gt = vcgtq_f32(s, vdupq_n_f32(tkv[v].thr));
        uint32x2_t m2 = vorr_u32(vget_low_u32(gt), vget_high_u32(gt));
        if (!(vget_lane_u32(m2, 0) | vget_lane_u32(m2, 1))) continue;
        float lv[4];
        vst1q_f32(lv, s);
        for (int j = 0; j < 4; j++) {
            if (!(lv[j] > tkv[v].thr)) continue;
            int slot = tkv[v].k < CH_TOPK ? tkv[v].k++ : tkv[v].mi;
            tkv[v].idx[slot] = i + j;
            tkv[v].val[slot] = lv[j];
            if (tkv[v].k == CH_TOPK) {
                tkv[v].mi = 0;
                for (int t = 1; t < tkv[v].k; t++) if (tkv[v].val[t] < tkv[v].val[tkv[v].mi]) tkv[v].mi = t;
                tkv[v].thr = tkv[v].val[tkv[v].mi];
            }
        }
    }
}
static void nb_topk(int r0, int r1) { for (int v = 0; v < nbc; v++) topk_add_v(v, r0, r1); }

/* Run tokens tok[0..nb) at positions n_past, n_past+1. best[v] = greedy next token after
   position v (when want_logits). Advances n_past by nb. */
static void forward_nb(const int *tok, int nb, int want_logits, int *best) {
    const int E = N_EMBD, F = N_FF, pos = n_past;
    const int8_t *head = (const int8_t *)IMG(H->head_off);
    const float *head_scale = (const float *)IMG(H->head_scale_off);
    float s[NBMAX];
    stats_t st_in[NBMAX];
    nbc = nb;
    for (int v = 0; v < nb; v++) Y[v] = nb == 1 ? ybuf : ysplit[v];

    PT_MARK();
    for (int v = 0; v < nb; v++) {
        for (int i = 0; i < E; i++) XR[v][i] = (float)head[(uint32_t)tok[v] * E + i] * head_scale[tok[v]];
        for (int i = 0; i < HEAD_DIM / 2; i++) {
            double a = (double)(pos + v) * inv_freq[i];
            RC[v][i] = (float)cos(a);
            RS[v][i] = (float)sin(a);
        }
        stats_init(&nv.st[v]);
        k_norm_stats(HB[v], XR[v], (const float *)IMG(LT[0].attn_norm_off), E, &nv.st[v]);
    }
    PT(P_EMB);

    for (uint32_t l = 0; l < H->n_layer; l++) {
        const bn_layer_t *L = &LT[l];

        for (int v = 0; v < nb; v++) st_in[v] = nv.st[v];
        nb_launch(IMG(L->qkv_off), H->ustride_qkv, HB, st_in, QKV_N, E, E, s);
        for (int v = 0; v < nb; v++) { nv.fq[v] = dq(L->s_q, s[v]); nv.fk[v] = dq(L->s_k, s[v]); nv.fv[v] = dq(L->s_v, s[v]); }
        stream_rows_nb(QKV_N, nb_epi_qkv);
        for (int v = 0; v < nb; v++) {
            k_rope(QB[v], N_HEAD, RC[v], RS[v]);
            k_rope(KB[v], N_KV_HEAD, RC[v], RS[v]);
            for (int g = 0; g < N_KV_HEAD; g++) {
                k_to_f16(kv_k + KVI(l, g, pos + v), KB[v] + g * HEAD_DIM, HEAD_DIM);
                k_to_f16(kv_v + KVI(l, g, pos + v), VB[v] + g * HEAD_DIM, HEAD_DIM);
            }
        }
        PT(P_ROPE);
        for (int v = 0; v < nb; v++) attention_into((int)l, pos + v, QB[v], OB[v]);
        PT(P_ATTN);

        for (int v = 0; v < nb; v++) {
            stats_init(&nv.st[v]);
            k_norm_stats(HB[v], OB[v], (const float *)IMG(L->attn_sub_norm_off), E, &nv.st[v]);
            st_in[v] = nv.st[v];
        }
        PT(P_NORM);
        nb_launch(IMG(L->o_off), H->ustride_o, HB, st_in, E, E, E, s);
        for (int v = 0; v < nb; v++) { nv.f[v] = dq(L->s_o, s[v]); stats_init(&nv.st[v]); }
        nv.w = (const float *)IMG(L->ffn_norm_off);
        stream_rows_nb(E, nb_epi_resid);

        for (int v = 0; v < nb; v++) st_in[v] = nv.st[v];
        nb_launch(IMG(L->gate_up_off), H->ustride_gate_up, HB, st_in, 2 * F, E, E, s);
        for (int v = 0; v < nb; v++) { nv.fg[v] = dq(L->s_gate, s[v]); nv.fu[v] = dq(L->s_up, s[v]); stats_init(&nv.st[v]); }
        nv.w = (const float *)IMG(L->ffn_sub_norm_off);
        stream_gate_up_nb();

        for (int v = 0; v < nb; v++) st_in[v] = nv.st[v];
        nb_launch(IMG(L->down_off), H->ustride_down, GB, st_in, E, F, F, s);
        for (int v = 0; v < nb; v++) { nv.f[v] = dq(L->s_down, s[v]); stats_init(&nv.st[v]); }
        nv.w = (const float *)IMG(l + 1 < H->n_layer ? LT[l + 1].attn_norm_off : H->output_norm_off);
        stream_rows_nb(E, nb_epi_resid);
    }
    n_past += nb;
    if (!want_logits) { gemv_w(0x54, 0); return; }

    /* LM head: coarse T5 ranking (both vectors in one pass) + exact int8 rescoring */
    for (int v = 0; v < nb; v++) {
        float amax = stats_amax(&nv.st[v]);
        k_quantize_s(XQ[v], HB[v], E, amax > 0.0f ? 127.0f / amax : 0.0f);
        tkv[v].k = 0; tkv[v].mi = 0; tkv[v].thr = -INFINITY;
    }
    PT(P_QUANT);
    for (int v = 0; v < nb; v++) memcpy(xq + v * t5_k(E), XQ[v], (size_t)t5_k(E));
    __asm__ volatile("dsb" ::: "memory");
    PT(P_FLUSH);
    gemv_w(0x54, nb == 2);
    gemv_start(GEMV_T5, IMG(H->coarse_off), H->ustride_coarse, xq, ybuf, (int)H->vocab, E);
    stream_rows_nb((int)H->vocab, nb_topk);
    gemv_w(0x54, 0);
    for (int v = 0; v < nb; v++) {
        int b = -1;
        float bv = -INFINITY;
        const float to_logit = stats_amax(&nv.st[v]) / 127.0f;   /* undo the activation quantization */
        for (int t = 0; t < 5; t++) { cand5[v].id[t] = -1; cand5[v].logit[t] = -INFINITY; }
        for (int j = 0; j < tkv[v].k; j++) {
            int r = tkv[v].idx[j];
            float val = (float)dot_i8(head + (size_t)r * E, XQ[v], E) * head_scale[r];
            if (val > bv || (val == bv && r < b)) { bv = val; b = r; }
            if (trace_cands) {
                float lg = val * to_logit;
                for (int t = 0; t < 5; t++)
                    if (lg > cand5[v].logit[t]) {
                        for (int m = 4; m > t; m--) { cand5[v].id[m] = cand5[v].id[m - 1]; cand5[v].logit[m] = cand5[v].logit[m - 1]; }
                        cand5[v].id[t] = r; cand5[v].logit[t] = lg;
                        break;
                    }
            }
        }
        best[v] = b;
    }
    PT(P_HEAD);
}

/* ---------------- drafting (tools/bitnet/ngram_table.py has the same rules) ---------------- */
static uint32_t fnv_ctx(int order, const int *t) {
    uint32_t h = 0x811C9DC5u ^ (uint32_t)order;
    for (int i = 0; i < order; i++)
        for (int sh = 0; sh <= 16; sh += 8) { h ^= ((uint32_t)t[i] >> sh) & 0xFFu; h *= 0x01000193u; }
    return h;
}

static int ngram_find(uint32_t key) {
    int lo = 0, hi = NGRAM_N - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (ngram_keys[mid] == key) return (int)ngram_vals[mid];
        if (ngram_keys[mid] < key) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}

/* guess the token after ctx[0..n): the conversation itself first, then the table */
static int draft_next(const int *ctx, int n) {
    if (n >= 3) {
        int a = ctx[n - 2], b = ctx[n - 1];
        for (int i = n - 3; i >= 0; i--)
            if (ctx[i] == a && ctx[i + 1] == b) return ctx[i + 2];
    }
    for (int o = NGRAM_MAX_ORDER; o >= 1; o--) {
        if (n < o) continue;
        int t = ngram_find(fnv_ctx(o, ctx + n - o));
        if (t >= 0) return t;
    }
    return -1;
}

static int use_spec = 1;      /* speculative decoding (engine v3) */
static int engine_v3;


/* ---------------- SD card load ---------------- */
static FATFS fatfs;

/* Sum of the image after the header; 0xFFFFFFFF if the header is not plausible. */
static int image_ok(uint32_t size_expect) {
    H = (const bn_header_t *)IMG(0);
    if (memcmp(H->magic, "BNZ2", 4) || H->version != 2 || H->k_pad_ff != (uint32_t)t5_k(N_FF) || (size_expect && H->total_bytes != size_expect) ||
        H->total_bytes > IMG_MAX || H->total_bytes < H->header_bytes ||
        H->n_embd != N_EMBD || H->n_ff != N_FF || H->n_head != N_HEAD || H->n_head_kv != N_KV_HEAD ||
        H->n_layer > N_LAYER_MAX)
        return 0;
    uint64_t t0 = now();
    uint32_t sum = 0;
    const uint32_t *p = (const uint32_t *)IMG(H->header_bytes);
    for (uint32_t i = 0; i < (H->total_bytes - H->header_bytes) / 4u; i++) sum += p[i];
    xil_printf("# checksum %08x (expect %08x) %s, %u ms\r\n", sum, H->checksum,
               sum == H->checksum ? "OK" : "BAD", ms_since(t0));
    if (sum != H->checksum) return 0;
    LT = (const bn_layer_t *)IMG(H->layer_table_off);
    return 1;
}

static int load_image(void) {
    FIL f;
    UINT got;
    /* DDR keeps its contents across a JTAG reload (no power cycle): reuse the image. */
    if (image_ok(0)) { xil_printf("# BITNET.BIN already in DDR, SD load skipped\r\n"); return 0; }
    uint64_t t0 = now();
    /* Logical drive "1:" = SD partition 1, "0:" = auto (superfloppy or first partition)
       (VolToPart in xilffs ff.c). Try the partition first: a card can carry a stale
       whole-disk boot sector at LBA 0 *and* a partition table pointing at the real
       volume (this one does: partition 1 at LBA 16). Windows uses the partition;
       FatFs auto-detect would pick the stale boot sector and see an empty disk. */
    static const char *const vol[2]  = {"1:/", "0:/"};
    static const char *const path[2] = {"1:/BITNET.BIN", "0:/BITNET.BIN"};
    FRESULT fr = FR_NO_FILE;
    int v;
    for (v = 0; v < 2; v++) {
        fr = f_mount(&fatfs, vol[v], 1);
        if (fr == FR_OK) fr = f_open(&f, path[v], FA_READ);
        xil_printf("# SD %s: FRESULT %d%s\r\n", vol[v], fr, fr == FR_OK ? " (BITNET.BIN found)" : "");
        if (fr == FR_OK) break;
        f_mount(0, vol[v], 0);
    }
    if (fr != FR_OK) {
        xil_printf("# ERROR: BITNET.BIN not found on the SD card (copy it to the card's root)\r\n");
        return -1;
    }
    uint32_t size = (uint32_t)f_size(&f);
    if (size > IMG_MAX) { xil_printf("# ERROR: image too big (%u)\r\n", size); return -1; }
    xil_printf("# loading BITNET.BIN: %u MB\r\n", size >> 20);
    const uint32_t chunk = 1u << 20;
    for (uint32_t off = 0; off < size; off += chunk) {
        uint32_t n = size - off < chunk ? size - off : chunk;
        if (f_read(&f, (void *)(uintptr_t)(IMG_ADDR + off), n, &got) != FR_OK || got != n) {
            xil_printf("# ERROR: SD read failed at %u\r\n", off);
            return -1;
        }
        if ((off & 0x3FFFFFFu) == 0 && off)
            xil_printf("#   %u MB (%u s)\r\n", off >> 20, ms_since(t0) / 1000u);
    }
    f_close(&f);
    Xil_DCacheFlush();
    uint32_t ms = ms_since(t0);
    xil_printf("# loaded in %u s (%u MB/s)\r\n", ms / 1000u, ms ? (size >> 10) / ms : 0);

    if (!image_ok(size)) { xil_printf("# ERROR: bad image (header or checksum)\r\n"); return -1; }
    return 0;
}

/* ---------------- UART line protocol ---------------- */
static char line[8192];

static int read_line(void) {
    int n = 0;
    for (;;) {
        char c = inbyte();
        if (c == '\r') continue;
        if (c == '\n') { line[n] = 0; return n; }
        if (n < (int)sizeof line - 1) line[n++] = c;
    }
}

static const char *parse_int(const char *s, int *v) {
    while (*s == ' ') s++;
    if (*s < '0' || *s > '9') return 0;
    int x = 0;
    while (*s >= '0' && *s <= '9') x = x * 10 + (*s++ - '0');
    *v = x;
    return s;
}

/* Speculative decoding: every pass feeds the current token plus a guess for the
   token after it (engine NB = 2). The model's prediction for the current token decides;
   if it equals the guess, the second position's prediction is valid too and the pass
   produced two tokens. A wrong guess only costs the A9 work of the second position. */
static int ctx_ids[MAX_CTX + 2];
static int ctx_n;
static uint32_t spec_passes, spec_drafts, spec_hits;

static void print_cands(int v) {
    if (!trace_cands) return;
    xil_printf("C");
    for (int t = 0; t < 5 && cand5[v].id[t] >= 0; t++) {
        int32_t q = (int32_t)(cand5[v].logit[t] * 100.0f);
        xil_printf(" %d:%s%d.%02d", cand5[v].id[t], q < 0 ? "-" : "", (q < 0 ? -q : q) / 100, (q < 0 ? -q : q) % 100);
    }
    xil_printf("\r\n");
}

static void generate_spec(const int *ids, int n, int max_new) {
    for (int i = 0; i < n && ctx_n < MAX_CTX; i++) ctx_ids[ctx_n++] = ids[i];
    pl_cycles = 0;
    prof_reset();
    uint64_t t0 = now();
    int best[2], next = -1;
    for (int i = 0; i < n; i += 2) {   /* prompt: two tokens per pass */
        int nb = n - i >= 2 ? 2 : 1;
        int last = i + nb >= n;
        forward_nb(ids + i, nb, last, best);
        if (last) { next = best[nb - 1]; print_cands(nb - 1); }
    }
    uint32_t prefill_ms = ms_since(t0), prefill_pl = pl_cycles;
    prof_print("prefill", n);
    prof_reset();

    t0 = now();
    pl_cycles = 0;
    spec_passes = spec_drafts = spec_hits = 0;
    int gen = 0;
    uint64_t ts = now();
    while (gen < max_new) {
        xil_printf("T %d %u\r\n", next, ms_since(ts));
        ts = now();
        gen++;
        if (next == TOK_EOT || next == TOK_EOS || gen == max_new) break;
        ctx_ids[ctx_n++] = next;
        int d = n_past + 2 < MAX_CTX ? draft_next(ctx_ids, ctx_n) : -1;
        int tok2[2] = {next, d};
        spec_passes++;
        if (d >= 0) {
            spec_drafts++;
            forward_nb(tok2, 2, 1, best);
            if (best[0] == d) {          /* guess confirmed: two tokens from one pass */
                spec_hits++;
                print_cands(0);
                xil_printf("T %d %u\r\n", d, ms_since(ts) / 2u);
                gen++;
                if (d == TOK_EOT || d == TOK_EOS || gen == max_new) break;
                ctx_ids[ctx_n++] = d;
                next = best[1];
                print_cands(1);
            } else {
                n_past--;                /* drop the guess's KV row */
                next = best[0];
                print_cands(0);
            }
        } else {
            forward_nb(tok2, 1, 1, best);
            next = best[0];
            print_cands(0);
        }
    }
    uint32_t decode_ms = ms_since(t0);
    prof_print("decode", (int)spec_passes);
    xil_printf("# speculation: %u passes, %u with a guess, %u guesses right -> %u.%02u tokens/pass\r\n",
               spec_passes, spec_drafts, spec_hits, (spec_passes + spec_hits) / (spec_passes ? spec_passes : 1),
               (100u * (spec_passes + spec_hits) / (spec_passes ? spec_passes : 1)) % 100u);
    xil_printf("E %d %u %d %u\r\n", n, prefill_ms, gen, decode_ms);
    xil_printf("# PL busy: prefill %u ms, decode %u ms\r\n", (uint32_t)((uint64_t)prefill_pl / GEMV_KHZ),
               (uint32_t)((uint64_t)pl_cycles / GEMV_KHZ));
}

static void cmd_generate(const char *s) {
    static int ids[MAX_CTX];
    int max_new, n = 0, id;
    if (!(s = parse_int(s, &max_new))) { xil_printf("X bad command\r\n"); return; }
    while (n < MAX_CTX && (s = parse_int(s, &id))) {
        if ((uint32_t)id >= H->vocab) { xil_printf("X bad token %d\r\n", id); return; }
        ids[n++] = id;
    }
    if (n == 0) { xil_printf("X no tokens\r\n"); return; }
    if (n_past + n + max_new > MAX_CTX) { xil_printf("X context full (%d), send R\r\n", n_past); return; }

    if (use_spec && engine_v3) { generate_spec(ids, n, max_new); return; }
    pl_cycles = 0;
    prof_reset();
    uint64_t t0 = now();
    int next = -1;
    for (int i = 0; i < n; i++) next = (use_overlap ? forward_ov : forward)(ids[i], i == n - 1);
    uint32_t prefill_ms = ms_since(t0), prefill_pl = pl_cycles;
    prof_print("prefill", n);
    prof_reset();

    t0 = now();
    pl_cycles = 0;
    int gen = 0;
    while (gen < max_new) {
        uint64_t ts = now();
        xil_printf("T %d", next);
        gen++;
        if (next == TOK_EOT || next == TOK_EOS || gen == max_new) { xil_printf(" 0\r\n"); break; }
        next = (use_overlap ? forward_ov : forward)(next, 1);
        xil_printf(" %u\r\n", ms_since(ts));
    }
    uint32_t decode_ms = ms_since(t0);
    prof_print("decode", gen - 1);
    xil_printf("E %d %u %d %u\r\n", n, prefill_ms, gen, decode_ms);
    xil_printf("# PL busy: prefill %u ms, decode %u ms\r\n", (uint32_t)((uint64_t)prefill_pl / GEMV_KHZ),
               (uint32_t)((uint64_t)pl_cycles / GEMV_KHZ));
}

int main(void) {
    xil_printf("\r\n# bitnet_zybo: BitNet b1.58 2B-4T on Zybo Z7-10\r\n");
    engine_v3 = gemv_r(0x1C) == GEMV_VERSION3;
    if (gemv_r(0x1C) != GEMV_VERSION && !engine_v3) {
        xil_printf("# ERROR: GEMV engine not found (version %08x): program the bitstream\r\n", gemv_r(0x1C));
        return 1;
    }
    /* x / y buffers: L1 only (see kernels.h), then push anything stale out of L2 */
    Xil_SetTlbAttributes(WORK_ADDR, WORK_SECTION_ATTR);
    Xil_SetTlbAttributes(XQ_ADDR, NORM_NONCACHE);
    Xil_DCacheFlush();
    if (load_image() != 0) { xil_printf("# HALT\r\n"); return 1; }
    for (int i = 0; i < HEAD_DIM / 2; i++)
        inv_freq[i] = 1.0 / pow((double)H->rope_theta, (double)(2 * i) / HEAD_DIM);
    xil_printf("# model: %u layers, vocab %u, ctx %d, CPU1 %s\r\nREADY\r\n", H->n_layer, H->vocab, MAX_CTX,
               job.alive == CPU1_MAGIC ? "up (attention split across both cores)" : "not started");

    for (;;) {
        if (read_line() == 0) continue;
        switch (line[0]) {
            case 'G': cmd_generate(line + 1); break;
            case 'R': n_past = 0; ctx_n = 0; xil_printf("OK\r\n"); break;
            case 'C': trace_cands = !trace_cands; xil_printf("OK cands=%d\r\n", trace_cands); break;
            case 'S': use_spec = !use_spec; xil_printf("OK spec=%d (engine v3 %s)\r\n", use_spec, engine_v3 ? "yes" : "no"); break;
            case 'V': verbose = !verbose; xil_printf("OK verbose=%d\r\n", verbose); break;
            case 'M': { int m = 0; parse_int(line + 1, &m); set_mem_mode(m);
                        xil_printf("OK mem_mode=%d\r\n", mem_mode); break; }
            case 'B': bench_kernels(); xil_printf("OK\r\n"); break;
            case 'O': use_overlap = !use_overlap; xil_printf("OK overlap=%d\r\n", use_overlap); break;
            case 'A': attn_v3 = !attn_v3; xil_printf("OK attn_v3=%d\r\n", attn_v3); break;
            case 'D': dual_core = !dual_core;
                      xil_printf("OK dual_core=%d cpu1=%s\r\n", dual_core, job.alive == CPU1_MAGIC ? "up" : "down"); break;
            case 'H': use_coarse_head = !use_coarse_head;
                      xil_printf("OK coarse_head=%d\r\n", use_coarse_head); break;
            case 'I': xil_printf("I ctx %d/%d layers %u\r\n", n_past, MAX_CTX, H->n_layer); break;
            default:  xil_printf("X unknown command\r\n"); break;
        }
    }
}
