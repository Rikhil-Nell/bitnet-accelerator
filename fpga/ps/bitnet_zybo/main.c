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
static int8_t  *const xq    = (int8_t *)(WORK_ADDR);                 /* <= 6912 B */
static int32_t *const ybuf  = (int32_t *)(WORK_ADDR + 0x4000);       /* <= 128256 x 4 B */
static float   *const kv_k  = (float *)(KV_ADDR);
static float   *const kv_v  = (float *)(KV_ADDR + N_LAYER_MAX * MAX_CTX * KV_DIM * 4);

#define A16 __attribute__((aligned(16)))
static float xres[N_EMBD] A16;      /* residual stream */
static float hbuf[N_FF] A16;        /* normed input / FFN activations */
static float gbuf[N_FF] A16;
static float qbuf[N_EMBD] A16;
static float obuf[N_EMBD] A16;
static float rope_cos[HEAD_DIM / 2] A16, rope_sin[HEAD_DIM / 2] A16;
static double inv_freq[HEAD_DIM / 2];

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
static float engine_ternary(const void *w, const float *v, int n, int k) {
    float s = k_quantize(xq, v, k);
    PT(P_QUANT);
    l1_clean(xq, (uint32_t)k);
    PT(P_FLUSH);
    gemv_start(GEMV_TERNARY, w, xq, ybuf, n, k);
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
static void attention(int l, int pos) {
    static float sc[4][MAX_CTX];
    const float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    const int G = N_HEAD / N_KV_HEAD;
    for (int g = 0; g < N_KV_HEAD; g++) {
        const float *kb = kv_k + (size_t)l * MAX_CTX * KV_DIM + g * HEAD_DIM;
        const float *vb = kv_v + (size_t)l * MAX_CTX * KV_DIM + g * HEAD_DIM;
        const float *q = qbuf + g * G * HEAD_DIM;
        float mx[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        for (int t = 0; t <= pos; t++) {
            const float *kt = kb + (size_t)t * KV_DIM;
            for (int o = 0; o < HEAD_DIM; o += 8) PF(kt + KV_DIM + o);
            for (int j = 0; j < G; j++) {
                float d = dot128(q + j * HEAD_DIM, kt) * inv_sqrt;
                sc[j][t] = d;
                if (d > mx[j]) mx[j] = d;
            }
        }
        for (int j = 0; j < G; j++) {
            float sum = 0.0f;
            for (int t = 0; t <= pos; t++) { float e = expf(sc[j][t] - mx[j]); sc[j][t] = e; sum += e; }
            float inv = 1.0f / sum;
            for (int t = 0; t <= pos; t++) sc[j][t] *= inv;
        }
        float *o = obuf + g * G * HEAD_DIM;
        memset(o, 0, sizeof(float) * G * HEAD_DIM);
        for (int t = 0; t <= pos; t++) {
            const float *vt = vb + (size_t)t * KV_DIM;
            for (int o = 0; o < HEAD_DIM; o += 8) PF(vt + KV_DIM + o);
            for (int j = 0; j < G; j++) axpy128(o + j * HEAD_DIM, sc[j][t], vt);
        }
    }
}

/* ---------------- coarse LM head ----------------
 * The int8 head is 328 MB per token. A ternary copy (82 MB, absmean threshold,
 * least-squares row scale) ranks all 128K rows on the engine; the top CH_TOPK are
 * then rescored exactly with the int8 rows. The exact winner was always within the
 * coarse top 16 over 250 decode steps (tools/bitnet/eval_coarse_head.py), so 64 is
 * a wide margin. Built at boot from the int8 head; lives after the image. */
#define CH_TOPK 64
static uint64_t *ch_w;      /* vocab x (E/32) words, engine ternary format */
static float    *ch_scale;  /* coarse row scale x int8 row scale */

static int build_coarse_head(void) {
    const int E = N_EMBD, V = (int)H->vocab, W = E / 32;
    const int8_t *head = (const int8_t *)IMG(H->head_off);
    const float *head_scale = (const float *)IMG(H->head_scale_off);
    uintptr_t a = (IMG_ADDR + H->total_bytes + 0xFFFFFu) & ~(uintptr_t)0xFFFFFu;
    ch_w = (uint64_t *)a;
    ch_scale = (float *)((a + (uintptr_t)V * W * 8u + 0xFFFu) & ~(uintptr_t)0xFFFu);
    uintptr_t end = (uintptr_t)(ch_scale + V);
    if (end > 0x40000000u) { xil_printf("# ERROR: coarse head does not fit (%08x)\r\n", (uint32_t)end); return -1; }
    uint64_t t0 = now();
    for (int r = 0; r < V; r++) {
        const int8_t *w = head + (size_t)r * E;
        int32x4_t acc = vdupq_n_s32(0);
        for (int i = 0; i < E; i += 16) {
            int8x16_t v = vabsq_s8(vld1q_s8(w + i));   /* |-128| does not occur: rows are in [-127, 127] */
            acc = vpadalq_s16(acc, vpaddlq_s8(v));
        }
        int32_t sa = vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1) + vgetq_lane_s32(acc, 2) + vgetq_lane_s32(acc, 3);
        /* keep |w| > 0.5 * mean|w|  <=>  2*E*|w| > sa */
        int32_t num = 0, den = 0;
        uint64_t *out = ch_w + (size_t)r * W;
        for (int g = 0; g < W; g++) {
            uint64_t word = 0;
            for (int i = 0; i < 32; i++) {
                int32_t x = w[g * 32 + i];
                int32_t ax = x < 0 ? -x : x;
                if (2 * E * ax > sa) {
                    word |= (uint64_t)(x > 0 ? 1u : 3u) << (2 * i);
                    num += ax;
                    den++;
                }
            }
            out[g] = word;
        }
        ch_scale[r] = (den ? (float)num / (float)den : 0.0f) * head_scale[r];
    }
    Xil_DCacheFlush();   /* the engine reads ch_w from DDR */
    xil_printf("# coarse head: %u MB at %08x, built in %u ms\r\n",
               (uint32_t)(((uintptr_t)V * W * 8u) >> 20), (uint32_t)a, ms_since(t0));
    return 0;
}

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
    l1_clean(xq, (uint32_t)E);
    PT(P_FLUSH);
    gemv_start(GEMV_TERNARY, ch_w, xq, ybuf, V, E);
    uint32_t cyc = gemv_wait();
    PT(P_PL);
    l1_invalidate(ybuf, (uint32_t)V * 4u);
    PT(P_INVAL);
    if (!cyc) xil_printf("# ERROR engine (coarse head)\r\n");
    pl_cycles += cyc;
    int k = top_k(ybuf, ch_scale, V, cand, cval);
    int best = -1;
    float bv = -INFINITY;
    for (int j = 0; j < k; j++) {
        int r = cand[j];
        float v = (float)dot_i8(head + (size_t)r * E, xq, E) * head_scale[r];
        if (v > bv || (v == bv && r < best)) { bv = v; best = r; }
    }
    PT(P_HEAD);
    return best;
}

static int use_coarse_head = 1;

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
        float *kc = kv_k + ((size_t)l * MAX_CTX + pos) * KV_DIM;
        float *vc = kv_v + ((size_t)l * MAX_CTX + pos) * KV_DIM;
        float s;

        /* attention */
        rms_norm(hbuf, xres, (const float *)IMG(L->attn_norm_off), E);
        PT(P_NORM);
        s = engine_ternary(IMG(L->qkv_off), hbuf, QKV_N, E);
        k_dequant(qbuf, ybuf, E, dq(L->s_q, s));
        k_dequant(kc, ybuf + E, KV_DIM, dq(L->s_k, s));
        k_dequant(vc, ybuf + E + KV_DIM, KV_DIM, dq(L->s_v, s));
        PT(P_DEQ);
        k_rope(qbuf, N_HEAD, rope_cos, rope_sin);
        k_rope(kc, N_KV_HEAD, rope_cos, rope_sin);
        PT(P_ROPE);
        attention((int)l, pos);
        PT(P_ATTN);
        rms_norm(hbuf, obuf, (const float *)IMG(L->attn_sub_norm_off), E);
        PT(P_NORM);
        s = engine_ternary(IMG(L->o_off), hbuf, E, E);
        k_dequant_add(xres, ybuf, E, dq(L->s_o, s));
        PT(P_DEQ);

        /* feed-forward: relu(gate)^2 * up */
        rms_norm(hbuf, xres, (const float *)IMG(L->ffn_norm_off), E);
        PT(P_NORM);
        s = engine_ternary(IMG(L->gate_up_off), hbuf, 2 * F, E);
        k_relu2_gate(gbuf, ybuf, ybuf + F, F, dq(L->s_gate, s), dq(L->s_up, s));
        PT(P_ELEM);
        rms_norm(hbuf, gbuf, (const float *)IMG(L->ffn_sub_norm_off), F);
        PT(P_NORM);
        s = engine_ternary(IMG(L->down_off), hbuf, E, F);
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
    k_quantize(xq, hbuf, E);
    PT(P_QUANT);
    if (use_coarse_head && ch_w) return head_coarse();
    /* Split the vocab so the four unit slices start 8 KB apart (mod 32 KB): if they
       start a multiple of 32 KB apart they hit the same DDR bank and bandwidth halves
       (measured 2.25 vs 4.05 GB/s). 4 x 32016 rows, then the remaining rows. */
    const int n1 = 4 * 32016;
    l1_clean(xq, (uint32_t)E);
    PT(P_FLUSH);
    gemv_start(GEMV_INT8, head, xq, ybuf, n1, E);
    uint32_t cyc = gemv_wait();
    gemv_start(GEMV_INT8, head + (uint32_t)n1 * E, xq, ybuf + n1, (int)H->vocab - n1, E);
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

/* ---------------- SD card load ---------------- */
static FATFS fatfs;

/* Sum of the image after the header; 0xFFFFFFFF if the header is not plausible. */
static int image_ok(uint32_t size_expect) {
    H = (const bn_header_t *)IMG(0);
    if (memcmp(H->magic, "BNZ1", 4) || H->version != 1 || (size_expect && H->total_bytes != size_expect) ||
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

    pl_cycles = 0;
    prof_reset();
    uint64_t t0 = now();
    int next = -1;
    for (int i = 0; i < n; i++) next = forward(ids[i], i == n - 1);
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
        next = forward(next, 1);
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
    if (gemv_r(0x1C) != GEMV_VERSION) {
        xil_printf("# ERROR: GEMV engine not found (version %08x): program the bitstream\r\n", gemv_r(0x1C));
        return 1;
    }
    /* x / y buffers: L1 only (see kernels.h), then push anything stale out of L2 */
    Xil_SetTlbAttributes(WORK_ADDR, WORK_SECTION_ATTR);
    Xil_DCacheFlush();
    if (load_image() != 0) { xil_printf("# HALT\r\n"); return 1; }
    if (build_coarse_head() != 0) ch_w = 0;
    for (int i = 0; i < HEAD_DIM / 2; i++)
        inv_freq[i] = 1.0 / pow((double)H->rope_theta, (double)(2 * i) / HEAD_DIM);
    xil_printf("# model: %u layers, vocab %u, ctx %d\r\nREADY\r\n", H->n_layer, H->vocab, MAX_CTX);

    for (;;) {
        if (read_line() == 0) continue;
        switch (line[0]) {
            case 'G': cmd_generate(line + 1); break;
            case 'R': n_past = 0; xil_printf("OK\r\n"); break;
            case 'V': verbose = !verbose; xil_printf("OK verbose=%d\r\n", verbose); break;
            case 'H': use_coarse_head = !use_coarse_head;
                      xil_printf("OK coarse_head=%d\r\n", use_coarse_head && ch_w != 0); break;
            case 'I': xil_printf("I ctx %d/%d layers %u\r\n", n_past, MAX_CTX, H->n_layer); break;
            default:  xil_printf("X unknown command\r\n"); break;
        }
    }
}
