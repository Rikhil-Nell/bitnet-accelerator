/*
 * bitnet_zybo: BitNet b1.58 2B-4T inference on the Zybo Z7-10 (engine v3, both A9 cores).
 *
 * Boot: load BITNET.BIN (tools/bitnet/pack_board.py) from the SD card into DDR (or reuse
 * it if it is still there), verify it, then serve a line protocol on UART1 (115200 8N1):
 *
 *   G <max_new> <id> <id> ...   append ids to the context and generate
 *                               -> "T <id> <ms>" per token, "C id:logit ..." top-5 if traced,
 *                                  then "E <n_prompt> <prefill_ms> <n_gen> <decode_ms>"
 *   R                           reset the context (new conversation)
 *   I                           info (context used)
 *   F <div>                     engine clock = 1000 MHz / div (boots at 12 = 83 MHz; 7 = 142.9
 *                               MHz needs a 5 V supply, USB power browns out above ~3.2 GB/s)
 *   D / N / C                   toggle: second core / speculative guessing / candidate trace
 *
 * Lines starting with '#' are logs (profile, speculation statistics). Every ternary matmul
 * and the LM-head ranking run on the PL engine; everything else runs here in float32,
 * following tools/bitnet/engine.py (the golden model). Design: docs/ARCHITECTURE.md.
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

static int n_past = 0;
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
enum { P_EMB, P_NORM, P_QUANT, P_FLUSH, P_PL, P_DEQ, P_ROPE, P_ATTN, P_HEAD, P_N };
static const char *const prof_name[P_N] = {
    "embed", "norm_stats", "quantize", "x_copy", "PL_wait", "epilogue",
    "rope", "attention", "head_rescore"};
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

static inline float dq(float w_scale, float s) { return s > 0.0f ? w_scale / s : 0.0f; }

/* GQA attention for one layer: 4 query heads share each KV head, so each K/V row
   is loaded once per group. */

/* ---------------- CPU1: attention worker ----------------
 * CPU1 (started by xsdb at cpu1_entry, cpu1.S) waits for jobs here. Memory is
 * coherent between the cores (SMP bit, shareable DDR), so a job is plain stores +
 * barriers; WFE/SEV avoid busy-polling the bus. */
#define CPU1_MAGIC 0xC0DE0001u
enum { JOB_ATTN = 0, JOB_QUANT = 1 };
static struct {
    volatile uint32_t seq, done, alive;
    volatile int type;
    volatile int l, pos, g0, g1;            /* JOB_ATTN: KV-head groups [g0, g1) of position pos */
    float *volatile q, *volatile o;
    int8_t *volatile qdst;                  /* JOB_QUANT: qdst[0..n) = round(qsrc * qs) */
    const float *volatile qsrc;
    volatile int qn;
    volatile float qs;
} __attribute__((aligned(32))) job;

static void cpu1_post(void) {
    __asm__ volatile("dmb" ::: "memory");
    job.seq = job.seq + 1u;
    __asm__ volatile("dsb\n\tsev" ::: "memory");
}
static void cpu1_wait(void) {
    while (job.done != job.seq) { }
    __asm__ volatile("dmb" ::: "memory");
}
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
        if (job.type == JOB_QUANT) {
            k_quantize_s(job.qdst, job.qsrc, job.qn, job.qs);
        } else {
            const int l = job.l, T = job.pos + 1;
            float *q = job.q, *o = job.o;
            for (int g = job.g0; g < job.g1; g++)
                attn_group(o + g * 4 * HEAD_DIM, q + g * 4 * HEAD_DIM, kv_k + KVI(l, g, 0),
                           kv_v + KVI(l, g, 0), T, sc4b);
        }
        __asm__ volatile("dmb" ::: "memory");
        job.done = seen;
        __asm__ volatile("dsb\n\tsev" ::: "memory");
    }
}


/* attention of the query in q (position pos, keys 0..pos) into o */
static void attention_into(int l, int pos, float *q, float *o) {
    static float32x4_t sc4[MAX_CTX + 1];
    int g_split = N_KV_HEAD;
    uint32_t s = 0;
    if (dual_core && job.alive == CPU1_MAGIC) {   /* CPU1 takes KV heads 3..4 */
        g_split = 3;
        job.type = JOB_ATTN;
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



/* Kernel micro-benchmark on hot (cached) data: ps per element. */

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


/* ---- incremental top-K over coarse scores ---- */



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

/* Epilogue statistics are accumulated per engine unit (rows of one unit always arrive
   in order) and merged in a fixed order, so results do not depend on the timing of the
   four units: a float sum in arrival order would differ in the last bits run to run. */
static stats_t stu[NBMAX][GEMV_NU];
static int cur_rpu;
static void stu_init(void) {
    for (int v = 0; v < NBMAX; v++) for (int u = 0; u < GEMV_NU; u++) stats_init(&stu[v][u]);
}
static void stu_merge(int nb) {
    for (int v = 0; v < nb; v++) {
        float32x4_t ss = stu[v][0].ss, am = stu[v][0].amax;
        for (int u = 1; u < GEMV_NU; u++) { ss = vaddq_f32(ss, stu[v][u].ss); am = vmaxq_f32(am, stu[v][u].amax); }
        nv.st[v].ss = ss;
        nv.st[v].amax = am;
    }
}

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
    cur_rpu = rpu;
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
        k_resid_norm(XR[v] + r0, HB[v] + r0, Y[v] + r0, nv.w + r0, r1 - r0, nv.f[v], &stu[v][r0 / cur_rpu]);
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
                    k_gate_norm(GB[v] + i0, Y[v] + i0, Y[v] + F + i0, nv.w + i0, i1 - i0, nv.fg[v], nv.fu[v], &stu[v][p]);
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
    const int split = nbc == 2 && dual_core && job.alive == CPU1_MAGIC;
    for (int v = nbc - 1; v >= 0; v--) {    /* vector 1 first, so CPU1 can start on it */
        float ss = stats_ss(&st[v]), amax = stats_amax(&st[v]);
        float r = 1.0f / sqrtf(ss / (float)n_norm + H->rms_eps);
        float qs = amax > 0.0f ? 127.0f / amax : 0.0f;
        s_out[v] = qs / r;
        if (split && v == 1) {
            job.type = JOB_QUANT; job.qdst = XQ[1]; job.qsrc = u[1]; job.qn = k; job.qs = qs;
            cpu1_post();
        } else {
            k_quantize_s(XQ[v], u[v], k, qs);
        }
    }
    if (split) cpu1_wait();
    PT(P_QUANT);
    for (int v = 0; v < nbc; v++) memcpy(xq + v * kp, XQ[v], (size_t)kp);
    __asm__ volatile("dsb" ::: "memory");
    PT(P_FLUSH);
    gemv_w(0x54, nbc == 2);
    gemv_start_nb(GEMV_T5, w, ustride, xq, ybuf, n_rows, k, nbc);
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
        if (nb == 2 && dual_core && job.alive == CPU1_MAGIC) {
            /* CPU1: all of position 2; CPU0: all of position 1 */
            job.type = JOB_ATTN; job.l = (int)l; job.pos = pos + 1; job.g0 = 0; job.g1 = N_KV_HEAD;
            job.q = QB[1]; job.o = OB[1];
            cpu1_post();
            static float32x4_t sc4a[MAX_CTX + 1];
            for (int g = 0; g < N_KV_HEAD; g++)
                attn_group(OB[0] + g * 4 * HEAD_DIM, QB[0] + g * 4 * HEAD_DIM, kv_k + KVI(l, g, 0),
                           kv_v + KVI(l, g, 0), pos + 1, sc4a);
            cpu1_wait();
        } else {
            for (int v = 0; v < nb; v++) attention_into((int)l, pos + v, QB[v], OB[v]);
        }
        PT(P_ATTN);

        for (int v = 0; v < nb; v++) {
            stats_init(&nv.st[v]);
            k_norm_stats(HB[v], OB[v], (const float *)IMG(L->attn_sub_norm_off), E, &nv.st[v]);
            st_in[v] = nv.st[v];
        }
        PT(P_NORM);
        nb_launch(IMG(L->o_off), H->ustride_o, HB, st_in, E, E, E, s);
        for (int v = 0; v < nb; v++) nv.f[v] = dq(L->s_o, s[v]);
        stu_init();
        nv.w = (const float *)IMG(L->ffn_norm_off);
        stream_rows_nb(E, nb_epi_resid);
        stu_merge(nb);

        for (int v = 0; v < nb; v++) st_in[v] = nv.st[v];
        nb_launch(IMG(L->gate_up_off), H->ustride_gate_up, HB, st_in, 2 * F, E, E, s);
        for (int v = 0; v < nb; v++) { nv.fg[v] = dq(L->s_gate, s[v]); nv.fu[v] = dq(L->s_up, s[v]); }
        stu_init();
        nv.w = (const float *)IMG(L->ffn_sub_norm_off);
        stream_gate_up_nb();
        stu_merge(nb);

        for (int v = 0; v < nb; v++) st_in[v] = nv.st[v];
        nb_launch(IMG(L->down_off), H->ustride_down, GB, st_in, E, F, F, s);
        for (int v = 0; v < nb; v++) nv.f[v] = dq(L->s_down, s[v]);
        stu_init();
        nv.w = (const float *)IMG(l + 1 < H->n_layer ? LT[l + 1].attn_norm_off : H->output_norm_off);
        stream_rows_nb(E, nb_epi_resid);
        stu_merge(nb);
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
    gemv_start_nb(GEMV_T5, IMG(H->coarse_off), H->ustride_coarse, xq, ybuf, (int)H->vocab, E, nb);
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
static int use_drafts = 1;    /* 0: same pass, never guess (for checking speculation is exact) */
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
        int d = use_drafts && n_past + 2 < MAX_CTX ? draft_next(ctx_ids, ctx_n) : -1;
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

    generate_spec(ids, n, max_new);
}

int main(void) {
    xil_printf("\r\n# bitnet_zybo: BitNet b1.58 2B-4T on Zybo Z7-10\r\n");
    if (gemv_r(0x1C) != GEMV_VERSION3) {
        xil_printf("# ERROR: GEMV engine not found (version %08x): program the bitstream\r\n", gemv_r(0x1C));
        return 1;
    }
    /* x / y buffers: L1 only (see kernels.h), then push anything stale out of L2 */
    Xil_SetTlbAttributes(WORK_ADDR, WORK_SECTION_ATTR);
    Xil_SetTlbAttributes(WORK_ADDR + 0x100000u, WORK_SECTION_ATTR);   /* y of 2 x 128256 rows > 1 MB */
    Xil_SetTlbAttributes(XQ_ADDR, NORM_NONCACHE);
    Xil_DCacheFlush();
    if (load_image() != 0) { xil_printf("# HALT\r\n"); return 1; }
    /* Start the engine at 83 MHz (2.7 GB/s): on USB power the board browns out above
       ~3.2 GB/s (sometimes at 3.2). With a 5 V supply, 'F 7' (or make board FCLK_DIV=7) gives 142.9 MHz. */
    Xil_Out32(0xF8000008u, 0xDF0Du);
    Xil_Out32(0xF8000170u, (Xil_In32(0xF8000170u) & ~(0x3Fu << 8)) | (12u << 8));
    Xil_Out32(0xF8000004u, 0x767Bu);
    xil_printf("# engine clock 83 MHz (send 'F 7' for 142.9 MHz on a 5 V supply)\r\n");
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
            case 'N': use_drafts = !use_drafts; xil_printf("OK drafts=%d\r\n", use_drafts); break;
            case 'F': {   /* FCLK0 = 1000 MHz IO PLL / div (7 = 142.9 MHz). Lower = less power. */
                int div = 0;
                parse_int(line + 1, &div);
                if (div >= 7 && div <= 20 && !(gemv_r(0x04) & 2u)) {   /* engine not busy */
                    Xil_Out32(0xF8000008u, 0xDF0Du);                  /* SLCR unlock */
                    uint32_t v = Xil_In32(0xF8000170u);               /* FPGA0_CLK_CTRL */
                    v = (v & ~(0x3Fu << 8)) | ((uint32_t)div << 8);   /* DIVISOR0 */
                    Xil_Out32(0xF8000170u, v);
                    Xil_Out32(0xF8000004u, 0x767Bu);                  /* SLCR lock */
                }
                uint32_t d0 = (Xil_In32(0xF8000170u) >> 8) & 0x3Fu, d1 = (Xil_In32(0xF8000170u) >> 20) & 0x3Fu;
                xil_printf("OK fclk0 = 1000 MHz / %u / %u\r\n", d0, d1);
                break;
            }
            case 'D': dual_core = !dual_core;
                      xil_printf("OK dual_core=%d cpu1=%s\r\n", dual_core, job.alive == CPU1_MAGIC ? "up" : "down"); break;
            case 'I': xil_printf("I ctx %d/%d layers %u\r\n", n_past, MAX_CTX, H->n_layer); break;
            default:  xil_printf("X unknown command\r\n"); break;
        }
    }
}
