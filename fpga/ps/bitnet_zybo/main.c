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
#include "ff.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xiltimer.h"
#include "gemv.h"

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

static float xres[N_EMBD];          /* residual stream */
static float hbuf[N_FF];            /* normed input / FFN activations */
static float gbuf[N_FF];
static float qbuf[N_EMBD];
static float obuf[N_EMBD];
static float scores[MAX_CTX];
static float rope_cos[HEAD_DIM / 2], rope_sin[HEAD_DIM / 2];
static double inv_freq[HEAD_DIM / 2];

static int n_past = 0;
static int verbose = 0;
static uint32_t pl_cycles = 0;

/* ---------------- small helpers ---------------- */
static uint64_t now(void) { XTime t; XTime_GetTime(&t); return t; }
static uint32_t ms_since(uint64_t t0) { return (uint32_t)((now() - t0) * 1000ull / (uint64_t)(COUNTS_PER_SECOND)); }

static void print_fixed(const char *tag, float v) {   /* xil_printf has no %f */
    int32_t i = (int32_t)(v * 10000.0f);
    xil_printf("%s%s%d.%04d", tag, i < 0 ? "-" : "", (i < 0 ? -i : i) / 10000, (i < 0 ? -i : i) % 10000);
}

static void rms_norm(float *out, const float *x, const float *w, int n) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float r = 1.0f / sqrtf((float)(ss / n) + H->rms_eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

/* Per-token absmax int8 (bitnet.cpp quantize_row_i8_s). Returns 127/amax (0 if all zero). */
static float quantize(const float *v, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) { float a = fabsf(v[i]); if (a > amax) amax = a; }
    float s = amax > 0.0f ? 127.0f / amax : 0.0f;
    for (int i = 0; i < n; i++) {
        float q = roundf(v[i] * s);
        xq[i] = (int8_t)(q > 127.0f ? 127.0f : (q < -128.0f ? -128.0f : q));
    }
    return s;
}

/* BitLinear: out[r] = (W . quant(v))[r] * w_scale / s */
static void bitlinear(const void *w, const float *v, int n, int k, float *const *outs,
                      const int *splits, const float *w_scales, int nparts) {
    float s = quantize(v, k);
    uint32_t cyc = gemv_run(GEMV_TERNARY, w, xq, ybuf, n, k);
    if (!cyc) xil_printf("# ERROR engine\r\n");
    pl_cycles += cyc;
    int r0 = 0;
    for (int p = 0; p < nparts; p++) {
        float f = s > 0.0f ? w_scales[p] / s : 0.0f;
        for (int i = 0; i < splits[p]; i++) outs[p][i] = (float)ybuf[r0 + i] * f;
        r0 += splits[p];
    }
}

static void rope(float *v, int nheads) {
    for (int h = 0; h < nheads; h++) {
        float *x = v + h * HEAD_DIM;
        for (int i = 0; i < HEAD_DIM / 2; i++) {
            float x1 = x[i], x2 = x[i + HEAD_DIM / 2];
            x[i]                = x1 * rope_cos[i] - x2 * rope_sin[i];
            x[i + HEAD_DIM / 2] = x2 * rope_cos[i] + x1 * rope_sin[i];
        }
    }
}

/* ---------------- forward pass for one token ---------------- */
static int forward(int token, int want_logits) {
    const int E = N_EMBD, F = N_FF, pos = n_past;
    const int8_t *head = (const int8_t *)IMG(H->head_off);
    const float *head_scale = (const float *)IMG(H->head_scale_off);

    /* embedding = int8 LM head row (tied weights) */
    for (int i = 0; i < E; i++) xres[i] = (float)head[(uint32_t)token * E + i] * head_scale[token];

    for (int i = 0; i < HEAD_DIM / 2; i++) {
        double a = (double)pos * inv_freq[i];
        rope_cos[i] = (float)cos(a);
        rope_sin[i] = (float)sin(a);
    }

    for (uint32_t l = 0; l < H->n_layer; l++) {
        const bn_layer_t *L = &LT[l];
        float *kc = kv_k + ((size_t)l * MAX_CTX + pos) * KV_DIM;
        float *vc = kv_v + ((size_t)l * MAX_CTX + pos) * KV_DIM;

        /* attention */
        rms_norm(hbuf, xres, (const float *)IMG(L->attn_norm_off), E);
        {
            float *outs[3] = {qbuf, kc, vc};
            const int splits[3] = {E, KV_DIM, KV_DIM};
            const float sc[3] = {L->s_q, L->s_k, L->s_v};
            bitlinear(IMG(L->qkv_off), hbuf, QKV_N, E, outs, splits, sc, 3);
        }
        rope(qbuf, N_HEAD);
        rope(kc, N_KV_HEAD);

        const float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
        for (int h = 0; h < N_HEAD; h++) {
            const int kvh = h / (N_HEAD / N_KV_HEAD);
            const float *q = qbuf + h * HEAD_DIM;
            float mx = -INFINITY;
            for (int t = 0; t <= pos; t++) {
                const float *kt = kv_k + ((size_t)l * MAX_CTX + t) * KV_DIM + kvh * HEAD_DIM;
                float d = 0.0f;
                for (int i = 0; i < HEAD_DIM; i++) d += q[i] * kt[i];
                d *= inv_sqrt;
                scores[t] = d;
                if (d > mx) mx = d;
            }
            float sum = 0.0f;
            for (int t = 0; t <= pos; t++) { scores[t] = expf(scores[t] - mx); sum += scores[t]; }
            float inv = 1.0f / sum;
            float *o = obuf + h * HEAD_DIM;
            for (int i = 0; i < HEAD_DIM; i++) o[i] = 0.0f;
            for (int t = 0; t <= pos; t++) {
                const float *vt = kv_v + ((size_t)l * MAX_CTX + t) * KV_DIM + kvh * HEAD_DIM;
                float p = scores[t] * inv;
                for (int i = 0; i < HEAD_DIM; i++) o[i] += p * vt[i];
            }
        }
        rms_norm(hbuf, obuf, (const float *)IMG(L->attn_sub_norm_off), E);
        {
            float *outs[1] = {gbuf};
            const int splits[1] = {E};
            const float sc[1] = {L->s_o};
            bitlinear(IMG(L->o_off), hbuf, E, E, outs, splits, sc, 1);
            for (int i = 0; i < E; i++) xres[i] += gbuf[i];
        }

        /* feed-forward: relu(gate)^2 * up */
        rms_norm(hbuf, xres, (const float *)IMG(L->ffn_norm_off), E);
        {
            float *outs[2] = {gbuf, hbuf};   /* hbuf is consumed by quantize() first */
            const int splits[2] = {F, F};
            const float sc[2] = {L->s_gate, L->s_up};
            bitlinear(IMG(L->gate_up_off), hbuf, 2 * F, E, outs, splits, sc, 2);
            for (int i = 0; i < F; i++) {
                float g = gbuf[i] > 0.0f ? gbuf[i] : 0.0f;
                gbuf[i] = g * g * hbuf[i];
            }
        }
        rms_norm(hbuf, gbuf, (const float *)IMG(L->ffn_sub_norm_off), F);
        {
            float *outs[1] = {gbuf};
            const int splits[1] = {E};
            const float sc[1] = {L->s_down};
            bitlinear(IMG(L->down_off), hbuf, E, F, outs, splits, sc, 1);
            for (int i = 0; i < E; i++) xres[i] += gbuf[i];
        }

        if (verbose) {
            double sa = 0.0;
            for (int i = 0; i < E; i++) sa += fabs(xres[i]);
            xil_printf("# L %d pos %d", (int)l, pos);
            print_fixed(" x0=", xres[0]);
            print_fixed(" x1=", xres[1]);
            print_fixed(" mean|x|=", (float)(sa / E));
            xil_printf("\r\n");
        }
    }
    n_past++;
    if (!want_logits) return -1;

    /* LM head: int8 engine, argmax of acc * row_scale (the 1/s factor is common) */
    rms_norm(hbuf, xres, (const float *)IMG(H->output_norm_off), E);
    quantize(hbuf, E);
    uint32_t cyc = gemv_run(GEMV_INT8, head, xq, ybuf, (int)H->vocab, E);
    if (!cyc) xil_printf("# ERROR engine (head)\r\n");
    pl_cycles += cyc;
    int best = 0;
    float bv = -INFINITY;
    for (uint32_t r = 0; r < H->vocab; r++) {
        float v = (float)ybuf[r] * head_scale[r];
        if (v > bv) { bv = v; best = (int)r; }
    }
    return best;
}

/* ---------------- SD card load ---------------- */
static FATFS fatfs;

static int load_image(void) {
    FIL f;
    UINT got;
    uint64_t t0 = now();
    if (f_mount(&fatfs, "0:/", 1) != FR_OK) { xil_printf("# ERROR: no SD card / filesystem\r\n"); return -1; }
    if (f_open(&f, "0:/BITNET.BIN", FA_READ) != FR_OK) { xil_printf("# ERROR: BITNET.BIN not found on SD\r\n"); return -1; }
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

    H = (const bn_header_t *)IMG(0);
    if (memcmp(H->magic, "BNZ1", 4) || H->version != 1 || H->total_bytes != size ||
        H->n_embd != N_EMBD || H->n_ff != N_FF || H->n_head != N_HEAD || H->n_head_kv != N_KV_HEAD ||
        H->n_layer > N_LAYER_MAX) {
        xil_printf("# ERROR: bad image header\r\n");
        return -1;
    }
    LT = (const bn_layer_t *)IMG(H->layer_table_off);
    t0 = now();
    uint32_t sum = 0;
    const uint32_t *p = (const uint32_t *)IMG(H->header_bytes);
    for (uint32_t i = 0; i < (H->total_bytes - H->header_bytes) / 4u; i++) sum += p[i];
    xil_printf("# checksum %08x (expect %08x) %s, %u ms\r\n", sum, H->checksum,
               sum == H->checksum ? "OK" : "BAD", ms_since(t0));
    return sum == H->checksum ? 0 : -1;
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
    uint64_t t0 = now();
    int next = -1;
    for (int i = 0; i < n; i++) next = forward(ids[i], i == n - 1);
    uint32_t prefill_ms = ms_since(t0), prefill_pl = pl_cycles;

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
    xil_printf("E %d %u %d %u\r\n", n, prefill_ms, gen, decode_ms);
    xil_printf("# PL busy: prefill %u ms, decode %u ms\r\n", prefill_pl / (GEMV_MHZ * 1000u),
               pl_cycles / (GEMV_MHZ * 1000u));
}

int main(void) {
    xil_printf("\r\n# bitnet_zybo: BitNet b1.58 2B-4T on Zybo Z7-10\r\n");
    if (gemv_r(0x1C) != GEMV_VERSION) {
        xil_printf("# ERROR: GEMV engine not found (version %08x): program the bitstream\r\n", gemv_r(0x1C));
        return 1;
    }
    if (load_image() != 0) { xil_printf("# HALT\r\n"); return 1; }
    for (int i = 0; i < HEAD_DIM / 2; i++)
        inv_freq[i] = 1.0 / pow((double)H->rope_theta, (double)(2 * i) / HEAD_DIM);
    xil_printf("# model: %u layers, vocab %u, ctx %d\r\nREADY\r\n", H->n_layer, H->vocab, MAX_CTX);

    for (;;) {
        if (read_line() == 0) continue;
        switch (line[0]) {
            case 'G': cmd_generate(line + 1); break;
            case 'R': n_past = 0; xil_printf("OK\r\n"); break;
            case 'V': verbose = !verbose; xil_printf("OK verbose=%d\r\n", verbose); break;
            case 'I': xil_printf("I ctx %d/%d layers %u\r\n", n_past, MAX_CTX, H->n_layer); break;
            default:  xil_printf("X unknown command\r\n"); break;
        }
    }
}
