/* extractor — motion-vector extraction on top of edge264.
 *
 * Built by this directory's Makefile ("make extractor"); the
 * motion-vector-extractors project installs the result as
 * executables/extractor8 and benchmarks it as method 8.
 *
 * CLI:
 *   extractor <input> <print mv> <output.csv> <is verbose> <thread count> <keyframes only>
 * stdout: "<frames> <mvs> <rss_kb> <decode_ms>".
 * Both are the contract the motion-vector-extractors benchmark harness expects
 * (ExtractorArgs in crates/mv-extract/ffmpeg_common.rs, and
 * collect_process_results() in crates/mv-bench/benchmark_extractors.rs).
 * Output (when print!=0): the compact CSV `frame,source,src_x,src_y,dst_x,dst_y`
 * every other method writes.
 *
 * WHY IT REACHES INTO edge264's INTERNALS
 * edge264's public API (edge264.h) exports decoded samples, not motion. The
 * vectors are there — every Edge264Macroblock carries mvs[LX][i4x4][compIdx]
 * and refIdx[LX][i8x8], and mb_buffers[pic] outlives the picture's decode —
 * they are simply not part of the public struct. So this file includes
 * src/edge264_internal.h and reads dec->mb_buffers[] directly, exactly as
 * edge264's own src/edge264_test.c and src/edge264_check.c do, so that the
 * submodule keeps carrying as little of this project as possible.
 *
 * It does carry one patch: upstream edge264 is 4:2:0-only, and the benchmark
 * corpus includes a High 4:2:2 clip (MCTTR0102b), so edge264_mv_extract.diff adds
 * 4:2:2 residual PARSING - enough to keep the entropy decoder in sync, which is
 * all motion vectors depend on. Chroma samples decode with 4:2:0 geometry and
 * are wrong; nothing here reads them. See README.md.
 *
 * WHAT IT EXPORTS
 * Same convention as the custom FFmpeg fork's compact MV side data (see
 * add_mb_compact() in ffmpeg/FFmpeg-8.0-custom/FFmpeg/libavcodec/mpegutils.c)
 * and extractor10: one row per inter partition, dst = partition centre in the
 * current picture, src = dst + mv/4 (quarter-pel, truncating), source -1 for
 * list 0 and +1 for list 1. Like FFmpeg, granularity stops at 8x8 — sub-8x8
 * partitions are represented by their 8x8 quadrant's top-left 4x4 block — and
 * a vector whose integer displacement truncates to zero is dropped.
 *
 * CONTAINERS
 * edge264 consumes Annex-B NAL units, so libavformat demuxes the mp4/avi and
 * h264_mp4toannexb converts. No pixel decoding happens in FFmpeg — this is the
 * same demux-only front-end extractor10 uses.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* FFmpeg first: edge264_internal.h defines a bare `mb` macro that would
 * otherwise rewrite identifiers inside libavcodec's headers. */
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>

#include "src/edge264_internal.h"
#undef mb
#undef mbA
#undef mbB
#undef mbC
#undef mbD

/* ===========================================================================
 * MV export filters — the makefile's MV_MIN_SIZE / MV_GRID, mirroring the
 * fork's mv_min_size / mv_grid AVOptions and extractor10's MvFilter. They
 * shrink the OUTPUT only; the picture is fully decoded either way.
 * ======================================================================== */
typedef struct {
    int min_size2;     /* squared length threshold, 0 = off */
    int grid;          /* cell size in pixels, 0 = off */
    int cols, rows;
    uint8_t *taken;    /* one flag per grid cell, cleared per picture */
} MvFilter;

static void flt_init(MvFilter *f, int min_size, int grid) {
    f->min_size2 = min_size * min_size;
    f->grid = grid;
    f->cols = f->rows = 0;
    f->taken = NULL;
}

/* Called once per picture. Frame dimensions are only known after an SPS has
 * been parsed, so the cell map is sized on first use and kept afterwards. */
static void flt_reset(MvFilter *f, int w, int h) {
    if (f->grid <= 0)
        return;
    int cols = (w + f->grid - 1) / f->grid, rows = (h + f->grid - 1) / f->grid;
    if (cols != f->cols || rows != f->rows) {
        free(f->taken);
        f->taken = calloc((size_t)cols * rows, 1);
        f->cols = cols;
        f->rows = rows;
        if (!f->taken)
            f->grid = 0;   /* out of memory: degrade to "no grid" rather than die */
        return;
    }
    memset(f->taken, 0, (size_t)f->cols * f->rows);
}

static void flt_free(MvFilter *f) { free(f->taken); f->taken = NULL; }

/* Size threshold first, grid last, so a cell is claimed by a vector that
 * actually passed the threshold (same order as the fork). */
static int flt_keep(MvFilter *f, int src_x, int src_y, int dst_x, int dst_y) {
    if (f->min_size2 > 0) {
        int dx = dst_x - src_x, dy = dst_y - src_y;
        if (dx * dx + dy * dy < f->min_size2)
            return 0;
    }
    if (f->grid > 0) {
        size_t cell = (size_t)(dst_y / f->grid) * f->cols + dst_x / f->grid;
        if (f->taken[cell])
            return 0;
        f->taken[cell] = 1;
    }
    return 1;
}

/* ===========================================================================
 * CSV output
 * ======================================================================== */
#define WBUF_SIZE (1 << 20)   /* matches MV_WRITER_BUF in ffmpeg_common.rs */

typedef struct {
    FILE *f;
    char buf[WBUF_SIZE];
    size_t n;
} Writer;

static void w_flush(Writer *w) {
    if (w->f && w->n) {
        fwrite(w->buf, 1, w->n, w->f);
        w->n = 0;
    }
}

static char *put_i32(char *p, int v) {
    if (v < 0) { *p++ = '-'; v = -v; }
    char tmp[12];
    int n = 0;
    do { tmp[n++] = (char)('0' + v % 10); v /= 10; } while (v);
    while (n)
        *p++ = tmp[--n];
    return p;
}

static void w_row(Writer *w, int frame, int source, int sx, int sy, int dx, int dy) {
    if (!w->f)
        return;
    if (w->n > WBUF_SIZE - 64)
        w_flush(w);
    char *p = w->buf + w->n;
    p = put_i32(p, frame);  *p++ = ',';
    p = put_i32(p, source); *p++ = ',';
    p = put_i32(p, sx);     *p++ = ',';
    p = put_i32(p, sy);     *p++ = ',';
    p = put_i32(p, dx);     *p++ = ',';
    p = put_i32(p, dy);     *p++ = '\n';
    w->n = (size_t)(p - w->buf);
}

/* ===========================================================================
 * Motion-vector export from one decoded picture
 * ======================================================================== */

/* inter_eqs_s patterns edge264 writes for whole-macroblock partitions (see the
 * mb_type branches in src/edge264_slice.c). Anything else was split into 8x8
 * blocks or finer, which FFmpeg also exports at 8x8. */
#define EQS_16x16 0x1b5fbbffu
#define EQS_8x16  0x1b1bbbbbu
#define EQS_16x8  0x1b5f1b5fu

/* Block indexing: mvs[] is [LX][i4x4][compIdx] and refIdx[] is [LX][i8x8], with
 * i4x4 running in H.264's 4x4 scan order (x444/y444 in edge264_internal.h), so
 * i8x8 == i4x4 >> 2 and block i8x8*4 is the top-left 4x4 of quadrant i8x8 —
 * which is the one FFmpeg samples motion_val at for each partition. The dst
 * offsets below are therefore the same ones ff_print_debug_info2_optimized()
 * computes: partition centres, 8 for a full-width/height side, 4 or 12 for a
 * half. */

/* One exported vector, held until its picture's display position is known. */
typedef struct { int16_t source, sx, sy, dx, dy; } MvRow;

/* One decoded picture's vectors, waiting in the reorder buffer. */
typedef struct {
    int32_t poc;      /* picture order count: display position within the GOP */
    MvRow *rows;
    int n, cap;
} PendingPic;

typedef struct {
    int l0_only;
    int collecting;   /* 0 when no CSV is wanted: count only, store nothing */
    MvFilter *flt;
    uint64_t count;
    PendingPic *cur;  /* picture currently being walked */
} ExportCtx;

static inline void emit(ExportCtx *e, int source, int dst_x, int dst_y, const int16_t *mv) {
    int src_x = dst_x + mv[0] / 4;   /* quarter-pel, truncating like FFmpeg's */
    int src_y = dst_y + mv[1] / 4;   /* motion_x / motion_scale */
    if (src_x == dst_x && src_y == dst_y)
        return;                       /* sub-pel motion, zero integer displacement */
    if (!flt_keep(e->flt, src_x, src_y, dst_x, dst_y))
        return;
    e->count++;
    if (!e->collecting)
        return;       /* nothing will be written, so do not pay to buffer it */
    PendingPic *p = e->cur;
    if (p->n == p->cap) {
        int cap = p->cap ? p->cap * 2 : 4096;
        MvRow *n = realloc(p->rows, (size_t)cap * sizeof *n);
        if (!n)
            return;                   /* out of memory: drop, never corrupt */
        p->rows = n;
        p->cap = cap;
    }
    p->rows[p->n++] = (MvRow){(int16_t)source, (int16_t)src_x, (int16_t)src_y,
                              (int16_t)dst_x, (int16_t)dst_y};
}

/* Walk one picture's macroblock array and emit its vectors.
 * `mbs` is dec->mb_buffers[pic], whose rows are pic_width_in_mbs + 1 entries
 * wide (the extra column holds the "unavailable" sentinel — see alloc_frame()
 * in src/edge264_headers.c). */
static void export_picture(ExportCtx *e, const Edge264Macroblock *mbs, int mb_w, int mb_h) {
    for (int mb_y = 0; mb_y < mb_h; mb_y++) {
        const Edge264Macroblock *row = mbs + (size_t)mb_y * (mb_w + 1);
        for (int mb_x = 0; mb_x < mb_w; mb_x++) {
            const Edge264Macroblock *m = row + mb_x;
            /* Intra macroblocks set refIdx to -1 across both lists; so does the
             * unavailable sentinel. One 64-bit test rejects them. */
            if (m->refIdx_l == (int64_t)-1)
                continue;
            unsigned eqs = little_endian32(m->f.inter_eqs_s);
            int x = mb_x * 16, y = mb_y * 16;

            for (int lx = 0; lx < 2; lx++) {
                if (lx && e->l0_only)
                    continue;
                /* FFmpeg gates list membership on mb_type, i.e. per macroblock
                 * rather than per partition (HAS_MV_EXT in mpegutils.c), so a
                 * list is exported for every partition as soon as any 8x8 uses
                 * it. Partitions that do not are zero and drop out in emit(). */
                const int8_t *refIdx = m->refIdx + lx * 4;
                if (refIdx[0] < 0 && refIdx[1] < 0 && refIdx[2] < 0 && refIdx[3] < 0)
                    continue;
                const int16_t *mvs = m->mvs + lx * 32;
                int source = lx ? 1 : -1;

                if (eqs == EQS_16x16) {
                    emit(e, source, x + 8, y + 8, mvs + 0 * 2);
                } else if (eqs == EQS_16x8) {
                    emit(e, source, x + 8, y + 4,  mvs + 0 * 2);
                    emit(e, source, x + 8, y + 12, mvs + 8 * 2);
                } else if (eqs == EQS_8x16) {
                    emit(e, source, x + 4,  y + 8, mvs + 0 * 2);
                    emit(e, source, x + 12, y + 8, mvs + 4 * 2);
                } else {
                    for (int i = 0; i < 4; i++)
                        emit(e, source, x + 4 + 8 * (i & 1), y + 4 + 8 * (i >> 1),
                             mvs + i * 4 * 2);
                }
            }
        }
    }
}

/* ===========================================================================
 * Access-unit handling
 * ---------------------------------------------------------------------------
 * The demuxed bytes are handed to edge264_decode_NAL() in place, with no copy.
 * Two things make that safe: serial decoding is synchronous, so the packet
 * outlives the call; and AVPacket buffers carry AV_INPUT_BUFFER_PADDING_SIZE
 * zeroed trailing bytes, which covers edge264_find_start_code()'s habit of
 * scanning in aligned 16-byte steps up to 15 bytes past `end`. Copying each
 * access unit into a private buffer first measured ~5% slower at 8 concurrent
 * streams - it is pure extra memory traffic in a workload that is already
 * memory-bound when many decoders run at once.
 * ======================================================================== */

/* ===========================================================================
 * Display-order reordering
 * ---------------------------------------------------------------------------
 * edge264 hands pictures back in decode order, not display order: its output
 * POCs are not monotonic (0,2,4,6,8,16,12,10,14,24,... on MCTTR0102b), whereas
 * every FFmpeg-based extractor writes frames in display order. Left alone, the
 * `frame` column would mean "decode position" here and "display position"
 * everywhere else, so per-frame comparisons and MV overlay videos misalign -
 * the vectors are right, they just land on the wrong frame.
 *
 * So hold pictures in a small window and emit the lowest (gop, poc) once the
 * window is full. REORDER_DEPTH only has to exceed the stream's reorder
 * distance; edge264 caps its DPB at 16 frames, so 16 cannot be exceeded by a
 * conforming stream.
 *
 * POC restarts at every IDR, which is why the sort key carries a GOP counter:
 * sorting on POC alone would interleave the end of one GOP with the start of
 * the next.
 * ======================================================================== */
#define REORDER_DEPTH 16

typedef struct {
    PendingPic pics[REORDER_DEPTH + 1];
    int n;
    int32_t last_out_poc; /* POC of the most recently written picture */
    int have_out;
    int out_frame;      /* display index actually written to the CSV */
} Reorder;

/* Write one picture's rows out under the next display index. */
static void reorder_write(Reorder *ro, PendingPic *p, Writer *w) {
    ro->last_out_poc = p->poc;
    ro->have_out = 1;
    for (int i = 0; i < p->n; i++)
        w_row(w, ro->out_frame, p->rows[i].source, p->rows[i].sx, p->rows[i].sy,
              p->rows[i].dx, p->rows[i].dy);
    ro->out_frame++;
    free(p->rows);
    p->rows = NULL;
    p->n = p->cap = 0;
}

/* Pop and write the earliest-display picture currently held. */
static void reorder_pop(Reorder *ro, Writer *w) {
    if (ro->n == 0)
        return;
    int best = 0;
    for (int i = 1; i < ro->n; i++)
        if (ro->pics[i].poc < ro->pics[best].poc)
            best = i;
    reorder_write(ro, &ro->pics[best], w);
    ro->pics[best] = ro->pics[--ro->n];
}

static void reorder_flush(Reorder *ro, Writer *w) {
    while (ro->n > 0)
        reorder_pop(ro, w);
}

/* Hand a freshly decoded picture to the window, flushing one if it is full.
 *
 * Detecting where POC numbering restarts is the whole difficulty. A POC going
 * backwards is NOT a restart - that is exactly the reordering this window
 * exists to undo. Nor is "POC == 0" sufficient: bigbunnyfull restarts at 224
 * mid-stream (decode order ... 236, 238, 240, then 224, 226, ...), reusing POCs
 * it had already issued, and a fixed backwards-jump threshold cannot separate
 * that from deep reordering.
 *
 * What is unambiguous: if an arriving picture's POC is no greater than one we
 * have already WRITTEN, it cannot belong to the sequence we are currently
 * emitting - its slot is gone. Treat that as the restart, drain what is held so
 * the two sequences cannot interleave, and begin again.
 */
static void reorder_push(Reorder *ro, PendingPic p, Writer *w) {
    /* Already written a picture at or past this POC: its slot is gone. */
    int restart = ro->have_out && p.poc <= ro->last_out_poc;
    /* Or the window still holds a picture with this exact POC - a POC cannot
     * repeat within one sequence, so numbering must have restarted. This is the
     * case a "written" test alone misses, because the window lags behind. */
    for (int i = 0; !restart && i < ro->n; i++)
        restart = ro->pics[i].poc == p.poc;
    if (restart) {
        reorder_flush(ro, w);
        ro->have_out = 0;
    }
    ro->pics[ro->n++] = p;
    if (ro->n > REORDER_DEPTH)
        reorder_pop(ro, w);
}

/* ===========================================================================
 * Decoder driving
 * ======================================================================== */
typedef struct {
    Edge264Decoder *dec;
    ExportCtx ex;
    Reorder ro;      /* holds pictures until their display position is known */
    Writer *w;
    int frames;      /* pictures decoded; CSV numbering comes from Reorder */
    int packets;     /* video access units demuxed, decoded or not */
    int keyframes_only;
    int verbose;
    /* NAL outcomes, so a stream edge264 cannot decode reports why instead of
     * quietly producing a header-only CSV. ENOTSUP on a sequence parameter set
     * is the interesting one: it means the stream uses an H.264 feature this
     * decoder does not implement (4:2:2 / 4:4:4 chroma, for instance), and
     * every slice after it then fails with EBADMSG. */
    int nal_unsup;
    int nal_bad;
    int sps_unsup;
} Run;

/* Drain everything the decoder is willing to hand back. Frames arrive in
 * display order, so r->frames is the same numbering avcodec_receive_frame()
 * gives the FFmpeg extractors. `borrow` holds our claim on the frame slot
 * until edge264_return_frame(), so the decoder cannot recycle its macroblock
 * array while we are reading motion out of it. */
static void drain(Run *r) {
    Edge264Frame out;
    while (edge264_get_frame(r->dec, &out, 1) == 0) {
        Edge264Decoder *dec = r->dec;
        int mb_w = dec->sps.pic_width_in_mbs;
        int mb_h = dec->sps.pic_height_in_mbs;
        /* return_arg is a bitfield of the frame slots this call handed out (two
         * for an MVC pair); FrameId identifies which of them is the base view. */
        unsigned slots = (unsigned)(uintptr_t)out.return_arg;
        for (unsigned s = slots; s; s &= s - 1) {
            int pic = __builtin_ctz(s);
            if (dec->FrameIds[pic] != out.FrameId)
                continue;
            const Edge264Macroblock *mbs = dec->mb_buffers[pic];
            if (mbs && mb_w > 0 && mb_h > 0) {
                PendingPic p = {0};
                /* Display position of this picture. edge264 hands them back in
                 * decode order, so this is what the CSV must be sorted by. */
                p.poc = dec->FieldOrderCnt[0][pic];
                r->ex.cur = &p;
                flt_reset(r->ex.flt, mb_w * 16, mb_h * 16);
                export_picture(&r->ex, mbs, mb_w, mb_h);
                reorder_push(&r->ro, p, r->w);
            }
            break;
        }
        r->frames++;
        edge264_return_frame(r->dec, out.return_arg);
    }
}

/* Feed one Annex-B chunk (one demuxed access unit, start codes included).
 * Mirrors the loop in edge264's src/edge264_test.c: ENOBUFS means the decoder
 * is out of frame slots and the same NAL must be retried after draining. */
static void feed(Run *r, const uint8_t *buf, size_t len) {
    const uint8_t *end0 = buf + len;
    r->packets++;
    const uint8_t *nal = edge264_find_start_code(buf, end0, 0);
    if (nal >= end0)
        return;
    nal += 3;
    while (nal < end0) {
        const uint8_t *end = edge264_find_start_code(nal, end0, 0);
        /* Keyframes-only drops non-IDR slices (type 1) and keeps parameter
         * sets, so every surviving picture is a self-contained IDR — the same
         * frames AVDISCARD_NONKEY leaves the FFmpeg extractors. */
        if (!r->keyframes_only || (*nal & 0x1f) != 1) {
            /* ENOBUFS means every frame slot is taken; drain() frees one, so
             * the retry makes progress. The bound only guards against a
             * decoder state where it never can, which would otherwise hang. */
            int res, tries = 0;
            int nal_type = *nal & 0x1f;
            do {
                res = edge264_decode_NAL(r->dec, nal, end, NULL, NULL);
                drain(r);
            } while (res == ENOBUFS && ++tries < 64);
            if (res == ENOTSUP) {
                r->nal_unsup++;
                if (nal_type == 7 || nal_type == 15)
                    r->sps_unsup++;
            } else if (res != 0 && res != ENODATA) {
                r->nal_bad++;
            }
        }
        nal = end + 3;
    }
}

/* ===========================================================================
 * libavformat demux front-end (container -> Annex-B, no pixel decode)
 * ======================================================================== */
static int run_demux(Run *r, const char *path) {
    int rc = -1;
    AVFormatContext *fmt = NULL;
    AVBSFContext *bsfc = NULL;
    AVPacket *pkt = NULL, *op = NULL;

    if (avformat_open_input(&fmt, path, NULL, NULL) < 0)
        return -1;
    if (avformat_find_stream_info(fmt, NULL) < 0)
        goto done;
    int vs = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (vs < 0)
        goto done;
    AVStream *st = fmt->streams[vs];
    if (st->codecpar->codec_id != AV_CODEC_ID_H264) {
        fprintf(stderr, "extractor: edge264 decodes H.264 only, '%s' is %s\n",
                path, avcodec_get_name(st->codecpar->codec_id));
        goto done;
    }

    const AVBitStreamFilter *bsf = av_bsf_get_by_name("h264_mp4toannexb");
    if (bsf) {
        if (av_bsf_alloc(bsf, &bsfc) < 0)
            goto done;
        avcodec_parameters_copy(bsfc->par_in, st->codecpar);
        if (av_bsf_init(bsfc) < 0) { av_bsf_free(&bsfc); bsfc = NULL; }
    }

    pkt = av_packet_alloc();
    op = av_packet_alloc();
    if (!pkt || !op)
        goto done;

    while (av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index == vs) {
            if (bsfc) {
                if (av_bsf_send_packet(bsfc, pkt) == 0) {
                    while (av_bsf_receive_packet(bsfc, op) == 0) {
                        feed(r, op->data, op->size);
                        av_packet_unref(op);
                    }
                }
            } else {
                feed(r, pkt->data, pkt->size);
            }
        }
        av_packet_unref(pkt);
    }
    if (bsfc) {
        av_bsf_send_packet(bsfc, NULL);
        while (av_bsf_receive_packet(bsfc, op) == 0) {
            feed(r, op->data, op->size);
            av_packet_unref(op);
        }
    }
    rc = 0;

done:
    if (op) av_packet_free(&op);
    if (pkt) av_packet_free(&pkt);
    if (bsfc) av_bsf_free(&bsfc);
    avformat_close_input(&fmt);
    return rc;
}

/* ===========================================================================
 * main
 * ======================================================================== */
static long rss_kb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f)
        return 0;
    char line[256];
    long kb = 0;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) { kb = strtol(line + 6, NULL, 10); break; }
    }
    fclose(f);
    return kb;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int env_int(const char *name, int dflt) {
    const char *v = getenv(name);
    return (v && *v) ? atoi(v) : dflt;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s <input file> <print mv> <output file> "
                        "<is verbose> <thread_count> <keyframes_only>\n", argv[0]);
        return 255;
    }
    const char *in_path = argv[1];
    int do_print = atoi(argv[2]);
    const char *out_path = argv[3];
    int verbose = atoi(argv[4]);
    int thread_count = atoi(argv[5]);
    int keyframes_only = atoi(argv[6]) == 1;

    /* Deliberately serial. edge264 does have a worker pool (edge264_alloc's
     * first argument), but this extractor reads motion straight out of
     * dec->mb_buffers[] after each picture is handed back, and the whole
     * MV-extraction comparison this benchmark exists for is run at
     * THREAD_COUNT=1. The thread-count argument is accepted for CLI
     * compatibility with extractor0..7 and otherwise ignored, so a
     * `make benchmark_threads` sweep reports method 11's serial cost at every
     * point rather than silently measuring something else. */
    (void)thread_count;
    const int n_threads = 0;

    /* Default on, like the fork's mv_l0_only and every other method under the
     * benchmark's L0_ONLY (see BENCH_ENV in the makefile). */
    int l0_only = env_int("L0_ONLY", 1) != 0;
    int mv_min_size = env_int("MV_MIN_SIZE", 0);
    int mv_grid = env_int("MV_GRID", 0);
    if (mv_min_size < 0) mv_min_size = 0;
    if (mv_min_size > 32767) mv_min_size = 32767;
    if (mv_grid < 0) mv_grid = 0;
    if (mv_grid > 32767) mv_grid = 32767;

    static Writer w;
    if (do_print) {
        w.f = fopen(out_path, "wb");
        if (!w.f) { perror("open output"); return 1; }
        fputs("frame,source,src_x,src_y,dst_x,dst_y\n", w.f);
    }

    Edge264Decoder *dec = edge264_alloc(n_threads, NULL, NULL, 0, NULL, NULL, NULL);
    if (dec)
        // Parse everything, reconstruct nothing - see edge264_mv_extract.diff. This is
        // edge264's counterpart to the custom FFmpeg fork's motion_vectors_only
        // AVOption, and the reason methods 5 and 11 are measuring comparable
        // work rather than MV-only against full decode.
        dec->mv_only = 1;
    if (!dec) {
        fprintf(stderr, "extractor: edge264_alloc failed\n");
        if (w.f) fclose(w.f);
        return 1;
    }

    MvFilter flt;
    flt_init(&flt, mv_min_size, mv_grid);

    Run r = { .dec = dec, .keyframes_only = keyframes_only, .verbose = verbose };
    r.ex.l0_only = l0_only;
    r.ex.collecting = do_print != 0;
    r.ex.flt = &flt;
    r.w = &w;

    double t0 = now_ms();
    int demux_rc = run_demux(&r, in_path);
    if (demux_rc == 0) {
        /* buf == end asks edge264 to bump every frame still in the DPB. */
        static const uint8_t eos = 0;
        int res, tries = 0;
        do {
            res = edge264_decode_NAL(dec, &eos, &eos, NULL, NULL);
            drain(&r);
        } while (res == ENOBUFS && ++tries < 64);
    }
    /* Anything still held in the reorder window belongs at the end of the
     * stream, in display order. */
    reorder_flush(&r.ro, &w);
    double decode_ms = now_ms() - t0;

    /* Reading the decoder's parsed parameter sets turns the generic "nothing
     * decoded" case into a specific one. Cheap, and we already have the
     * internal header for the motion vectors. */
    int cavlc_422 = 0;
    if (dec->sps.ChromaArrayType == 2) {
        cavlc_422 = 1;
        for (int i = 0; i < 4; i++)
            if (dec->PPS[i].entropy_coding_mode_flag)
                cavlc_422 = 0;
    }

    long mem = rss_kb();
    w_flush(&w);
    if (w.f)
        fclose(w.f);
    edge264_free(&dec);
    flt_free(&flt);

    if (demux_rc != 0) {
        fprintf(stderr, "extractor: could not demux '%s'\n", in_path);
        printf("0 0 %ld 0.000\n", mem);
        return 3;
    }
    /* Keyframes-only decodes ~2-5%% of the pictures, so reporting decoded
     * pictures would make ms/frame incomparable with the all-frame methods.
     * extractor6.rs reports the video packet count for exactly this reason;
     * match it. CSV rows keep the decoded-picture numbering either way. */
    int reported = keyframes_only ? r.packets : r.frames;

    /* Say why nothing came out. Without this the extractor looks broken: it
     * writes a header-only CSV, reports "0 0", and exits 0. stderr is inherited
     * by the benchmark's children (only stdout is piped - see spawn_processes
     * in crates/mv-bench/benchmark_extractors.rs), so this reaches the console
     * of a `make benchmark` run. */
    if (r.frames == 0 && r.packets > 0) {
        if (r.sps_unsup)
            fprintf(stderr,
                "extractor: edge264 rejected this stream's sequence parameter set "
                "(ENOTSUP) - it decodes Progressive High 4:2:0 only, so 4:2:2 / 4:4:4 "
                "and interlaced clips produce no vectors. %d NAL units failed across "
                "%d access units.\n",
                r.nal_unsup + r.nal_bad, r.packets);
        else if (cavlc_422)
            fprintf(stderr,
                "extractor: this is a CAVLC 4:2:2 stream. The edge264 fork in this "
                "repo implements 4:2:2 residual parsing for CABAC only - its CAVLC "
                "chroma DC needs the nC==-2 coeff_token table (H.264 Table 9-5), which "
                "is not built. The CABAC encode of the same clip works.\n");
        else
            fprintf(stderr,
                "extractor: edge264 decoded no pictures from this stream's %d access "
                "units (%d NAL units unsupported, %d rejected as malformed).\n",
                r.packets, r.nal_unsup, r.nal_bad);
    } else if (r.nal_unsup || r.nal_bad) {
        fprintf(stderr,
            "extractor: %d NAL units failed to decode (%d unsupported, %d malformed); "
            "the CSV covers the %d pictures that did.\n",
            r.nal_unsup + r.nal_bad, r.nal_unsup, r.nal_bad, r.frames);
    }

    if (verbose)
        fprintf(stderr, "extractor (edge264): pictures=%d packets=%d mvs=%llu serial l0_only=%d\n",
                r.frames, r.packets, (unsigned long long)r.ex.count, l0_only);

    printf("%d %llu %ld %.3f\n", reported, (unsigned long long)r.ex.count, mem, decode_ms);
    return 0;
}
