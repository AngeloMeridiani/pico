#include "picolib_intrument.h"
#include <string.h>

typedef struct {
    const char *name;   /* pointer you pass (assumed valid for program lifetime or file-scope string) */
    double accum;       /* accumulated excluded seconds */
    double last_start;  /* last MPI_Wtime() when EXCL_BEGIN called */
    int    depth;       /* nesting depth for this tag */
    int    in_use;      /* 1 if slot is used */
} picolib_excl_entry_t;

static picolib_excl_entry_t g_tags[PICOLIB_EXCL_MAX_TAGS];

static int _picolib_find_tag(const char *tag) {
    for (int i = 0; i < PICOLIB_EXCL_MAX_TAGS; ++i) {
        if (g_tags[i].in_use && strcmp(g_tags[i].name, tag) == 0) return i;
    }
    return -1;
}

static int _picolib_ensure_tag(const char *tag) {
    int idx = _picolib_find_tag(tag);
    if (idx >= 0) return idx;
    for (int i = 0; i < PICOLIB_EXCL_MAX_TAGS; ++i) {
        if (!g_tags[i].in_use) {
            g_tags[i].in_use    = 1;
            g_tags[i].name      = tag;
            g_tags[i].accum     = 0.0;
            g_tags[i].last_start= 0.0;
            g_tags[i].depth     = 0;
            return i;
        }
    }
    return -1; /* no space */
}

int PICOLIB_EXCL_RESET_ALL(void) {
    for (int i = 0; i < PICOLIB_EXCL_MAX_TAGS; ++i) {
        g_tags[i].in_use = 0;
        g_tags[i].name   = NULL;
        g_tags[i].accum  = 0.0;
        g_tags[i].last_start = 0.0;
        g_tags[i].depth  = 0;
    }
    return 0;
}

int PICOLIB_EXCL_CLEAR_ACCUMS(void) {
    for (int i = 0; i < PICOLIB_EXCL_MAX_TAGS; ++i) {
        if (g_tags[i].in_use) {
            g_tags[i].accum = 0.0;
            /* keep name, depth must be 0 between iterations ideally */
            if (g_tags[i].depth != 0) {
                /* If a tag is still open, close it implicitly */
                g_tags[i].accum += MPI_Wtime() - g_tags[i].last_start;
                g_tags[i].depth = 0;
            }
        }
    }
    return 0;
}

int PICOLIB_EXCL_BEGIN(const char *tag) {
    if (!tag) return -2;
    int idx = _picolib_ensure_tag(tag);
    if (idx < 0) return -1; /* out of slots */
    /* Support nesting: only stamp a start time on the transition from depth 0 -> 1 */
    if (g_tags[idx].depth == 0) {
        g_tags[idx].last_start = MPI_Wtime();
    }
    g_tags[idx].depth += 1;
    return 0;
}

int PICOLIB_EXCL_END(const char *tag) {
    if (!tag) return -2;
    int idx = _picolib_find_tag(tag);
    if (idx < 0) return -3; /* tag not found */
    if (g_tags[idx].depth <= 0) return -4; /* unmatched END */
    g_tags[idx].depth -= 1;
    if (g_tags[idx].depth == 0) {
        g_tags[idx].accum += MPI_Wtime() - g_tags[idx].last_start;
    }
    return 0;
}

double PICOLIB_EXCL_GET(const char *tag) {
    int idx = _picolib_find_tag(tag);
    if (idx < 0) return 0.0;
    /* If currently open (depth>0), include time up to now without closing */
    if (g_tags[idx].depth > 0) {
        return g_tags[idx].accum + (MPI_Wtime() - g_tags[idx].last_start);
    }
    return g_tags[idx].accum;
}

int PICOLIB_EXCL_LIST(const char **names, double *values, int max) {
    int n = 0;
    for (int i = 0; i < PICOLIB_EXCL_MAX_TAGS && n < max; ++i) {
        if (g_tags[i].in_use) {
            names[n]  = g_tags[i].name;
            /* Include open time as well (non-destructive) */
            double v = g_tags[i].accum;
            if (g_tags[i].depth > 0) v += MPI_Wtime() - g_tags[i].last_start;
            values[n] = v;
            ++n;
        }
    }
    return n;
}
