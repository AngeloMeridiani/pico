/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */
#include <string.h>
#include <stdio.h>
#include <mpi.h>
#include "picolib.h"

#if defined PICO_INSTRUMENT && !defined PICO_NCCL && !defined PICO_MPI_CUDA_AWARE
// ----------------------------------------------------------------------------------------------
//                                Internal Data Structures
// ----------------------------------------------------------------------------------------------
typedef struct {
  const char* tag_name;
  double      accum;
  double      last_start;
  int         depth;
  int         active;
} picolib_tag_t;



typedef struct {
  picolib_tag_t*  tag;
  int             out_len;
  double*         out_buf;
} picolib_tag_handler_t;

static picolib_tag_t pico_tags[PICOLIB_MAX_TAGS];
static picolib_tag_handler_t pico_handles[PICOLIB_MAX_TAGS];
static int picolib_handles_built = 0;
// ----------------------------------------------------------------------------------------------
//                    Functions behind the PICOLIB_TAG_BEGIN/END macros
// ----------------------------------------------------------------------------------------------

/**
* @brief Find the index of a tag by name.
*
* @param tag The name of the tag to find.
* @return The index of the tag, or -1 if not found.
*
* @note This function is not to be called directly.
*/
static inline int _picolib_find_tag(const char *tag) {
  for (int i = 0; i < PICOLIB_MAX_TAGS; ++i) {
    if (pico_tags[i].active && strcmp(pico_tags[i].tag_name, tag) == 0) 
      return i;
  }
  return -1;
}

/**
 * @brief Ensure that a tag exists, creating it if necessary.
 *
 * @param tag The name of the tag to ensure.
 * @return The index of the tag, or -1 on error (e.g., maximum tags exceeded).
 *
 * @note This function is not to be called directly.
 */
static inline int _picolib_ensure_tag(const char *tag) {
  int idx = _picolib_find_tag(tag);
  if (idx >= 0) return idx;
  for (int i = 0; i < PICOLIB_MAX_TAGS; ++i) {
    if (!pico_tags[i].active) {
      pico_tags[i].active     = 1;
      pico_tags[i].tag_name   = tag;
      pico_tags[i].accum      = 0.0;
      pico_tags[i].last_start = 0.0;
      pico_tags[i].depth      = 0;
      return i;
    }
  }
  return -1;
}

int picolib_tag_begin(const char *tag) {
  if (!tag) {
    fprintf(stderr, "Error: NULL tag passed to picolib_tag_begin.\n");
    return -1;
  }
  int idx = _picolib_ensure_tag(tag);
  if (idx < 0) {
    fprintf(stderr, "Error: Maximum number of tags (%d) exceeded.\n", PICOLIB_MAX_TAGS);
    return -1;
  }

  if (pico_tags[idx].depth < 0) {
    fprintf(stderr, "Error: Tag '%s' has invalid depth '%d' before beginning.\n", 
            tag, pico_tags[idx].depth);
    return -1;
  }

  if (pico_tags[idx].depth == 0){
    pico_tags[idx].last_start = MPI_Wtime();
  }
  pico_tags[idx].depth++;
  return 0;
}

int picolib_tag_end(const char *tag) {
  if (!tag) {
    fprintf(stderr, "Error: NULL tag passed to picolib_tag_end.\n");
    return -1;
  }

  int idx = _picolib_find_tag(tag);
  if (idx < 0) {
    fprintf(stderr, "Error: Tag '%s' was not initialized before ending.\n", tag);
    return -1;
  }

  if (pico_tags[idx].depth <= 0) {
    fprintf(stderr, "Error: Tag '%s' was not properly begun before ending.\n", tag);
    return-1;
  }

  pico_tags[idx].depth -= 1;
  if (pico_tags[idx].depth == 0){
    pico_tags[idx].accum += MPI_Wtime() - pico_tags[idx].last_start;
  }
  return 0;
}


// ----------------------------------------------------------------------------------------------
//                    Functions for managing tags
// ----------------------------------------------------------------------------------------------


/**
 * @brief Initialize all tags to unused state.
 *
 * @note This function is not to be called directly; use picolib_init_tags() instead.
 */
static inline void picolib_initialize_all_tags(void) {
  for (int i = 0; i < PICOLIB_MAX_TAGS; ++i) {
    pico_tags[i].tag_name   = NULL;
    pico_tags[i].accum  = 0.0;
    pico_tags[i].last_start = 0.0;
    pico_tags[i].depth = 0;
    pico_tags[i].active = 0;
  }
}

/**
 * @brief Initialize all bindings to unused state.
 *
 *  @note This function is not to be called directly; use picolib_init_tags() instead.
 */
static inline void picolib_initialize_all_bindings(void) {
  for (int i = 0; i < PICOLIB_MAX_TAGS; ++i) {
    pico_handles[i].tag     = NULL;
    pico_handles[i].out_len = 0;
    pico_handles[i].out_buf = NULL;
  }
}

void picolib_init_tags(void) {
  picolib_initialize_all_tags();
  picolib_initialize_all_bindings();
  picolib_handles_built = 0;
}


int picolib_count_tags(void) {
  int n = 0;
  for (int i = 0; i < PICOLIB_MAX_TAGS; ++i) {
    if (pico_tags[i].active) ++n;
  }
  return n;
}

int picolib_get_tag_names(const char **names, int count) {
  if (names == NULL || count <= 0) {
    fprintf(stderr, "Error: Invalid arguments to picolib_get_tag_names.\n");
    return -1;
  }

  int written = 0;
  for (int i = 0; i < PICOLIB_MAX_TAGS && written < count; ++i) {
    if (!pico_tags[i].active) continue;

    if (pico_tags[i].tag_name == NULL) {
      fprintf(stderr, "Error: Inconsistent state: active tag with NULL name.\n");
      return -1;
    }
    names[written++] = pico_tags[i].tag_name;
  }

  if (written != count) {
    fprintf(stderr, "Error: Mismatch in tag count. Expected %d, found %d.\n", count, written);
    return -1;
  }
  return 0;
}

int picolib_build_handles(double **bufs, int k, int out_len) {
  if (!bufs || k <= 0 || out_len <= 0) {
    fprintf(stderr, "Error: Invalid arguments to picolib_build_handles.\n");
    return -1;
  }

  int tag_cnt = 0;
  for (int i = 0; i < PICOLIB_MAX_TAGS; ++i)
    if (pico_tags[i].active) ++tag_cnt;

  if (tag_cnt != k){
    fprintf(stderr, "Error: Number of active tags (%d) does not match number of buffers (%d).\n",
            tag_cnt, k);
    return -1;
  }

  int seen = 0;
  for (int i = 0; i < PICOLIB_MAX_TAGS && seen < k; ++i) {
    if (!pico_tags[i].active) continue;

    if (!bufs[seen]){
      fprintf(stderr, "Error: NULL buffer provided for tag '%s'.\n", pico_tags[i].tag_name);
      return -1;
    }

    pico_handles[seen].tag = &pico_tags[i];
    pico_handles[seen].out_buf = bufs[seen];
    pico_handles[seen].out_len = out_len;
    ++seen;
  }
  if (seen != k){
    fprintf(stderr, "Error: Mismatch in tag count. Expected %d, found %d.\n", k, seen);
    return -1;
  }
  picolib_handles_built = k;
  return 0;
}


int picolib_clear_tags(void) {
  for (int i = 0; i < PICOLIB_MAX_TAGS; ++i) {
    if (!pico_tags[i].active) continue;

    if (pico_tags[i].depth != 0) {
      fprintf(stderr, "Error: Tag '%s' was not properly ended before clearing.\n", pico_tags[i].tag_name);
      return -1;
    }
    pico_tags[i].last_start = 0.0;
    pico_tags[i].accum = 0.0;
  }
  return 0;
}

int picolib_snapshot_store(int iter_idx) {
  int k = picolib_handles_built;
  if (iter_idx < 0 || k <= 0) {
    fprintf(stderr, "Error: Invalid arguments to picolib_snapshot_store (iter_idx=%d, k=%d).\n", iter_idx, k);
    return -1;
  }

  for (int i = 0; i < k; ++i) {
    picolib_tag_t *tag = pico_handles[i].tag;
    if (!tag || !tag->active || !pico_handles[i].out_buf) {
      fprintf(stderr, "Error: Invalid handle at index %d.\n", i);
      return -1;
    }
    if (iter_idx >= pico_handles[i].out_len) {
      fprintf(stderr, "Error: Iteration index %d out of bounds for handle %d (length %d).\n",
              iter_idx, i, pico_handles[i].out_len);
      return -1;
    }
    if (tag->depth != 0) {
      fprintf(stderr, "Error: Tag '%s' must be closed (depth==0) before snapshot.\n", tag->tag_name);
      return -1;
    }
    double v = tag->accum;
    if (v < 0.0) {
      fprintf(stderr, "Error: Inconsistent state: tag '%s' has negative accumulated time.\n", tag->tag_name);
      return -1;
    }
    pico_handles[i].out_buf[iter_idx] = v;
  }
  return 0;
}

#else

int picolib_tag_begin(const char *tag) {
  return 0;
}

int picolib_tag_end(const char *tag) {
  return 0;
}

void picolib_init_tags(void) {
}


int picolib_count_tags(void) {
  return 0;
}

int picolib_get_tag_names(const char **names, int count) {
  return 0;
}

int picolib_build_handles(double **bufs, int k, int out_len) {
  return 0;
}

int picolib_get_handles_built(void) { return 0; }

int picolib_clear_tags(void) {
  return 0;
}

int picolib_snapshot_store(int iter_idx) {
  return 0;
}

#endif // PICO_INSTRUMENT

