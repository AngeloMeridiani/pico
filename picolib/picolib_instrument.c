/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */
#include <string.h>
#include <stdio.h>
#include <mpi.h>
#include "picolib_instrument.h"

// ----------------------------------------------------------------------------------------------
//                                Internal Data Structures
// ----------------------------------------------------------------------------------------------
typedef struct {
  const char  *tag_name;
  double      accum;
  double      last_start;
  int         depth;
  int         active;
} picolib_tag_entry_t;

static picolib_tag_entry_t pico_tags[PICOLIB_MAX_TAGS];

// ----------------------------------------------------------------------------------------------
//                                Internal Helper Functions
// ----------------------------------------------------------------------------------------------
static int _picolib_find_tag(const char *tag) {
  for (int i = 0; i < PICOLIB_MAX_TAGS; ++i) {
    if (pico_tags[i].active && strcmp(pico_tags[i].tag_name, tag) == 0) 
      return i;
  }
  return -1;
}

static int _picolib_ensure_tag(const char *tag) {
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


// ----------------------------------------------------------------------------------------------
//                    Functions behind the PICOLIB_TAG_BEGIN/END macros
// ----------------------------------------------------------------------------------------------
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

  if (pico_tags[idx].depth != 0) {
    fprintf(stderr, "Error: Tag '%s' was not properly ended before beginning again.\n", tag);
    return -1;
  }

  pico_tags[idx].depth++;
  if (pico_tags[idx].depth != 1) {
    fprintf(stderr, "Error: Tag '%s' has invalid depth after beginning.\n", tag);
    return -1;
  }
  pico_tags[idx].last_start = MPI_Wtime();
  return 0;
}

int picolib_tag_end(const char *tag) {
  if (!tag) {
    fprintf(stderr, "Error: NULL tag passed to picolib_tag_begin.\n");
    return -1;
  }

  int idx = _picolib_find_tag(tag);
  if (idx < 0) {
    fprintf(stderr, "Error: Tag '%s' was not initialized before ending.\n", tag);
    return -1;
  }

  if (pico_tags[idx].depth <= 0) {
    fprintf(stderr, "Error: Tag '%s' was not properly begun before ending.\n", tag);
    return -1;
  }

  pico_tags[idx].depth -= 1;
  if (pico_tags[idx].depth != 0) {
    fprintf(stderr, "Error: Tag '%s' has invalid depth after ending.\n", tag);
    return -1;
  }

  pico_tags[idx].accum += MPI_Wtime() - pico_tags[idx].last_start;
  return 0;
}


// ----------------------------------------------------------------------------------------------
//                    Functions for managing tags
// ----------------------------------------------------------------------------------------------

int picolib_count_tags(void) {
  int n = 0;
  for (int i = 0; i < PICOLIB_MAX_TAGS; ++i) {
    if (pico_tags[i].active) ++n;
  }
  return n;
}

void picolib_reset_all_tags(void) {
  for (int i = 0; i < PICOLIB_MAX_TAGS; ++i) {
    pico_tags[i].tag_name   = NULL;
    pico_tags[i].accum  = 0.0;
    pico_tags[i].last_start = 0.0;
    pico_tags[i].depth = 0;
    pico_tags[i].active = 0;
  }
}

int picolib_get_tag_names(const char **names, int count) {
  if (names == NULL || count <= 0) {
    fprintf(stderr, "Error: Invalid arguments to picolib_get_tag_names.\n");
    return -1;
  }

  int written = 0;
  for (int i = 0; i < PICOLIB_MAX_TAGS && written < count; ++i) {
    if (pico_tags[i].active) {
      if (pico_tags[i].tag_name == NULL) {
        fprintf(stderr, "Error: Inconsistent state: active tag with NULL name.\n");
        return -1;
      }
      names[written++] = pico_tags[i].tag_name;
    }
  }

  if (written != count) {
    fprintf(stderr, "Error: Mismatch in tag count. Expected %d, found %d.\n", count, written);
    return -1;
  }
  return 0;
}

int picolib_clear_tags(void) {
  for (int i = 0; i < PICOLIB_MAX_TAGS; ++i) {
    if (!pico_tags[i].active) continue;

    if (pico_tags[i].depth > 0) {
      fprintf(stderr, "Error: Tag '%s' was not properly ended before clearing.\n", pico_tags[i].tag_name);
      return -1;
    }
    pico_tags[i].accum = 0.0;
  }
  return 0;
}

