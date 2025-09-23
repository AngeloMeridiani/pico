#ifndef PICOLIB_INTRUMENT_H
#define PICOLIB_INTRUMENT_H

#include <mpi.h>

#define PICOLIB_EXCL_MAX_TAGS 4

/* Returns 0 on success, negative on error. */
int PICOLIB_EXCL_RESET_ALL(void);                /* Clear all accumulated times, end any open tags. */
int PICOLIB_EXCL_BEGIN(const char *tag);         /* Start excluding under 'tag' (nesting per-tag OK). */
int PICOLIB_EXCL_END(const char *tag);           /* Stop excluding under 'tag'. */
double PICOLIB_EXCL_GET(const char *tag);        /* Current accumulated seconds for 'tag'. (No reset) */

/* Export all (name, seconds) pairs into user buffers.
   - names: output buffer of size 'max' for pointers to tag names (const char* you provided)
   - values: output buffer of size 'max' for seconds
   Returns the number of active tags written (0..EXCL_MAX_TAGS). */
int PICOLIB_EXCL_LIST(const char **names, double *values, int max);

/* Optional: clear accumulators but keep tag registrations (faster between iterations). */
int PICOLIB_EXCL_CLEAR_ACCUMS(void);

/* Convenience macro to wrap an excluded block under a tag. */
#define PICOLIB_EXCLUDE_BLOCK(TAG, CODE_BLOCK) \
    do { PICOLIB_EXCL_BEGIN(TAG); { CODE_BLOCK; } PICOLIB_EXCL_END(TAG); } while (0)


#endif /* PICOLIB_INTRUMENT_H */
