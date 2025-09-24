/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */
#ifndef PICOLIB_INSTRUMENT_H
#define PICOLIB_INSTRUMENT_H

#define PICOLIB_MAX_TAGS 32

// ----------------------------------------------------------------------------------------------
//                        PUBLIC API Maros for instrumentation
// ----------------------------------------------------------------------------------------------
int picolib_tag_begin(const char *tag);
int picolib_tag_end(const char *tag);

#define PICO_TAG_BEGIN(TAG) do {       \
  if (picolib_tag_begin((TAG)) != 0) {  \
    return -1;                          \
  }                                     \
} while (0)

#define PICO_TAG_END(TAG) do {         \
  if (picolib_tag_end((TAG)) != 0) {    \
    return -1;                          \
  }                                     \
} while (0)


// ----------------------------------------------------------------------------------------------
//                   Functions for managing tags, used in pico core
// ----------------------------------------------------------------------------------------------
/**
 * @brief Reset all tags to unused state. Pico core call this once 
 *        at the start of the benchmarking run.
 */
void picolib_reset_all_tags(void);

/**
 * @brief Count of currenly used tags for sizing arrays in pico core
 *
 * @return number of tags in use (<= PICOLIB_MAX_TAGS)
 */
int picolib_count_tags(void);

/**
 * @brief Writes the names of the currently used tags into the provided array.
 *
 * @param names Array to write the tag names into.
 * @param max Current number of tags in use (must be retrieved via picolib_count_tags()).
 *
 * @return 0 on success, -1 on error.
 */
int picolib_get_tag_names(const char **names, int count);

/**
 * @brief Clear all tags, resetting their state.
 *
 * @return 0 on success, -1 on error.
 */
int picolib_clear_tags(void);


#endif /* PICOLIB_INSTRUMENT_H */
