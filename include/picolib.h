/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */

#ifndef PICOLIB_H
#define PICOLIB_H

#include <mpi.h>
#include <stddef.h>

#define ALLREDUCE_MPI_ARGS        const void *sbuf, void *rbuf, size_t count, \
                                  MPI_Datatype dtype, MPI_Op op, MPI_Comm comm
#define ALLGATHER_MPI_ARGS        const void *sbuf, size_t scount, MPI_Datatype sdtype, \
                                  void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm
#define ALLTOALL_MPI_ARGS         const void *sbuf, size_t scount, MPI_Datatype sdtype, \
                                  void *rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm
#define BCAST_MPI_ARGS            void *buf, size_t count, MPI_Datatype dtype, int root, MPI_Comm comm
#define GATHER_MPI_ARGS           const void *sbuf, size_t scount, MPI_Datatype sdtype, \
                                  void *rbuf, size_t rcount, MPI_Datatype rdtype, int root, MPI_Comm comm
#define REDUCE_MPI_ARGS           const void *sbuf, void *rbuf, size_t count, \
                                  MPI_Datatype dtype, MPI_Op op, int root, MPI_Comm comm
#define REDUCE_SCATTER_MPI_ARGS   const void *sbuf, void *rbuf, const int rcounts[], \
                                  MPI_Datatype dtype, MPI_Op op, MPI_Comm comm
#define SCATTER_MPI_ARGS          const void *sbuf, size_t scount, MPI_Datatype sdtype, \
                                  void *rbuf, size_t rcount, MPI_Datatype rdtype, int root, MPI_Comm comm
#ifdef PICO_NCCL
#include <nccl.h>
#include <cuda_runtime.h>

#define ALLREDUCE_NCCL_ARGS       const void *sbuf, void *rbuf, size_t count, ncclDataType_t dtype, \
                                  ncclRedOp_t op, ncclComm_t nccl_comm, cudaStream_t stream
#define ALLGATHER_NCCL_ARGS       const void *sbuf, void* rbuf, size_t count, \
                                  ncclDataType_t dtype, ncclComm_t nccl_comm, cudaStream_t stream
#define ALLTOALL_NCCL_ARGS        const void *sbuf, void* rbuf, size_t count,\
                                  ncclDataType_t dtype, ncclComm_t nccl_comm, cudaStream_t stream
#define BCAST_NCCL_ARGS           void *buf, size_t count, ncclDataType_t dtype, \
                                  int root, ncclComm_t nccl_comm, cudaStream_t stream
#define GATHER_NCCL_ARGS          const void *sbuf, void *rbuf, size_t count, ncclDataType_t dtype, \
                                  int root, ncclComm_t nccl_comm, cudaStream_t stream
#define REDUCE_NCCL_ARGS          const void *sbuf, void *rbuf, size_t count, ncclDataType_t dtype, \
                                  ncclRedOp_t op, int root, ncclComm_t nccl_comm, cudaStream_t stream
#define REDUCE_SCATTER_NCCL_ARGS  const void *sbuf, void *rbuf, size_t rcount, ncclDataType_t dtype, \
                                  ncclRedOp_t op, ncclComm_t nccl_comm, cudaStream_t stream
#define SCATTER_NCCL_ARGS         const void *sbuf, void *rbuf, size_t count, ncclDataType_t dtype, \
                                  int root, ncclComm_t nccl_comm, cudaStream_t stream
#endif

extern size_t bine_allreduce_segsize;

int allreduce_recursivedoubling(ALLREDUCE_MPI_ARGS);
int allreduce_ring(ALLREDUCE_MPI_ARGS);
int allreduce_rabenseifner(ALLREDUCE_MPI_ARGS);
int allreduce_bine_lat(ALLREDUCE_MPI_ARGS);
int allreduce_bine_bdw_static(ALLREDUCE_MPI_ARGS);
int allreduce_bine_bdw_remap(ALLREDUCE_MPI_ARGS);
int allreduce_bine_bdw_remap_segmented(ALLREDUCE_MPI_ARGS);
int allreduce_bine_block_by_block_any_even(ALLREDUCE_MPI_ARGS);

int allgather_k_bruck(ALLGATHER_MPI_ARGS);
int allgather_recursivedoubling(ALLGATHER_MPI_ARGS);
int allgather_ring(ALLGATHER_MPI_ARGS);
int allgather_sparbit(ALLGATHER_MPI_ARGS);
int allgather_bine_block_by_block(ALLGATHER_MPI_ARGS);
int allgather_bine_block_by_block_any_even(ALLGATHER_MPI_ARGS);
int allgather_bine_send_remap(ALLGATHER_MPI_ARGS);
int allgather_bine_2_blocks(ALLGATHER_MPI_ARGS);
int allgather_bine_2_blocks_dtype(ALLGATHER_MPI_ARGS);

int alltoall_bine(ALLTOALL_MPI_ARGS);

int bcast_scatter_allgather(BCAST_MPI_ARGS);
int bcast_bine_lat(BCAST_MPI_ARGS);
int bcast_bine_lat_reversed(BCAST_MPI_ARGS);
int bcast_bine_lat_new(BCAST_MPI_ARGS);
int bcast_bine_lat_i_new(BCAST_MPI_ARGS);
int bcast_bine_bdw_remap(BCAST_MPI_ARGS);

int gather_bine(GATHER_MPI_ARGS);

int reduce_bine_lat(REDUCE_MPI_ARGS);
int reduce_bine_bdw(REDUCE_MPI_ARGS);

int reduce_scatter_recursivehalving(REDUCE_SCATTER_MPI_ARGS);
int reduce_scatter_recursive_distance_doubling(REDUCE_SCATTER_MPI_ARGS);
int reduce_scatter_ring(REDUCE_SCATTER_MPI_ARGS);
int reduce_scatter_butterfly(REDUCE_SCATTER_MPI_ARGS);
int reduce_scatter_bine_send_remap(REDUCE_SCATTER_MPI_ARGS);
int reduce_scatter_bine_permute_remap(REDUCE_SCATTER_MPI_ARGS);
int reduce_scatter_bine_block_by_block(REDUCE_SCATTER_MPI_ARGS);
int reduce_scatter_bine_block_by_block_any_even(REDUCE_SCATTER_MPI_ARGS);

int scatter_bine(SCATTER_MPI_ARGS);


/**
 * Instrumentation support
 */
#if defined PICO_INSTRUMENT && !defined PICO_NCCL && !defined PICO_MPI_CUDA_AWARE
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
 * @brief Init all tags and bindings to unused state. Pico core call this once 
 *        at the start of the benchmarking run.
 */
void picolib_init_tags(void);

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
 * @brief Binds the provided buffers to the currently active tags.
 *
 * @param bufs Array of buffers, one for each active tag.
 * @param k Number of active tags (must be retrieved via picolib_count_tags()).
 * @param out_len Length of each buffer (must be = benchmaking iter).
 *
 * @return 0 on success, -1 on error.
 */
int picolib_buiuld_handles(double **bufs, int k, int out_len);


/**
 * @brief Clear all active tag accum and last start values.
 *
 * @return 0 on success, -1 on error.
 */
int picolib_clear_tags(void);

/**
 * @brief Store the current accum values of all active tags into the provided buffers.
 *
 * @param iter_idx Current benchmarking iteration index.
 * @param k Number of active tags (must be retrieved via picolib_count_tags()).
 *
 * @return 0 on success, -1 on error.
 */
int picolib_snapshot_store(int iter_idx, int k);

#endif // PICO_INSTRUMENT


#endif // PICOLIB_H
