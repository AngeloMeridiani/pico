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

#endif
