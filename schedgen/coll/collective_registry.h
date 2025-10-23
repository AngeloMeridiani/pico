/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */
#ifndef SCHEDGEN_COLL_COLLECTIVE_REGISTRY_H
#define SCHEDGEN_COLL_COLLECTIVE_REGISTRY_H

#include <functional>
#include <map>
#include <string>
#include <vector>

class Goal;
struct CollectiveContext;

enum class CollectiveKind {
  Barrier,
  Allreduce,
  Iallreduce,
  Bcast,
  Allgather,
  Reduce,
  Alltoall
};

struct CollectiveContext {
  int root = 0;
  int replace_comp_time = -1;
  int segmentsize = 0;
};

using CollectiveGenerator =
    std::function<void(Goal *, int rank, int comm_size, int data_size,
                       const CollectiveContext &ctx)>;

void register_collective_algorithm(CollectiveKind kind,
                                   const std::string &name,
                                   CollectiveGenerator generator);

CollectiveGenerator
lookup_collective_algorithm(CollectiveKind kind, const std::string &name);

std::vector<std::string> list_collective_algorithms(CollectiveKind kind);

const char *collective_kind_to_string(CollectiveKind kind);

#endif // SCHEDGEN_COLL_COLLECTIVE_REGISTRY_H
