#ifndef DGDFT_FLOPS_H
#define DGDFT_FLOPS_H
#include <mpi.h>

#include "environment.hpp"
#include "utility.hpp"
#include <iostream>


namespace dgdft {

struct flops_count_t {
  double time;
  unsigned long flops[2];
};

void flops_count_init();

flops_count_t flops_count();

flops_count_t flops_count_sum(const flops_count_t &cnt,
                              const MPI_Comm &flops_comm);

} // namespace dgdft

#endif // !DGDFT_FLOPS_H
