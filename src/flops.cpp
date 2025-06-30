#include "flops.h"
//#include <swperf.h>
#include "environment.hpp"
#include "utility.hpp"
#include <iostream>



namespace dgdft {

void flops_count_init() {
  penv_slave_fd_float_init();
  penv_host1_float_muletc_init();
}

flops_count_t flops_count() {
  flops_count_t cnt;
  cnt.time = MPI_Wtime();
  penv_host1_float_muletc_count(&cnt.flops[0]);
  penv_slave_fd_float_sum_count(&cnt.flops[1]);
  return cnt;
}

flops_count_t flops_count_sum(const flops_count_t &cnt,
                              const MPI_Comm &flops_comm) {
  MPI_Comm rowComm, colComm;
  flops_count_t cnt_end;
  flops_count_t cnt_sum;
  int mpiRank;
  int mpiSize;

  // MPI_Barrier(flops_comm);
  cnt_end = flops_count();
  cnt_end.time -= cnt.time;
  cnt_end.flops[0] -= cnt.flops[0];
  cnt_end.flops[1] -= cnt.flops[1];

  MPI_Comm_rank(flops_comm, &mpiRank);
  MPI_Comm_size(flops_comm, &mpiSize);

  int rowColor = mpiRank / 4096;
  int colColor = mpiRank % 4096;
  MPI_Comm_split(flops_comm, rowColor, 0, &rowComm);
  MPI_Comm_split(flops_comm, colColor, 0, &colComm);
  MPI_Reduce(cnt_end.flops, &cnt_sum.flops, 2, MPI_UNSIGNED_LONG, MPI_SUM, 0,
             rowComm);
  MPI_Reduce(cnt_sum.flops, &cnt_end.flops, 2, MPI_UNSIGNED_LONG, MPI_SUM, 0,
             colComm);
  if (mpiRank == 0) {
    return cnt_end;
  }
  {
    return {0, 0, 0};
  }
}

} // namespace dgdft
