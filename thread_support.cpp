#include <cstdio>

#include <iostream>
#include <vector>

#include <errno.h>

#include "thread_support.hpp"

void pin_threads()
{
#pragma omp parallel default(none), shared(std::cerr)
  {
    int tid = omp_get_thread_num();
    int ret = tmc_cpus_set_my_cpu(tid);

    if (ret != 0) {
      perror(NULL);
      std::cerr << "Error pinning thread: " << tid << " to tile: " << tid << std::endl;
    }
  }
} // pin_threads

void check_pinned_threads()
{
  std::vector<int> pinning(omp_get_max_threads());

#pragma omp parallel default(none), shared(std::cerr, pinning)
  {
    int tid = omp_get_thread_num();
    int tile = tmc_cpus_get_my_cpu();

    if (tile == -1) {
      perror(NULL);
      std::cerr << "Thread: " << tid << " not bound to any tile" << std::endl;
    }
    else if (tid != tile) {
      std::cerr << "Thread: " << tid << " bound to the wrong tile: " << tile << std::endl;
    }
    pinning[tid] = tile;
  }

  for (int i = 0; i < pinning.size(); i++)
    std::cout << "Thread: " << i << ", pinned on tile: " << pinning[i] << std::endl;
} // check_pinned_threads
