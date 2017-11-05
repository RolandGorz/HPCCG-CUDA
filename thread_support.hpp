#ifndef __THREAD_SUPPORT_H
#define __THREAD_SUPPORT_H

#include <omp.h>

#include <tmc/cpus.h>

void pin_threads();
void check_pinned_threads();

#endif // __THREAD_SUPPORT_H
