#ifndef HeterogeneousCore_Examples_test_kernel_cuh
#define HeterogeneousCore_Examples_test_kernel_cuh

#include <cuda_runtime.h>

__global__
void addNoise(int n, float* cellCharge, float* cellToa, bool weightMode, float* devRand, uint8_t* cellType, uint* word);

void addNoiseWrapper(int n, float* cellCharge, float* cellToa, bool weightMode, float* devRand, uint8_t* cellType, uint* word);

#endif
