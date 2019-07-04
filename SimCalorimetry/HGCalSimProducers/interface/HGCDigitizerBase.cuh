#ifndef HeterogeneousCore_digi_kernel_cuh
#define HeterogeneousCore_digi_kernel_cuh

#include <cuda_runtime.h>
#include <curand.h>

__global__
void addNoise(int n, float* cellCharge, float* cellToa, bool weightMode, float* rand, uint16_t* cellType, uint* word);

void addNoiseWrapper(int n, float* cellCharge, float* cellToa, bool weightMode, float* rand, uint16_t* cellType, uint* word, curandGenerator_t &gen);

#endif
