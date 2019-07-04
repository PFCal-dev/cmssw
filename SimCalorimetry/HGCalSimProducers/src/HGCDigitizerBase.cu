#include <cuda.h>
#include <curand.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <assert.h>
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"


__global__
void addNoise(int n, float* cellCharge, float* cellToa, bool weightMode, float* rand, uint16_t* cellType, uint32_t* word)
{
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) //protection
  {
   float rawCharge(cellCharge[i]);
   float toa(cellToa[i]);
   float randNum(rand[i]);
   uint16_t type(cellType[i]);

   float noise[] = {0.168,0.336,0.256};

   if(weightMode && rawCharge>0)
     toa = toa/rawCharge;

   float totalCharge = rawCharge;
   totalCharge += std::max(randNum*noise[type], 0.f);
   if(totalCharge<0.f) totalCharge=0.f;

   bool passThr=(totalCharge>0.672);
   uint16_t finalCharge=(uint16_t)(fminf( totalCharge, 100.)/0.0977);
   uint16_t finalToA=(uint16_t)(toa/0.0244);

   word[i] = ( (passThr<<31) |
             ((finalToA & 0x3ff) <<13) |
             ((finalCharge & 0xfff)));

  }
}


void addNoiseWrapper(int n, float* cellCharge, float* cellToa, bool weightMode, float* rand, uint16_t* cellType, uint32_t* word, curandGenerator_t &gen)
{

  //Generate n floats on device
  std::cout << "--> N " << n << std::endl;
  curandGenerateNormal(gen, rand, n, 0.f, 1.f);
  std::cout << "--> DONE CURAND" << std::endl;

  //call function on the GPU
  addNoise<<<(n+255)/256, 256>>>(n, cellCharge, cellToa, weightMode, rand, cellType, word);
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaGetLastError());
  std::cout << "--> DONE NOISE" << std::endl;
  std::cout << std::endl;
}
