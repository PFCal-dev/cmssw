#include <cuda.h>
#include <curand.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <assert.h>
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"


__global__
void addNoise(const int n, const float* cellCharge, const float* cellToa, const bool weightMode, const float* rand, const uint16_t* cellType, uint32_t* word)
{
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) //protection
  {
   float rawCharge(cellCharge[i]);
   float toa(cellToa[i]);
   float randNum(rand[i]);
   uint16_t type(cellType[i]);

   constexpr float noise[3] = {0.168,0.336,0.256};

   if(weightMode && rawCharge>0)
     toa = toa/rawCharge;

   float totalCharge = rawCharge;
   totalCharge += randNum*noise[type];
   if(totalCharge<0.f) totalCharge=0.f;

   constexpr float inv_lsb = 1./0.0977;
   constexpr float inv_time_lsb = 1./0.0244;
   bool passThr=(totalCharge>0.672f);
   uint16_t finalCharge=(uint16_t)(fminf( totalCharge, 100.f)*inv_lsb);
   uint16_t finalToA=(uint16_t)(toa*inv_time_lsb);

   word[i] = ( (passThr<<31) |
             ((finalToA & 0x3ff) <<13) |
             ((finalCharge & 0xfff)));

  }
}


void addNoiseWrapper(int n, float* cellCharge, float* cellToa, bool weightMode, float* rand, uint16_t* cellType, uint32_t* word, curandGenerator_t &gen)
{

  //Generate n floats on device
  //std::cout << "--> N " << n << std::endl;
  curandGenerateNormal(gen, rand, n, 0.f, 1.f);
  //std::cout << "--> DONE CURAND" << std::endl;

  //call function on the GPU
  addNoise<<<(n+255)/256, 256>>>(n, cellCharge, cellToa, weightMode, rand, cellType, word);
  // cudaCheck(cudaDeviceSynchronize());
  // cudaCheck(cudaGetLastError());
  //std::cout << "--> DONE NOISE" << std::endl;
  //std::cout << std::endl;
}
