#include <cuda.h>
#include <curand.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

__global__
void addNoise(int n, float* cellCharge, float* cellToa, bool weightMode, float* devRand, uint8_t* cellType, uint* word)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  float rawCharge(cellCharge[i]);
  float toa(cellToa[i]);
  float randNum(devRand[i]);
  uint8_t type(cellType[i]);

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


void addNoiseWrapper(int n, float* cellCharge, float* cellToa, bool weightMode, float* devRand, uint8_t* cellType, uint* word,curandGenerator_t &gen)
{    

  //Generate n floats on device
  std::cout << "--> N " << n << std::endl;
  curandGenerateNormal(gen, devRand, n, 0.f, 1.f);
  std::cout << "--> DONE" << std::endl;

  //call function on the GPU
  addNoise<<<(n+255)/256, 256>>>(n, cellCharge, cellToa, weightMode, devRand, cellType, word);
  std::cout << "--> DONE NOISE" << std::endl;
}         
