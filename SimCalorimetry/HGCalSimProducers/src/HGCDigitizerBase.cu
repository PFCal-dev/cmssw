#include <cuda.h>
#include <curand.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

__global__
void addNoise(int n, float* cellCharge, float* cellToa, bool weightMode, float* devRand, uint8_t* cellType, uint* word)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x + blockDim.x*gridDim.x*blockDim.y*blockIdx.y;
  printf("\ni = %d", i);
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
  std::cout << "\n--> n = " << n << std::endl;
  curandGenerateNormal(gen, devRand, n, 0.f, 1.f);
  std::cout << "--> DONE generation" << std::endl;

  //call function on the GPU
  int device_id = 0;
  int maxThreadsPerBlock = -1;
  cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device_id);

  int minBlocksNum = n / maxThreadsPerBlock  +  n % maxThreadsPerBlock;
  int blocksDimXNum = n / maxThreadsPerBlock  +  n % maxThreadsPerBlock;
  int maxBlockDimX = -1;
  cudaDeviceGetAttribute(&maxBlockDimX, cudaDevAttrMaxBlockDimX, device_id);

  int blocksDimYNum = 1;
  int maxBlockDimY = -1;
  cudaDeviceGetAttribute(&maxBlockDimY, cudaDevAttrMaxBlockDimY, device_id);
  if (minBlocksNum > maxBlockDimX) {
    std::cout <<"minBlocksNum (" <<minBlocksNum <<") for a 1D-block 1D kernel is larger than maxBlockDimX (" <<maxBlockDimX <<"), will consider a 1D-block 2D kernel!" <<std::endl;
    blocksDimXNum = maxBlockDimX;
    int minBlocksYNum = minBlocksNum / maxBlockDimX + minBlocksNum % maxBlockDimX;

    if (minBlocksYNum > maxBlockDimY) {
      std::cout <<"minBlocksYNum (" <<minBlocksYNum <<") for a 1D-block 2D kernel is larger than maxBlockDimY (" <<maxBlockDimY <<"), will run a 1D-block 3D kernel!" <<std::endl;
      return;
    }

    blocksDimYNum = minBlocksYNum;
  }

  dim3 dimGrid(blocksDimXNum, blocksDimYNum);
  std::cout <<"Running kernel with " <<blocksDimXNum <<" x-blocks, " <<blocksDimYNum <<" y-blocks and " <<maxThreadsPerBlock <<" threads per block" <<std::endl;
  addNoise<<<dimGrid, maxThreadsPerBlock>>>(n, cellCharge, cellToa, weightMode, devRand, cellType, word);
  std::cout << "--> DONE NOISE" << std::endl;
}         
