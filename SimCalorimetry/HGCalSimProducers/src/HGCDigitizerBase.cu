#include <cuda.h>
#include <curand.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

__global__
void addNoise(int n, float* cellCharge, float* cellToa, bool weightMode, float* devRand, uint16_t* cellType, uint* word)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x + blockDim.x*gridDim.x*blockDim.y*blockIdx.y;
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


void addNoiseWrapper(int n, float* cellCharge, float* cellToa, bool weightMode, float* devRand, uint16_t* cellType, uint* word,curandGenerator_t &gen)
{    
  int device_id = atoi(getenv("CUDA_VISIBLE_DEVICES")); // run nvidia-smi and make sure this GPU is free
  //cudaSetDevice(device_id);
  std::cout <<"Using GPU id " <<device_id <<std::endl;

  //Generate n floats on device
  std::cout << "\n--> n = " << n << std::endl;
  curandGenerateNormal(gen, devRand, n, 0.f, 1.f);
  std::cout << "--> DONE generation" << std::endl;

  //call function on the GPU
  int maxThreadsPerBlock = 1024;
  //cudaError_t err0 = cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device_id);
  //if (err0) {printf("err = %d, %s\n", err0, cudaGetErrorString(err0)); return;}

  int minBlocksNum = n / maxThreadsPerBlock  +  n % maxThreadsPerBlock;
  int gridDimX = minBlocksNum;
  int maxGridDimX = 1024;
  //cudaError_t err1 = cudaDeviceGetAttribute(&maxGridDimX, cudaDevAttrMaxGridDimX, device_id);
  //if (err1) {printf("err = %d, %s\n", err1, cudaGetErrorString(err1)); return;}
  maxGridDimX = -1; // cudaDeviceGetAttribute gives "err = 101, invalid device ordinal"

  int gridDimY = 1;
  int maxGridDimY = 1024;
  //cudaError_t err2 = cudaDeviceGetAttribute(&maxGridDimY, cudaDevAttrMaxGridDimY, device_id);
  //if (err2) {printf("err = %d, %s\n", err2, cudaGetErrorString(err2)); return;}

  if (maxGridDimX > 0 && minBlocksNum > maxGridDimX) {
    std::cout <<"minBlocksNum (" <<minBlocksNum <<") for a 1D-block 1D grid is larger than maxGridDimX (" <<maxGridDimX <<"), will consider a 1D-block 2D grid!" <<std::endl;
    gridDimX = maxGridDimX;
    int minBlocksYNum = minBlocksNum / maxGridDimX + minBlocksNum % maxGridDimX;

    if (minBlocksYNum > maxGridDimY) {
      std::cout <<"minBlocksYNum (" <<minBlocksYNum <<") for a 1D-block 2D grid is larger than maxGridDimY (" <<maxGridDimY <<"), will run a 1D-block 3D grid!" <<std::endl;
      return;
    }

    gridDimY = minBlocksYNum;
  }

  dim3 dimGrid(gridDimX);
  if (gridDimY > 1) // do not specify gridDimY if =1 to avoid printf issue in the kernel
    dimGrid = dim3(gridDimX, gridDimY);
  std::cout <<"Running kernel with " <<gridDimX <<" x-blocks, " <<gridDimY <<" y-blocks and " <<maxThreadsPerBlock <<" threads per block" <<std::endl;
  addNoise<<<dimGrid, maxThreadsPerBlock>>>(n, cellCharge, cellToa, weightMode, devRand, cellType, word);
  cudaDeviceSynchronize();
  std::cout << "--> DONE NOISE" << std::endl;
}         
