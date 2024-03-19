#include "npGPUArray.cuh"
#include <curand.h>
#include <curand_kernel.h>


#pragma once

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    }} while(0)

// for uniform distribution
template<typename TP>
__global__ void kernelInitializeRandomUnif(TP* arr, int size, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);  // Initialize curand state for each thread
        arr[idx] = curand_uniform(&state);  // Generate a random value
    }
}

// for uniform distribution
template<typename TP>
__global__ void kernelInitializeRandomUnif(TP* arr, int size, int lo, int hi, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);  // Initialize curand state for each thread
        arr[idx] = (curand_uniform(&state) * (hi - lo) + lo);  // Generate a random value
    }
}

// for normal distribution
template<typename TP>
__global__ void kernelInitializeRandomNorm(TP* arr, int size, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);  // Initialize curand state for each thread
        arr[idx] = curand_normal(&state);  // Generate a random value
    }
}

namespace np {
	class Random {
	public:
        // from uniform distribution
        template<typename TP>
        static ArrayGPU<TP> rand(int rows, int cols, int lo, int hi, unsigned long long seed) {
            ArrayGPU<TP> ar(rows, cols);

            const int BLOCK_SIZE = (GPU_NUM_CUDA_CORE == 64)?8:16;
            dim3 block(BLOCK_SIZE * BLOCK_SIZE);
            dim3 grid(ceil(rows * cols, block.x));
            kernelInitializeRandomUnif<TP> << <grid, block >> > (ar.mat, rows * cols, lo, hi, seed);
            cudaDeviceSynchronize();

            return ar;
        }
        template<typename TP>
		static ArrayGPU<TP> rand(int rows, int cols, unsigned long long seed) {
			ArrayGPU<TP> ar(rows, cols);

            const int BLOCK_SIZE = 16;
            dim3 block(BLOCK_SIZE * BLOCK_SIZE);
            dim3 grid( ceil( rows * cols , block.x ) );
            kernelInitializeRandomUnif<TP> << <grid, block >> > (ar.mat, rows * cols, seed);
            cudaDeviceSynchronize();

            return ar;
		}

        template<typename TP>
        static ArrayGPU<TP> rand(int rows = 1, int cols = 1) {
            return rand<TP>(rows, cols, time(NULL));
        }

        // from normal distribution
        template<typename TP>
        static ArrayGPU<TP> randn(int rows, int cols, unsigned long long seed) {
            ArrayGPU<TP> ar(rows, cols);

            const int BLOCK_SIZE = 16;
            dim3 block(BLOCK_SIZE * BLOCK_SIZE);
            dim3 grid(ceil(rows * cols, block.x));
            kernelInitializeRandomNorm<TP> << <grid, block >> > (ar.mat, rows * cols, seed);
            cudaDeviceSynchronize();

            return ar;
        }

        template<typename TP>
        static ArrayGPU<TP> randn(int rows = 1, int cols = 1) {
            return randn<TP>(rows, cols, time(NULL));
        }
	};
}