#pragma once

#include<cuda_runtime.h>
#include<curand.h>

#include<iostream>

// cuda error checking macro
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    }} while(0)


// curand error checking macro
#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    }} while(0)