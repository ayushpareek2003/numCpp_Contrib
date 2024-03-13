#include <iostream>
#include <cuda_runtime.h>

#pragma once
/*
	API Structure ->
	            
*/
#include <cuda_runtime.h>

// cuda error checking macro
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
inline void check(cudaError_t err, const char* const func, const char* const file, const int line)
{
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA Runtime Error at: " << file << ":" << line
			<< std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		// We don't exit when we encounter CUDA errors in this example.
		//std::exit(EXIT_FAILURE);
	}
}

// kernel to copy a matrix stored in row major order to column major order and vice-versa.
__global__ void kernelTransposeInMem(float* in, float* out, int M, int N) {
	/*
		in: input array
		out: output array
		M, N: dimension of input array
	*/

	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;

	if (r < M && c < N) {
		out[c * M + r] = in[r * N + c];
	}
}


namespace np {

	template<typename TP>
	class ArrayGPU{
	public:
		TP* mat;
		int rows, cols;

		// initialise array with all values set to Val
		ArrayGPU(int rows = 1, int cols = 1, TP Val = 0) {
			this->rows = rows;
			this->cols = cols;

			CHECK_CUDA_ERROR( cudaMalloc((void**)&mat, this->rows * this->cols * sizeof(TP)) );
			CHECK_CUDA_ERROR( cudaMemset(mat, Val, this->rows * this->cols * sizeof(TP)) );
		}

		// pointer to host memory.
		void copyFromCPU(TP* h_array) {
			CHECK_CUDA_ERROR(cudaMemcpy(mat, h_array, this->rows * this->cols * sizeof(TP), cudaMemcpyHostToDevice));
		}

		// pointer to device memory.
		void copyFromGPU(TP* d_array) {
			CHECK_CUDA_ERROR(cudaMemcpy(mat, d_array, this->rows * this->cols * sizeof(TP), cudaMemcpyDeviceToDevice));
		}

		ArrayGPU T() {
			ArrayGPU out(this->cols, this->rows);

			const int BLOCK_SIZE = 16;
			dim3 block(BLOCK_SIZE, BLOCK_SIZE);
			dim3 grid(ceil(this->cols / block.x), ceil(this->rows / block.y));

			kernelTransposeInMem<<<grid, block>>>(this->mat, out.mat, this->rows, this->cols);
			cudaDeviceSynchronize();

			return out;

		}

		TP at(int r, int c) {
			return at(r * this->cols + c);
		}

		TP at(int idx) {
			TP val;
		}

		~ArrayGPU() {
			cudaFree(mat);
		}
	};
}