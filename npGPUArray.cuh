#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cublas_v2.h>

#include "customKernels.cuh"

#define ceil(x, y) (x + y - 1) / y

#pragma once

cublasHandle_t cbls_handle;

/*
	API Structure ->

*/



namespace np {
	int GPU_NUM_CUDA_CORE = 0;
	int GPU_NUM_SM = 0;

	int _ConvertSMVer2Cores(int major, int minor) {
		// Refer to the CUDA Compute Capability documentation for the number of cores per multiprocessor
		// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
		switch ((major << 4) + minor) {
		case 0x10: return 8;  // Tesla
		case 0x11: return 8;  // Tesla
		case 0x12: return 8;  // Tesla
		case 0x13: return 8;  // Tesla
		case 0x20: return 32; // Fermi
		case 0x21: return 48; // Fermi
		case 0x30: return 192; // Kepler
		case 0x32: return 192; // Kepler
		case 0x35: return 192; // Kepler
		case 0x37: return 192; // Kepler
		case 0x50: return 128; // Maxwell
		case 0x52: return 128; // Maxwell
		case 0x53: return 128; // Maxwell
		case 0x60: return 64;  // Pascal
		case 0x61: return 128; // Pascal
		case 0x62: return 128; // Pascal
		case 0x70: return 64;  // Volta
		case 0x72: return 64;  // Volta
		case 0x75: return 64;  // Turing
		case 0x80: return 64;  // Ampere
		case 0x86: return 128;  // Ampere
		default: return -1;    // Unknown
		}
	}



	void getGPUConfig(int deviceId = 0) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, deviceId);
		GPU_NUM_CUDA_CORE = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
		GPU_NUM_SM = deviceProp.multiProcessorCount;
	}
	template<typename TP>
	class ArrayGPU {
	private:
	public:
		TP* mat;
		int rows, cols;
		

		ArrayGPU(int rows = 1, int cols = 1) {
			this->rows = rows;
			this->cols = cols;

			CUDA_CALL(cudaMalloc((void**)&mat, this->rows * this->cols * sizeof(TP)));

			if (cbls_handle == nullptr) {
				cublasCreate(&cbls_handle);
			}
		}

		// initialise array with all values set to Val
		ArrayGPU(int rows, int cols, TP Val) {
			this->rows = rows;
			this->cols = cols;

			CUDA_CALL(cudaMalloc((void**)&mat, this->rows * this->cols * sizeof(TP)));

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(this->rows * this->cols, block.x));

			kernelInitMatBroadcast<TP> << <grid, block >> > (mat, Val, this->rows * this->cols);
			cudaDeviceSynchronize();

			if (cbls_handle == nullptr) {
				cublasCreate(&cbls_handle);
			}
		}

		void reshape(int newRows, int newCols) {
			if (newRows * newCols == this->rows * this->cols) {
				this->rows = newRows;
				this->cols = newCols;
			}
		}

		int size() {
			return this->rows * this->cols;
		}
		
		// pointer to host memory.
		void copyFromCPU(TP* h_array) {
			CUDA_CALL(cudaMemcpy(mat, h_array, this->rows * this->cols * sizeof(TP), cudaMemcpyHostToDevice));
		}

		// pointer to device memory.
		void copyFromGPU(TP* d_array) {
			CUDA_CALL(cudaMemcpy(mat, d_array, this->rows * this->cols * sizeof(TP), cudaMemcpyDeviceToDevice));
		}


		void print() {
			kernelPrintMat<TP> <<<1, 1>>>(mat, this->rows, this->cols);
			cudaDeviceSynchronize();
		}

		ArrayGPU<TP> T() {
			ArrayGPU<TP> out(this->cols, this->rows);

			const int TILE_WIDTH = (GPU_NUM_CUDA_CORE == 64) ? 8 : 16;
			const int ROW_BLOCK = 8;
			dim3 block(TILE_WIDTH, ROW_BLOCK);
			dim3 grid(ceil(this->cols, TILE_WIDTH), ceil(this->rows, TILE_WIDTH));

			switch (GPU_NUM_CUDA_CORE) {
			case 64: 
				kernelTransposeInMem<TP, 8, ROW_BLOCK> << <grid, block >> > (this->mat, out.mat, this->rows, this->cols); break;
			default:
				kernelTransposeInMem<TP, 16, ROW_BLOCK> << <grid, block >> > (this->mat, out.mat, this->rows, this->cols); break;
			}
			cudaDeviceSynchronize();

			return out;

		}

		// get value at r, c
		TP at(int r, int c) {
			return at(r * this->cols + c);
		}

		// get value at an index
		TP at(int idx) {
			TP val;

			CUDA_CALL(cudaMemcpy(&val, mat + idx, sizeof(TP), cudaMemcpyDeviceToHost));

			return val;
		}

		//get values from multiple indexes
		ArrayGPU<TP> at(ArrayGPU<int>& idxs) {
			int size = max(idxs.rows, idxs.cols);
			ArrayGPU<TP> ans(size);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(size, block.x));
			kernelGetMatValues<TP> << <grid, block >> > (mat, ans.mat, idxs.mat, size);
			cudaDeviceSynchronize();

			return ans;
		}

		// get values from multiple indexes
		ArrayGPU<TP> at(ArrayGPU<int> &r, ArrayGPU<int> &c) {
			/*
				r = (0, 1, 2, 3, 4, 5, 6)
				c = (7, 6, 4, 2, 1, 8, 9)
				fetch all (ri , ci) elements
			*/
			int size = max(r.rows, r.cols);
			ArrayGPU<TP> ans(size);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(size, block.x));
			kernelGetMatValues<TP> << <grid, block >> > (mat, this->cols, ans.mat, r.mat, c.mat, size);
			cudaDeviceSynchronize();

			return ans;
		}

		// set value at r, c
		void set(int r, int c, TP val) {
			int idx = r * this->cols + c;
			set(idx, val);
		}

		// set value at idx
		void set(int idx, TP val) {
			CUDA_CALL(cudaMemcpy(mat + idx, &val, sizeof(TP), cudaMemcpyHostToDevice));
		}

		// set values from multiple indexes
		void set(ArrayGPU<int>& idxs, ArrayGPU<TP>& val) {
			/*
				r = (0, 1, 2, 3, 4, 5, 6)
				c = (7, 6, 4, 2, 1, 8, 9)
				val = (1, 2, 3, 4, 5, 6, 7)
			set all (ri , ci) elements to vali
			*/
			int size = max(idxs.rows, idxs.cols); // one dimension will always be 1.

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(size, block.x));
			kernelSetMatValues<TP> << <grid, block >> > (mat, val.mat, idxs.mat, size);
			cudaDeviceSynchronize();
		}

		// set values from multiple indexes
		void set(ArrayGPU<int>& r, ArrayGPU<int>& c, ArrayGPU<TP> &val) {
			/*
				r = (0, 1, 2, 3, 4, 5, 6)
				c = (7, 6, 4, 2, 1, 8, 9)
				val = (1, 2, 3, 4, 5, 6, 7)
			set all (ri , ci) elements to vali
			*/
			int size = max(r.rows, r.cols); // one dimension will always be 1.

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(size, block.x));
			kernelSetMatValues<TP><<<grid, block>>>(mat, this->cols, val.mat, r.mat, c.mat, size);
			cudaDeviceSynchronize();
		}
		

		// defining dot product

		ArrayGPU<TP> dot(ArrayGPU<TP>& B) {
			/*
				C = A @ B
				input:
					A: shape MxK
					B: shape KxN
				output:
					C: shape MxN

			*/
			// condition for dot product
			if (this->cols == B.rows) {
				ArrayGPU<TP> res(this->rows, B.cols);

				const float alpha = 1.0f;
				const float beta = 0.0f;

				//C = A . B k lie.
				cublasSgemm(cbls_handle,  // 
					CUBLAS_OP_N, CUBLAS_OP_N,
					B.cols, this->rows, this->cols, // B cols, A rows, A cols
					&alpha,
					B.mat, B.cols,					// B, B cols
					this->mat, this->cols,				// A, A cols
					&beta,
					res.mat, B.cols);				// C, B cols

				return res;
			}
		}

		ArrayGPU<TP> Tdot(ArrayGPU<TP>& B) {
			/*
				C = A.T @ B
				input:
					A: shape KxM
					B: shape KxN
				output:
					C: shape MxN

			*/
			// condition for dot product
			if (this->rows == B.rows) {
				ArrayGPU<TP> res(this->cols, B.cols);

				const float alpha = 1.0f;
				const float beta = 0.0f;

				//C = AT . B
				cublasSgemm(cbls_handle,  // 
					CUBLAS_OP_N, CUBLAS_OP_T,
					B.cols, this->cols, this->rows, // B cols, A cols, A rows
					&alpha,
					B.mat, B.cols,					// B, B cols
					this->mat, this->cols,				// A, A cols
					&beta,
					res.mat, B.cols);				// C, B cols

				return res;
			}

		}

		ArrayGPU<TP> dotT(ArrayGPU<TP>& B) {
			/*
				C = A @ B.T
				input:
					A: shape MxK
					B: shape NxK
				output:
					C: shape MxN

			*/
			// condition for dot product
			if (this->cols == B.cols) {
				ArrayGPU<TP> res(this->rows, B.rows);

				const float alpha = 1.0f;
				const float beta = 0.0f;

				cublasSgemm(cbls_handle,  // 
					CUBLAS_OP_T, CUBLAS_OP_N,
					B.rows, this->rows, this->cols, // B cols, A rows, A cols
					&alpha,
					B.mat, B.cols,					// B, B cols
					this->mat, this->cols,			     	// A, A cols
					&beta,
					res.mat, B.rows);				// C, B cols

				return res;
			}

		}


		// add functions
		ArrayGPU<TP> operator+(ArrayGPU<TP>& B) {
			if (this->rows == 1 && this->cols == 1) {
				// A is scalar
				ArrayGPU<TP> res(B.rows, B.cols);

				const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(res.size(), block.x));

				kernelMatAddScalar<TP> << <grid, block >> > (B.mat, this->at(0), res.mat, res.size());
				cudaDeviceSynchronize();
				return res;
			}
			else if (B.rows == 1 && B.cols == 1) {
				// B is scalar
				ArrayGPU<TP> res(this->rows, this->cols);

				const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(res.size(), block.x));

				kernelMatAddScalar<TP> << <grid, block >> > (this->mat, B.at(0), res.mat, res.size());
				cudaDeviceSynchronize();
				return res;
			}
			else if (this->rows == 1 || this->cols == 1) {
				//A is vector
				//A vector ki dim, is equal to either col or row of B
				int vecDim = max(this->rows, this->cols);

				if (vecDim == B.rows) {
					// along rows add kr
					ArrayGPU<TP> res(B.rows, B.cols);
					const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatAddVecAlongRows<TP> << <grid, block >> > (B.mat, this->mat, res.mat, res.size(), B.cols);
					cudaDeviceSynchronize();

					return res;
				}
				else if (vecDim == B.cols) {
					//along cols add kr 
					ArrayGPU<TP> res(B.rows, B.cols);
					const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatAddVecAlongCols<TP> << <grid, block >> > (B.mat, this->mat, res.mat, res.size(), B.cols);
					cudaDeviceSynchronize();

					return res;
				}
			}
			else if (B.cols == 1 || B.rows == 1) {
				// B is vetor
				// B vector ki dim, is eq to either col or row of B
				int vecDim = max(B.rows, B.cols);

				if (vecDim == this->rows) {
					// along rows add kr
					ArrayGPU<TP> res(this->rows, this->cols);
					const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatAddVecAlongRows<TP> << <grid, block >> > (this->mat, B.mat, res.mat, res.size(), this->cols);
					cudaDeviceSynchronize();

					return res;
				}
				else if (vecDim == this->cols) {
					//along cols add kr 
					ArrayGPU<TP> res(this->rows, this->cols);
					const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatAddVecAlongCols<TP> << <grid, block >> > (this->mat, B.mat, res.mat, res.size(), this->cols);
					cudaDeviceSynchronize();

					return res;
				}

			}
			else if (this->rows == B.rows && this->cols == B.cols) {
				// A and B both are matrices of same dimensions
				ArrayGPU<TP> res(this->rows, this->cols);
				const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(res.size(), block.x));
				kernelMatAddMat<TP> << <grid, block >> > (this->mat, B.mat, res.mat, res.size());
				cudaDeviceSynchronize();
				return res;
			}
		}

		ArrayGPU<TP> operator+(TP Scalar) {
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatAddScalar<TP> << <grid, block >> > (this->mat, Scalar, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}

		// minus
		ArrayGPU<TP> operator-(ArrayGPU<TP>& B) {
			if (this->rows == 1 && this->cols == 1) {
				// A is scalar
				ArrayGPU<TP> res(B.rows, B.cols);

				const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(res.size(), block.x));

				kernelMatSubScalar<TP> << <grid, block >> > (B.mat, this->at(0), res.mat, res.size());
				cudaDeviceSynchronize();
				return res;
			}
			else if (B.rows == 1 && B.cols == 1) {
				// B is scalar
				ArrayGPU<TP> res(this->rows, this->cols);

				const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(res.size(), block.x));

				kernelMatSubScalar<TP> << <grid, block >> > (this->mat, B.at(0), res.mat, res.size());
				cudaDeviceSynchronize();
				return res;
			}
			else if (this->rows == 1 || this->cols == 1) {
				//A is vector
				//A vector ki dim, is equal to either col or row of B
				int vecDim = max(this->rows, this->cols);

				if (vecDim == B.rows) {
					// along rows add kr
					ArrayGPU<TP> res(B.rows, B.cols);
					const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatSubVecAlongRows<TP> << <grid, block >> > (B.mat, this->mat, res.mat, res.size(), B.cols);
					cudaDeviceSynchronize();

					return res;
				}
				else if (vecDim == B.cols) {
					//along cols add kr 
					ArrayGPU<TP> res(B.rows, B.cols);
					const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatSubVecAlongCols<TP> << <grid, block >> > (B.mat, this->mat, res.mat, res.size(), B.cols);
					cudaDeviceSynchronize();

					return res;
				}
			}
			else if (B.cols == 1 || B.rows == 1) {
				// B is vetor
				// B vector ki dim, is eq to either col or row of B
				int vecDim = max(B.rows, B.cols);

				if (vecDim == this->rows) {
					// along rows add kr
					ArrayGPU<TP> res(this->rows, this->cols);
					const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatSubVecAlongRows<TP> << <grid, block >> > (this->mat, B.mat, res.mat, res.size(), this->cols);
					cudaDeviceSynchronize();

					return res;
				}
				else if (vecDim == this->cols) {
					//along cols add kr 
					ArrayGPU<TP> res(this->rows, this->cols);
					const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatSubVecAlongCols<TP> << <grid, block >> > (this->mat, B.mat, res.mat, res.size(), this->cols);
					cudaDeviceSynchronize();

					return res;
				}

			}
			else if (this->rows == B.rows && this->cols == B.cols) {
				// A and B both are matrices of same dimensions
				ArrayGPU<TP> res(this->rows, this->cols);
				const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(res.size(), block.x));
				kernelMatSubMat<TP> << <grid, block >> > (this->mat, B.mat, res.mat, res.size());
				cudaDeviceSynchronize();
				return res;
			}
		}

		ArrayGPU<TP> operator-(TP Scalar) {
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatSubScalar<TP> << <grid, block >> > (this->mat, Scalar, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}


		// unary negation operator
		ArrayGPU<TP> operator-() const {
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatMulScalar<TP> << <grid, block >> > (this->mat, -1, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}

		// multiply
		ArrayGPU<TP> operator*(ArrayGPU<TP>& B) {
			if (this->rows == 1 && this->cols == 1) {
				// A is scalar
				ArrayGPU<TP> res(B.rows, B.cols);

				const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(res.size(), block.x));

				kernelMatMulScalar<TP> << <grid, block >> > (B.mat, this->at(0), res.mat, res.size());
				cudaDeviceSynchronize();
				return res;
			}
			else if (B.rows == 1 && B.cols == 1) {
				// B is scalar
				ArrayGPU<TP> res(this->rows, this->cols);

				const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(res.size(), block.x));

				kernelMatMulScalar<TP> << <grid, block >> > (this->mat, B.at(0), res.mat, res.size());
				cudaDeviceSynchronize();
				return res;
			}
			else if (this->rows == 1 || this->cols == 1) {
				//A is vector
				//A vector ki dim, is equal to either col or row of B
				int vecDim = max(this->rows, this->cols);

				if (vecDim == B.rows) {
					// along rows add kr
					ArrayGPU<TP> res(B.rows, B.cols);
					const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatMulVecAlongRows<TP> << <grid, block >> > (B.mat, this->mat, res.mat, res.size(), B.cols);
					cudaDeviceSynchronize();

					return res;
				}
				else if (vecDim == B.cols) {
					//along cols add kr 
					ArrayGPU<TP> res(B.rows, B.cols);
					const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatMulVecAlongCols<TP> << <grid, block >> > (B.mat, this->mat, res.mat, res.size(), B.cols);
					cudaDeviceSynchronize();

					return res;
				}
			}
			else if (B.cols == 1 || B.rows == 1) {
				// B is vetor
				// B vector ki dim, is eq to either col or row of B
				int vecDim = max(B.rows, B.cols);

				if (vecDim == this->rows) {
					// along rows add kr
					ArrayGPU<TP> res(this->rows, this->cols);
					const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatMulVecAlongRows<TP> << <grid, block >> > (this->mat, B.mat, res.mat, res.size(), this->cols);
					cudaDeviceSynchronize();

					return res;
				}
				else if (vecDim == this->cols) {
					//along cols add kr 
					ArrayGPU<TP> res(this->rows, this->cols);
					const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatMulVecAlongCols<TP> << <grid, block >> > (this->mat, B.mat, res.mat, res.size(), this->cols);
					cudaDeviceSynchronize();

					return res;
				}

			}
			else if (this->rows == B.rows && this->cols == B.cols) {
				// A and B both are matrices of same dimensions
				ArrayGPU<TP> res(this->rows, this->cols);
				const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(res.size(), block.x));
				kernelMatMulMat<TP> << <grid, block >> > (this->mat, B.mat, res.mat, res.size());
				cudaDeviceSynchronize();
				return res;
			}
		}

		ArrayGPU<TP> operator*(TP Scalar) {
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatMulScalar<TP> << <grid, block >> > (this->mat, Scalar, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}

		//divide
		ArrayGPU<TP> operator/(ArrayGPU<TP>& B) {
			if (this->rows == 1 && this->cols == 1) {
				// A is scalar
				ArrayGPU<TP> res(B.rows, B.cols);

				const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(res.size(), block.x));

				kernelMatDivScalar<TP> << <grid, block >> > (B.mat, this->at(0), res.mat, res.size());
				cudaDeviceSynchronize();
				return res;
			}
			else if (B.rows == 1 && B.cols == 1) {
				// B is scalar
				ArrayGPU<TP> res(this->rows, this->cols);

				const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(res.size(), block.x));

				kernelMatDivScalar<TP> << <grid, block >> > (this->mat, B.at(0), res.mat, res.size());
				cudaDeviceSynchronize();
				return res;
			}
			else if (this->rows == 1 || this->cols == 1) {
				//A is vector
				//A vector ki dim, is equal to either col or row of B
				int vecDim = max(this->rows, this->cols);

				if (vecDim == B.rows) {
					// along rows add kr
					ArrayGPU<TP> res(B.rows, B.cols);
					const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatDivVecAlongRows<TP> << <grid, block >> > (B.mat, this->mat, res.mat, res.size(), B.cols);
					cudaDeviceSynchronize();

					return res;
				}
				else if (vecDim == B.cols) {
					//along cols add kr 
					ArrayGPU<TP> res(B.rows, B.cols);
					const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatDivVecAlongCols<TP> << <grid, block >> > (B.mat, this->mat, res.mat, res.size(), B.cols);
					cudaDeviceSynchronize();

					return res;
				}
			}
			else if (B.cols == 1 || B.rows == 1) {
				// B is vetor
				// B vector ki dim, is eq to either col or row of B
				int vecDim = max(B.rows, B.cols);

				if (vecDim == this->rows) {
					// along rows add kr
					ArrayGPU<TP> res(this->rows, this->cols);
					const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatDivVecAlongRows<TP> << <grid, block >> > (this->mat, B.mat, res.mat, res.size(), this->cols);
					cudaDeviceSynchronize();

					return res;
				}
				else if (vecDim == this->cols) {
					//along cols add kr 
					ArrayGPU<TP> res(this->rows, this->cols);
					const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatDivVecAlongCols<TP> << <grid, block >> > (this->mat, B.mat, res.mat, res.size(), this->cols);
					cudaDeviceSynchronize();

					return res;
				}

			}
			else if (this->rows == B.rows && this->cols == B.cols) {
				// A and B both are matrices of same dimensions
				ArrayGPU<TP> res(this->rows, this->cols);
				const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(res.size(), block.x));
				kernelMatDivMat<TP> << <grid, block >> > (this->mat, B.mat, res.mat, res.size());
				cudaDeviceSynchronize();
				return res;
			}
		}

		ArrayGPU<TP> operator/(TP Scalar) {
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatDivScalar<TP> << <grid, block >> > (this->mat, Scalar, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}

		//returns an array of 0s and 1s depending on true or false of the conditions.
		// element wise comparison

		// >
		ArrayGPU<TP> operator>(ArrayGPU<TP>& B) {
			if (B.rows == 1 && B.cols == 1) {
				// B is scalar
				ArrayGPU<TP> res(this->rows, this->cols);
				const int BLOCK_SIZE = GPU_NUM_CUDA_CORE; 
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(this->size(), block.x));

				kernelMatIsGreaterThanScalar<TP> << <grid, block >> > (this->mat, B.at(0), res.mat, this->size());
				cudaDeviceSynchronize();
				return res;
			}
			else if (B.rows == this->rows && B.cols == this->cols) {
				// both have same dimensions
				ArrayGPU<TP> res(this->rows, this->cols);
				const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(this->size(), block.x));

				kernelMatIsGreaterThanMat<TP> << <grid, block >> > (this->mat, B.mat, res.mat, this->size());
				cudaDeviceSynchronize();
				return res;
			}
		}

		ArrayGPU<TP> operator>(TP Scalar) {
			// Scalar 
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(this->size(), block.x));

			kernelMatIsGreaterThanScalar<TP> << <grid, block >> > (this->mat, Scalar, res.mat, this->size());
			cudaDeviceSynchronize();
			return res;
		}

		// <
		ArrayGPU<TP> operator<(ArrayGPU<TP>& B) {
			if (B.rows == 1 && B.cols == 1) {
				// B is scalar
				ArrayGPU<TP> res(this->rows, this->cols);
				const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(this->size(), block.x));

				kernelMatIsLessThanScalar<TP> << <grid, block >> > (this->mat, B.at(0), res.mat, this->size());
				cudaDeviceSynchronize();
				return res;
			}
			else if (B.rows == this->rows && B.cols == this->cols) {
				// both have same dimensions
				ArrayGPU<TP> res(this->rows, this->cols);
				const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(this->size(), block.x));

				kernelMatIsLessThanMat<TP> << <grid, block >> > (this->mat, B.mat, res.mat, this->size());
				cudaDeviceSynchronize();
				return res;
			}
		}

		ArrayGPU<TP> operator<(TP Scalar) {
			// Scalar 
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(this->size(), block.x));

			kernelMatIsLessThanScalar<TP> << <grid, block >> > (this->mat, Scalar, res.mat, this->size());
			cudaDeviceSynchronize();
			return res;
		}

		// >=
		ArrayGPU<TP> operator>=(ArrayGPU<TP>& B) {
			if (B.rows == 1 && B.cols == 1) {
				// B is scalar
				ArrayGPU<TP> res(this->rows, this->cols);
				const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(this->size(), block.x));

				kernelMatIsGreaterThanEqScalar<TP> << <grid, block >> > (this->mat, B.at(0), res.mat, this->size());
				cudaDeviceSynchronize();
				return res;
			}
			else if (B.rows == this->rows && B.cols == this->cols) {
				// both have same dimensions
				ArrayGPU<TP> res(this->rows, this->cols);
				const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(this->size(), block.x));

				kernelMatIsGreaterThanEqMat<TP> << <grid, block >> > (this->mat, B.mat, res.mat, this->size());
				cudaDeviceSynchronize();
				return res;
			}
		}

		ArrayGPU<TP> operator>=(TP Scalar) {
			// Scalar 
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(this->size(), block.x));

			kernelMatIsGreaterThanEqScalar<TP> << <grid, block >> > (this->mat, Scalar, res.mat, this->size());
			cudaDeviceSynchronize();
			return res;
		}

		// <=
		ArrayGPU<TP> operator<=(ArrayGPU<TP>& B) {
			if (B.rows == 1 && B.cols == 1) {
				// B is scalar
				ArrayGPU<TP> res(this->rows, this->cols);
				const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(this->size(), block.x));

				kernelMatIsLessThanEqScalar<TP> << <grid, block >> > (this->mat, B.at(0), res.mat, this->size());
				cudaDeviceSynchronize();
				return res;
			}
			else if (B.rows == this->rows && B.cols == this->cols) {
				// both have same dimensions
				ArrayGPU<TP> res(this->rows, this->cols);
				const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(this->size(), block.x));

				kernelMatIsLessThanEqMat<TP> << <grid, block >> > (this->mat, B.mat, res.mat, this->size());
				cudaDeviceSynchronize();
				return res;
			}
		}

		ArrayGPU<TP> operator<=(TP Scalar) {
			// Scalar 
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(this->size(), block.x));

			kernelMatIsLessThanEqScalar<TP> << <grid, block >> > (this->mat, Scalar, res.mat, this->size());
			cudaDeviceSynchronize();
			return res;
		}
		

		//sum. along axis or total
		ArrayGPU<TP> sum(int axis = -1) {
			if (axis == -1) {
				// return total sum
				const int BLOCK_SIZE = ( (GPU_NUM_CUDA_CORE == 64) ? 64 : 128 ) * 2;
				dim3 block(BLOCK_SIZE);
				dim3 grid(std::min<int>(ceil(this->size(), block.x), GPU_NUM_SM * 2));
				ArrayGPU<TP> res(1, 1);
				// device pointer tmp
				float* tmp_d;
				CUDA_CALL(cudaMalloc((void**)&tmp_d, grid.x));
				switch (GPU_NUM_CUDA_CORE) {
				case 64:
					kernelReduceSum<TP, 64 * 2> << <grid, block >> > (this->mat, tmp_d, this->size());
					cudaDeviceSynchronize();
					// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
					kernelReduceSum<TP, 64 * 2> << <1, block >> > (tmp_d, res.mat, grid.x);
					cudaDeviceSynchronize();
					break;
				default:
					kernelReduceSum<TP, 128 * 2> << <grid, block >> > (this->mat, tmp_d, this->size());
					cudaDeviceSynchronize();
					// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
					kernelReduceSum<TP, 128 * 2> << <1, block >> > (tmp_d, res.mat, grid.x);
					cudaDeviceSynchronize();
					break;

				}
				

				CUDA_CALL(cudaFree(tmp_d));

				return res;
			}
			else if (axis == 0) {
				// sum along columns. dimension=numCols
				return this->T().sum(1).T();

			}
			else if (axis == 1) {
				// sum along rows. output dim = numRows
				ArrayGPU<TP> res(this->rows);

				const int BLOCK_SIZE = ( (GPU_NUM_CUDA_CORE == 64)?64:128 ) * 2;
				dim3 block(BLOCK_SIZE);
				dim3 grid(std::min<int>(ceil(this->cols, block.x), GPU_NUM_SM * 2));

				float* tmp_d;
				CUDA_CALL(cudaMalloc((void**)&tmp_d, this->rows * grid.x));
				switch (GPU_NUM_CUDA_CORE) {
				case 64:
					for (int i = 0; i < this->rows; ++i) {
						kernelReduceSum<TP, 64 * 2> << <grid, block >> > (this->mat + i * this->cols, tmp_d + i * grid.x, this->cols);
					}

					cudaDeviceSynchronize();


					for (int i = 0; i < this->rows; ++i) {
						kernelReduceSum<TP, 64 * 2> << <1, block >> > (tmp_d + i * grid.x, res.mat + i, grid.x);
					}

					cudaDeviceSynchronize();
					break;
				default:
					for (int i = 0; i < this->rows; ++i) {
						kernelReduceSum<TP, 128 * 2> << <grid, block >> > (this->mat + i * this->cols, tmp_d + i * grid.x, this->cols);
					}

					cudaDeviceSynchronize();


					for (int i = 0; i < this->rows; ++i) {
						kernelReduceSum<TP, 128 * 2> << <1, block >> > (tmp_d + i * grid.x, res.mat + i, grid.x);
					}

					cudaDeviceSynchronize();

				}
				return res;
			}
		}

		//max. along axis or total
		ArrayGPU<TP> max(int axis = -1) {
			if (axis == -1) {
				// return total sum
				const int BLOCK_SIZE = ((GPU_NUM_CUDA_CORE == 64) ? 64 : 128) * 2;
				dim3 block(BLOCK_SIZE);
				dim3 grid(std::min<int>(ceil(this->size(), block.x), GPU_NUM_SM * 2));
				ArrayGPU<TP> res(1, 1);
				// device pointer tmp
				float* tmp_d;
				CUDA_CALL(cudaMalloc((void**)&tmp_d, grid.x));
				switch (GPU_NUM_CUDA_CORE) {
				case 64:
					kernelReduceMax<TP, 64 * 2> << <grid, block >> > (this->mat, tmp_d, this->size());
					cudaDeviceSynchronize();
					// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
					kernelReduceMax<TP, 64 * 2> << <1, block >> > (tmp_d, res.mat, grid.x);
					cudaDeviceSynchronize();
					break;
				default:
					kernelReduceMax<TP, 128 * 2> << <grid, block >> > (this->mat, tmp_d, this->size());
					cudaDeviceSynchronize();
					// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
					kernelReduceMax<TP, 128 * 2> << <1, block >> > (tmp_d, res.mat, grid.x);
					cudaDeviceSynchronize();
					break;

				}


				CUDA_CALL(cudaFree(tmp_d));

				return res;
			}
			else if (axis == 0) {
				// sum along columns. dimension=numCols
				return this->T().max(1).T();

			}
			else if (axis == 1) {
				// sum along rows. output dim = numRows
				ArrayGPU<TP> res(this->rows);

				const int BLOCK_SIZE = ((GPU_NUM_CUDA_CORE == 64) ? 64 : 128) * 2;
				dim3 block(BLOCK_SIZE);
				dim3 grid(std::min<int>(ceil(this->cols, block.x), GPU_NUM_SM * 2));

				float* tmp_d;
				CUDA_CALL(cudaMalloc((void**)&tmp_d, this->rows * grid.x));
				switch (GPU_NUM_CUDA_CORE) {
				case 64:
					for (int i = 0; i < this->rows; ++i) {
						kernelReduceMax<TP, 64 * 2> << <grid, block >> > (this->mat + i * this->cols, tmp_d + i * grid.x, this->cols);
					}

					cudaDeviceSynchronize();


					for (int i = 0; i < this->rows; ++i) {
						kernelReduceMax<TP, 64 * 2> << <1, block >> > (tmp_d + i * grid.x, res.mat + i, grid.x);
					}

					cudaDeviceSynchronize();
					break;
				default:
					for (int i = 0; i < this->rows; ++i) {
						kernelReduceMax<TP, 128 * 2> << <grid, block >> > (this->mat + i * this->cols, tmp_d + i * grid.x, this->cols);
					}

					cudaDeviceSynchronize();


					for (int i = 0; i < this->rows; ++i) {
						kernelReduceMax<TP, 128 * 2> << <1, block >> > (tmp_d + i * grid.x, res.mat + i, grid.x);
					}

					cudaDeviceSynchronize();

				}
				return res;
			}
		}
		

		//min. along axis or total
		ArrayGPU<TP> min(int axis = -1) {
			if (axis == -1) {
				// return total sum
				const int BLOCK_SIZE = ((GPU_NUM_CUDA_CORE == 64) ? 64 : 128) * 2;
				dim3 block(BLOCK_SIZE);
				dim3 grid(std::min<int>(ceil(this->size(), block.x), GPU_NUM_SM * 2));

				ArrayGPU<TP> res(1, 1);
				// device pointer tmp
				float* tmp_d;
				CUDA_CALL(cudaMalloc((void**)&tmp_d, grid.x));
				switch (GPU_NUM_CUDA_CORE) {
				case 64:
					kernelReduceMin<TP, 64 * 2> << <grid, block >> > (this->mat, tmp_d, this->size());
					cudaDeviceSynchronize();
					// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
					kernelReduceMin<TP, 64 * 2> << <1, block >> > (tmp_d, res.mat, grid.x);
					cudaDeviceSynchronize();
					break;
				default:
					kernelReduceMin<TP, 128 * 2> << <grid, block >> > (this->mat, tmp_d, this->size());
					cudaDeviceSynchronize();
					// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
					kernelReduceMin<TP, 128 * 2> << <1, block >> > (tmp_d, res.mat, grid.x);
					cudaDeviceSynchronize();
					break;

				}


				CUDA_CALL(cudaFree(tmp_d));

				return res;
			}
			else if (axis == 0) {
				// sum along columns. dimension=numCols
				return this->T().min(1).T();

			}
			else if (axis == 1) {
				// sum along rows. output dim = numRows
				ArrayGPU<TP> res(this->rows);

				const int BLOCK_SIZE = ((GPU_NUM_CUDA_CORE == 64) ? 64 : 128) * 2;
				dim3 block(BLOCK_SIZE);
				dim3 grid(std::min<int>(ceil(this->cols, block.x), GPU_NUM_SM * 2));

				float* tmp_d;
				CUDA_CALL(cudaMalloc((void**)&tmp_d, this->rows * grid.x));
				switch (GPU_NUM_CUDA_CORE) {
				case 64:
					for (int i = 0; i < this->rows; ++i) {
						kernelReduceMin<TP, 64 * 2> << <grid, block >> > (this->mat + i * this->cols, tmp_d + i * grid.x, this->cols);
					}

					cudaDeviceSynchronize();


					for (int i = 0; i < this->rows; ++i) {
						kernelReduceMin<TP, 64 * 2> << <1, block >> > (tmp_d + i * grid.x, res.mat + i, grid.x);
					}

					cudaDeviceSynchronize();
					break;
				default:
					for (int i = 0; i < this->rows; ++i) {
						kernelReduceMin<TP, 128 * 2> << <grid, block >> > (this->mat + i * this->cols, tmp_d + i * grid.x, this->cols);
					}

					cudaDeviceSynchronize();


					for (int i = 0; i < this->rows; ++i) {
						kernelReduceMin<TP, 128 * 2> << <1, block >> > (tmp_d + i * grid.x, res.mat + i, grid.x);
					}

					cudaDeviceSynchronize();

				}
				return res;
			}
		}

		// argmax
		ArrayGPU<int> argmax(int axis = -1) {
			if (axis == -1) {
				// return total sum
				const int BLOCK_SIZE = ((GPU_NUM_CUDA_CORE == 64) ? 64 : 128) * 2;
				dim3 block(BLOCK_SIZE);
				dim3 grid(std::min<int>(ceil(this->size(), block.x), GPU_NUM_SM * 2));

				ArrayGPU<TP> res(1);
				ArrayGPU<int> resIdx(1);
				// device pointer tmp
				float* tmp_A_d;
				CUDA_CALL(cudaMalloc((void**)&tmp_A_d, grid.x));
				int* tmp_A_Idx_d;
				CUDA_CALL(cudaMalloc((void**)&tmp_A_Idx_d, grid.x));

				switch (GPU_NUM_CUDA_CORE) {
				case 64:
					kernelReduceArgMax<TP, 64 * 2> << <grid, block >> > (this->mat, tmp_A_d, tmp_A_Idx_d, this->size());
					cudaDeviceSynchronize();
					// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
					kernelReduceArgMax<TP, 64 * 2> << <1, block >> > (tmp_A_d, tmp_A_Idx_d, res.mat, resIdx.mat, grid.x);
					cudaDeviceSynchronize();
					break;
				default:
					kernelReduceArgMax<TP, 128 * 2> << <grid, block >> > (this->mat, tmp_A_d, tmp_A_Idx_d, this->size());
					cudaDeviceSynchronize();
					// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
					kernelReduceArgMax<TP, 128 * 2> << <1, block >> > (tmp_A_d, tmp_A_Idx_d, res.mat, resIdx.mat, grid.x);
					cudaDeviceSynchronize();
				}


				

				CUDA_CALL(cudaFree(tmp_A_d));

				return resIdx;
			}
			else if (axis == 0) {
				// sum along columns. dimension=numCols
				return this->T().argmax(1).T();

			}
			else if (axis == 1) {
				// sum along rows. output dim = numRows
				ArrayGPU<TP> res(this->rows);
				ArrayGPU<int> resIdx(this->rows);

				const int BLOCK_SIZE = ((GPU_NUM_CUDA_CORE == 64) ? 64 : 128) * 2;
				dim3 block(BLOCK_SIZE);
				dim3 grid(std::min<int>(ceil(this->cols, block.x), GPU_NUM_SM * 2));

				float* tmp_A_d;
				CUDA_CALL(cudaMalloc((void**)&tmp_A_d, this->rows * grid.x));
				int* tmp_A_Idx_d;
				CUDA_CALL(cudaMalloc((void**)&tmp_A_Idx_d, this->rows * grid.x));

				switch(GPU_NUM_CUDA_CORE) {
				case 64:
					for (int i = 0; i < this->rows; ++i) {
						kernelReduceArgMax<TP, 64 * 2> << <grid, block >> > (this->mat + i * this->cols, tmp_A_d + i * grid.x, tmp_A_Idx_d + i * grid.x, this->cols);
					}

					cudaDeviceSynchronize();


					for (int i = 0; i < this->rows; ++i) {
						kernelReduceArgMax<TP, 64 * 2> << <1, block >> > (tmp_A_d + i * grid.x, tmp_A_Idx_d + i * grid.x, res.mat + i, resIdx.mat + i, grid.x);
					}

					cudaDeviceSynchronize();
					break;
				default:
					for (int i = 0; i < this->rows; ++i) {
						kernelReduceArgMax<TP, 128 * 2> << <grid, block >> > (this->mat + i * this->cols, tmp_A_d + i * grid.x, tmp_A_Idx_d + i * grid.x, this->cols);
					}

					cudaDeviceSynchronize();


					for (int i = 0; i < this->rows; ++i) {
						kernelReduceArgMax<TP, 128 * 2> << <1, block >> > (tmp_A_d + i * grid.x, tmp_A_Idx_d + i * grid.x, res.mat + i, resIdx.mat + i, grid.x);
					}

					cudaDeviceSynchronize();
				}

				return resIdx;
			}
		}
		// argmin
		//min along axis or total
		ArrayGPU<int> argmin(int axis = -1) {
			if (axis == -1) {
				// return total sum
				const int BLOCK_SIZE = ((GPU_NUM_CUDA_CORE == 64) ? 64 : 128) * 2;
				dim3 block(BLOCK_SIZE);
				dim3 grid(std::min<int>(ceil(this->size(), block.x), GPU_NUM_SM * 2));
				ArrayGPU<TP> res(1);
				ArrayGPU<int> resIdx(1);
				// device pointer tmp
				float* tmp_A_d;
				CUDA_CALL(cudaMalloc((void**)&tmp_A_d, grid.x));
				int* tmp_A_Idx_d;
				CUDA_CALL(cudaMalloc((void**)&tmp_A_Idx_d, grid.x));

				switch (GPU_NUM_CUDA_CORE) {
				case 64:
					kernelReduceArgMin<TP, 64 * 2> << <grid, block >> > (this->mat, tmp_A_d, tmp_A_Idx_d, this->size());
					cudaDeviceSynchronize();
					// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
					kernelReduceArgMin<TP, 64 * 2> << <1, block >> > (tmp_A_d, tmp_A_Idx_d, res.mat, resIdx.mat, grid.x);
					cudaDeviceSynchronize();
					break;
				default:
					kernelReduceArgMin<TP, 128 * 2> << <grid, block >> > (this->mat, tmp_A_d, tmp_A_Idx_d, this->size());
					cudaDeviceSynchronize();
					// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
					kernelReduceArgMin<TP, 128 * 2> << <1, block >> > (tmp_A_d, tmp_A_Idx_d, res.mat, resIdx.mat, grid.x);
					cudaDeviceSynchronize();
				}




				CUDA_CALL(cudaFree(tmp_A_d));

				return resIdx;
			}
			else if (axis == 0) {
				// sum along columns. dimension=numCols
				return this->T().argmin(1).T();

			}
			else if (axis == 1) {
				// sum along rows. output dim = numRows
				ArrayGPU<TP> res(this->rows);
				ArrayGPU<int> resIdx(this->rows);

				const int BLOCK_SIZE = ((GPU_NUM_CUDA_CORE == 64) ? 64 : 128) * 2;
				dim3 block(BLOCK_SIZE);
				dim3 grid(std::min<int>(ceil(this->cols, block.x), GPU_NUM_SM * 2));

				float* tmp_A_d;
				CUDA_CALL(cudaMalloc((void**)&tmp_A_d, this->rows * grid.x));
				int* tmp_A_Idx_d;
				CUDA_CALL(cudaMalloc((void**)&tmp_A_Idx_d, this->rows * grid.x));

				switch(GPU_NUM_CUDA_CORE) {
				case 64:
					for (int i = 0; i < this->rows; ++i) {
						kernelReduceArgMin<TP, 64 * 2> << <grid, block >> > (this->mat + i * this->cols, tmp_A_d + i * grid.x, tmp_A_Idx_d + i * grid.x, this->cols);
					}

					cudaDeviceSynchronize();


					for (int i = 0; i < this->rows; ++i) {
						kernelReduceArgMin<TP, 64 * 2> << <1, block >> > (tmp_A_d + i * grid.x, tmp_A_Idx_d + i * grid.x, res.mat + i, resIdx.mat + i, grid.x);
					}

					cudaDeviceSynchronize();
					break;
				default:
					for (int i = 0; i < this->rows; ++i) {
						kernelReduceArgMin<TP, 128 * 2> << <grid, block >> > (this->mat + i * this->cols, tmp_A_d + i * grid.x, tmp_A_Idx_d + i * grid.x, this->cols);
					}

					cudaDeviceSynchronize();


					for (int i = 0; i < this->rows; ++i) {
						kernelReduceArgMin<TP, 128 * 2> << <1, block >> > (tmp_A_d + i * grid.x, tmp_A_Idx_d + i * grid.x, res.mat + i, resIdx.mat + i, grid.x);
					}

					cudaDeviceSynchronize();
				}

				return resIdx;
			}
		}

		// sort
		// argsort
		
		~ArrayGPU() {
			cudaFree(mat);
		}
	};
}