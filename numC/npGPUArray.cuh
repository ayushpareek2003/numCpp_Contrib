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

	template<typename TP>
	class ArrayGPU {
	private:
	public:
		TP* mat;
		int rows, cols;

		// initialise array with all values set to Val
		ArrayGPU(int rows = 1, int cols = 1, TP Val = 0) {
			this->rows = rows;
			this->cols = cols;

			CUDA_CALL(cudaMalloc((void**)&mat, this->rows * this->cols * sizeof(TP)));

			const int BLOCK_SIZE = 128;
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

			const int BLOCK_SIZE = 16;
			dim3 block(BLOCK_SIZE, BLOCK_SIZE);
			dim3 grid(ceil(this->cols, block.x), ceil(this->rows, block.y));

			kernelTransposeInMem<TP> << <grid, block >> > (this->mat, out.mat, this->rows, this->cols);
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

		// get values from multiple indexes
		ArrayGPU<TP> at(ArrayGPU<int> &r, ArrayGPU<int> &c) {
			/*
				r = (0, 1, 2, 3, 4, 5, 6)
				c = (7, 6, 4, 2, 1, 8, 9)
				fetch all (ri , ci) elements
			*/
			int size = max(r.rows, r.cols);
			ArrayGPU<TP> ans(size);

			const int BLOCK_SIZE = 128;
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
		void set(ArrayGPU<int>& r, ArrayGPU<int>& c, ArrayGPU<TP> &val) {
			/*
				r = (0, 1, 2, 3, 4, 5, 6)
				c = (7, 6, 4, 2, 1, 8, 9)
				val = (1, 2, 3, 4, 5, 6, 7)
			set all (ri , ci) elements to vali
			*/
			int size = max(r.rows, r.cols); // one dimension will always be 1.

			const int BLOCK_SIZE = 16;
			dim3 block(BLOCK_SIZE * BLOCK_SIZE);
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

				const int BLOCK_SIZE = 128;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(res.size(), block.x));

				kernelMatAddScalar<TP> << <grid, block >> > (B.mat, this->at(0), res.mat, res.size());
				cudaDeviceSynchronize();
				return res;
			}
			else if (B.rows == 1 && B.cols == 1) {
				// B is scalar
				ArrayGPU<TP> res(this->rows, this->cols);

				const int BLOCK_SIZE = 128;
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
					const int BLOCK_SIZE = 128;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatAddVecAlongRows<TP> << <grid, block >> > (B.mat, this->mat, res.mat, res.size(), B.cols);
					cudaDeviceSynchronize();

					return res;
				}
				else if (vecDim == B.cols) {
					//along cols add kr 
					ArrayGPU<TP> res(B.rows, B.cols);
					const int BLOCK_SIZE = 128;
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
					const int BLOCK_SIZE = 128;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatAddVecAlongRows<TP> << <grid, block >> > (this->mat, B.mat, res.mat, res.size(), this->cols);
					cudaDeviceSynchronize();

					return res;
				}
				else if (vecDim == this->cols) {
					//along cols add kr 
					ArrayGPU<TP> res(this->rows, this->cols);
					const int BLOCK_SIZE = 128;
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
				const int BLOCK_SIZE = 128;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(res.size(), block.x));
				kernelMatAddMat<TP> << <grid, block >> > (this->mat, B.mat, res.mat, res.size());
				cudaDeviceSynchronize();
				return res;
			}
		}

		ArrayGPU<TP> operator+(TP Scalar) {
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = 128;
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

				const int BLOCK_SIZE = 128;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(res.size(), block.x));

				kernelMatSubScalar<TP> << <grid, block >> > (B.mat, this->at(0), res.mat, res.size());
				cudaDeviceSynchronize();
				return res;
			}
			else if (B.rows == 1 && B.cols == 1) {
				// B is scalar
				ArrayGPU<TP> res(this->rows, this->cols);

				const int BLOCK_SIZE = 128;
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
					const int BLOCK_SIZE = 128;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatSubVecAlongRows<TP> << <grid, block >> > (B.mat, this->mat, res.mat, res.size(), B.cols);
					cudaDeviceSynchronize();

					return res;
				}
				else if (vecDim == B.cols) {
					//along cols add kr 
					ArrayGPU<TP> res(B.rows, B.cols);
					const int BLOCK_SIZE = 128;
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
					const int BLOCK_SIZE = 128;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatSubVecAlongRows<TP> << <grid, block >> > (this->mat, B.mat, res.mat, res.size(), this->cols);
					cudaDeviceSynchronize();

					return res;
				}
				else if (vecDim == this->cols) {
					//along cols add kr 
					ArrayGPU<TP> res(this->rows, this->cols);
					const int BLOCK_SIZE = 128;
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
				const int BLOCK_SIZE = 128;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(res.size(), block.x));
				kernelMatSubMat<TP> << <grid, block >> > (this->mat, B.mat, res.mat, res.size());
				cudaDeviceSynchronize();
				return res;
			}
		}

		ArrayGPU<TP> operator-(TP Scalar) {
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = 128;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatSubScalar<TP> << <grid, block >> > (this->mat, Scalar, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}

		// multiply
		ArrayGPU<TP> operator*(ArrayGPU<TP>& B) {
			if (this->rows == 1 && this->cols == 1) {
				// A is scalar
				ArrayGPU<TP> res(B.rows, B.cols);

				const int BLOCK_SIZE = 128;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(res.size(), block.x));

				kernelMatMulScalar<TP> << <grid, block >> > (B.mat, this->at(0), res.mat, res.size());
				cudaDeviceSynchronize();
				return res;
			}
			else if (B.rows == 1 && B.cols == 1) {
				// B is scalar
				ArrayGPU<TP> res(this->rows, this->cols);

				const int BLOCK_SIZE = 128;
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
					const int BLOCK_SIZE = 128;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatMulVecAlongRows<TP> << <grid, block >> > (B.mat, this->mat, res.mat, res.size(), B.cols);
					cudaDeviceSynchronize();

					return res;
				}
				else if (vecDim == B.cols) {
					//along cols add kr 
					ArrayGPU<TP> res(B.rows, B.cols);
					const int BLOCK_SIZE = 128;
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
					const int BLOCK_SIZE = 128;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatMulVecAlongRows<TP> << <grid, block >> > (this->mat, B.mat, res.mat, res.size(), this->cols);
					cudaDeviceSynchronize();

					return res;
				}
				else if (vecDim == this->cols) {
					//along cols add kr 
					ArrayGPU<TP> res(this->rows, this->cols);
					const int BLOCK_SIZE = 128;
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
				const int BLOCK_SIZE = 128;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(res.size(), block.x));
				kernelMatMulMat<TP> << <grid, block >> > (this->mat, B.mat, res.mat, res.size());
				cudaDeviceSynchronize();
				return res;
			}
		}

		ArrayGPU<TP> operator*(TP Scalar) {
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = 128;
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

				const int BLOCK_SIZE = 128;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(res.size(), block.x));

				kernelMatDivScalar<TP> << <grid, block >> > (B.mat, this->at(0), res.mat, res.size());
				cudaDeviceSynchronize();
				return res;
			}
			else if (B.rows == 1 && B.cols == 1) {
				// B is scalar
				ArrayGPU<TP> res(this->rows, this->cols);

				const int BLOCK_SIZE = 128;
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
					const int BLOCK_SIZE = 128;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatDivVecAlongRows<TP> << <grid, block >> > (B.mat, this->mat, res.mat, res.size(), B.cols);
					cudaDeviceSynchronize();

					return res;
				}
				else if (vecDim == B.cols) {
					//along cols add kr 
					ArrayGPU<TP> res(B.rows, B.cols);
					const int BLOCK_SIZE = 128;
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
					const int BLOCK_SIZE = 128;
					dim3 block(BLOCK_SIZE);
					dim3 grid(ceil(res.size(), block.x));
					kernelMatDivVecAlongRows<TP> << <grid, block >> > (this->mat, B.mat, res.mat, res.size(), this->cols);
					cudaDeviceSynchronize();

					return res;
				}
				else if (vecDim == this->cols) {
					//along cols add kr 
					ArrayGPU<TP> res(this->rows, this->cols);
					const int BLOCK_SIZE = 128;
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
				const int BLOCK_SIZE = 128;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(res.size(), block.x));
				kernelMatDivMat<TP> << <grid, block >> > (this->mat, B.mat, res.mat, res.size());
				cudaDeviceSynchronize();
				return res;
			}
		}

		ArrayGPU<TP> operator/(TP Scalar) {
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = 128;
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
				const int BLOCK_SIZE = 128; 
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(this->size(), block.x));

				kernelMatIsGreaterThanScalar<TP> << <grid, block >> > (this->mat, B.at(0), res.mat, this->size());
				cudaDeviceSynchronize();
				return res;
			}
			else if (B.rows == this->rows && B.cols == this->cols) {
				// both have same dimensions
				ArrayGPU<TP> res(this->rows, this->cols);
				const int BLOCK_SIZE = 128;
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
			const int BLOCK_SIZE = 128;
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
				const int BLOCK_SIZE = 128;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(this->size(), block.x));

				kernelMatIsLessThanScalar<TP> << <grid, block >> > (this->mat, B.at(0), res.mat, this->size());
				cudaDeviceSynchronize();
				return res;
			}
			else if (B.rows == this->rows && B.cols == this->cols) {
				// both have same dimensions
				ArrayGPU<TP> res(this->rows, this->cols);
				const int BLOCK_SIZE = 128;
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
			const int BLOCK_SIZE = 128;
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
				const int BLOCK_SIZE = 128;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(this->size(), block.x));

				kernelMatIsGreaterThanEqScalar<TP> << <grid, block >> > (this->mat, B.at(0), res.mat, this->size());
				cudaDeviceSynchronize();
				return res;
			}
			else if (B.rows == this->rows && B.cols == this->cols) {
				// both have same dimensions
				ArrayGPU<TP> res(this->rows, this->cols);
				const int BLOCK_SIZE = 128;
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
			const int BLOCK_SIZE = 128;
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
				const int BLOCK_SIZE = 128;
				dim3 block(BLOCK_SIZE);
				dim3 grid(ceil(this->size(), block.x));

				kernelMatIsLessThanEqScalar<TP> << <grid, block >> > (this->mat, B.at(0), res.mat, this->size());
				cudaDeviceSynchronize();
				return res;
			}
			else if (B.rows == this->rows && B.cols == this->cols) {
				// both have same dimensions
				ArrayGPU<TP> res(this->rows, this->cols);
				const int BLOCK_SIZE = 128;
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
			const int BLOCK_SIZE = 128;
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
			}
			else if (axis == 0) {
				// sum along columns. dimension=numCols
			}
			else if (axis == 1) {
				// sum along rows. output dim = numRows
			}
		}

		//max. along axis or total
		ArrayGPU<TP> max(int axis = -1) {
			if (axis == -1) {
				// return overall max
			}
			else if (axis == 0) {
				// max along columns. dimension=numCols
			}
			else if (axis == 1) {
				// max along rows. output dim = numRows
			}
		}

		
		~ArrayGPU() {
			cudaFree(mat);
		}
	};
}