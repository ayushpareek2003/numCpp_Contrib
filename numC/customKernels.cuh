#include <cuda_runtime.h>
#include <cmath>
#pragma once


// cuda error checking macro
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    }} while(0)

// 1 x 1 grid to print matrices
template<typename TP>
__global__ void kernelPrintMat(TP* in, int M, int N) {
	// 1, 1 grid. only to print matrix
	for (int r = 0; r < M; ++r) {
		for (int c = 0; c < N; ++c) {
			if (std::is_same<TP, int>::value) {
				printf("%d ", in[r * N + c]);
			}
			else if (std::is_same<TP, float>::value) {
				printf("%f ", in[r * N + c]);
			}
			else if (std::is_same<TP, double>::value) {
				printf("%lf ", in[r * N + c]);
			}
			else if (std::is_same<TP, char>::value) {
				printf("%c ", in[r * N + c]);
			}
			else {
				// Handle other types here
				printf("Unsupported type");
			}
		}
		printf("\n");
	}
}


// kernel to copy a matrix stored in row major order to column major order and vice-versa.
template<typename TP>
__global__ void kernelTransposeInMem(TP* in, TP* out, int M, int N) {
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



// kernel to initialise all values of nparray to val.
template<typename TP>
__global__ void kernelInitMatBroadcast(TP* in, TP Val, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		in[idx] = Val;
	}
}



//kernel to initialised arange array -> 0 to n - 1, array size N
template<typename TP>
__global__ void kernelInitMatArange(TP* in, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		in[idx] = idx;
	}
}



// kernel to get values at a range of indexes.
template<typename TP>
__global__ void kernelGetMatValues(TP* in, int rdin, TP* out, int* rows, int* cols, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		out[idx] = in[rows[idx] * rdin + cols[idx]];
	}
}

// kernel to set values at a range of indexes.
template<typename TP>
__global__ void kernelSetMatValues(TP* in, int rdin, TP* val, int* rows, int* cols, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		in[rows[idx] * rdin + cols[idx]] = val[idx];
	}
}


//ARITHMATIC FUNCTIONs

// addition functions

//add corrosponding elements of 2 matrices 
// C = A + B
template<typename TP>
__global__ void kernelMatAddMat(TP* A, TP* B, TP* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		C[idx] = A[idx] + B[idx];
	}
}

//add matrix and a scalar.
// C = A + Scal (broadcasting)
//add scalar to all values of the matrix
template<typename TP>
__global__ void kernelMatAddScalar(TP* A, TP Scal, TP* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		C[idx] = A[idx] + Scal;
	}
}

//add matrix  and vector, mat.rows = vec.dim 
//C = A + V (broadcasting)
// shapeA = M x N matrix
template<typename TP>
__global__ void kernelMatAddVecAlongRows(TP* A, TP* V, TP* C, int size, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int r = idx / N;
	//int c = idx % N; 

	if (idx < size) {
		C[idx] = A[idx] + V[r];
	}
}

//add matrix  and vector, mat.cols = vec.dim 
//C = A + V (broadcasting)
// shapeA = M x N matrix
template<typename TP>
__global__ void kernelMatAddVecAlongCols(TP* A, TP* V, TP* C, int size, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//int r = idx / N;
	int c = idx % N;

	if (idx < size) {
		C[idx] = A[idx] + V[c];
	}
}

// subtraction functions

//subtract corrosponding elements of 2 matrices 
// C = A - B
template<typename TP>
__global__ void kernelMatSubMat(TP* A, TP* B, TP* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		C[idx] = A[idx] - B[idx];
	}
}

//subtract matrix and a scalar.
// C = A - Scal (broadcasting)
//subtract scalar from all values of the matrix
template<typename TP>
__global__ void kernelMatSubScalar(TP* A, TP Scal, TP* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		C[idx] = A[idx] - Scal;
	}
}

//sub matrix  and vector, mat.rows = vec.dim 
//C = A - V (broadcasting)
// shapeA = M x N matrix
template<typename TP>
__global__ void kernelMatSubVecAlongRows(TP* A, TP* V, TP* C, int size, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int r = idx / N;
	//int c = idx % N; 

	if (idx < size) {
		C[idx] = A[idx] - V[r];
	}
}

//sub matrix and vector, mat.cols = vec.dim 
//C = A - V (broadcasting)
// shapeA = M x N matrix
template<typename TP>
__global__ void kernelMatSubVecAlongCols(TP* A, TP* V, TP* C, int size, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//int r = idx / N;
	int c = idx % N;

	if (idx < size) {
		C[idx] = A[idx] - V[c];
	}
}


// multiplication functions

//mul corrosponding elements of 2 matrices 
// C = A * B
template<typename TP>
__global__ void kernelMatMulMat(TP* A, TP* B, TP* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		C[idx] = A[idx] * B[idx];
	}
}

//mul matrix and a scalar.
// C = A * Scal (broadcasting)
//mul scalar to all values of the matrix
template<typename TP>
__global__ void kernelMatMulScalar(TP* A, TP Scal, TP* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		C[idx] = A[idx] * Scal;
	}
}

//mul matrix  and vector, mat.rows = vec.dim 
//C = A * V (broadcasting)
// shapeA = M x N matrix
template<typename TP>
__global__ void kernelMatMulVecAlongRows(TP* A, TP* V, TP* C, int size, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int r = idx / N;
	//int c = idx % N; 

	if (idx < size) {
		C[idx] = A[idx] * V[r];
	}
}

//mul matrix  and vector, mat.cols = vec.dim 
//C = A * V (broadcasting)
// shapeA = M x N matrix
template<typename TP>
__global__ void kernelMatMulVecAlongCols(TP* A, TP* V, TP* C, int size, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//int r = idx / N;
	int c = idx % N;

	if (idx < size) {
		C[idx] = A[idx] * V[c];
	}
}




// division functions

//div corrosponding elements of 2 matrices 
// C = A / B
template<typename TP>
__global__ void kernelMatDivMat(TP* A, TP* B, TP* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		C[idx] = A[idx] / B[idx];
	}
}


//div matrix and a scalar.
// C = A / Scal (broadcasting)
//div scalar to all values of the matrix
template<typename TP>
__global__ void kernelMatDivScalar(TP* A, TP Scal, TP* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		C[idx] = A[idx] / Scal;
	}
}

//div matrix  and vector, mat.rows = vec.dim 
//C = A / V (broadcasting)
// shapeA = M x N matrix
template<typename TP>
__global__ void kernelMatDivVecAlongRows(TP* A, TP* V, TP* C, int size, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int r = idx / N;
	//int c = idx % N; 

	if (idx < size) {
		C[idx] = A[idx] / V[r];
	}
}

//div matrix  and vector, mat.cols = vec.dim 
//C = A / V (broadcasting)
// shapeA = M x N matrix
template<typename TP>
__global__ void kernelMatDivVecAlongCols(TP* A, TP* V, TP* C, int size, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//int r = idx / N;
	int c = idx % N;

	if (idx < size) {
		C[idx] = A[idx] / V[c];
	}
}


// compare 2 matrix ( element wise ) and put max value in result matrix.
//A = MxN
//B = MxN
//Ci = max(Ai, Bi). (elementwise)
template<typename TP>
__global__ void kernelMatMaximumMat(TP* A, TP *B, TP* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		if (A[idx] > B[idx])
			C[idx] = A[idx];
		else
			C[idx] = B[idx];
	}
}

// compare a matrix and a scalar and put max of them in result matrix.
//A = MxN
//B = scalar
//Ci = max(Ai, B). (elementwise)
template<typename TP>
__global__ void kernelMatMaximumScalar(TP* A, TP B, TP* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		if (A[idx] > B)
			C[idx] = A[idx];
		else
			C[idx] = B;
	}
}

// compare 2 matrix. tell if former greater.
//A = MxN
//B = scalar
//Ci = Ai > Bi. (elementwise)
template<typename TP>
__global__ void kernelMatIsGreaterThanMat(TP* A, TP *B, TP* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		if (A[idx] > B[idx])
			C[idx] = 1;
		else
			C[idx] = 0;
	}
}

// compare a matrix and scalar tell if former greater.
//A = MxN
//B = scalar
//Ci = Ai > B. (elementwise)
template<typename TP>
__global__ void kernelMatIsGreaterThanScalar(TP* A, TP B, TP* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		if (A[idx] > B)
			C[idx] = 1;
		else
			C[idx] = 0;
	}
}


// compare 2 matrix. tell if former smaller.
//A = MxN
//B = scalar
//Ci = Ai < Bi. (elementwise)
template<typename TP>
__global__ void kernelMatIsLessThanMat(TP* A, TP* B, TP* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		if (A[idx] < B[idx])
			C[idx] = 1;
		else
			C[idx] = 0;
	}
}

// compare a matrix and scalar tell if former smaller.
//A = MxN
//B = scalar
//Ci = Ai < B. (elementwise)
template<typename TP>
__global__ void kernelMatIsLessThanScalar(TP* A, TP B, TP* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		if (A[idx] < B)
			C[idx] = 1;
		else
			C[idx] = 0;
	}
}

// compare 2 matrix. tell if former greater.
//A = MxN
//B = scalar
//Ci = Ai >= Bi. (elementwise)
template<typename TP>
__global__ void kernelMatIsGreaterThanEqMat(TP* A, TP* B, TP* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		if (A[idx] >= B[idx])
			C[idx] = 1;
		else
			C[idx] = 0;
	}
}

// compare a matrix and scalar tell if former greater.
//A = MxN
//B = scalar
//Ci = Ai > B. (elementwise)
template<typename TP>
__global__ void kernelMatIsGreaterThanEqScalar(TP* A, TP B, TP* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		if (A[idx] >= B)
			C[idx] = 1;
		else
			C[idx] = 0;
	}
}


// compare 2 matrix. tell if former smaller.
//A = MxN
//B = scalar
//Ci = Ai < Bi. (elementwise)
template<typename TP>
__global__ void kernelMatIsLessThanEqMat(TP* A, TP* B, TP* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		if (A[idx] <= B[idx])
			C[idx] = 1;
		else
			C[idx] = 0;
	}
}

// compare a matrix and scalar tell if former smaller.
//A = MxN
//B = scalar
//Ci = Ai < B. (elementwise)
template<typename TP>
__global__ void kernelMatIsLessThanEqScalar(TP* A, TP B, TP* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		if (A[idx] <= B)
			C[idx] = 1;
		else
			C[idx] = 0;
	}
}


// np.exp
// A = MxN
// C = MxN
//Ci = exp(Ai)
template<typename TP>
__global__ void kernelExpMat(TP* A, TP* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		C[idx] = expf(A[idx]);
	}
}

// np.log
// A = MxN
// C = MxN
//Ci = exp(Ai)
template<typename TP>
__global__ void kernelLogMat(TP* A, TP* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		C[idx] = logf(A[idx]);
	}
}
