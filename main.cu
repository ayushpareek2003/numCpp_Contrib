#include "npGPUArray.cuh"
#include "npRandom.cuh"

#include "npFunctions.cuh"
#include <time.h>

int main() {
	np::getGPUConfig();
	std::cout << np::GPU_NUM_CUDA_CORE;
	auto a = np::Random::randn<float>(10,10);	
	printf("\nA: \n");
	a.print();

	printf("\n argmin(A): \n");
	a.argmin().print();
	printf("\n argmin(A, 0)\n");
	a.argmin(0).print();
	printf("\n argmin(A, 1)\n");
	a.argmin(1).print();


	return 0;
}
