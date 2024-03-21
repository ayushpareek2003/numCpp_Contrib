#include "npGPUArray.cuh"
#include "npRandom.cuh"

#include "npFunctions.cuh"
#include <time.h>

int main() {
	np::getGPUConfig();
	printf("\nGPU CONFIG: NUM_CORES: %d, NUM_SMs = %d\n", np::GPU_NUM_CUDA_CORE, np::GPU_NUM_SM);

	auto a = np::Random::rand<float>(10);
	a.reshape(5, 2);
	a.print();

	printf("\n TOT SUM: \n");
	a.sum().print();

	printf("\n A:\n");
	a.print();
	

	printf("\n SUM (axis = 0): \n");
	a.sum(0).print();

	printf("\n SUM (axis = 1): \n");
	a.sum(1).print();

	cudaDeviceReset();

	return 0;
}
