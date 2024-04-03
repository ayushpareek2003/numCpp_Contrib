#include <numC/npGPUArray.cuh>
#include <numC/gpuConfig.cuh>
#include <numC/npRandom.cuh>
#include <numC/npFunctions.cuh>

// standard libs
#include <iostream>
#include <time.h>


int main() {
	np::getGPUConfig(0);
	printf("\nGPU CONFIG: NUM_CORES: %d, NUM_SMs = %d\n", np::GPU_NUM_CUDA_CORE, np::GPU_NUM_SM);

	auto a = np::Random::rand<int>(10, 10, 40, 60);
	std::cout<<a<<std::endl;

	a.set(np::arange<int>(10), np::arange<int>(10), a.at(np::arange<int>(10), np::arange<int>(10)) * 2);
	
	std::cout<<a<<std::endl;

	cudaDeviceReset();
	return 0;
}
