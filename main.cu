#include "src/npGPUArray.cuh"
#include "src/npRandom.cuh"
#include "src/npFunctions.cuh"


#include <time.h>


int main() {
	np::getGPUConfig();
	printf("\nGPU CONFIG: NUM_CORES: %d, NUM_SMs = %d\n", np::GPU_NUM_CUDA_CORE, np::GPU_NUM_SM);

	auto a = np::Random::rand<int>(5, 2, 40, 60);

	std::cout<<"A: \n"<<a<<"\n";
	std::cout<<"A.T: \n"<<a.T()<<"\n";
	// std::cout<<"A.T.T: \n"<<a.T().T()<<"\n";

	auto a_neg = -a;
	std::cout<<"\n-A: \n"<<a_neg<<"\n";

	std::cout<<"\nTOTAL SUM: "<<a.sum()<<"\n";
	std::cout<<"\nSUM ALONG COLS: "<<a.T().sum(1)<<"\n";
	std::cout<<"\nSUM ALONG ROWS: "<<a.sum(1)<<"\n";

	cudaDeviceReset();
	return 0;
}
