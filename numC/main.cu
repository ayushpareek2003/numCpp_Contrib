#include "npGPUArray.cuh"
#include "npRandom.cuh"

#include "npFunctions.cuh"

int main() {
	auto a = np::Random::randn<float>(5, 5);
	a.print();
	printf("\n\n\n");
	np::exp<float>(a).print();

	printf("\n\n\n");
	np::log(np::exp<float>(a)).print();



	return 0;
}
