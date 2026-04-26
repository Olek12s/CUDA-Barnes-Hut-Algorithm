#include <iostream>
#include <vector>
#include <algorithm>
#include <execution>
#include <random>
#include <chrono>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

__host__ void cuda_sort_example(float* arr, int n)
{
	thrust::device_vector<float> d_vec(arr, arr + n);
	thrust::sort(d_vec.begin(), d_vec.end());
	cudaMemcpy(arr, thrust::raw_pointer_cast(d_vec.data()), n*sizeof(float), cudaMemcpyDeviceToHost);
}

void cudaTest()
{
	std::cout << "Test CUDA & OpenGL code\n";

	// Set up CUDA device
	cudaDeviceProp properties;

	cudaGetDeviceProperties(&properties,0);

	int fact = 1024;
	int driverVersion, runtimeVersion;

	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);

	int gpuClockKHz;
	int memClockKHz;

	cudaDeviceGetAttribute(
		&gpuClockKHz,
		cudaDevAttrClockRate,
		0);

	cudaDeviceGetAttribute(
		&memClockKHz,
		cudaDevAttrMemoryClockRate,
		0);

	std::cout << "************************************************************************" << std::endl;
	std::cout << "                          GPU Device Properties                         " << std::endl;
	std::cout << "************************************************************************" << std::endl;
	std::cout << "Name:                                    " << properties.name << std::endl;
	std::cout << "CUDA driver/runtime version:             " << driverVersion/1000 << "." << (driverVersion%100)/10 << "/" << runtimeVersion/1000 << "." << (runtimeVersion%100)/10 << std::endl;
	std::cout << "CUDA compute capabilitiy:                " << properties.major << "." << properties.minor << std::endl;
	std::cout << "Number of multiprocessors:               " << properties.multiProcessorCount << std::endl;
	std::cout << "GPU clock rate: "
			  << gpuClockKHz / 1024
			  << " (MHz)" << std::endl;

	std::cout << "Memory clock rate: "
			  << memClockKHz / 1024
			  << " (MHz)" << std::endl;
	std::cout << "Memory bus width:                        " << properties.memoryBusWidth << "-bit" << std::endl;
	//std::cout << "Theoretical memory bandwidth:            " << (properties.memoryClockRate/fact*(properties.memoryBusWidth/8)*2)/fact <<" (GB/s)" << std::endl;
	std::cout << "Device global memory:                    " << properties.totalGlobalMem/(fact*fact) << " (MB)" << std::endl;
	std::cout << "Shared memory per block:                 " << properties.sharedMemPerBlock/fact <<" (KB)" << std::endl;
	std::cout << "Constant memory:                         " << properties.totalConstMem/fact << " (KB)" << std::endl;
	std::cout << "Maximum number of threads per block:     " << properties.maxThreadsPerBlock << std::endl;
	std::cout << "Maximum thread dimension:                [" << properties.maxThreadsDim[0] << ", " << properties.maxThreadsDim[1] << ", " << properties.maxThreadsDim[2] << "]" << std::endl;
	std::cout << "Maximum grid size:                       [" << properties.maxGridSize[0] << ", " << properties.maxGridSize[1] << ", " << properties.maxGridSize[2] << "]" << std::endl;
	std::cout << "**************************************************************************" << std::endl;
	std::cout << "                                                                          " << std::endl;
	std::cout << "**************************************************************************" << std::endl;
}
