#include <vector>

#include "pfc_threading.h"
#include<string>  
#include "kernel.cuh"

bool foundDevice() {
	auto count{ 0 };
	gpuErrchk(cudaGetDeviceCount(&count));
	if (count < 1) return false;

	auto device_no{ 0 };
	gpuErrchk(cudaSetDevice(device_no));

	return true;
}

void printMatrix(float* matrix, int width, int const height) {
	for (int y = 0; y < height; ++y) {
		std::cout << "| ";
		for (int x = 0; x < width; ++x) {
			std::cout << matrix[y * width + x] << " ";
		}
		std::cout << "|" << std::endl;
	}
	std::cout << std::endl;
}

void runTranspose() {
	auto const size = N * N;
	auto const mem_size = size * sizeof(float);
	float matrix_src[size];
	float matrix_dest[size];

	for (int i = 0; i < size; ++i) {
		matrix_src[i] = i;
	}

	auto* hostpointer_src{ matrix_src };
	auto* hostpointer_dest{ matrix_dest };

	printMatrix(hostpointer_src, N, N);

	float* device_src = nullptr;
	float* device_dest = nullptr;

	gpuErrchk(cudaMalloc(&device_src, mem_size))
		gpuErrchk(cudaMalloc(&device_dest, mem_size));

	gpuErrchk(cudaMemcpy(device_src, hostpointer_src, mem_size, cudaMemcpyKind::cudaMemcpyHostToDevice));

	dim3 const tib{ 32, 32, 1 };
	dim3 const big{ 1 };

	callTranspose(big, tib, device_dest, device_src);
	gpuErrchk(cudaGetLastError());

	gpuErrchk(cudaMemcpy(hostpointer_dest, device_dest, mem_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));

	printMatrix(hostpointer_dest, N, N);
	gpuErrchk(cudaDeviceReset());
}




void runMult() {
	auto const TIB_SIZE = 32;

	auto m1_host_ptr = std::make_unique<float[]>(M1_SIZE);
	auto m2_host_ptr = std::make_unique<float[]>(M2_SIZE);
	auto m3_host_ptr = std::make_unique<float[]>(M3_SIZE);

	for (int i = 0; i < M1_SIZE; ++i) {
		m1_host_ptr[i] = i + 1;
	}
	printMatrix(m1_host_ptr.get(), M, N_);
	for (int i = 0; i < M2_SIZE; ++i) {
		m2_host_ptr[i] = 3-i;
	}
	printMatrix(m2_host_ptr.get(), O, M);

	float* m1_device_ptr = nullptr;
	float* m2_device_ptr = nullptr;
	float* m3_device_ptr = nullptr;

	cudaMalloc(&m1_device_ptr, M1_SIZE * sizeof(int));
	cudaMalloc(&m2_device_ptr, M2_SIZE * sizeof(int));
	cudaMalloc(&m3_device_ptr, M3_SIZE * sizeof(int));

	cudaMemcpy(m1_device_ptr, m1_host_ptr.get(), M1_SIZE * sizeof(float),
		cudaMemcpyHostToDevice);
	cudaMemcpy(m2_device_ptr, m2_host_ptr.get(), M2_SIZE * sizeof(float),
		cudaMemcpyHostToDevice);

	dim3 const tib = { TIB_SIZE, TIB_SIZE, 1 };
	dim3 const big = { (O + tib.x - 1) / tib.x, (N_ + tib.y - 1) / tib.y, 1 };
	callMult(big, tib, m1_device_ptr, m2_device_ptr, m3_device_ptr);

	cudaDeviceSynchronize();
	cudaMemcpy(m3_host_ptr.get(), m3_device_ptr, M3_SIZE * sizeof(float),
		cudaMemcpyDeviceToHost);

	printMatrix(m3_host_ptr.get(), O, N_);


	cudaFree(m1_device_ptr);
	cudaFree(m2_device_ptr);
	cudaFree(m3_device_ptr);
}

int main() {

	if (!foundDevice()) {
		std::cout << "!!!!!!!!!! FAILED, no device found !!!!!!!!!!" << std::endl;
		return 1;
	}

	runTranspose();

	runMult();

	return 0;
}