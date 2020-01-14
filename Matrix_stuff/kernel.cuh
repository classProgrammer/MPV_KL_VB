#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include<iostream>
#include "pfc_types.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code == cudaSuccess) return;

	fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(code);
}


//__constant__ static const int M = 3;
//__constant__ static const int I = 2;
__constant__ static const int N = 3;
//
//__constant__ static const int M1_SIZE = M * I; // y * x matrix
//__constant__ static const int M2_SIZE = N * M; // y * x matrix
//__constant__ static const int M3_SIZE = N * I; // y * x matrix

// Transpose


void callTranspose(dim3 const& big, dim3 const& tib, float* dest, float* src);
void callTransposeV2(dim3 const& big, dim3 const& tib, float* dest, float* src);


// MULT
__constant__ static const int N_ = 3;
__constant__ static const int M = 2;
__constant__ static const int O = 3;
__constant__ static const int M1_SIZE = N_ * M; // y * x matrix
__constant__ static const int M2_SIZE = M * O; // y * x matrix
__constant__ static const int M3_SIZE = N_ * O; // y * x matrix
void callMult(dim3 const& big, dim3 const& tib, float* m1, float* m2, float* m3);