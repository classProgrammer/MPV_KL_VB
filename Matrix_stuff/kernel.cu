#include "kernel.cuh"

// ---------------------------------------------------------
// ----------------- MATRIX TRANSPOSE ----------------------
// ---------------------------------------------------------

__forceinline__ __device__
int matrixPos(int x, int y, int xSize) {
	return y * xSize + x;
}

__global__ __inline__ void transpose_kernel(float* dest, float* src) {
	auto const x{ blockIdx.x * blockDim.x + threadIdx.x };
	auto const y{ blockIdx.y * blockDim.y + threadIdx.y };

	if (x >= N || y >= N) return;

	dest[matrixPos(y,x, N)] = src[matrixPos(x,y,N)];
}

__global__ __inline__ void transpose_kernel_v2(float* dest, float* src) {
	auto const i{ blockIdx.x * blockDim.x + threadIdx.x };
	auto const x{ i % N };
	auto const y{ i / N };

	if (x >= N || y >= N) return;

	auto const from = x * N + y;
	auto const to = y * N + x;

	dest[to] = src[from];
}

void callTranspose(dim3 const& big, dim3 const& tib, float* dest, float* src) {
	transpose_kernel << < big, tib >> > (dest, src);
}

void callTransposeV2(dim3 const& big, dim3 const& tib, float* dest, float* src) {
	transpose_kernel_v2 << < big, tib >> > (dest, src);
}

// ---------------------------------------------------------
// ------------------- MATRIX MULT -------------------------
// ---------------------------------------------------------


__forceinline__ __device__
void cpy(float* src, float* dst, int x, int y) {
	int pos = matrixPos(x, y, O);
	dst[pos] = src[pos];
}

__forceinline__ __device__
int mul(float* m1, float* m2, int x3, int y3) {
	int res = 0;
	for (auto m = 0; m < M; m++) {
		res += m1[matrixPos(m, y3, M)] * m2[matrixPos(x3, m, O)];
	}
	return res;
}
__forceinline__ __global__
void mult_kernel(float* m1, float* m2, float* m3) {
	__shared__ float m3_s[M3_SIZE];
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= O || y >= N_) return;

	m3_s[matrixPos(x, y, O)] = mul(m1, m2, x, y);
	cpy(m3_s, m3, x, y);
}

void callMult(dim3 const& big, dim3 const& tib, float* m1, float* m2, float* m3) {
	mult_kernel << < big, tib >> > (m1, m2, m3);
}