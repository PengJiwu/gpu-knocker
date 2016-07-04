/**
 * Implements linear programmin GPU kernels.
 */

#include <stdio.h>

#include "lpKernels.cuh"

__global__ void solveLP(const uint32_t * const population, float *fitness) {
	if (blockIdx.x * gridDim.x + threadIdx.x == 0) {
		printf("DUMMY solveLP\n");
	}
}
