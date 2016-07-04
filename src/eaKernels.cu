/**
 * Implementation of evolutionary algorithm GPU kernels.
 */

#include "eaKernels.cuh"

#include <stdio.h>

__global__ void createPopulation(uint32_t *population,
		curandStatePhilox4_32_10 *rngState) {
	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id == 0) {
		printf("DUMMY createPopulation\n");
	}

	curandStatePhilox4_32_10 localState = rngState[id];
	curand(&localState);
	rngState[id] = localState;
}

__global__ void crossoverPopulation(const uint32_t * const population,
		uint32_t *temporaryPopulation, curandStatePhilox4_32_10 *rngState) {
	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id == 0) {
		printf("DUMMY crossoverPopulation\n");
	}

	curandStatePhilox4_32_10 localState = rngState[id];
	curand(&localState);
	rngState[id] = localState;
}

__global__ void initializeRNG(curandStatePhilox4_32_10 *rngState) {
	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(0, id, 0, &rngState[id]);
}

__global__ void migratePopulation(uint32_t *population,
		const float * const statistics, uint32_t iteration) {
	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id == 0) {
		printf("DUMMY migratePopulation\n");
	}
}

__global__ void mutatePopulation(uint32_t *population,
		curandStatePhilox4_32_10 *rngState) {
	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id == 0) {
		printf("DUMMY mutatePopulation\n");
	}

	curandStatePhilox4_32_10 localState = rngState[id];
	curand(&localState);
	rngState[id] = localState;
}

__global__ void selectPopulation(const uint32_t * const population,
		uint32_t *temporaryPopulation, const float * const fitness,
		curandStatePhilox4_32_10 *rngState) {
	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id == 0) {
		printf("DUMMY selectPopulation\n");
	}

	curandStatePhilox4_32_10 localState = rngState[id];
	curand(&localState);
	rngState[id] = localState;
}
