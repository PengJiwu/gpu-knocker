/**
 * Implements linear programming GPU kernels.
 */

#include "lpKernels.cuh"

#include <stdint.h>
#include <stdio.h>

#include "helper.cuh"

__device__ float fitnessFunction(uint32_t * population, uint32_t island,
		uint32_t individual) {
	// TODO replace
	uint32_t count = 0;
	for (uint32_t gene = 0; gene < parametersGPU.individualSizeInt; gene++) {
		uint32_t value = *getGene(population, island, individual, gene);
		// bit magic - see https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
		uint32_t temp = value - ((value >> 1) & 0x55555555);
		temp = (temp & 0x33333333) + ((temp >> 2) & 0x33333333);
		count += (((temp + (temp >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
	}
	return (float) count;
}

__global__ void solveLP(uint32_t *population, float *fitness) {
	// TODO replace
	for (uint32_t island = blockIdx.x; island < parametersGPU.islandAmount;
			island += gridDim.x) {
		for (uint32_t individual = threadIdx.x;
				individual < parametersGPU.populationSize; individual +=
						blockDim.x) {
			float fitnessValue = fitnessFunction(population, island,
					individual);
			setFitness(fitness, island, individual, fitnessValue);
		}
	}
}
