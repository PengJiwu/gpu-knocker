/**
 * Implements helper functions.
 */

#include "helper.cuh"

#include "parameters.cuh"

__device__ void copyIndividual(uint32_t *fromPopulation, uint32_t fromIsland,
		uint32_t fromIndividual, uint32_t *toPopulation, uint32_t toIsland,
		uint32_t toIndividual) {
	for (uint32_t gene = 0; gene < parametersGPU.individualSizeInt; gene++) {
		uint32_t *toGene = getGene(toPopulation, toIsland, toIndividual, gene);
		*toGene = *getGene(fromPopulation, fromIsland, fromIndividual, gene);
	}
}

__device__ float getFitness(float *fitness, uint32_t island,
		uint32_t individual) {
	return fitness[island * parametersGPU.populationSize + individual];
}

__device__ uint32_t *getGene(uint32_t *population, uint32_t island,
		uint32_t individual, uint32_t gene) {
	return &population[gene * parametersGPU.islandAmount
			* parametersGPU.populationSize
			+ island * parametersGPU.populationSize + individual];
}

__device__ float maximum(float value1, float value2) {
	return (value1 > value2) ? value1 : value2;
}

__device__ float minimum(float value1, float value2) {
	return (value1 < value2) ? value1 : value2;
}

__device__ void lock(uint32_t *mutex) {
	while (atomicCAS(mutex, 0, 1) != 0) {
	}
}

__device__ void setFitness(float *fitness, uint32_t island, uint32_t individual,
		float value) {
	fitness[island * parametersGPU.populationSize + individual] = value;
}

__device__ void unlock(uint32_t *mutex) {
	atomicExch(mutex, 0);
}

__device__ float warpReduceMax(float value) {
	for (uint32_t offset = warpSize / 2; offset > 0; offset /= 2) {
		float shuffle = __shfl_down(value, offset, warpSize);
		value = maximum(value, shuffle);
	}
	return value;
}

__device__ float warpReduceMin(float value) {
	for (uint32_t offset = warpSize / 2; offset > 0; offset /= 2) {
		float shuffle = __shfl_down(value, offset, warpSize);
		value = minimum(value, shuffle);
	}
	return value;
}

__device__ float warpReduceSum(float value) {
	for (uint32_t offset = warpSize / 2; offset > 0; offset /= 2) {
		value += __shfl_down(value, offset, warpSize);
	};
	return value;
}
