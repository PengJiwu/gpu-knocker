/**
 * Implements helper functions.
 */

#include "helperHost.cuh"

float getFitnessHost(float *fitness, uint32_t island, uint32_t individual,
		Parameters *parameters) {
	return fitness[island * parameters->populationSize + individual];
}

uint32_t *getGeneHost(uint32_t *population, uint32_t island,
		uint32_t individual, uint32_t gene, Parameters *parameters) {
	return &population[gene * parameters->islandAmount
			* parameters->populationSize + island * parameters->populationSize
			+ individual];
}

uint32_t getNumberKnockoutsHost(uint32_t *population, uint32_t island,
		uint32_t individual, Parameters *parameters) {
	uint32_t count = 0;
	for (uint32_t gene = 0; gene < parameters->individualSizeInt; gene++) {
		uint32_t value = ~*getGeneHost(population, island, individual, gene,
				parameters);
		// bit magic - see https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
		uint32_t temp = value - ((value >> 1) & 0x55555555);
		temp = (temp & 0x33333333) + ((temp >> 2) & 0x33333333);
		count += (((temp + (temp >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
	}
	return count;
}

void setFitnessHost(float *fitness, uint32_t island, uint32_t individual,
		float value, Parameters *parameters) {
	fitness[island * parameters->populationSize + individual] = value;
}
