/**
 * Implements helper functions.
 */

#include "helperHost.cuh"

uint32_t *getGeneHost(uint32_t *population, uint32_t island,
		uint32_t individual, uint32_t gene, Parameters *parameters) {
	return &population[gene * parameters->islandAmount
			* parameters->populationSize + island * parameters->populationSize
			+ individual];
}

void setFitnessHost(float *fitness, uint32_t island, uint32_t individual,
		float value, Parameters *parameters) {
	fitness[island * parameters->populationSize + individual] = value;
}
