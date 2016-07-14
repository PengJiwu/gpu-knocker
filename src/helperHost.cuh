/**
 * Defines helper functions on host.
 */

#ifndef HELPERHOST_CUH_
#define HELPERHOST_CUH_

#include "parameters.cuh"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Returns pointer to specified gene.
 *
 * @param population Population.
 * @param island Number of island.
 * @param individual Number of individual.
 * @param gene Number of gene.
 * @param parameters Parameters.
 * @return Pointer to specified gene.
 */
uint32_t *getGeneHost(uint32_t *population, uint32_t island,
		uint32_t individual, uint32_t gene, Parameters *parameters);

/**
 * Sets desired fitness value
 *
 * @param fitness Fitness.
 * @param island Island.
 * @param individual Individual.
 * @param value Value.
 * @param parameters Parameters.
 */
void setFitnessHost(float *fitness, uint32_t island, uint32_t individual,
		float value, Parameters *parameters);

#ifdef __cplusplus
}
#endif

#endif /* HELPERHOST_CUH_ */
