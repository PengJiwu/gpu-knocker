/**
 * Defines evolutionary algorithm GPU kernels.
 */

#ifndef EAKERNELS_CUH_
#define EAKERNELS_CUH_

#include <curand_kernel.h>
#include <stdint.h>

#include "parameters.cuh"
#include "statistics.cuh"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initializes population with random data.
 *
 * @param population Population to initialize.
 * @param rngState RNGState.
 */
__global__ void createPopulation(uint32_t *population,
		curandStatePhilox4_32_10 *rngState);

/**
 * Crosses individuals of population over.
 *
 * @param population Population.
 * @param population Temporary population.
 * @param rngState RNGState.
 */
__global__ void crossoverPopulation(const uint32_t * const population,
		uint32_t *temporaryPopulation, curandStatePhilox4_32_10 *rngState);

/**
 * Initializes population with random data.
 *
 * @param state State of random number generator to initialize.
 */
__global__ void initializeRNG(curandStatePhilox4_32_10 *rngState);

/**
 * Migrates individuals between islands.
 *
 * @param population Population.
 * @param statistics Statistics
 * @param iteration Iteration.
 */
__global__ void migratePopulation(uint32_t *population,
		const float * const statistics, uint32_t iteration);

/**
 * Mutates some individuals of the population.
 *
 * @param population Population.
 * @param rngState RNGState.
 */
__global__ void mutatePopulation(uint32_t *population,
		curandStatePhilox4_32_10 *rngState);

/**
 * Stores best individuals from population in temporary population.
 *
 * @param population Population.
 * @param temporaryPopulation Temporary population.
 * @param fitness Fitness.
 * @param rngState RNGState.
 */
__global__ void selectPopulation(const uint32_t * const population,
		uint32_t *temporaryPopulation, const float * const fitness,
		curandStatePhilox4_32_10 *rngState);

#ifdef __cplusplus
}
#endif

#endif /* EAKERNELS_CUH_ */
