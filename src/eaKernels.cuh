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
 * Flips a single bit
 *
 * @param gene Gene.
 * @param rngState RNGState.
 */
__device__ void bitFlipMutation(uint32_t *gene,
		curandStatePhilox4_32_10 *rngState);

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
__global__ void crossoverPopulation(uint32_t *population,
		curandStatePhilox4_32_10 *rngState);

/**
 * Initializes population with random data.
 *
 * @param seed Seed for random number generator.
 * @param state State of random number generator to initialize.
 */
__global__ void initializeRNG(uint32_t seed,
		curandStatePhilox4_32_10 *rngState);

/**
 * Migrates individuals between islands.
 *
 * @param population Population.
 * @param fitness Fitness.
 * @param statistics Statistics.
 */
__global__ void migratePopulation(uint32_t *population, float *fitness,
		float *statistics);

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
__global__ void selectPopulation(uint32_t *population,
		uint32_t *temporaryPopulation, float *fitness,
		curandStatePhilox4_32_10 *rngState);

/**
 * Selects individual by tournament selection.
 *
 * @param population Population.
 * @param temporaryPopulation TemporaryPopulation.
 * @param fitness Fitness.
 * @param island Island.
 * @param individual Individual.
 * @param rngState RNGState.
 */
__device__ void tournamentSelection(uint32_t *population,
		uint32_t *temporaryPopulation, float *fitness, uint32_t island,
		uint32_t individual, curandStatePhilox4_32_10 *rngState);

/**
 * Creates new individual by uniform crossover.
 *
 * @param population Population.
 * @param temporaryPopulation TemporaryPopulation.
 * @param rngState RNGState.
 */
__device__ void uniformCrossover(uint32_t *population, uint32_t island,
		uint32_t individual, curandStatePhilox4_32_10 *rngState);

#ifdef __cplusplus
}
#endif

#endif /* EAKERNELS_CUH_ */
