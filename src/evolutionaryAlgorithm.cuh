/**
 * Defines evolutionary algorithm.
 */

#ifndef EVOLUTIONARYALGORITHM_CUH_
#define EVOLUTIONARYALGORITHM_CUH_

#include <curand_kernel.h>
#include <stdint.h>

#include "lpSolver.cuh"
#include "parameters.cuh"
#include "statistics.cuh"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Holds data for the evolutionary algorithm.
 */
typedef struct EvolutionaryAlgorithm {
	/**
	 * Fitness for evolutionary algorithm.
	 */
	float *fitness;

	/**
	 * Population for evolutionary algorithm.
	 */
	uint32_t *population;

	/**
	 * State of random number generator for each thread.
	 */
	curandStatePhilox4_32_10 *rngState;

	/**
	 * Temporary population for evolutionary algorithm.
	 */
	uint32_t *temporaryPopulation;
} EvolutionaryAlgorithm;

/**
 * Create evolutionary algorithm with default values.
 *
 * @param parameters Parameters.
 * @return Initialized EvolutionaryAlgorithm.
 */
EvolutionaryAlgorithm *createEvolutionaryAlgorithm(Parameters *parameters);

/**
 * Clears memory for EvolutionaryAlgorithm.
 *
 * @param evolutionaryAlgorithm EvolutionaryAlgorithm to be deleted.
 */
void deleteEvolutionaryAlgorithm(EvolutionaryAlgorithm *evolutionaryAlgorithm);

/**
 * Print the best knockout with least knockouts.
 *
 * @param lpSolver
 * @param statistics
 * @param parameters
 */
void printBestKnockout(LPSolver* lpSolver, Statistics* statistics,
		Parameters* parameters);

/**
 * Prints knockouts with fitness value equal to the islands maximum.
 *
 * @param lpSolver LPSolver.
 * @param statistics Statistics.
 * @param parameters Parameters.
 */
void printBestKnockouts(LPSolver* lpSolver, Statistics* statistics,
		Parameters* parameters);

/**
 * Prints knockouts of specified individual.
 *
 * @param individual
 * @param island
 * @param lpSolver
 * @param parameters
 */
void printKnockout(uint32_t individual, uint32_t island, LPSolver* lpSolver,
		Parameters* parameters);

/**
 * Starts the evolutionary algorithm.
 *
 * @param evolutionaryAlgorithm EvolutionaryAlgorithm.
 * @param lpSolver LPSolver.
 * @param parameters Parameters.
 * @param statistics Statistics.
 */
void runEvolutionaryAlgorithm(EvolutionaryAlgorithm *evolutionaryAlgorithm,
		LPSolver *lpSolver, Statistics *statistics, Parameters *parameters);

/**
 * Swaps pointer to population and temporary population.
 *
 * @param population Population.
 * @param temporaryPopulation TemporaryPopulation.
 */
void swapTemporaryPopulation(uint32_t **population,
		uint32_t **temporaryPopulation);

#ifdef __cplusplus
}
#endif

#endif /* EVOLUTIONARYALGORITHM_CUH_ */
