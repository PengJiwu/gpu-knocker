/**
 * Implementation of evolutionary algorithm.
 */

#include "evolutionaryAlgorithm.cuh"

#include <stdlib.h>
#include <string.h>

#include "cudaCheck.cuh"
#include "eaKernels.cuh"
#include "lpKernels.cuh"

EvolutionaryAlgorithm *createEvolutionaryAlgorithm(Parameters *parameters) {
	EvolutionaryAlgorithm *evolutionaryAlgorithm =
			(EvolutionaryAlgorithm *) malloc(sizeof(EvolutionaryAlgorithm));
	cudaCheck(
			cudaMalloc(&evolutionaryAlgorithm->fitness,
					parameters->individualSizeInt * parameters->populationSize
							* sizeof(uint32_t)));
	cudaCheck(
			cudaMalloc(&evolutionaryAlgorithm->population,
					parameters->individualSizeInt * parameters->populationSize
							* sizeof(uint32_t)));
	cudaCheck(
			cudaMalloc(&evolutionaryAlgorithm->rngState,
					parameters->blockSize * parameters->gridSize
							* sizeof(curandStatePhilox4_32_10)));
	cudaCheck(
			cudaMalloc(&evolutionaryAlgorithm->temporaryPopulation,
					parameters->individualSizeInt * parameters->populationSize
							* sizeof(uint32_t)));

	return evolutionaryAlgorithm;
}

void deleteEvolutionaryAlgorithm(EvolutionaryAlgorithm *evolutionaryAlgorithm) {
	cudaCheck(cudaFree(evolutionaryAlgorithm->fitness));
	cudaCheck(cudaFree(evolutionaryAlgorithm->population));
	cudaCheck(cudaFree(evolutionaryAlgorithm->rngState));
	cudaCheck(cudaFree(evolutionaryAlgorithm->temporaryPopulation));
	free(evolutionaryAlgorithm);
}

void evaluatePopulation(LPSolver *lpSolver,
		EvolutionaryAlgorithm *evolutionaryAlgorithm,
		const Parameters * const parameters) {
	printf("DUMMY evaluatePopulation\n");
	solveLP<<<parameters->gridSize, parameters->blockSize>>>(
			evolutionaryAlgorithm->population, evolutionaryAlgorithm->fitness);
}

char *runEvolutionaryAlgorithm(EvolutionaryAlgorithm *evolutionaryAlgorithm,
		LPSolver *lpSolver, Statistics *statistics,
		const Parameters * const parameters) {
	initializeRNG<<<parameters->gridSize, parameters->blockSize>>>(
			evolutionaryAlgorithm->rngState);
	createPopulation<<<parameters->gridSize, parameters->blockSize>>>(
			evolutionaryAlgorithm->population, evolutionaryAlgorithm->rngState);
	evaluatePopulation(lpSolver, evolutionaryAlgorithm, parameters);

	for (uint32_t iteration = 0; iteration < parameters->iterationAmount;
			iteration++) {
		// migration after specified interval
		if ((iteration + 1) % parameters->migrationInterval == 0) {
			migratePopulation<<<parameters->gridSize, parameters->blockSize>>>(
					evolutionaryAlgorithm->population,
					statistics->iterationData, iteration);
		}

		selectPopulation<<<parameters->gridSize, parameters->blockSize>>>(
				evolutionaryAlgorithm->population,
				evolutionaryAlgorithm->temporaryPopulation,
				evolutionaryAlgorithm->fitness,
				evolutionaryAlgorithm->rngState);
		crossoverPopulation<<<parameters->gridSize, parameters->blockSize>>>(
				evolutionaryAlgorithm->population,
				evolutionaryAlgorithm->temporaryPopulation,
				evolutionaryAlgorithm->rngState);
		cudaDeviceSynchronize();
		swapTemporaryPopulation(&evolutionaryAlgorithm->population,
				&evolutionaryAlgorithm->temporaryPopulation);

		mutatePopulation<<<parameters->gridSize, parameters->blockSize>>>(
				evolutionaryAlgorithm->population,
				evolutionaryAlgorithm->rngState);

		cudaDeviceSynchronize();

		evaluatePopulation(lpSolver, evolutionaryAlgorithm, parameters);
		gatherStatistics(statistics, evolutionaryAlgorithm->fitness, iteration,
				parameters);
	}

	char *knockouts = (char *) malloc(18 * sizeof(char));
	strcpy(knockouts, "42,DUMMY,knockout");
	return knockouts;
}

void swapTemporaryPopulation(uint32_t **population,
		uint32_t **temporaryPopulation) {
	uint32_t *temp = *population;
	*population = *temporaryPopulation;
	*temporaryPopulation = temp;
}
