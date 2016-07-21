/**
 * Implementation of evolutionary algorithm.
 */

#include "evolutionaryAlgorithm.cuh"

#include <stdlib.h>
#include <string.h>

#include "cudaCheck.cuh"
#include "eaKernels.cuh"

EvolutionaryAlgorithm *createEvolutionaryAlgorithm(Parameters *parameters) {
	EvolutionaryAlgorithm *evolutionaryAlgorithm =
			(EvolutionaryAlgorithm *) malloc(sizeof(EvolutionaryAlgorithm));
	cudaCheck(
			cudaMalloc(&evolutionaryAlgorithm->fitness,
					parameters->populationSize * parameters->islandAmount
							* sizeof(float)));
	cudaCheck(
			cudaMalloc(&evolutionaryAlgorithm->population,
					parameters->individualSizeInt * parameters->populationSize
							* parameters->islandAmount * sizeof(uint32_t)));
	cudaCheck(
			cudaMalloc(&evolutionaryAlgorithm->rngState,
					parameters->blockSize * parameters->gridSize
							* sizeof(curandStatePhilox4_32_10)));
	cudaCheck(
			cudaMalloc(&evolutionaryAlgorithm->temporaryPopulation,
					parameters->individualSizeInt * parameters->populationSize
							* parameters->islandAmount * sizeof(uint32_t)));

	return evolutionaryAlgorithm;
}

void deleteEvolutionaryAlgorithm(EvolutionaryAlgorithm *evolutionaryAlgorithm) {
	cudaCheck(cudaFree(evolutionaryAlgorithm->fitness));
	cudaCheck(cudaFree(evolutionaryAlgorithm->population));
	cudaCheck(cudaFree(evolutionaryAlgorithm->rngState));
	cudaCheck(cudaFree(evolutionaryAlgorithm->temporaryPopulation));
	free(evolutionaryAlgorithm);
}

char *runEvolutionaryAlgorithm(EvolutionaryAlgorithm *evolutionaryAlgorithm,
		LPSolver *lpSolver, Statistics *statistics, Parameters *parameters) {
	initializeRNG<<<parameters->gridSize, parameters->blockSize>>>(0,
			evolutionaryAlgorithm->rngState);
	createPopulation<<<parameters->gridSize, parameters->blockSize>>>(
			evolutionaryAlgorithm->population, evolutionaryAlgorithm->rngState);
	evaluatePopulation(evolutionaryAlgorithm->population,
			evolutionaryAlgorithm->fitness, lpSolver, parameters);

	for (uint32_t iteration = 0; iteration < parameters->iterationAmount;
			iteration++) {
		// migration after specified interval
		if ((iteration + 1) % parameters->migrationInterval == 0) {
			migratePopulation<<<parameters->gridSize, parameters->blockSize,
					(parameters->migrationSize + 2) * sizeof(uint32_t)>>>(
					evolutionaryAlgorithm->population,
					evolutionaryAlgorithm->fitness, statistics->iterationData);
		}

		selectPopulation<<<parameters->gridSize, parameters->blockSize>>>(
				evolutionaryAlgorithm->population,
				evolutionaryAlgorithm->temporaryPopulation,
				evolutionaryAlgorithm->fitness,
				evolutionaryAlgorithm->rngState);
		crossoverPopulation<<<parameters->gridSize, parameters->blockSize>>>(
				evolutionaryAlgorithm->temporaryPopulation,
				evolutionaryAlgorithm->rngState);
		mutatePopulation<<<parameters->gridSize, parameters->blockSize>>>(
				evolutionaryAlgorithm->temporaryPopulation,
				evolutionaryAlgorithm->rngState);

		cudaDeviceSynchronize();
		swapTemporaryPopulation(&evolutionaryAlgorithm->population,
				&evolutionaryAlgorithm->temporaryPopulation);

		evaluatePopulation(evolutionaryAlgorithm->population,
				evolutionaryAlgorithm->fitness, lpSolver, parameters);
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
