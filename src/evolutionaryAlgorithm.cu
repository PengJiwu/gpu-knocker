/**
 * Implementation of evolutionary algorithm.
 */

#include "evolutionaryAlgorithm.cuh"

#include <stdlib.h>
#include <string.h>

#include "cudaCheck.cuh"
#include "eaKernels.cuh"
#include "helperHost.cuh"

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

void printBestKnockout(LPSolver* lpSolver, Statistics* statistics,
		Parameters* parameters) {
	float islandMax = 0;
	for (uint32_t island = 0; island < parameters->islandAmount; island++) {
		float currentIslandMax = statistics->data[((parameters->iterationAmount
				- 1) * parameters->islandAmount + island) * 3];
		if (islandMax < currentIslandMax) {
			islandMax = currentIslandMax;
		}
	}
	uint32_t bestIndividual;
	uint32_t islandBestIndividual;
	uint32_t leastKnockouts = 4294967295;
	for (uint32_t island = 0; island < parameters->islandAmount; island++) {
		for (uint32_t individual = 0; individual < parameters->populationSize;
				individual++) {
			float fitness = getFitnessHost(lpSolver->copyFitness, island,
					individual, parameters);
			if (fitness == islandMax) {
				uint32_t knockouts = getNumberKnockoutsHost(
						lpSolver->copyPopulation, island, individual,
						parameters);
				if (knockouts < leastKnockouts) {
					bestIndividual = individual;
					islandBestIndividual = island;
					leastKnockouts = knockouts;
				}
			}
		}
	}
	printf("Best knockout with least knockouts: %f", islandMax);
	printKnockout(bestIndividual, islandBestIndividual, lpSolver, parameters);
}

void printBestKnockouts(LPSolver* lpSolver, Statistics* statistics,
		Parameters* parameters) {
	for (uint32_t island = 0; island < parameters->islandAmount; island++) {
		float islandMax = statistics->data[((parameters->iterationAmount - 1)
				* parameters->islandAmount + island) * 3];
		for (uint32_t individual = 0; individual < parameters->populationSize;
				individual++) {
			float fitness = getFitnessHost(lpSolver->copyFitness, island,
					individual, parameters);
			if (fitness == islandMax) {
				printf("%f", fitness);
				printKnockout(individual, island, lpSolver, parameters);
			}
		}
	}
}

void printKnockout(uint32_t individual, uint32_t island, LPSolver* lpSolver,
		Parameters* parameters) {
	for (uint32_t position = 0; position < parameters->individualSize;
			position++) {
		uint32_t geneNumber = position / 32;
		uint32_t gene = *getGeneHost(lpSolver->copyPopulation, island,
				individual, geneNumber, parameters);
		uint32_t bit = (gene >> ((position + 31) % 32)) & 0x00000001;
		if (bit == 0 && (position + 1 != parameters->biomass)
				&& (position + 1 != parameters->product)
				&& (position + 1 != parameters->substrate)
				&& (position + 1 != parameters->maintenance)) {
			printf(",%d", position + 1);
		}
	}
	printf("\n");
}

void runEvolutionaryAlgorithm(EvolutionaryAlgorithm *evolutionaryAlgorithm,
		LPSolver *lpSolver, Statistics *statistics, Parameters *parameters) {
	initializeRNG<<<parameters->gridSize, parameters->blockSize>>>(0,
			evolutionaryAlgorithm->rngState);
	createPopulation<<<parameters->gridSize, parameters->blockSize>>>(
			evolutionaryAlgorithm->population, evolutionaryAlgorithm->rngState);
	mutatePopulation<<<parameters->gridSize, parameters->blockSize>>>(
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
}

void swapTemporaryPopulation(uint32_t **population,
		uint32_t **temporaryPopulation) {
	uint32_t *temp = *population;
	*population = *temporaryPopulation;
	*temporaryPopulation = temp;
}
