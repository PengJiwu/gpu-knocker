/**
 * Implementation of evolutionary algorithm GPU kernels.
 */

#include "eaKernels.cuh"

#include <float.h>
#include <stdio.h>

#include "helper.cuh"

__device__ void bitFlipMutation(uint32_t *gene,
		curandStatePhilox4_32_10 *rngState) {
	*gene ^= 0x80000000 >> (curand(rngState) % 32);
}

__global__ void createPopulation(uint32_t *population,
		curandStatePhilox4_32_10 *rngState) {
	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

	curandStatePhilox4_32_10 localState = rngState[id];
	uint32_t populationSize = parametersGPU.individualSizeInt
			* parametersGPU.populationSize * parametersGPU.islandAmount;
	for (uint32_t gene = id; gene < populationSize;
			gene += blockDim.x * gridDim.x) {
		population[gene] = 0xFFFFFFFF;
	}
	rngState[id] = localState;
}

__global__ void crossoverPopulation(uint32_t *population,
		curandStatePhilox4_32_10 *rngState) {
	// TODO cache in shared memory
	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

	curandStatePhilox4_32_10 localState = rngState[id];
	for (uint32_t island = blockIdx.x; island < parametersGPU.islandAmount;
			island += gridDim.x) {
		for (uint32_t individual = (uint32_t) (parametersGPU.populationSize
				* parametersGPU.selectionRate) + threadIdx.x;
				individual < parametersGPU.populationSize; individual +=
						blockDim.x) {
			uniformCrossover(population, island, individual, &localState);
		}
	}
	rngState[id] = localState;
}

__global__ void initializeRNG(uint32_t seed,
		curandStatePhilox4_32_10 *rngState) {
	// separate kernel for performance reasons
	// see http://docs.nvidia.com/cuda/curand/device-api-overview.html#performance-notes
	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, id, 0, &rngState[id]);
}

__global__ void migratePopulation(uint32_t *population, float *fitness,
		float *statistics) {
	extern __shared__ uint32_t sharedMemory[];
	uint32_t *amountFound = &sharedMemory[parametersGPU.migrationSize];
	uint32_t *amountStored = &sharedMemory[parametersGPU.migrationSize + 1];
	for (uint32_t island = blockIdx.x; island < parametersGPU.islandAmount;
			island += gridDim.x) {
		if (threadIdx.x == 0) {
			*amountFound = 0;
			*amountStored = 0xFFFFFFFF;
		}
		__syncthreads();
		uint32_t nextIsland = (island + 1) % parametersGPU.islandAmount;

		float currentIslandAverage = statistics[island * 3 + 2];
		float nextIslandAverage = statistics[nextIsland * 3 + 2];

		for (uint32_t individual = threadIdx.x;
				individual < parametersGPU.populationSize
						&& *amountFound < parametersGPU.migrationSize;
				individual += blockDim.x) {
			if (getFitness(fitness, island, individual)
					> currentIslandAverage) {
				uint32_t oldAmount = atomicAdd(amountFound, 1);
				if (oldAmount < parametersGPU.migrationSize) {
					atomicAdd(amountStored, 1);
					sharedMemory[oldAmount] = individual;
				}
			}
		}
		__syncthreads();
		for (uint32_t individual = threadIdx.x;
				individual < parametersGPU.populationSize
						&& *amountStored < parametersGPU.migrationSize;
				individual += blockDim.x) {
			if (getFitness(fitness, nextIsland, individual) < nextIslandAverage
					&& *amountStored < parametersGPU.migrationSize) {
				uint32_t oldAmount = atomicSub(amountStored, 1);
				if (oldAmount < parametersGPU.migrationSize) {
					copyIndividual(population, island, sharedMemory[oldAmount],
							population, nextIsland, individual);
					setFitness(fitness, nextIsland, individual,
							getFitness(fitness, island,
									sharedMemory[oldAmount]));
				}
			}
		}
	}
}

__global__ void mutatePopulation(uint32_t *population,
		curandStatePhilox4_32_10 *rngState) {
	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

	curandStatePhilox4_32_10 localState = rngState[id];
	for (uint32_t island = blockIdx.x; island < parametersGPU.islandAmount;
			island += gridDim.x) {
		for (uint32_t individual = threadIdx.x;
				individual
						< parametersGPU.populationSize
								* parametersGPU.mutationRate; individual +=
						blockDim.x) {
			uint32_t selectedIndividual = curand(&localState)
					% parametersGPU.populationSize;
			for (uint32_t mutations = 0;
					mutations < parametersGPU.mutationStrength; mutations++) {
				bitFlipMutation(
						getGene(population, island, selectedIndividual,
								curand(&localState)
										% parametersGPU.individualSizeInt),
						&localState);
			}
		}
	}
	rngState[id] = localState;
}

__global__ void selectPopulation(uint32_t *population,
		uint32_t *temporaryPopulation, float *fitness,
		curandStatePhilox4_32_10 *rngState) {
	// TODO cache in shared memory
	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

	curandStatePhilox4_32_10 localState = rngState[id];
	for (uint32_t island = blockIdx.x; island < parametersGPU.islandAmount;
			island += gridDim.x) {
		for (uint32_t individual = threadIdx.x;
				individual
						< (uint32_t) (parametersGPU.populationSize
								* parametersGPU.selectionRate); individual +=
						blockDim.x) {
			tournamentSelection(population, temporaryPopulation, fitness,
					island, individual, &localState);
		}
	}
	rngState[id] = localState;
}

__device__ void tournamentSelection(uint32_t *population,
		uint32_t *temporaryPopulation, float *fitness, uint32_t island,
		uint32_t individual, curandStatePhilox4_32_10 *rngState) {
	uint32_t selectedIndividual;
	float selectedFitness = -FLT_MAX;
	for (uint32_t round = 0; round < parametersGPU.tournamentSize; round++) {
		uint32_t select = curand(rngState) % parametersGPU.populationSize;
		if (getFitness(fitness, island, select) > selectedFitness) {
			selectedIndividual = select;
			selectedFitness = getFitness(fitness, island, select);
		}
	}
	copyIndividual(population, island, selectedIndividual, temporaryPopulation,
			island, individual);
}

__device__ void uniformCrossover(uint32_t *population, uint32_t island,
		uint32_t individual, curandStatePhilox4_32_10 *rngState) {
	uint32_t parent1 = curand(rngState)
			% (uint32_t) (parametersGPU.populationSize
					* parametersGPU.selectionRate);
	uint32_t parent2 = curand(rngState)
			% (uint32_t) (parametersGPU.populationSize
					* parametersGPU.selectionRate);
	for (uint32_t gene = 0; gene < parametersGPU.individualSizeInt; gene++) {
		uint32_t mask = curand(rngState);
		uint32_t *newGene = getGene(population, island, individual, gene);
		*newGene = (mask & *getGene(population, island, parent1, gene))
				| (~mask & *getGene(population, island, parent2, gene));
	}
}
