/**
 * Implementation of statistics.
 */

#include "statistics.cuh"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include "cudaCheck.cuh"
#include "eaKernels.cuh"
#include "helper.cuh"

Statistics *createStatistics(Parameters *parameters) {
	Statistics *statistics = (Statistics *) malloc(sizeof(Statistics));
	statistics->data = (float *) malloc(
			3 * parameters->iterationAmount * parameters->islandAmount
					* sizeof(float));
	for (uint32_t island = 0;
			island < parameters->iterationAmount * parameters->islandAmount;
			island++) {
		statistics->data[island * 3] = -FLT_MAX;
		statistics->data[island * 3 + 1] = FLT_MAX;
		statistics->data[island * 3 + 2] = 0;
	}
	cudaCheck(
			cudaMalloc(&statistics->iterationData,
					3 * parameters->islandAmount * sizeof(float)));
	cudaCheck(
			cudaMemcpy(statistics->iterationData, statistics->data,
					3 * parameters->islandAmount * sizeof(float),
					cudaMemcpyHostToDevice));

	return statistics;
}

void deleteStatistics(Statistics *statistics) {
	free(statistics->data);
	cudaCheck(cudaFree(statistics->iterationData));
	free(statistics);
}

__global__ void gatherKernel(float *fitness, float *statistics) {
	// see https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
	// and http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
	extern __shared__ float blockData[];
	int warpIndex = threadIdx.x / 32;
	int warpLane = threadIdx.x % 32;

	for (uint32_t island = blockIdx.x; island < parametersGPU.islandAmount;
			island += gridDim.x) {
		float localData[3] = { -FLT_MAX, FLT_MAX, 0 };
		for (uint32_t individual = threadIdx.x;
				individual < parametersGPU.populationSize; individual +=
						blockDim.x) {
			if (individual < parametersGPU.populationSize) {
				float fitnessValue = getFitness(fitness, island, individual);
				localData[0] = maximum(localData[0], fitnessValue);
				localData[1] = minimum(localData[1], fitnessValue);
				localData[2] += fitnessValue
						/ (float) parametersGPU.populationSize;
			}
		}
		localData[0] = warpReduceMax(localData[0]);
		localData[1] = warpReduceMin(localData[1]);
		localData[2] = warpReduceSum(localData[2]);
		if (warpLane == 0) {
			blockData[warpIndex] = localData[0];
			blockData[warpIndex + warpSize] = localData[1];
			blockData[warpIndex + 2 * warpSize] = localData[2];
		}

		__syncthreads();

		if (threadIdx.x < blockDim.x / warpSize) {
			localData[0] = blockData[warpLane];
			localData[1] = blockData[warpLane + warpSize];
			localData[2] = blockData[warpLane + 2 * warpSize];
		} else {
			localData[0] = -FLT_MAX;
			localData[1] = FLT_MAX;
			localData[2] = 0;
		}
		if (warpIndex == 0) {
			localData[0] = warpReduceMax(localData[0]);
			localData[1] = warpReduceMin(localData[1]);
			localData[2] = warpReduceSum(localData[2]);
		}
		if (threadIdx.x == 0) {
			statistics[island * 3] = localData[0];
			statistics[island * 3 + 1] = localData[1];
			statistics[island * 3 + 2] = localData[2];
		}
	}
}

void gatherStatistics(Statistics *statistics, float *fitness,
		uint32_t iteration, Parameters *parameters) {
	cudaCheck(
			cudaMemcpy(statistics->iterationData,
					&statistics->data[3 * parameters->islandAmount * iteration],
					3 * parameters->islandAmount * sizeof(float),
					cudaMemcpyHostToDevice));

	gatherKernel<<<parameters->gridSize, parameters->blockSize,
			3 * parameters->blockSize * sizeof(float)>>>(fitness,
			statistics->iterationData);

	cudaCheck(
			cudaMemcpy(
					&statistics->data[3 * parameters->islandAmount * iteration],
					statistics->iterationData,
					3 * parameters->islandAmount * sizeof(float),
					cudaMemcpyDeviceToHost));
}

void printStatistics(Statistics *statistics, Parameters *parameters) {
	for (uint32_t iteration = 0; iteration < parameters->iterationAmount;
			iteration++) {
		printf("iteration %u: (MAX,MIN,AVG)\n", iteration);
		for (uint32_t island = 0; island < parameters->islandAmount; island++) {
			printf("island[%u]=%f|%f|%f ", island,
					statistics->data[(iteration * parameters->islandAmount
							+ island) * 3],
					statistics->data[(iteration * parameters->islandAmount
							+ island) * 3 + 1],
					statistics->data[(iteration * parameters->islandAmount
							+ island) * 3 + 2]);
		}
		printf("\n");
	}
}
