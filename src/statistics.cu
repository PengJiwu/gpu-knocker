/**
 * Implementation of statistics.
 */

#include "statistics.cuh"

#include <stdio.h>
#include <stdlib.h>

#include "cudaCheck.cuh"

Statistics *createStatistics(const Parameters * const parameters) {
	Statistics *statistics = (Statistics *) malloc(sizeof(Statistics));
	statistics->data = (float *) malloc(
			3 * parameters->iterationAmount * parameters->islandAmount
					* sizeof(float));
	cudaCheck(
			cudaMalloc(&statistics->iterationData,
					3 * parameters->islandAmount * sizeof(float)));

	return statistics;
}

void deleteStatistics(Statistics *statistics) {
	free(statistics->data);
	cudaCheck(cudaFree(statistics->iterationData));
	free(statistics);
}

void gatherStatistics(Statistics *statistics, const float * const fitness,
		const uint32_t iteration, const Parameters * const parameters) {
	printf("DUMMY gatherStatistics\n");
}

void printStatistics(const Statistics * const statistics,
		const Parameters * const parameters) {
	printf("DUMMY printStatistics\n");
}
