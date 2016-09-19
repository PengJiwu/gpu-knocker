/**
 * Starts the actual algorithm.
 */

#include "gpuKnocker.cuh"

#include <stdio.h>
#include <stdlib.h>

#include "cudaCheck.cuh"
#include "evolutionaryAlgorithm.cuh"
#include "lpSolver.cuh"
#include "parameters.cuh"
#include "statistics.cuh"

void knock(char *mps, char *parameter) {
	Parameters *parameters = createParameters();

	parseParameters(parameter, mps, parameters);

	cudaEvent_t custart, custop;
	cudaCheck(cudaEventCreate(&custart));
	cudaCheck(cudaEventCreate(&custop));
	cudaCheck(cudaEventRecord(custart, 0));

	LPSolver *lpSolver = createLPSolver(parameters);
	if (parameters->isVerbose) {
		printParameters(parameters);
	}

	EvolutionaryAlgorithm *evolutionaryAlgorithm = createEvolutionaryAlgorithm(
			parameters);
	Statistics *statistics = createStatistics(parameters);

	runEvolutionaryAlgorithm(evolutionaryAlgorithm, lpSolver, statistics,
			parameters);
	if (parameters->isVerbose) {
		printStatisticsFull(statistics, parameters);
		printBestKnockouts(lpSolver, statistics, parameters);
	} else {
		printStatisticsAggregated(statistics, parameters);
		printBestKnockout(lpSolver, statistics, parameters);
	}

	deleteEvolutionaryAlgorithm(evolutionaryAlgorithm);
	deleteLPSolver(lpSolver);
	deleteStatistics(statistics);

	cudaCheck(cudaEventRecord(custop, 0));
	cudaCheck(cudaEventSynchronize(custop));
	float elapsedTime;
	cudaCheck(cudaEventElapsedTime(&elapsedTime, custart, custop));
	if (parameters->isBenchmark) {
		printf("%3.1f\n", elapsedTime);
	} else {
		printf("This took %3.1f ms.\n", elapsedTime);
	}
	cudaCheck(cudaEventDestroy(custart));
	cudaCheck(cudaEventDestroy(custop));

	deleteParameters(parameters);

	cudaDeviceReset();
}
