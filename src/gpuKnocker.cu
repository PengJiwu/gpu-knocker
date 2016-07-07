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

char *knock(char *mps, char *target, char *parameter) {
	Parameters *parameters = createParameters();

	parseParameters(parameter, mps, target, parameters);
	if (parameters->isVerbose) {
		printParameters(parameters);
	}

	cudaEvent_t custart, custop;
	cudaCheck(cudaEventCreate(&custart));
	cudaCheck(cudaEventCreate(&custop));
	cudaCheck(cudaEventRecord(custart, 0));

	EvolutionaryAlgorithm *evolutionaryAlgorithm = createEvolutionaryAlgorithm(
			parameters);
	LPSolver *lpSolver = createLPSolver(parameters);
	Statistics *statistics = createStatistics(parameters);

	preprocessLPProblem(lpSolver, parameters);
	char *knockouts = runEvolutionaryAlgorithm(evolutionaryAlgorithm, lpSolver,
			statistics, parameters);
	if (parameters->isVerbose) {
		printStatistics(statistics, parameters);
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

	return knockouts;
}
