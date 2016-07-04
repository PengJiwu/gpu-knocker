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

char *knock(char *mps, char *target, const char * const parameter) {
	cudaEvent_t custart, custop;
	cudaCheck(cudaEventCreate(&custart));
	cudaCheck(cudaEventCreate(&custop));
	cudaCheck(cudaEventRecord(custart, 0));

	Parameters *parameters = createParameters();

	parseParameters(parameter, mps, target, parameters);
	printParameters(parameters);

	EvolutionaryAlgorithm *evolutionaryAlgorithm = createEvolutionaryAlgorithm(
			parameters);
	LPSolver *lpSolver = createLPSolver(parameters);
	Statistics *statistics = createStatistics(parameters);

	preprocessLPProblem(lpSolver, parameters);
	char *knockouts = runEvolutionaryAlgorithm(evolutionaryAlgorithm, lpSolver,
			statistics, parameters);

	deleteEvolutionaryAlgorithm(evolutionaryAlgorithm);
	deleteLPSolver(lpSolver);
	deleteStatistics(statistics);

	deleteParameters(parameters);

	cudaCheck(cudaEventRecord(custop, 0));
	cudaCheck(cudaEventSynchronize(custop));
	float elapsedTime;
	cudaCheck(cudaEventElapsedTime(&elapsedTime, custart, custop));
	printf("This took %3.1f ms.\n", elapsedTime);
	cudaCheck(cudaEventDestroy(custart));
	cudaCheck(cudaEventDestroy(custop));

	return knockouts;
}
