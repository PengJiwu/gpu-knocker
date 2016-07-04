/**
 * Starts the actual algorithm.
 */

#include "gpuKnocker.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parameters.cuh"

char *knock(char *mps, char *target, const char * const parameter) {
	Parameters *parameters = createParameters();

	parseParameters(parameter, mps, target, parameters);
	printParameter(parameters->cpu);

	// TODO DUMMY
	char *knockout = malloc(3 * sizeof(char));
	strcpy(knockout, "OK");

	deleteParameters(parameters);

	return knockout;
}
