/**
 * Implementation of parameters.
 */

#include "parameters.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cudaCheck.cuh"

__constant__ Parameters parametersGPU;

Parameters *createParameters() {
	Parameters *parameters = (Parameters *) malloc(sizeof(Parameters));

	// evolutionary algorithm parameters
	parameters->individualSize = 1024;
	parameters->individualSizeInt = (parameters->individualSize + 31) / 32; // 32 bits per unsigned integer, round up
	parameters->islandAmount = 1;
	parameters->iterationAmount = 1000;
	parameters->migrationInterval = 10000;
	parameters->migrationSize = 5;
	parameters->mutationRate = 0.2f;
	parameters->mutationStrength = 1;
	parameters->populationSize = 100;
	parameters->selectionRate = 0.3f;
	parameters->tournamentSize = 5;
	// general parameters
	parameters->isBenchmark = false;
	parameters->isVerbose = false;
	// GPU parameters
	parameters->blockSize = 192;
	parameters->gridSize = 12;
	parameters->warpSize = 32;
	// linear programming parameters
	strcpy(parameters->lpInputFile, "lpProblem.mps");
	strcpy(parameters->target, "target.csv");

	cudaCheck(
			cudaMemcpyToSymbol(parametersGPU, &parameters, sizeof(Parameters),
					0, cudaMemcpyHostToDevice));

	return parameters;
}

void deleteParameters(Parameters *parameters) {
	free(parameters);
}

void parseParameters(char *parameter, char *mps, char *target,
		Parameters *parameters) {
	FILE *parameterFile;

	parameterFile = fopen(parameter, "r");
	if (parameterFile == NULL) {
		fprintf(stderr, "Couldn't open file %s\n", parameter);
		exit(EXIT_FAILURE);
	}

	char *line = NULL;
	char *key = (char *) malloc(20);
	char *value = (char *) malloc(10);
	size_t len = 0;
	ssize_t read;
	while ((read = getline(&line, &len, parameterFile)) != -1) {
		if (line[0] == '#' || read == 1) {
			// ignore lines starting with # and empty lines
			continue;
		} else {
			sscanf(line, "%s %s", key, value);
			// evolutionary algorithm parameters
			if (strcmp(key, "individualSize") == 0) {
				parameters->individualSize = atoi(value);
				parameters->individualSizeInt = (parameters->individualSize + 31) / 32;
			} else if (strcmp(key, "islandAmount") == 0) {
				parameters->islandAmount = atoi(value);
			} else if (strcmp(key, "iterationAmount") == 0) {
				parameters->iterationAmount = atoi(value);
			} else if (strcmp(key, "migrationInterval") == 0) {
				parameters->migrationInterval = atoi(value);
			} else if (strcmp(key, "migrationSize") == 0) {
				parameters->migrationSize = atoi(value);
			} else if (strcmp(key, "mutationRate") == 0) {
				parameters->mutationRate = atof(value);
			} else if (strcmp(key, "mutationStrength") == 0) {
				parameters->mutationStrength = atof(value);
			} else if (strcmp(key, "populationSize") == 0) {
				parameters->populationSize = atoi(value);
			} else if (strcmp(key, "selectionRate") == 0) {
				parameters->selectionRate = atof(value);
			} else if (strcmp(key, "tournamentSize") == 0) {
				parameters->tournamentSize = atoi(value);
			}
			// general parameters
			else if (strcmp(key, "isBenchmark") == 0) {
				parameters->isBenchmark = strcmp(value, "false") ? true : false;
			} else if (strcmp(key, "isVerbose") == 0) {
				parameters->isVerbose = strcmp(value, "false") ? true : false;
			}
			// GPU parameters
			else if (strcmp(key, "blockSize") == 0) {
				parameters->blockSize = atoi(value);
			} else if (strcmp(key, "gridSize") == 0) {
				parameters->gridSize = atoi(value);
			} else if (strcmp(key, "warpSize") == 0) {
				parameters->warpSize = atoi(value);
			}
			// default
			else {
				printf("Unknown key \"%s\" with value \"%s\"was ignored.", key,
						value);
			}
		}
	}

	fclose(parameterFile);
	if (line) {
		free(line);
	}
	free(key);
	free(value);

	strcpy(parameters->lpInputFile, mps);
	strcpy(parameters->target, target);
}

void printParameters(Parameters *parameters) {
	printf(
			"EAParameters: individualSize=%u individualSizeInt=%u islandAmount=%u iterationAmount=%u migrationInterval=%u migrationSize=%u mutationRate=%f mutationStrength=%d populationSize=%u selectionRate=%f tournamentSize=%u\n",
			parameters->individualSize, parameters->individualSizeInt,
			parameters->islandAmount, parameters->iterationAmount,
			parameters->migrationInterval, parameters->migrationSize,
			parameters->mutationRate, parameters->mutationStrength,
			parameters->populationSize, parameters->selectionRate,
			parameters->tournamentSize);
	printf("GeneralParameters: isBenchmark=%d isVerbose=%d\n",
			parameters->isBenchmark, parameters->isVerbose);
	printf("GPUParamerters: blockSize=%u gridSize=%u warpSize=%u\n",
			parameters->blockSize, parameters->gridSize, parameters->warpSize);
	printf("LPParamerters: lpInputFile=%s target=%s\n", parameters->lpInputFile,
			parameters->target);
}
