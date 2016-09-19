/**
 * Starts the program.
 */

#include <stdio.h>
#include <stdlib.h>

#include "gpuKnocker.cuh"

int main(int argc, char *argv[]) {
	if (argc == 3) {
		char *mps = argv[1];
		char *parameter = argv[2];

		knock(mps, parameter);
	} else {
		printf(
				"Please provide a MPS and config file like \"gpu-knocker lpProblem.mps parameter.conf\"\n");
	}

	return EXIT_SUCCESS;
}
