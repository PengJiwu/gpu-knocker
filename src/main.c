/**
 * Starts the program.
 */

#include <stdio.h>
#include <stdlib.h>

#include "gpuKnocker.h"

int main(int argc, char *argv[]) {
	char *mps = "lpProblem.mps";
	char *parameters = "parameter.conf";
	char *target = "target.csv";
	char *knockout = knock(mps, parameters, target);

	printf("%s\n", knockout);
	free(knockout);

	return EXIT_SUCCESS;
}
