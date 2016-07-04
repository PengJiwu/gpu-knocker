/**
 * Starts the program.
 */

#include <stdio.h>
#include <stdlib.h>

#include "gpuKnocker.h"

int main(int argc, char *argv[]) {
	char *mps = "lpProblem.mps";
	char *target = "target.csv";
	char *parameter = "parameter.conf";

	char *knockout = knock(mps, target, parameter);

	printf("%s\n", knockout);
	free(knockout);

	return EXIT_SUCCESS;
}
