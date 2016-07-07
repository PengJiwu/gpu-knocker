/**
 * Implementation of linear programming solver.
 */

#include "lpSolver.cuh"

#include <stdlib.h>
#include <stdio.h>

#include "lpKernels.cuh"

LPSolver *createLPSolver(Parameters *parameters) {
	LPSolver *lpSolver = (LPSolver *) malloc(sizeof(LPSolver));
	lpSolver->dummy = 0;

	return lpSolver;
}

void deleteLPSolver(LPSolver *lpSolver) {
	free(lpSolver);
}

void preprocessLPProblem(LPSolver *lpSolver, Parameters *parameters) {
	// TODO implement
}
