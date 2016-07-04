/**
 * Defines linear programming solver.
 */

#ifndef LPSOLVER_CUH_
#define LPSOLVER_CUH_

#include "parameters.cuh"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Holds all parameters necessary for the linear programming solver.
 */
typedef struct LPSolver {
	int dummy;
} LPSolver;

/**
 * Create LPSolver with default values.
 *
 * @param parameters Parameters.
 * @return Initialized LPSolver.
 */
LPSolver *createLPSolver(Parameters *parameters);

/**
 * Clears memory for LPSolver.
 *
 * @param lpSolver LPSolver to be deleted.
 */
void deleteLPSolver(LPSolver *lpSolver);

/**
 * Clears memory for LPSolver.
 *
 * @param lpSolver LPSolver to be deleted.
 */
void deleteLPSolver(LPSolver *lpSolver);

/**
 * Reads and preprocesses the linear programming problem.
 *
 * @param lpSolver LPSolver.
 * @param parameters Parameters.
 */
void preprocessLPProblem(LPSolver *lpSolver,
		const Parameters * const parameters);

#ifdef __cplusplus
}
#endif

#endif /* LPSOLVER_CUH_ */
