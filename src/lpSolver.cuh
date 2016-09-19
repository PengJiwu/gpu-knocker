/**
 * Defines linear programming solver.
 */

#ifndef LPSOLVER_CUH_
#define LPSOLVER_CUH_

#include "parameters.cuh"

#include <glpk.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Holds all parameters necessary for the linear programming solver.
 */
typedef struct LPSolver {
	/**
	 * Copy of fitness in CPU memory.
	 */
	float *copyFitness;

	/**
	 * Copy of population in CPU memory.
	 */
	uint32_t *copyPopulation;

	/**
	 * Parameter for GLPK simplex solver.
	 */
	glp_smcp lpParameter;

	/**
	 * GLPK original problem data.
	 */
	glp_prob *lpProblem;

	/**
	 * GLPK problem data for modification.
	 */
	glp_prob *lpProblemWork;
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
 * Evaluates all individuals.
 *
 * @param population Population.
 * @param fitness Fitness.
 * @param lpSolver LPSolver.
 * @param parameters Parameters.
 */
void evaluatePopulation(uint32_t *population, float *fitness,
		LPSolver *lpSolver, Parameters *parameters);

/**
 * Preprocesses the LPSolver. Also copies parameters to GPU.
 *
 * @param lpSolver LPSolver.
 * @param parameters Parameters.
 */
void preprocessLPSolver(LPSolver *lpSolver, Parameters *parameters);

#ifdef __cplusplus
}
#endif

#endif /* LPSOLVER_CUH_ */
