/**
 * Defines linear programming GPU kernels.
 */

#ifndef LPKERNELS_CUH_
#define LPKERNELS_CUH_

#include <stdint.h>

#include "parameters.cuh"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Calculates fitness.
 *
 * @param population Population.
 * @param island Number of island.
 * @param individual Number of individual.
 * @return Returns fitness value of individual.
 */
__device__ float fitnessFunction(uint32_t *population, uint32_t island,
		uint32_t individual);

/**
 * Solves LP.
 *
 * @param population Population.
 * @param fitness Fitness.
 */
__global__ void solveLP(uint32_t *population, float *fitness);

#ifdef __cplusplus
}
#endif

#endif /* LPKERNELS_CUH_ */
