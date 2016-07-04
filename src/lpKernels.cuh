/**
 * Defines linear programmin GPU kernels.
 */

#ifndef LPKERNELS_CUH_
#define LPKERNELS_CUH_

#include <stdint.h>

#include "parameters.cuh"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Solves LP.
 *
 * @param population Population.
 * @param fitness Fitness.
 */
__global__ void solveLP(const uint32_t * const population, float *fitness);

#ifdef __cplusplus
}
#endif

#endif /* LPKERNELS_CUH_ */
