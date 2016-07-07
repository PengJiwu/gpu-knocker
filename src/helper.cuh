/**
 * Defines helper functions.
 */

#ifndef HELPER_CUH_
#define HELPER_CUH_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Returns pointer to specified gene.
 *
 * @param fromPopulation Population to copy from.
 * @param fromIsland Island to copy from.
 * @param fromIndividual Individual to copy from.
 * @param toPopulation Population to copy to.
 * @param toIsland Island to copy to.
 * @param toIndividual Individual to copy to.
 */
__device__ void copyIndividual(uint32_t *fromPopulation, uint32_t fromIsland,
		uint32_t fromIndividual, uint32_t *toPopulation, uint32_t toIsland,
		uint32_t toIndividual);

/**
 * Returns fitness value.
 *
 * @param fitness Fitness.
 * @param island Island.
 * @param individual Individual.
 * @return Fitness value.
 */
__device__ float getFitness(float *fitness, uint32_t island,
		uint32_t individual);

/**
 * Returns pointer to specified gene.
 *
 * @param population Population.
 * @param island Number of island.
 * @param individual Number of individual.
 * @param gene Number of gene.
 * @return Pointer to specified gene.
 */
__device__ uint32_t *getGene(uint32_t *population, uint32_t island,
		uint32_t individual, uint32_t gene);

/**
 * Returns maximum of both values.
 *
 * @param value1 Value1.
 * @param value2 Value2.
 * @return Maximum.
 */
__device__ float maximum(float value1, float value2);

/**
 * Returns minimum of both values.
 *
 * @param value1 Value1.
 * @param value2 Value2.
 * @return Minimum.
 */
__device__ float minimum(float value1, float value2);

/**
 * Locks by setting mutex to 1.
 *
 * @param mutex Mutex.
 */
__device__ void lock(uint32_t *mutex);

/**
 * Sets desired fitness value
 *
 * @param fitness Fitness.
 * @param island Island.
 * @param individual Individual.
 * @param value Value.
 */
__device__ void setFitness(float *fitness, uint32_t island, uint32_t individual,
		float value);

/**
 * Unlocks by setting mutex to 0.
 *
 * @param mutex Mutex.
 */
__device__ void unlock(uint32_t *mutex);

/**
 * Returns maximum of values in a warp.
 *
 * @param value Value to compare to.
 * @return Maximum of warp.
 */
__device__ float warpReduceMax(float value);

/**
 * Returns minimum of values in a warp.
 *
 * @param value Value to compare to.
 * @return Mimium of warp.
 */
__device__ float warpReduceMin(float value);

/**
 * Returns sum of values in a warp.
 *
 * @param value Value to sum.
 * @return Sum of warp.
 */
__device__ float warpReduceSum(float value);

#ifdef __cplusplus
}
#endif

#endif /* GENERALKERNELS_CUH_ */
