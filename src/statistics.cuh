/**
 * Defines statistics.
 */

#ifndef STATISTICS_CUH_
#define STATISTICS_CUH_

#include <stdint.h>

#include "parameters.cuh"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Holds statistic data.
 */
typedef struct Statistics {
	/**
	 * Pointer statistic data. Data is stored as max,min,avg for each island.
	 */
	float *data;

	/**
	 * Pointer statistic data for the current iteration. Data is stored as max,min,avg for each island.
	 */
	float *iterationData;
} Statistics;

/**
 * Creates Statistics with default values.
 *
 * @param parameters Parameters.
 * @return Initialized Statistics.
 */
Statistics *createStatistics(const Parameters * const parameters);

/**
 * Clears memory for Statistics.
 *
 * @param statistics Statistics to be deleted.
 */
void deleteStatistics(Statistics *statistics);

/**
 * Gathers statistics for the iteration.
 *
 * @param statistics Statistics stored here.
 * @param fitness Fitness values to build statistics from.
 * @param iteration Current iteration.
 * @param parameters Parameters.
 */
void gatherStatistics(Statistics *statistics, const float * const fitness,
		const uint32_t iteration, const Parameters * const parameters);

/**
 * Prints statistics to console.
 *
 * @param statistics Statistics to print.
 * @param parameters Parameters.
 */
void printStatistics(const Statistics * const statistics,
		const Parameters * const parameters);

#ifdef __cplusplus
}
#endif

#endif /* STATISTICS_CUH_ */
