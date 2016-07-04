/**
 * Defines general parameter methods.
 */

#ifndef PARAMETERS_CUH_
#define PARAMETERS_CUH_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

/**
 * Holds parameters.
 */
typedef struct Parameters {
	// evolutionary algorithm parameters

	/**
	 * Size of an individual.
	 */
	uint32_t individualSize;

	/**
	 * Size of an individual in int's.
	 */
	uint32_t individualSizeInt;

	/**
	 * Number of islands.
	 */
	uint32_t islandAmount;

	/**
	 * Maximum number of iterations.
	 */
	uint32_t iterationAmount;

	/**
	 * Number of intervals after which migration happens.
	 */
	uint32_t migrationInterval;

	/**
	 * Number of individuals that are migrated.
	 */
	uint32_t migrationSize;

	/**
	 * Mutation Rate.
	 */
	float mutationRate;

	/**
	 * Size of a population.
	 */
	uint32_t populationSize;

	/**
	 * Selection rate.
	 */
	float selectionRate;

	/**
	 * Size of tournament.
	 */
	uint32_t tournamentSize;

	// general parameters

	/**
	 * Run in benchmark mode or not. If true only print performance information.
	 */
	bool isBenchmark;

	/**
	 * Print additional information or not.
	 */
	bool isVerbose;

	// GPU parameters

	/**
	 * Number of threads per block.
	 */
	uint32_t blockSize;

	/**
	 * Number of blocks per grid.
	 */
	uint32_t gridSize;

	/**
	 * Number of threads per warp.
	 */
	uint32_t warpSize;

	// LP parameters

	/**
	 * File to read LP problem from. Maximum length is 99 characters.
	 */
	char lpInputFile[100];

	/**
	 * File to read target from. Maximum length is 99 characters.
	 */
	char target[100];
} Parameters;

/**
 * Pointer to parameters on GPU.
 */
extern __constant__ Parameters parametersGPU;

/**
 * Creates Parameters with default values.
 *
 * @return Initialized Parameters.
 */
Parameters *createParameters();

/**
 * Clears memory for Parameters.
 *
 * @param parameters Parameters to be deleted.
 */
void deleteParameters(Parameters *parameters);

/**
 * Reads parameters from file.
 *
 * @param parameterFile File to read parameters from.
 * @param mps MPS file to read LP problem from.
 * @param target CSV file to read target from.
 * @param parameter Parameters stored here.
 */
void parseParameters(const char * const parameterFile, const char * const mps,
		const char * const target, Parameters *parameters);

/**
 * Prints parameters to console.
 *
 * @param parameters Parameters to print.
 */
void printParameters(const Parameters * const parameters);

#ifdef __cplusplus
}
#endif

#endif /* PARAMETERS_CUH_ */
