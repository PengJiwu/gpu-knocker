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
	 * Number of islands. Should be an integer multiple of gridSize.
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
	 * Number of runs of mutation operator.
	 */
	uint32_t mutationStrength;

	/**
	 * Size of a population. Should be an integer multiple of blockSize.
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
	 * Number of threads per block. Should be an integer multiple of cores per multiprocessor.
	 */
	uint32_t blockSize;

	/**
	 * Number of blocks per grid. Should be an integer multiple of multiprocessor count.
	 */
	uint32_t gridSize;

	/**
	 * Number of threads per warp. This is 32 on current CUDA architectures.
	 */
	uint32_t warpSize;

	// LP parameters

	/**
	 * File to read LP problem from. Maximum length is 99 characters.
	 */
	char lpInputFile[100];

	/**
	 * Index (one-based) of biomass reaction. Used for calculating the fitness value via BPCY. Prevented from being knocked out.
	 */
	uint32_t biomass;

	/**
	 * Index (one-based) of biomass reaction. Used for calculating the fitness value via BPCY. Prevented from being knocked out.
	 */
	uint32_t product;

	/**
	 * Index (one-based) of substrate reaction. Used for calculating the fitness value via BPCY. Prevented from being knocked out.
	 */
	uint32_t substrate;

	/**
	 * Index (one-based) of maintenance reaction. Prevented from being knocked out.
	 */
	uint32_t maintenance;
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
 * Reads parameters from file. Doesn't copy the parameters to the GPU.
 *
 * @param parameterFile File to read parameters from.
 * @param mps MPS file to read LP problem from.
 * @param parameters Parameters stored here.
 */
void parseParameters(char *parameterFile, char *mps, Parameters *parameters);

/**
 * Prints parameters to console.
 *
 * @param parameters Parameters to print.
 */
void printParameters(Parameters *parameters);

#ifdef __cplusplus
}
#endif

#endif /* PARAMETERS_CUH_ */
