/**
 * Defines CUDA runtime error checking
 */

#ifndef CUDACHECK_CUH_
#define CUDACHECK_CUH_

#include <stdio.h>
#include <stdlib.h>

//#define NDEBUG // include to remove asserts and cudaCheck
#define cudaCheck(call) __cudaCheck(call, __FILE__, __LINE__)

inline void __cudaCheck(cudaError err, const char* file, int line) {
#ifndef NDEBUG
	if (err != cudaSuccess) {
		fprintf(stderr, "%s(%d): CUDA error: %s\n", file, line,
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#endif
}

#endif /* CUDACHECK_CUH_ */
