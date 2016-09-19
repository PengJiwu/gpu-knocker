/**
 * Defines the main interface.
 */

#ifndef GPUKNOCKER_H_
#define GPUKNOCKER_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Starts the algorithm.
 *
 * @param mps File containing the linear programming problem in MPS format.
 * @param parameter File containing the parameters. Unspecified parameters will be set to default values.
 */
void knock(char *mps, char *parameter);

#ifdef __cplusplus
}
#endif

#endif /* GPUKNOCKER_H_ */
