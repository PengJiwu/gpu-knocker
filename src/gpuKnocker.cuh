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
 * @param target File containing a comma separated list of metabolites to overproduce.
 * @param parameter File containing the parameters. Unspecified parameters will be set to default values.
 * @return Returns knockouts and the achieved target values. Needs to be freed.
 */
char *knock(char *mps, char *target, char *parameter);

#ifdef __cplusplus
}
#endif

#endif /* GPUKNOCKER_H_ */
