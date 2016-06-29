/**
 * Defines the main interface.
 */

#ifndef GPUKNOCKER_H_
#define GPUKNOCKER_H_

/**
 * Starts the algorithm.
 *
 * @param mps File containing the linear programming problem in MPS format.
 * @param parameters File containing the parameters. Unspecified parameters will be set to default values.
 * @param target File containing a comma separated list of metabolites to overproduce.
 * @return Returns knockouts and the achieved target values. Needs to be freed.
 */
char *knock(const char const *mps, const char const *parameters,
		const char const *target);

#endif /* GPUKNOCKER_H_ */
