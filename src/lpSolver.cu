/**
 * Implementation of linear programming solver.
 */

#include "lpSolver.cuh"

#include <stdlib.h>
#include <stdio.h>

#include "cudaCheck.cuh"
#include "helperHost.cuh"
#include "lpKernels.cuh"

LPSolver *createLPSolver(Parameters *parameters) {
	LPSolver *lpSolver = (LPSolver *) malloc(sizeof(LPSolver));
	// disable all output from GLPK
	glp_term_out(GLP_OFF);
	glp_init_smcp(&lpSolver->lpParameter);
	lpSolver->lpProblem = glp_create_prob();
	lpSolver->lpProblemWork = glp_create_prob();

	preprocessLPSolver(lpSolver, parameters);

	lpSolver->copyFitness = (float *) malloc(
			parameters->populationSize * parameters->islandAmount
					* sizeof(float));
	lpSolver->copyPopulation = (uint32_t *) malloc(
			parameters->individualSizeInt * parameters->populationSize
					* parameters->islandAmount * sizeof(uint32_t));

	return lpSolver;
}

void deleteLPSolver(LPSolver *lpSolver) {
	free(lpSolver->copyFitness);
	free(lpSolver->copyPopulation);
	glp_delete_prob(lpSolver->lpProblem);
	glp_delete_prob(lpSolver->lpProblemWork);
	free(lpSolver);
}

void evaluatePopulation(uint32_t *population, float *fitness,
		LPSolver *lpSolver, Parameters *parameters) {
	// TODO replace
	//solveLP<<<parameters->gridSize, parameters->blockSize>>>(population, fitness);
	cudaCheck(
			cudaMemcpy(lpSolver->copyPopulation, population,
					parameters->individualSizeInt * parameters->populationSize
							* parameters->islandAmount * sizeof(uint32_t),
					cudaMemcpyDeviceToHost));
	for (uint32_t island = 0; island < parameters->islandAmount; island++) {
		for (uint32_t individual = 0; individual < parameters->populationSize;
				individual++) {
			glp_copy_prob(lpSolver->lpProblemWork, lpSolver->lpProblem, GLP_ON);
			for (uint32_t i = 0; i < parameters->individualSize; i++) {
				uint32_t geneNumber = i / 32;
				uint32_t gene = *getGeneHost(lpSolver->copyPopulation, island,
						individual, geneNumber, parameters);
				uint32_t bit = (gene >> ((i + 31) % 32)) & 0x00000001;
				if (bit == 0) {
					glp_set_col_bnds(lpSolver->lpProblemWork, i + 1, GLP_FX,
							0.0, 0.0);
				}
			}
			glp_simplex(lpSolver->lpProblemWork, &lpSolver->lpParameter);
			setFitnessHost(lpSolver->copyFitness, island, individual,
					glp_get_obj_val(lpSolver->lpProblemWork), parameters);
		}
	}
	cudaCheck(
			cudaMemcpy(fitness, lpSolver->copyFitness,
					parameters->populationSize * parameters->islandAmount
							* sizeof(float), cudaMemcpyHostToDevice));
}

void preprocessLPSolver(LPSolver *lpSolver, Parameters *parameters) {
	glp_read_mps(lpSolver->lpProblem, GLP_MPS_FILE, NULL,
			parameters->lpInputFile);
	glp_set_obj_dir(lpSolver->lpProblem, GLP_MAX);
	parameters->individualSize = glp_get_num_cols(lpSolver->lpProblem);
	parameters->individualSizeInt = (parameters->individualSize + 31) / 32;
	cudaCheck(
			cudaMemcpyToSymbol(parametersGPU, parameters, sizeof(Parameters), 0,
					cudaMemcpyHostToDevice));
	// solve for the first time with presolver on to speed up calculation
	lpSolver->lpParameter.presolve = GLP_ON;
	glp_simplex(lpSolver->lpProblem, &lpSolver->lpParameter);
	lpSolver->lpParameter.presolve = GLP_OFF;
}
