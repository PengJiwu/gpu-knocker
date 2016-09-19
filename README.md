# gpu-knocker

This project implements a tool for metabolic engineering to find knockout targets to optimize production of substances, partially implemented in CUDA.

The project is developed on Linux (CentOS 6) with a K20 (compute capability 3.5). Other plattforms aren't tested.

## Parameters

Parameters can be provided by a file to override defaults. The file should contain key value pairs (case sensitive). Below is an example with all parameters set to their default values:

```
# parameters as `key value` pairs
# case sensitive

# EA parameters
individualSize 1024
islandAmount 1
iterationAmount 1000
migrationInterval 10000
migrationSize 5
mutationRate 0.2
mutationStrength 1
populationSize 100
selectionRate 0.3
tournamentSize 5

# general parameters
isBenchmark false
isVerbose false

# GPU parameters
blockSize 192
gridSize 12
warpSize 32

# LP parameters
biomass 0
product 0
substrate 0
maintenance 0
```

The numbers for biomass, product, substrate and maintenance refer to the numbers of the reactions in the MPS file (one based). These reactions are protected from being knocked out and are used to compute the biomass-product coupled yield (BPCY).

## Build

```
nvcc -o gpu-knocker src/*.c src/*.cu -lglpk -arch=sm_35 --relocatable-device-code=true -O3
```

## Run

```
./gpu-knocker lpProblem.mps parameter.conf
```

## Status

- [x] parameter handling
- [x] evolutionary algorithm in CUDA
- [x] print proposed knockouts
- [x] MPS reader
- [ ] ~~linear programming solver~~ deemed out of scope, for prototype see https://github.com/SethosII/cuda-revised-simplex
- [x] integrate [GLPK](https://www.gnu.org/software/glpk/)
