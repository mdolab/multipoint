#!/bin/bash
set -e
# Set these to allow MPI oversubscription because the tests need to run on 3 procs but the test runner may have fewer
export OMPI_MCA_rmaps_base_oversubscribe=1 # This works for OpenMPI <= 4
export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe # This works from OpenMPI >= 5
testflo -v . -n 1 --coverage --coverpkg multipoint
