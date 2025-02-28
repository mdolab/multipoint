#!/bin/bash
set -e
# Set these to allow MPI oversubscription because the tests need to run on 3 procs but the test runner may only have 2
export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe
testflo -v . -n 1 --coverage --coverpkg multipoint
