#!/bin/bash
set -e
# Set these to allow MPI oversubscription because the tests need to run on 3 procs but the test runner may only have 2
export OMPI_MCA_rmaps_base_oversubscribe=true
testflo -v . -n 1 --coverage --coverpkg multipoint
