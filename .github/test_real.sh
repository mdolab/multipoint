#!/bin/bash
set -e
testflo -v . -n 1 --coverage --coverpkg multipoint
