#!/bin/bash

# ===============================================================
# OVERVIEW
# ===============================================================
#
# Source this file in order to set the proper ENV variables for
# this module. 
# 
# ===============================================================
# USAGE
# ===============================================================
# 
# $ source ./setenv.sh
# 
# ===============================================================

echo "Setting ENV variables for module: DL_TRAIN"

# --- Set ENV for current module
export DL_TRAIN_ROOT=$PWD
export PYTHONPATH=$PYTHONPATH:$PWD
