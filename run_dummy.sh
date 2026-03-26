#!/bin/bash

# run this script to run the pipeline locally

# path to config file
CONFIG=config/config_dummy.yml

# number of available CPUs on the local machine
ncpu=35

# unlock the workflow in case it was locked by a previous run
snakemake --unlock --configfile $CONFIG

# run the workflow
snakemake --cores $ncpu --configfile $CONFIG