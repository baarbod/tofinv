#!/bin/bash

# run this script to run the pipeline locally on the dummy data

# path to config file
CONFIG=config/config_dummy.yml

# number of available CPUs and memory (in MB) on the local machine
ncpu=24
mem=128000

# unlock the workflow in case it was locked by a previous run
snakemake --unlock --configfile $CONFIG

# run the workflow
snakemake --cores $ncpu --resources mem_mb=$mem --configfile $CONFIG --latency-wait 30 --restart-times 2 --printshellcmds --rerun-incomplete --keep-going