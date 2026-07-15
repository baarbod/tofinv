#!/bin/bash

# run this script to run the pipeline locally

# path to config file
CONFIG= ## INSERT PATH TO CONFIG FILE HERE ##

# number of available CPUs and memory (in MB) on the local machine
ncpu=24
mem=128000

# unlock the workflow in case it was locked by a previous run
snakemake --unlock --configfile $CONFIG

# run the workflow
snakemake --cores $ncpu --resources mem_mb=$mem --configfile $CONFIG --latency-wait 30 --restart-times 2 --printshellcmds --rerun-incomplete --keep-going