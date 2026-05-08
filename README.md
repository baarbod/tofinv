# tofinv

This repository contains an end-to-end pipeline that estimates cerebrospinal fluid (CSF) flow velocity from fMRI inflow signals due to inflow effects. 

This framework is based on our prior work. If you use this pipeline, please cite this:

> **Ashenagar et al., 2025** > Modeling dynamic inflow effects in fMRI to quantify cerebrospinal fluid flow
> ([https://doi.org/10.1162/IMAG.a.9](https://doi.org/10.1162/IMAG.a.9))

## PREREQUISITES

* **Operating System:** A Linux environment is required to run this pipeline.
* **Python:** Version `3.10.14` is required.
* **Hardware:** The pipeline speed scales with number of available CPUs. At least 1 GPU is recommended for neural network training stages.
* **Software:** This pipeline depends on Synthseg from FreeSurfer (7.3.2+).

## GETTING STARTED

### Setup and Installation

Set up a virtual environment to manage dependencies:
```bash
# Create and activate environment
python3.10 -m venv .venv
source .venv/bin/activate

# Clone and install
git clone https://github.com/baarbod/tofinv.git
cd tofinv
pip install .
```

NOTE: If your system has Freesurfer 7.3.2+, then ensure the module is loaded when running the pipeline. If you want to use an Apptainer/Singularity container instead, change the `fs_container` and `bind_container` parameters in the config from `null` to the corresponding paths.

### Updating
If you have already installed the pipeline and want to update to the latest version (including updates to external dependencies like tofmodel and autocsfmask), follow these steps:

```bash
cd tofinv
git pull
pip install --upgrade .
```

### Test run with dummy data

Before running on your own data, do a test run on the provided dummy data to ensure your environment is set up correctly and the pipeline executes as expected. The configuration file for this test `config/config_dummy.yml` is already pre-configured for the dummy dataset.

Run the pipeline:
```bash
bash run_dummy.sh
```

### Running on your own data

Once you have confirmed that the dummy run works successfully, you can try using your own data. 

You will need to create a new input text file for your data. Make a copy of the preconfigured `config/input_dummy_test.txt` file, rename it, and fill in the paths to your specific files. 

Each row in this file corresponds to one input sample. Please maintain the exact same formatting as the dummy file. Here is the breakdown of what each column in a row represents, in order:

1. **Subject Name**
2. **Session Name**
3. **Run Name**
4. **Raw fMRI Volume:** Path to the raw fMRI volume.
5. **Raw SBRef Volume:** Path to the raw single-band reference volume (if unavailable, use the functional data averaged across time).
6. **T1-Weighted Anatomical Image:** Path to the anatomical image (use the `orig.mgz` file generated from `recon-all`).
7. **Anatomical Segmentation Volume:** Path to the segmentation volume (use the `aseg.mgz` file generated from `recon-all`).
8. **Registration File:** Path to pregenerated output FreeSurfer registration file.

Configure your parameters. 
Use `config/config_base.yml` as your starting point (do not use `config_dummy.yml` for real data). 
   
Update the configuration file.
Make a copy of `config_base.yml` and name it however you want. Ensure you update the following to match your dataset:
- File and directory paths
- Specific fMRI acquisition parameters


Execute the pipeline: 
Run Snakemake pointing to your newly updated configuration file. You can use the `run_dummy.sh` as a template for this.

### Running in SLURM mode

If you are on a SLURM cluster, see the example script below for running the pipeline in slurm mode. This script launches a master job which then orchestrates requesting jobs for pipeline steps. This master job does not need many resources; just ensure it has plently of time. The actual resources used in slurm mode are currently hardcoded for each rule in the Snakefile so modify them there. 

```bash
#!/bin/bash
#SBATCH --job-name=snakemaster
#SBATCH --output=logs/master_%j.out
#SBATCH --error=logs/master_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --partition=PARTION_NAME

source path/to/pythonenv/.venv/bin/activate

cd path/to/tofinv

CONFIG=config/config_base.yml

snakemake --configfile $CONFIG --unlock

snakemake --slurm --configfile $CONFIG --jobs 100 --latency-wait 30 --restart-times 2 --printshellcmds --rerun-incomplete --keep-going --group-components sampling=10 simulation=5


```



### CONTACT & SUPPORT

For any questions, comments, suggestions, or issues running the pipeline, please feel free to reach out!

Baarbod Ashenagar 
bashen@mit.edu

