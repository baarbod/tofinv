# tofinv

This repository contains an end-to-end pipeline that estimates cerebrospinal fluid (CSF) flow velocity from fMRI inflow signals in the 4th ventricle. 

This framework is based on our prior work. If you use this pipeline, please cite this:

> **Ashenagar et al., 2025** > Modeling dynamic inflow effects in fMRI to quantify cerebrospinal fluid flow
> ([https://doi.org/10.1162/IMAG.a.9](https://doi.org/10.1162/IMAG.a.9))

## PREREQUISITES

* **Operating System:** A Linux environment is required to run this pipeline.
* **Python:** Version `3.10.14`. I tested the code on this version so use other versions at your own risk.
* **Hardware:** While the pipeline can run locally, it is recommended to run this on a computing cluster (e.g., using SLURM). For testing on the dummy data (see below) you can run locally.

## GETTING STARTED

### Phase 0: Setup and Installation

Create a new directory (optional but recommended)
```bash
mkdir -p repos
cd repos
```
If you already have a python environment, you can skip the next couple of steps. \
To create an isolated Python virtual environment, run the following command (replace `python3.10` with your specific path if needed):
```bash
python3.10 -m venv .venv
```
Then activate the environment
```bash
source .venv/bin/activate
``` 

Clone the repository
```bash
git clone https://github.com/baarbod/tofinv.git
```
Navigate into the cloned repository
```bash
cd tofinv
```
Install the package and all dependencies
```bash
pip install .
```

The pipeline currently uses the Synthseg tool which requires Freesurfer 7.3.2+.
You will need to install a Freesurfer container and then later in the config file you'll specify the path to the container as well as bind directories for where your data is stored. \

For the next phase where we test on dummy data, you don't need to have this container set up, but you will need when you're ready to run on you real data.


### Phase 1: Test run with dummy data

Before running on your own data, we recommend executing a test run using generated dummy data to ensure your environment is set up correctly and the pipeline executes as expected.

Generate the dummy data (This will create a dummy_data folder in your repository.):
```bash
python generate_dummy_data.py
```

Run the pipeline:
The configuration file for this test `config/config_dummy.yml` is already pre-configured for the dummy dataset. You can launch the test run using the provided bash script (make sure to set the number of CPUs in this script based on your system):
```bash
bash run_dummy.sh
```

### Phase 2: Running on your own data

Once you have confirmed that the dummy run works successfully, you can try using your own data. 

You will need to create a new input text file for your data. Make a copy of the preconfigured `config/input_dummy_test.txt` file, rename it, and fill in the paths to your specific files. 

Each row in this file corresponds to one input sample. Please maintain the exact same formatting as the dummy file. Here is the breakdown of what each column in a row represents, in order:

1. **Subject Name**
2. **Session Name**
3. **Run Name**
4. **Raw fMRI Volume:** Path to the raw fMRI volume.
5. **Raw SBRef Volume:** Path to the raw single-band reference (SBRef) volume.
6. **T1-Weighted Anatomical Image:** Path to the anatomical image (e.g., MPRAGE).
7. **Anatomical Segmentation Volume:** Path to the segmentation volume (e.g., the `aseg.nii` file generated from `recon-all`).
8. **Registration File:** Path to the functional-to-anatomical registration file.

Configure your parameters. 
Use `config/config_base.yml` as your starting point (do not use `config_dummy.yml` for real data). 
   
Update the configuration file.
Make a copy of `config_base.yml` and name it however you want. Ensure you update the following to match your dataset:
- File and directory paths
- Specific fMRI acquisition parameters


Execute the pipeline: 
Run Snakemake pointing to your newly updated configuration file. You can use the `run_dummy.sh` as a template for this.

### TODO
Add a section about running the pipeline on a computing cluster (i.e. SLURM)


### CONTACT & SUPPORT

For any questions, comments, suggestions, or issues running the pipeline, please feel free to reach out!

Baarbod Ashenagar 
bashen@mit.edu

