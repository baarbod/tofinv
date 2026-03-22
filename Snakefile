import pandas as pd
import os
import shutil
from pathlib import Path

# --- Globals & Paths ---
OUTDIR = config["paths"]["output_dir"]

# Decoupled Sub-directories
PREPDIR  = f"{OUTDIR}/1_preprocessing"
DATADIR  = f"{OUTDIR}/2_aggregated_data"
SYNTHDIR = f"{OUTDIR}/3_synthetic_data"
EXPDIR   = f"{OUTDIR}/4_experiments"
EVALDIR  = f"{OUTDIR}/5_evaluations"

# Grab the config file path passed via the CLI (--configfile)
if workflow.overwrite_configfiles:
    CONFIG_YML = workflow.overwrite_configfiles[0]
else:
    raise ValueError("Missing config file! Please run with: snakemake --configfile /path/to/config.yml")

# --- Data Loading ---
manifest = pd.read_csv(
    config["paths"]["input_manifest"], 
    header=0 if "subject" in open(config["paths"]["input_manifest"]).readline() else None,
    quotechar='"',
    names=["sub", "ses", "run", "func", "sbref", "anat", "aseg", "reg"]
)
manifest.set_index(["sub", "ses", "run"], drop=False, inplace=True)
manifest.sort_index(inplace=True)

# Pre-calculate lists for cleaner expand() statements later
SUB_SES_RUN = list(zip(manifest['sub'], manifest['ses'], manifest['run']))
UNIQUE_SUB_SES = manifest.drop_duplicates(['sub', 'ses'])
SUB_SES_UNIQUE = list(zip(UNIQUE_SUB_SES['sub'], UNIQUE_SUB_SES['ses']))
EXPERIMENTS = list(config["train_configs"].keys())

# =============================================================================
# --- Helper Functions ---
# =============================================================================

def get_func_path(wildcards):
    return manifest.loc[(wildcards.sub, wildcards.ses, wildcards.run), "func"]

def get_sbref_path(wildcards):
    return manifest.loc[(wildcards.sub, wildcards.ses, wildcards.run), "sbref"]

def get_area_inputs(wildcards):
    if (wildcards.sub, wildcards.ses) not in manifest.index.droplevel("run"):
        return {"func": [], "anat": [], "aseg": [], "reg": []}
    subset = manifest.loc[(wildcards.sub, wildcards.ses)].iloc[0]
    return {
        "func": subset["func"], 
        "anat": subset["anat"], 
        "aseg": subset["aseg"], 
        "reg": subset["reg"]
    }

def get_train_args(wildcards):
    return config["train_configs"][wildcards.exp]

# =============================================================================
# --- Rules ---
# =============================================================================

# rule all:
#     input:
#         f"{OUTDIR}/summary_report.pdf"
rule all:
    input:
        # Require the saved config file
        f"{OUTDIR}/config_used.yml",
        
        # Require the final evaluation files
        [f"{EVALDIR}/{exp}/{sub}/{ses}/{run}/velocity_predicted.txt" 
         for exp in EXPERIMENTS 
         for sub, ses, run in SUB_SES_RUN]

# ---------------------------------------------------------
# STAGE 0: UTILITIES
# ---------------------------------------------------------
rule save_config:
    input:
        CONFIG_YML
    output:
        f"{OUTDIR}/config_used.yml"
    localrule: True # Runs instantly on the master node without submitting a job
    shell:
        "cp {input} {output}"

# ---------------------------------------------------------
# STAGE 1: PREPROCESSING (Subject-level)
# ---------------------------------------------------------
rule automask:
    input:
        func = get_func_path,
        sbref = get_sbref_path
    output:
        signal = f"{PREPDIR}/{{sub}}/{{ses}}/{{run}}/automask/signal.txt",
        fixed_sbref = f"{PREPDIR}/{{sub}}/{{ses}}/{{run}}/automask/SBRef_fixed.nii.gz",
        synthseg = f"{PREPDIR}/{{sub}}/{{ses}}/{{run}}/automask/SBRef_fixed_synthseg.nii.gz"
    params:
        outdir = f"{PREPDIR}/{{sub}}/{{ses}}/{{run}}/automask",
        fs_container = config["paths"]["fs_container"],
        bind_container = config["paths"]["bind_container"],
        nslice = config["params"]["nslice"]
    resources:
        runtime = 30, nodes = 1, cpus_per_task = 1, mem_mb = 24000,
        slurm_partition = "mit_preemptable"
    shell:
        "python -m tofinv.masking --func {input.func} --sbref {input.sbref} "
        "--nslice_to_keep {params.nslice} --outdir {params.outdir} "
        "--container {params.fs_container} --container_bind {params.bind_container}"

rule noise:
    input:
        func = get_func_path,
        synthseg = rules.automask.output.synthseg,
        fixed_sbref = rules.automask.output.fixed_sbref
    output:
        noise_file = f"{PREPDIR}/{{sub}}/{{ses}}/{{run}}/noise/noise.txt",
    params:
        outdir = f"{PREPDIR}/{{sub}}/{{ses}}/{{run}}/noise"
    resources:
        runtime = 30, nodes = 1, cpus_per_task = 1, mem_mb = 20000, 
        slurm_partition = "mit_preemptable"
    shell:
        "python -m tofinv.noise --func {input.func} --synthseg {input.synthseg} "
        "--sbref {input.fixed_sbref} --outdir {params.outdir}"

rule area:
    input: unpack(get_area_inputs)
    output:
        area_file = f"{PREPDIR}/{{sub}}/{{ses}}/area/area.txt",
        area_dir = directory(f"{PREPDIR}/{{sub}}/{{ses}}/area")
    resources:
        runtime = 30, nodes = 1, cpus_per_task = 1, mem_mb = 16000,
        slurm_partition = "mit_normal"
    shell:
        "python -m tofinv.area --func {input.func} --anat {input.anat} "
        "--aseg {input.aseg} --reg {input.reg} --outdir {output.area_dir}"

rule optim:
    input:
        signal = rules.automask.output.signal,
        area = rules.area.output.area_file,
        config = CONFIG_YML
    output:
        done = touch(f"{PREPDIR}/{{sub}}/{{ses}}/{{run}}/optim/.optim_done")
    params:
        outdir = f"{PREPDIR}/{{sub}}/{{ses}}/{{run}}/optim"
    resources:
        runtime = 240, nodes = 1, cpus_per_task = 20, mem_mb = 42000,
        slurm_partition = "mit_preemptable"
    shell:
        "python -m tofinv.optim --signal {input.signal} --area {input.area} "
        "--config {input.config} --outdir {params.outdir}"

# ---------------------------------------------------------
# STAGE 2: AGGREGATED DATA
# ---------------------------------------------------------
rule aggregate_noise:
    input:
        files = expand(f"{PREPDIR}/{{sub}}/{{ses}}/{{run}}/noise/noise.txt", zip, sub=manifest['sub'], ses=manifest['ses'], run=manifest['run'])
    output:
        noise_data = f"{DATADIR}/noise_data.pkl"
    params:
        length = config["params"]["sample_length"],
        search_dir = PREPDIR
    resources:
        runtime = 20, nodes = 1, cpus_per_task = 1, mem_mb = 20000,               
        slurm_partition = "mit_preemptable"
    shell:
        "python -m tofinv.noise --collect --outdir {params.search_dir} "
        "--outfile {output.noise_data} --desired_sample_length {params.length}"

rule aggregate_area:
    input:
        areas = expand(f"{PREPDIR}/{{sub}}/{{ses}}/area/area.txt", zip, sub=UNIQUE_SUB_SES['sub'], ses=UNIQUE_SUB_SES['ses'])
    output:
        area_collection = f"{DATADIR}/area_collection.pkl"
    params:
        search_dir = PREPDIR
    resources:
        runtime = 15, mem_mb = 4000, slurm_partition = "mit_normal"
    shell:
        "python -m tofinv.area --collect --outdir {params.search_dir} --outfile {output.area_collection}"

# def get_successful_optim_runs(wildcards):
#     all_paths = expand(
#         f"{PREPDIR}/{{sub}}/{{ses}}/{{run}}/optim", 
#         zip, 
#         sub=manifest['sub'], 
#         ses=manifest['ses'], 
#         run=manifest['run']
#     )
#     existing_paths = [f for f in all_paths if os.path.exists(f)]
#     return [os.path.dirname(p) for p in existing_paths]

rule aggregate_optim:
    input:
        dones = expand(f"{PREPDIR}/{{sub}}/{{ses}}/{{run}}/optim/.optim_done", zip, sub=manifest['sub'], ses=manifest['ses'], run=manifest['run'])
        # optims = get_successful_optim_runs
    output:
        optimized_velocity = f"{DATADIR}/crude_optim_velocity_amps.pkl"
    params:
        search_dir = PREPDIR
    resources:
        runtime = 120, nodes = 1, cpus_per_task = 20, mem_mb = 64000, slurm_partition = "mit_normal"
    shell:
        "python -m tofinv.optim --collect --outdir {params.search_dir} --outfile {output.optimized_velocity}"

# ---------------------------------------------------------
# STAGE 3: SYNTHETIC DATA
# ---------------------------------------------------------
rule synthdata_sampling:
    input:
        voptim = rules.aggregate_optim.output.optimized_velocity,
        area_collection = rules.aggregate_area.output.area_collection,
        config = CONFIG_YML
    output: 
        pkl = f"{SYNTHDIR}/inputs_batched/task{{batch}}.pkl"
    group: "sampling"
    threads: config["params"]["threads_high"]
    resources:
        runtime = 20, nodes = 1, cpus_per_task = 1, mem_mb = 8000, slurm_partition = "mit_preemptable"
    shell:
        "python -m tofinv.synthdata.sampling --config {input.config} --taskid {wildcards.batch} "
        "--optim_path {input.voptim} --area_path {input.area_collection} --output {output.pkl}"

rule synthdata_sort:
    input:
        batches = expand(f"{SYNTHDIR}/inputs_batched/task{{batch}}.pkl", batch=range(1, config["synthetic"]['num_batches'] + 1))
    output:
        sort_done = temp(touch(f"{SYNTHDIR}/inputs_batched_sorted/.sort_done"))
    params:
        outdir = f"{SYNTHDIR}/inputs_batched_sorted",
        batch_size = config["params"]["batch_size"]
    resources:
        runtime = 60, nodes = 1, cpus_per_task = 1, mem_mb = 32000, slurm_partition = "mit_normal",
    shell:
        "python -m tofinv.synthdata.processing sort --input_dir {SYNTHDIR}/inputs_batched "
        "--output_dir {params.outdir} --batch_size {params.batch_size}"

rule synthdata_simulate:
    input: 
        sort_done = rules.synthdata_sort.output.sort_done
    output: 
        batch_done = touch(f"{SYNTHDIR}/simulations_batched/batch_{{batch}}/.sim_done")
    params:
        input_dir = f"{SYNTHDIR}/inputs_batched_sorted",
        output_dir = f"{SYNTHDIR}/simulations_batched/batch_{{batch}}"
    group: "simulation"
    threads: config["params"]["threads_high"]
    resources:
        runtime = 240, nodes = 1, cpus_per_task = 5, mem_mb = 16000, slurm_partition = "mit_preemptable"
    shell:
        "python -m tofinv.synthdata.simulation --input_dir {params.input_dir} "
        "--task_id {wildcards.batch} --output_dir {params.output_dir}"

rule synthdata_combine:
    input:
        dones = expand(f"{SYNTHDIR}/simulations_batched/batch_{{batch}}/.sim_done", batch=range(1, config["synthetic"]['num_batches'] + 1))
    output:
        final_pkl = f"{SYNTHDIR}/dataset.pkl"
    threads: config["params"]["threads_high"]
    resources:
        runtime = 120, nodes = 1, cpus_per_task = 64, mem_mb = 100000, slurm_partition = "mit_normal"
    shell:
        "python -m tofinv.synthdata.processing combine --sim_dir {SYNTHDIR}/simulations_batched --output_dir {SYNTHDIR}"

# ---------------------------------------------------------
# STAGE 4: EXPERIMENTS (Training)
# ---------------------------------------------------------
rule train_surrogate:
    input:
        dataset = rules.synthdata_combine.output.final_pkl
    output:
        weights = f"{EXPDIR}/surrogate_model/surrogate_model_weights.pth",
        plot_done = touch(f"{EXPDIR}/surrogate_model/surrogate_plots/.plot_done")
    params:
        plot_dir = f"{EXPDIR}/surrogate_model/surrogate_plots"
    resources:
        runtime = 360, nodes = 1, cpus_per_task = 4, mem_mb = 64000, slurm_partition = "mit_preemptable", slurm_extra = "--gres=gpu:4"
    shell:
        "python -m tofinv.surrogate --dataset {input.dataset} --out_weights {output.weights} --outdir {params.plot_dir}"

rule train_model:
    input:
        noise_data = rules.aggregate_noise.output.noise_data,
        dataset = rules.synthdata_combine.output.final_pkl,
        surrogate = rules.train_surrogate.output.weights
    output:
        model = f"{EXPDIR}/{{exp}}/best_model.pth"
    params:
        train_args = get_train_args,
        epochs = config["params"]["train_epochs"],
        batch = config["params"]["train_batch_size"],
        lr = config["params"]["train_lr"],
        outdir = f"{EXPDIR}/{{exp}}"
    resources:
        runtime = 240, nodes = 1, cpus_per_task = 1, mem_mb = 64000, slurm_partition = "mit_preemptable", slurm_extra = "--gres=gpu:1"
    shell:
        """
        python -m tofinv.train \
            --epochs {params.epochs} \
            --batch {params.batch} \
            --lr {params.lr} \
            --dataset {input.dataset} \
            --noisedir {DATADIR} \
            --surrogate_path {input.surrogate} \
            --outdir {params.outdir} \
            --out_weights {output.model} \
            {params.train_args}
        """

# ---------------------------------------------------------
# STAGE 5: EVALUATIONS
# ---------------------------------------------------------
rule evaluate:
    input:
        signal = rules.automask.output.signal,
        model = rules.train_model.output.model,
        area = rules.area.output.area_file,
        config = CONFIG_YML
    output:
        velocity = f"{EVALDIR}/{{exp}}/{{sub}}/{{ses}}/{{run}}/velocity_predicted.txt"
    threads: config["params"]["threads_low"]
    params:
        outdir = f"{EVALDIR}/{{exp}}/{{sub}}/{{ses}}/{{run}}"
    resources:
        runtime = 240, nodes = 1, cpus_per_task = 1, mem_mb = 24000, slurm_partition = "mit_preemptable"
    shell:
        "python -m tofinv.evaluation "
        "--signal {input.signal} --area {input.area} --model {input.model} "
        "--config {input.config} --outdir {params.outdir} --ncpu {threads}"