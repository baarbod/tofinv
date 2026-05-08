import os
import pickle
import logging
import argparse
import sys
import multiprocessing
from multiprocessing import Pool
from functools import partial
import numpy as np
import time

import tofinv.utils as utils
from tofmodel.forward import posfunclib as pfl
from tofmodel.forward import simulate as tm

# --- Main Process Logging ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def setup_worker_logger(log_level=logging.INFO):
    """
    Configures logging for child processes. 
    This prevents workers from stepping on each other's output.
    """
    worker_logger = multiprocessing.get_logger()
    if not worker_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # Process name (e.g., ForkPoolWorker-1) helps trace errors to specific cores
        formatter = logging.Formatter('%(asctime)s [%(processName)s] %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        worker_logger.addHandler(handler)
    worker_logger.setLevel(log_level)

def simulate_sample(isample, sample_data, output_dir):
    """Simulation for a single parameter set."""
    worker_logger = multiprocessing.get_logger()
    input_data = sample_data['input_data']
    sample_file = os.path.join(output_dir, f"sample_{isample:03}.pkl")
    
    # Check if we should skip (useful for resuming interrupted cluster jobs)
    if os.path.exists(sample_file):
        worker_logger.info(f"Sample {isample} already exists. Skipping.")
        return True

    try:
        p = input_data['param'].scan_param
        
        # Velocity and Timing Setup
        v_orig = utils.define_velocity_fourier(
            input_data['velocity_input'], p.num_pulse, input_data['rand_phase'], input_data['v_offset']
        )
        v_up = utils.upsample(v_orig, v_orig.size*100+1, p.repetition_time).flatten()
        t = np.arange(0, p.repetition_time*p.num_pulse, p.repetition_time/100)
        
        # Position Function
        t_base, v_base = utils.add_baseline_period(t, v_up, p.repetition_time*p.num_pulse_baseline_offset)
        x_func = partial(pfl.compute_position_numeric_spatial, 
                         tr_vect=t_base, vts=v_base, 
                         xarea=input_data['xarea_sample'], area=input_data['area_sample'])
        
        # Forward Model Simulation 
        s_raw = tm.simulate_inflow(
            p.repetition_time, p.echo_time, p.num_pulse+p.num_pulse_baseline_offset, 
            p.slice_width, p.flip_angle, p.t1_time, p.t2_time, p.num_slice, 
            p.alpha_list, p.MBF, x_func, ncpu=1, varysliceprofile=True, 
            dx=0.005, offset_fact=0, enable_logging=False
        )
        
        s = s_raw[p.num_pulse_baseline_offset:, :input_data['param'].nslice_to_use]
            
        # Save
        with open(sample_file, "wb") as f:
            pickle.dump({
                'X': s, 'v': v_orig,
                'xarea': input_data['xarea_sample'],
                'area': input_data['area_sample'],
                'input': input_data
            }, f)
            
        return True
    except Exception as e:
        worker_logger.error(f"FAILURE: Sample {isample} index error: {e}")
        return False
    
def run_batch(input_dir, task_id, output_dir):
    target_pattern = f"task{int(task_id):03}.pkl" 
    input_file = None
    
    # Locate batch file
    for file in os.listdir(input_dir):
        if file.endswith(target_pattern):
            input_file = os.path.join(input_dir, file)
            break
            
    if input_file is None:
        logger.error(f"Task ID {task_id} specified, but {target_pattern} not found in {input_dir}")
        sys.exit(1)

    logger.info(f"--- Starting Batch {task_id} ---")
    with open(input_file, 'rb') as f:
        samples = pickle.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_workers = min(len(samples), len(os.sched_getaffinity(0)))
    logger.info(f"Batch {task_id}: {len(samples)} samples total. Using {num_workers} workers.")

    tstart = time.time()
    
    # Using starmap to execute simulations in parallel
    with Pool(processes=num_workers, initializer=setup_worker_logger) as pool:
        # Pass enumerate to keep track of sample IDs for filenames
        results = pool.starmap(simulate_sample, [(i, s, output_dir) for i, s in enumerate(samples)])
    
    tfinish = time.time()
    success_count = sum(results)
    
    logger.info(f"--- Batch {task_id} Summary ---")
    logger.info(f"Runtime: {tfinish-tstart:.2f}s (Avg: {(tfinish-tstart)/len(samples):.2f}s per sample)")
    logger.info(f"Success Rate: {success_count}/{len(samples)}")
    
    if success_count < len(samples):
        logger.warning(f"{len(samples) - success_count} samples failed. Check worker logs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Inflow Signal Simulator")
    parser.add_argument("--input_dir", required=True, help="Directory containing taskXXX.pkl files")
    parser.add_argument("--task_id", required=True, help="ID to match taskXXX.pkl")
    parser.add_argument("--output_dir", required=True, help="Directory for individual sample pkls")
    args = parser.parse_args()

    run_batch(args.input_dir, args.task_id, args.output_dir)