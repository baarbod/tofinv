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

import tofmodel.inverse.utils as utils
from tofmodel.forward import posfunclib as pfl
from tofmodel.forward import simulate as tm

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def setup_worker_logger(log_level=logging.INFO):
    worker_logger = multiprocessing.get_logger()
    if not worker_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('[%(processName)s] %(message)s')
        handler.setFormatter(formatter)
        worker_logger.addHandler(handler)
    worker_logger.setLevel(log_level)

def simulate_sample(isample, sample_data, output_dir):
    """Core physics simulation for a single parameter set."""
    worker_logger = multiprocessing.get_logger()
    input_data = sample_data['input_data']
    sample_file = os.path.join(output_dir, f"sample_{isample:03}.pkl")
    
    try:
        p = input_data['param'].scan_param
        
        # 1. Velocity and Timing Setup
        v_orig = utils.define_velocity_fourier(
            input_data['velocity_input'], p.num_pulse, input_data['rand_phase'], input_data['v_offset']
        )
        v_up = utils.upsample(v_orig, v_orig.size*100+1, p.repetition_time).flatten()
        t = np.arange(0, p.repetition_time*p.num_pulse, p.repetition_time/100)
        
        # 2. Position Function (incorporating vessel area)
        t_base, v_base = utils.add_baseline_period(t, v_up, p.repetition_time*p.num_pulse_baseline_offset)
        x_func = partial(pfl.compute_position_numeric_spatial, 
                         tr_vect=t_base, vts=v_base, 
                         xarea=input_data['xarea_sample'], area=input_data['area_sample'])
        
        # 3. Bloch Simulation
        s_raw = tm.simulate_inflow(p.repetition_time, p.echo_time, p.num_pulse+p.num_pulse_baseline_offset, 
                                   p.slice_width, p.flip_angle, p.t1_time, p.t2_time, p.num_slice, 
                                   p.alpha_list, p.MBF, x_func, ncpu=1)
        
        # Slice to requested output size
        s = s_raw[p.num_pulse_baseline_offset:, :input_data['param'].nslice_to_use]
            
        # 4. Save
        with open(sample_file, "wb") as f:
            pickle.dump({
                'X': s, 'v': v_orig,
                'xarea': input_data['xarea_sample'],
                'area': input_data['area_sample'],
                'input': input_data
            }, f)
            
        return True
    except Exception as e:
        worker_logger.error(f"Sample {isample} failed: {e}")
        return False
    
def run_batch(input_dir, task_id, output_dir):
    """Finds the correct pkl in input_dir based on task_id and runs simulation."""
    # Logic to find the file manually
    input_file = None
    # Adjust naming pattern here to match what synthData_sort actually produces
    target_pattern = f"task{int(task_id):03}.pkl" 
    
    for file in os.listdir(input_dir):
        if file.endswith(target_pattern):
            input_file = os.path.join(input_dir, file)
            break
            
    if input_file is None:
        logger.error(f"No file found in {input_dir} matching {target_pattern}")
        sys.exit(1)

    with open(input_file, 'rb') as f:
        samples = pickle.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_workers = min(len(samples), len(os.sched_getaffinity(0)))
    logger.info(f"Loaded {input_file}. Simulating {len(samples)} samples using {num_workers} CPU workers.")

    tstart = time.time()
    with Pool(processes=num_workers, initializer=setup_worker_logger) as pool:
        results = pool.starmap(simulate_sample, [(i, s, output_dir) for i, s in enumerate(samples)])
    tfinish = time.time()
    success_count = sum(results)
    logger.info(f"Batch {task_id} completed in {tfinish-tstart:.2f} seconds: {success_count}/{len(samples)} successful.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    run_batch(args.input_dir, args.task_id, args.output_dir)
