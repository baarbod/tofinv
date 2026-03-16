import argparse
import logging
import os
import pickle
import sys
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
from omegaconf import OmegaConf
from scipy.interpolate import interp1d

import tofinv.utils as utils
from tofinv.optim import transform
from tofmodel.forward import posfunclib as pfl
from tofmodel.forward import simulate as tm

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def compute_init_positions(input_data):
    p = input_data['param'].scan_param
    v = utils.define_velocity_fourier(input_data['velocity_input'], p.num_pulse, input_data['rand_phase'], input_data['v_offset'])
    v_up = utils.upsample(v, v.size * 100 + 1, p.repetition_time).flatten()
    t = np.arange(0, p.repetition_time * p.num_pulse, p.repetition_time / 100)
    t_base, v_base = utils.add_baseline_period(t, v_up, p.repetition_time * p.num_pulse_baseline_offset)
    x_func = partial(pfl.compute_position_numeric_spatial, 
                     tr_vect=t_base, vts=v_base, 
                     xarea=input_data['xarea_sample'], area=input_data['area_sample'])
    timings, _ = tm.get_pulse_targets(p.repetition_time, 
                                      p.num_slice, 
                                      p.num_pulse + p.num_pulse_baseline_offset, 
                                      np.array(p.alpha_list, ndmin=2).T)
    lb, ub = tm.get_init_position_bounds(x_func, np.unique(timings), p.slice_width, p.num_slice)
    return np.arange(lb, ub + 0.01, 0.01)

def generate_batch(param, optim_data_list, area_lookup, task_id):
    ds = param.synthetic
    batch_size = ds.num_samples // ds.num_batches
    logger.info(f"Task {task_id}: Generating batch of size {batch_size}")
    
    base_seed = 42
    rng = np.random.default_rng(base_seed + task_id)
    
    frequencies = np.fft.rfftfreq(ds.num_time_point, d=param.scan_param.repetition_time)
    batch_params = []
    n_optim = len(optim_data_list)
    keys_area = list(area_lookup.keys())
    
    for i in range(batch_size):
        if (i + 1) % 10 == 0:
            logger.info(f"Task {task_id}: Sampling iteration {i + 1}/{batch_size}")
            
        ind_random = rng.integers(0, n_optim)
        vbase, optim_param, sub_name = optim_data_list[ind_random]
        jitter_vector = rng.uniform(0.9, 1.1, size=len(optim_param))
        res_x_jittered = optim_param * jitter_vector
        random_voffset = res_x_jittered[-1]
        phase_base = np.angle(np.fft.rfft(vbase))
        vopt = transform(vbase, res_x_jittered, phase_base)
        random_phase = np.angle(np.fft.rfft(vopt))
        vopt_demean = vopt - vopt.mean()
        V_freq = np.fft.rfft(vopt_demean, axis=0)
        amp_raw = np.abs(V_freq)
        
        if param.synthetic.areamode == 'collection':
            if sub_name in area_lookup:
                xarea_raw, area_raw = area_lookup[sub_name]
            else:
                logger.warning(f"Subject {sub_name} not found in area collection. Picking random.")
                random_sub = rng.choice(keys_area)
                xarea_raw, area_raw = area_lookup[random_sub]
            x_new = np.linspace(0, 1, ds.num_time_point)
            x_old = np.linspace(0, 1, xarea_raw.size)
            xarea = np.interp(x_new, x_old, xarea_raw)
            area = np.interp(x_new, x_old, area_raw)
        else:
            xarea, area = area_lookup["default"]
            
        batch_params.append({
            'frequencies': tuple(frequencies),
            'v_offset': random_voffset,
            'rand_phase': random_phase,
            'velocity_input': tuple(amp_raw),
            'param': param,
            'xarea_sample': xarea,
            'area_sample': area,
            'task_id': task_id
        })

    start_time = time.time()

    n_workers = min(batch_size, len(os.sched_getaffinity(0)))
    logger.info(f"Starting Task {task_id}: Computing positions for {batch_size} samples using {n_workers} workers.")
    
    with Pool(processes=n_workers) as pool:
        x0_list = pool.map(compute_init_positions, batch_params)

    end_time = time.time()
    logger.info(f"Task {task_id}: Parallel computation finished in {end_time - start_time:.2f} seconds.")

    return [{'input_data': p, 'x0_array': x0} for p, x0 in zip(batch_params, x0_list)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output", required=True, help="Path to save the batch .pkl")
    parser.add_argument("--optim_path", required=True)
    parser.add_argument("--area_path", required=True)
    parser.add_argument("--taskid", type=int, default=1)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    
    with open(args.optim_path, 'rb') as f:
        optim_data_list = pickle.load(f)
    
    if cfg.synthetic.areamode == 'straight_tube':
        x_fixed = np.linspace(-3, 3, cfg.synthetic.num_time_point)
        area_lookup = { "default": (x_fixed, np.ones_like(x_fixed)) }
    elif cfg.synthetic.areamode == 'collection': 
        with open(args.area_path, 'rb') as f:
            area_raw_list = pickle.load(f)
        area_lookup = {sub: (xa, a) for xa, a, sub in area_raw_list}

    batch_data = generate_batch(cfg, optim_data_list, area_lookup, args.taskid)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(batch_data, f)
    
    logger.info(f"Successfully saved batch to {args.output}")