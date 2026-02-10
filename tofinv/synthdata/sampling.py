import argparse
import logging
import os
import pickle
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
from omegaconf import OmegaConf
from scipy.interpolate import interp1d

import tofinv.utils as utils
from tofinv.optim import transform
from tofmodel.forward import posfunclib as pfl
from tofmodel.forward import simulate as tm

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def get_resampled_area(x, a, n_points, offset=0, scale=1.0):
    """Handles the alignment and resampling of the vessel area profile."""
    # Logic to find central widest point while avoiding edge artifacts
    edge_buffer = 20
    if len(a) > 2 * edge_buffer:
        central_widest_idx = np.argmax(a[edge_buffer:-edge_buffer]) + edge_buffer
    else:
        central_widest_idx = np.argmax(a)
    
    widest_pos = x[central_widest_idx]
    
    # Normalize coordinates: center at widest point, apply offset, apply scale
    x_centered = x - widest_pos - offset
    a_scaled = a * scale
    
    # Interpolate to target feature size
    f = interp1d(x_centered, a_scaled, kind='linear', fill_value='extrapolate')
    x_new = np.linspace(x_centered.min(), x_centered.max(), n_points)
    return x_new, f(x_new)

def compute_init_positions(input_data):
    """Worker function to compute starting positions for protons."""
    p = input_data['param'].scan_param
    
    # Generate velocity curve
    v = utils.define_velocity_fourier(
        input_data['velocity_input'], p.num_pulse, input_data['rand_phase'], input_data['v_offset']
    )
    v_up = utils.upsample(v, v.size * 100 + 1, p.repetition_time).flatten()
    t = np.arange(0, p.repetition_time * p.num_pulse, p.repetition_time / 100)
    
    t_base, v_base = utils.add_baseline_period(t, v_up, p.repetition_time * p.num_pulse_baseline_offset)
    
    # Define position function over space/time
    x_func = partial(
        pfl.compute_position_numeric_spatial, 
        tr_vect=t_base, vts=v_base, 
        xarea=input_data['xarea_sample'], area=input_data['area_sample']
    )
    
    # Identify pulse targets to find simulation bounds
    timings, _ = tm.get_pulse_targets(
        p.repetition_time, p.num_slice, p.num_pulse + p.num_pulse_baseline_offset, 
        np.array(p.alpha_list, ndmin=2).T
    )
    
    lb, ub = tm.get_init_position_bounds(x_func, np.unique(timings), p.slice_width, p.num_slice)
    return np.arange(lb, ub + 0.01, 0.01)

def generate_batch(param, data_dir, task_id):
    """Main logic for sampling parameters and calculating initial conditions."""
    ds = param.synthetic
    batch_size = ds.num_samples // ds.num_batches
    
    base_seed = 42
    rng = np.random.default_rng(base_seed + task_id)
    
    # Load optimized velocity base
    with open(os.path.join(data_dir, 'crude_optim_velocity_amps.pkl'), 'rb') as f:
        optim_data_list = pickle.load(f)

    # define information for cross-sectional areas
    if param.synthetic.areamode == 'straight_tube':
        # Default fallback
        x_fixed = np.linspace(-3, 3, param.synthetic.num_time_point)
        area_lookup = { "default": (x_fixed, np.ones_like(x_fixed)) }
    elif param.synthetic.areamode == 'collection': 
        with open(os.path.join(data_dir, 'area_collection.pkl'), 'rb') as f:
            # We updated this to be a list of (xarea, area, subject)
            area_raw_list = pickle.load(f)
        area_lookup = {sub: (xa, a) for xa, a, sub in area_raw_list}

    # 3. Build parameter list
    frequencies = np.fft.rfftfreq(ds.num_time_point, d=param.scan_param.repetition_time)
    batch_params = []
    
    for _ in range(batch_size):

        ind_random = np.random.randint(0, len(optim_data_list))
        vbase, res, sub_name = optim_data_list[ind_random]
        
        # jitter the optimized parameters
        jitter_range = (0.95, 1.05)
        jitter_vector = np.random.uniform(jitter_range[0], jitter_range[1], size=len(res.x))
        res_x_jittered = res.x * jitter_vector
        random_voffset = res_x_jittered[-1]
        random_phase = np.random.uniform(0, 2*np.pi, frequencies.size)
        
        # 1. Transform velocity to get the "optimized" version
        # We use the real phase from the original recording
        phase_base = np.angle(np.fft.rfft(vbase))
        vopt = transform(vbase, res_x_jittered, phase_base)
        
        # 2. Extract Spectral Components
        V_freq = np.fft.rfft(vopt, axis=0)
        amp_raw = np.abs(V_freq)
        
        # --- SELECT MATCHING AREA ---
        if param.synthetic.areamode == 'collection':
            if sub_name in area_lookup:
                xarea_raw, area_raw = area_lookup[sub_name]
            else:
                # Fallback if a subject exists in velocity but not in area collection
                logger.warning(f"Subject {sub_name} not found in area collection. Picking random.")
                random_sub = np.random.choice(list(area_lookup.keys()))
                xarea_raw, area_raw = area_lookup[random_sub]

            # Apply geometric augmentations
            offset = np.random.uniform(param.synthetic.slc1_offset_lower, param.synthetic.slc1_offset_upper)
            scale = np.random.uniform(param.synthetic.area_scale_lower, param.synthetic.area_scale_upper)
            xarea, area = get_resampled_area(xarea_raw, area_raw, param.synthetic.num_time_point, offset, scale)
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

    # 3. Parallel computation of proton starting positions
    n_workers = min(batch_size, len(os.sched_getaffinity(0)))
    logger.info(f"Starting Task {task_id}: Computing positions for {batch_size} samples using {n_workers} workers.")
    
    with Pool(processes=n_workers) as pool:
        x0_list = pool.map(compute_init_positions, batch_params)

    return [{'input_data': p, 'x0_array': x0} for p, x0 in zip(batch_params, x0_list)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--data_dir", required=True, help="Path to input data (KDEs, areas)")
    parser.add_argument("--output", required=True, help="Path to save the batch .pkl")
    parser.add_argument("--taskid", type=int, default=1)
    args = parser.parse_args()

    # Load configuration
    cfg = OmegaConf.load(args.config)
    
    # Run and save
    batch_data = generate_batch(cfg, args.data_dir, args.taskid)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(batch_data, f)
    
    logger.info(f"Successfully saved batch to {args.output}")