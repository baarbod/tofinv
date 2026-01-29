import argparse
import logging
from pathlib import Path
from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# Internal TOF imports
from tofmodel.inverse.models import TOFinverse
import tofinv.utils as utils
from tofmodel.forward import posfunclib as pfl
from tofmodel.forward import simulate as tm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Data & Model Loading ---

def load_config(config_path):
    return OmegaConf.load(config_path)

def load_network(state_filename, param, device='cpu'):
    checkpoint = torch.load(state_filename, map_location=torch.device(device), weights_only=True)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    ds = param.synthetic
    model = TOFinverse(
        nfeature_in=ds.num_input_features, 
        nfeature_out=ds.num_output_features, 
        input_size=ds.num_time_point,
        output_size=ds.num_time_point
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_data(spath, area_path, param):
    """Loads signal and area data, resamples area to match model input size."""
    offset = param.scan_param.num_pulse_baseline_offset
    n_slices = param.nslice_to_use
    
    sraw = np.loadtxt(spath)[offset:, :n_slices]
    A = np.loadtxt(area_path)
    xarea_raw, area_raw = A[:, 0], A[:, 1]

    new_len = param.scan_param.num_pulse
    x_new = np.linspace(0, 1, new_len)
    x_old = np.linspace(0, 1, xarea_raw.size)
    
    xarea = np.interp(x_new, x_old, xarea_raw)
    area = np.interp(x_new, x_old, area_raw)
    return sraw, xarea, area

# --- Signal Processing ---

def get_steady_state_constant(sp):
    """Computes the theoretical steady state signal (mT_ss)."""
    fa_rad = np.radians(sp.flip_angle)
    exp_tr_t1 = np.exp(-sp.repetition_time / sp.t1_time)
    exp_te_t2 = np.exp(-sp.echo_time / sp.t2_time)
    
    return np.sin(fa_rad) * exp_te_t2 * (1 - exp_tr_t1) / (1 - exp_tr_t1 * np.cos(fa_rad))

# --- Main Logic Blocks ---

def run_velocity_inference(model, s_data, xarea, area):
    return utils.input_batched_signal_into_NN_area(s_data, model, xarea, area)

def run_forward_model(velocity_NN, xarea, area, param, ncpu=8):
    sp = param.scan_param
    # Higher resolution velocity for simulation
    v_up = utils.upsample(velocity_NN, velocity_NN.size * 100 + 1, sp.repetition_time).flatten()
    t = np.arange(0, sp.repetition_time * velocity_NN.size, sp.repetition_time / 100)
    
    t_base, v_base = utils.add_baseline_period(t, v_up, sp.repetition_time * sp.num_pulse_baseline_offset)
    
    pos_func = partial(pfl.compute_position_numeric_spatial, tr_vect=t_base, vts=v_base, xarea=xarea, area=area)
    
    s_sim_raw = tm.simulate_inflow(
        sp.repetition_time, sp.echo_time, velocity_NN.size + sp.num_pulse_baseline_offset,
        sp.slice_width, sp.flip_angle, sp.t1_time, sp.t2_time, sp.num_slice, 
        sp.alpha_list, sp.MBF, pos_func, ncpu=ncpu
    )
    return s_sim_raw[sp.num_pulse_baseline_offset:, :param.nslice_to_use]

# --- IO & Visualization ---

def save_and_plot(outdir, velocity, sraw_scaled, ssim_scaled, tr):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Save raw data
    np.savetxt(outdir / 'signal_data.txt', sraw_scaled)
    np.savetxt(outdir / 'velocity_predicted.txt', velocity)
    np.savetxt(outdir / 'signal_simulation.txt', ssim_scaled)

    # Time domain plot
    t_vec = tr * np.arange(velocity.size)
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    axes[0].plot(t_vec, sraw_scaled[:velocity.size])
    axes[0].set_title("Input Signal (Scaled)")
    axes[1].plot(t_vec, velocity, color='black')
    axes[1].set_title("Inferred Velocity")
    axes[2].plot(t_vec, ssim_scaled)
    axes[2].set_title("Simulated Signal (Scaled)")
    plt.tight_layout()
    fig.savefig(outdir / "summary_plot.png")

# --- Execution Entry Point ---

def main():
    parser = argparse.ArgumentParser(description='Run velocity inference using TOF framework')
    parser.add_argument('--signal', type=str, required=True)
    parser.add_argument('--area', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ncpu', type=int, default=8)
    args = parser.parse_args()

    # 1. Setup
    param = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_network(args.model, param, device)
    
    # 2. Data Prep
    logger.info("Loading data...")
    sraw, xarea, area = load_data(args.signal, args.area, param)
    sraw_scaled = utils.scale_by_baseline(sraw)
    
    # 3. Inference
    logger.info("Running inference...")
    velocity = run_velocity_inference(model, sraw_scaled, xarea, area)
    
    # 4. Simulation Verification
    logger.info("Running forward model...")
    ssim = run_forward_model(velocity, xarea, area, param, ncpu=args.ncpu)
    
    # 5. Normalization for Comparison
    mT_ss = get_steady_state_constant(param.scan_param)
    ssim_scaled = ssim / mT_ss
    
    # 6. Save
    logger.info(f"Saving results to {args.outdir}")
    save_and_plot(args.outdir, velocity, sraw_scaled, ssim_scaled, param.scan_param.repetition_time)

if __name__ == "__main__":
    main()