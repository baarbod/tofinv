import argparse
import logging
import sys
import time
from pathlib import Path
from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from tofinv.nn_models import TOFinverse
import tofinv.utils as utils
from tofmodel.forward import posfunclib as pfl
from tofmodel.forward import simulate as tm

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    return OmegaConf.load(config_path)

def load_network(state_filename, param, device='cpu'):
    logger.info(f"Loading weights from {state_filename} to {device}")
    checkpoint = torch.load(state_filename, map_location=torch.device(device), weights_only=True)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    model = TOFinverse(
        nflow_in=param.nslice_to_use, 
        nfeature_out=1, 
        context_dim=32
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_data(spath, area_path, param):
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

def run_velocity_inference(model, s_data, xarea, area_scaled):
    t0 = time.time()
    velocity = utils.input_batched_signal_into_NN_area(s_data, model, xarea, area_scaled)
    logger.info(f"Inference complete in {time.time()-t0:.3f}s")
    return velocity

def run_forward_model(velocity_NN, xarea, area, param, ncpu=8, enable_logging=False):
    sp = param.scan_param
    v_up = utils.upsample(velocity_NN, velocity_NN.size * 100 + 1, sp.repetition_time).flatten()
    t = np.arange(0, sp.repetition_time * velocity_NN.size, sp.repetition_time / 100)
    
    t_base, v_base = utils.add_baseline_period(t, v_up, sp.repetition_time * sp.num_pulse_baseline_offset)
    
    pos_func = partial(pfl.compute_position_numeric_spatial, tr_vect=t_base, vts=v_base, xarea=xarea, area=area)
    
    if enable_logging:
        logger.info(f"Starting forward model simulation using {ncpu} cores...")
    t0 = time.time()
    s_sim_raw = tm.simulate_inflow(
        sp.repetition_time, sp.echo_time, velocity_NN.size + sp.num_pulse_baseline_offset,
        sp.slice_width, sp.flip_angle, sp.t1_time, sp.t2_time, sp.num_slice, 
        sp.alpha_list, sp.MBF, pos_func, ncpu=ncpu, varysliceprofile=True, dx=0.005, offset_fact=0, enable_logging=enable_logging
    )
    if enable_logging:
        logger.info(f"Simulation complete in {time.time()-t0:.2f}s")
    return s_sim_raw[sp.num_pulse_baseline_offset:, :param.nslice_to_use]

def save_and_plot(outdir, velocity, sraw_scaled, ssim_scaled, tr):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    np.savetxt(outdir / 'signal_data.txt', sraw_scaled)
    np.savetxt(outdir / 'velocity_predicted.txt', velocity)
    np.savetxt(outdir / 'signal_simulation.txt', ssim_scaled)

    fig, axes = plt.subplots(3, 2, figsize=(16, 10), constrained_layout=True)
    t_vec = tr * np.arange(velocity.size)
    n = velocity.size
    freqs = np.fft.rfftfreq(n, d=tr)
    
    def get_spectrum(signal):
        fft_res = np.fft.rfft(signal, axis=0)
        return np.abs(fft_res)
    
    sraw_view = sraw_scaled[:velocity.size]
    sraw_spec = get_spectrum(sraw_view)
    ssim_spec = get_spectrum(ssim_scaled)
    vel_spec = get_spectrum(velocity)

    # Scaling axes for comparison
    ylim_time = (min(np.min(sraw_view), np.min(ssim_scaled)) * 1.1, max(np.max(sraw_view), np.max(ssim_scaled)) * 1.1)
    ylim_freq = (max(1e-6, np.min(sraw_spec[1:])), np.max(sraw_spec[1:]) * 2.0)

    # Plotting logic remains same...
    axes[0, 0].plot(t_vec, sraw_view); axes[0, 0].set_title("Input Signal (Time Domain)")
    axes[0, 0].set_ylim(ylim_time)
    
    axes[0, 1].plot(freqs[1:], sraw_spec[1:]); axes[0, 1].set_title("Input Signal (Frequency Domain)")
    axes[0, 1].set_yscale('log'); axes[0, 1].set_ylim(ylim_freq)

    axes[1, 0].plot(t_vec, velocity, color='black', label='Velocity')
    axes[1, 0].axhline(np.mean(velocity), color='red', linestyle='--', label=f'Mean: {np.mean(velocity):.3f}')
    axes[1, 0].set_title("Inferred Velocity (Time Domain)"); axes[1, 0].legend()

    axes[1, 1].plot(freqs[1:], vel_spec[1:], color='black'); axes[1, 1].set_title("Inferred Velocity (Freq Domain)")
    axes[1, 1].set_yscale('log')

    axes[2, 0].plot(t_vec, ssim_scaled); axes[2, 0].set_title("Simulated Signal (Time Domain)")
    axes[2, 0].set_ylim(ylim_time)

    axes[2, 1].plot(freqs[1:], ssim_spec[1:]); axes[2, 1].set_title("Simulated Signal (Freq Domain)")
    axes[2, 1].set_yscale('log'); axes[2, 1].set_ylim(ylim_freq)

    save_path = outdir / "summary_plot.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Summary visualization saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Run velocity inference using TOF framework')
    parser.add_argument('--signal', type=str, required=True)
    parser.add_argument('--area', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ncpu', type=int, default=8)
    args = parser.parse_args()

    param = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info("--- Starting Inference Workflow ---")
    model = load_network(args.model, param, device)
    
    logger.info("Loading and scaling input data...")
    sraw, xarea, area = load_data(args.signal, args.area, param)
    sraw_scaled = utils.scale_data(sraw)
    area_scaled = utils.scale_area(xarea, area)

    logger.info("Running neural network inference...")
    velocity = run_velocity_inference(model, sraw_scaled, xarea, area_scaled)
    
    logger.info("Verifying via forward model...")
    ssim = run_forward_model(velocity, xarea, area, param, ncpu=args.ncpu)
    ssim_scaled = utils.scale_data(ssim)
    
    logger.info(f"Finalizing output at {args.outdir}")
    save_and_plot(args.outdir, velocity, sraw_scaled, ssim_scaled, param.scan_param.repetition_time)
    logger.info("Workflow completed successfully.")

if __name__ == "__main__":
    main()