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
        sp.alpha_list, sp.MBF, pos_func, ncpu=ncpu, varysliceprofile=False, dx=0.005, offset_fact=0, enable_logging=True
    )
    return s_sim_raw[sp.num_pulse_baseline_offset:, :param.nslice_to_use]

# --- IO & Visualization ---

def save_and_plot(outdir, velocity, sraw_scaled, ssim_scaled, tr):
    """
    Saves data to text files and generates a 3x2 summary plot:
    - Left Column: Time Domain (Input, Velocity, Simulated) - Matched Scales for Signals
    - Right Column: Frequency Domain (Input, Velocity, Simulated) - Matched Scales for Signals, DC removed
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Save Raw Data ---
    np.savetxt(outdir / 'signal_data.txt', sraw_scaled)
    np.savetxt(outdir / 'velocity_predicted.txt', velocity)
    np.savetxt(outdir / 'signal_simulation.txt', ssim_scaled)

    # --- 2. Setup Plot Layout ---
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), constrained_layout=True)
    
    # Time vector
    t_vec = tr * np.arange(velocity.size)
    
    # Frequency vector (one-sided FFT)
    n = velocity.size
    freqs = np.fft.rfftfreq(n, d=tr)
    
    # Helper to compute Magnitude Spectrum
    def get_spectrum(signal):
        fft_res = np.fft.rfft(signal, axis=0)
        return np.abs(fft_res)

    # --- 3. Compute Spectra First (needed for scaling) ---
    # Slice sraw to match velocity length
    sraw_view = sraw_scaled[:velocity.size]
    
    sraw_spec = get_spectrum(sraw_view)
    ssim_spec = get_spectrum(ssim_scaled)
    vel_spec = get_spectrum(velocity)

    # --- 4. Calculate Shared Limits (Time Domain) ---
    t_min = min(np.min(sraw_view), np.min(ssim_scaled))
    t_max = max(np.max(sraw_view), np.max(ssim_scaled))
    t_range = t_max - t_min if t_max != t_min else 1.0
    ylim_time = (t_min - 0.05 * t_range, t_max + 0.05 * t_range)

    # --- 5. Calculate Shared Limits (Freq Domain) ---
    # Important: Ignore index 0 (DC component) for limit calculation
    f_min = min(np.min(sraw_spec[1:]), np.min(ssim_spec[1:]))
    f_max = max(np.max(sraw_spec[1:]), np.max(ssim_spec[1:]))
    
    # Since we are in log space, ensure min is > 0 (it should be for Magnitude)
    # If 0 exists, clip to a small epsilon or the second smallest value
    if f_min <= 0: f_min = 1e-10 
    
    # Log-space padding (visual padding)
    # We can just let matplotlib handle the log autoscaling if we set the same limits,
    # but explicit limits ensure exact matching.
    ylim_freq = (f_min * 0.5, f_max * 2.0) # Multiply/Divide for log padding

    # ==========================================
    # ROW 0: Input Signal
    # ==========================================
    # Time Domain
    axes[0, 0].plot(t_vec, sraw_view)
    axes[0, 0].set_title("Input Signal (Time Domain)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_ylim(ylim_time)  # <--- Matched Time Scale
    axes[0, 0].grid(True, alpha=0.3)

    # Frequency Domain (Skipping DC)
    axes[0, 1].plot(freqs[1:], sraw_spec[1:])
    axes[0, 1].set_title("Input Signal (Frequency Domain)")
    axes[0, 1].set_ylabel("Magnitude")
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_ylim(ylim_freq)  # <--- Matched Freq Scale
    axes[0, 1].grid(True, alpha=0.3, which='both')

    # ==========================================
    # ROW 1: Inferred Velocity
    # ==========================================
    # Time Domain
    axes[1, 0].plot(t_vec, velocity, color='black', linewidth=1, label='Velocity')
    mean_vel = np.mean(velocity)
    axes[1, 0].axhline(mean_vel, color='red', linestyle='--', linewidth=1.5, 
                       label=f'Mean: {mean_vel:.3f}')
    axes[1, 0].legend(loc='upper right', framealpha=0.9)
    axes[1, 0].set_title("Inferred Velocity (Time Domain)")
    axes[1, 0].set_ylabel("Velocity") 
    axes[1, 0].grid(True, alpha=0.3)

    # Frequency Domain (Skipping DC) - Velocity has its own scale
    axes[1, 1].plot(freqs[1:], vel_spec[1:], color='black')
    axes[1, 1].set_title("Inferred Velocity (Frequency Domain)")
    axes[1, 1].set_ylabel("Magnitude")
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3, which='both')

    # ==========================================
    # ROW 2: Simulated Signal
    # ==========================================
    # Time Domain
    axes[2, 0].plot(t_vec, ssim_scaled)
    axes[2, 0].set_title("Simulated Signal (Time Domain)")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("Amplitude")
    axes[2, 0].set_ylim(ylim_time)  # <--- Matched Time Scale
    axes[2, 0].grid(True, alpha=0.3)

    # Frequency Domain (Skipping DC)
    axes[2, 1].plot(freqs[1:], ssim_spec[1:])
    axes[2, 1].set_title("Simulated Signal (Frequency Domain)")
    axes[2, 1].set_xlabel("Frequency (Hz)")
    axes[2, 1].set_ylabel("Magnitude")
    axes[2, 1].set_yscale('log')
    axes[2, 1].set_ylim(ylim_freq)  # <--- Matched Freq Scale
    axes[2, 1].grid(True, alpha=0.3, which='both')

    # --- Save ---
    save_path = outdir / "summary_plot.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {save_path}")
    
    
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
    sraw_scaled = utils.scale_data(sraw)

    # 3. Inference
    logger.info("Running inference...")
    velocity = run_velocity_inference(model, sraw_scaled, xarea, area)
    
    # 4. Simulation Verification
    logger.info("Running forward model...")
    ssim = run_forward_model(velocity, xarea, area, param, ncpu=args.ncpu)
    
    # 5. Normalization for Comparison
    ssim_scaled = utils.scale_data(ssim)

    # 6. Save
    logger.info(f"Saving results to {args.outdir}")
    save_and_plot(args.outdir, velocity, sraw_scaled, ssim_scaled, param.scan_param.repetition_time)

if __name__ == "__main__":
    main()