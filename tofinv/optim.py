# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from omegaconf import OmegaConf
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_evaluations
from scipy.signal import find_peaks

import tofmodel.inverse.evaluation as eval
import tofmodel.inverse.utils as utils

# --- HELPER FUNCTIONS (Exposed for both Optimization and Aggregation) ---

def psd_channels(x, fs=1/0.378):
    X = np.fft.rfft(x, axis=0)
    return np.abs(X)

def transform(vbase, params, phase_input):
    v_scale, fg_end, mu, alpha = params
    v = vbase * v_scale
    V = np.fft.rfft(v)
    gain_curve = np.linspace(1.0, fg_end, len(V)) ** mu
    mag_norm = np.abs(V) / (np.max(np.abs(V)) + 1e-9)
    weight = mag_norm ** alpha
    weighted_gain = 1 + (gain_curve - 1) * weight
    V_scaled = V * weighted_gain
    return utils.define_velocity_fourier(V_scaled, 300, phase_input, 0)

def rfft_freqs_and_spectrum(x, demean=True):
    if demean: x = x - x.mean()
    freqs = np.fft.rfftfreq(len(x), d=0.378)
    mag = psd_channels(x)
    return freqs, mag

def compute_err(x1, x2, rms):
    """Multichannel Weighted Log-MAE: Peak detection per channel."""
    eps = 1e-12
    min_len = min(len(x1), len(x2))
    
    log_x1 = np.log(x1[:min_len, :] + eps)
    log_x2 = np.log(x2[:min_len, :] + eps)
    
    n_freqs = x1.shape[0]
    freq_weights = np.linspace(1, 1, n_freqs) ** 2 
    
    total_loss = 0
    for i in range(3):
        target_chan = x2[:min_len, i]
        peaks, _ = find_peaks(target_chan, prominence=np.mean(target_chan))
        
        weights = np.ones(min_len)
        weights[peaks] = 1.0
        
        diff = np.abs(log_x1[:, i] - log_x2[:, i])
        weighted_diff = diff * weights * freq_weights
        chan_loss = np.sum(weighted_diff)
        total_loss += chan_loss * (1 / (rms[i] + 1e-6))
        
    return total_loss

# --- CORE OPTIMIZATION LOGIC ---

def run_optimization(signal_path, area_path, config_path, outdir):
    param = OmegaConf.load(config_path)
    num_offset = param.scan_param.num_pulse_baseline_offset
    
    outdir = Path(outdir)
    plots_dir = outdir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    # Note: Use the area_path provided by args, but fallback logic can be added
    s_raw_full, xarea, area = eval.load_data(signal_path, area_path, param)
    
    window_size = 300
    nwindows = 1#s_raw_full.shape[0] // window_size
    all_v_bases, all_opt_params = [], []

    for i in range(nwindows):
        print(f"[*] Optimizing Window {i}...")
        s_raw = s_raw_full[i*window_size : (i+1)*window_size, :]
        v_base = (s_raw[:, 0] - s_raw[:, 0].mean()) / (s_raw[:, 0].max() + 1e-9)
        phase = np.angle(np.fft.rfft(v_base))

        s_target = eval.scale_data(s_raw[num_offset:, :])
        psd_measured = psd_channels(s_target)
        rms = np.sqrt(np.mean(psd_measured**2, axis=0))

        def objective(p):
            v_opt = transform(v_base, p, phase)
            ssim = eval.run_forward_model(v_opt, xarea, area, param)[num_offset:, :]
            psd_sim = psd_channels(eval.scale_data(ssim))
            return compute_err(psd_sim, psd_measured, rms) if not np.any(np.isnan(psd_sim)) else 1e6

        res = gp_minimize(objective, [(1e-4, 1.0), (1.0, 20.0), (0.5, 3.0), (1e-4, 2.0)], 
                          n_calls=20, n_initial_points=10, acq_func="gp_hedge")
        
        all_v_bases.append(v_base)
        all_opt_params.append(res.x)

        # ---------------- DIAGNOSTIC FIGURES ----------------
        # 1. Generate the optimized velocity and model prediction
        v_opt = transform(v_base, res.x, phase)
        
        # Run forward model (using variables available in loop scope)
        ssim_opt_full = eval.run_forward_model(v_opt, xarea, area, param)
        ssim_opt = ssim_opt_full[num_offset:, :]
        
        # Scale for visualization
        s_plot = eval.scale_data(s_target)
        ssim_opt_plot = eval.scale_data(ssim_opt)

        # 2. Time-Domain Figure
        fig_td, axes = plt.subplots(1, 4, figsize=(14, 3))
        axes[0].plot(v_base, color='gray')
        axes[0].set_title('Base-v (Input)')
        
        axes[1].plot(v_opt, color='black')
        axes[1].set_title('Opt-v (Transformed)')
        
        for ch in range(3):
            axes[2].plot(s_plot[:, ch], alpha=0.7)
        axes[2].set_title('Measured Signal (S)')
        
        for ch in range(3):
            axes[3].plot(ssim_opt_plot[:, ch], alpha=0.7)
        axes[3].set_title('Simulated (Opt)')
        
        for ax in axes: ax.set_xlabel('Frames')
        plt.tight_layout()
        fig_td.savefig(plots_dir / f'Results_TD_win{i}.png')

        # 3. Frequency-Domain Figure
        fig_fd, axes = plt.subplots(1, 4, figsize=(14, 3))
        freqs, m_base = rfft_freqs_and_spectrum(v_base)
        _, m_opt = rfft_freqs_and_spectrum(v_opt)
        
        axes[0].plot(freqs, m_base, color='gray')
        axes[0].set_title('Base PSD')
        
        axes[1].plot(freqs, m_opt, color='black')
        axes[1].set_title('Opt PSD')
        
        for ch in range(3):
            f, m = rfft_freqs_and_spectrum(s_plot[:, ch])
            axes[2].plot(f, m, alpha=0.7)
        axes[2].set_title('Measured PSD')
        
        for ch in range(3):
            f, m = rfft_freqs_and_spectrum(ssim_opt_plot[:, ch])
            axes[3].plot(f, m, alpha=0.7)
        axes[3].set_title('Simulated PSD')
        
        for ax in axes: ax.set_xlabel('Freq (Hz)')
        plt.tight_layout()
        fig_fd.savefig(plots_dir / f'Results_FD_win{i}.png')
        plt.close(fig_fd)
        
        # 4. Bayesian Optimization Plots
        # Convergence
        conv_ax = plot_convergence(res)
        conv_ax.set_title(f"Convergence (Window {i})")
        conv_ax.get_figure().savefig(plots_dir / f'convergence_win{i}.png')
        
        # Evaluations (Hyperparameter space search)
        eval_axes = plot_evaluations(res)
        # plot_evaluations returns an array of axes; we grab the figure from the first one
        eval_axes.get_figure().savefig(plots_dir / f'evaluations_win{i}.png')
        
        plt.close('all')

    # Final Save
    np.savetxt(outdir / 'base_velocity.txt', np.column_stack(all_v_bases))
    np.savetxt(outdir / 'optim_param.txt', np.array(all_opt_params))

# --- AGGREGATION LOGIC ---

def aggregate_optim(search_dir, outfile):
    search_path = Path(search_dir)
    amps = []
    
    # Find all param files and their corresponding base velocities
    for p_file in search_path.rglob("optim_param.txt"):
        v_file = p_file.parent / "base_velocity.txt"
        if not v_file.exists(): continue
        
        params_mat = np.atleast_2d(np.loadtxt(p_file))
        vels_mat = np.atleast_2d(np.loadtxt(v_file))
        if vels_mat.shape[0] > vels_mat.shape[1]: vels_mat = vels_mat.T

        for p, v in zip(params_mat, vels_mat):
            v_opt = transform(v, p, np.angle(np.fft.rfft(v)))
            _, mag = rfft_freqs_and_spectrum(v_opt)
            if mag.size == 151: amps.append(mag)

    X = np.array(amps)
    X = X[np.all(X >= 0, axis=1)]
    with open(outfile, "wb") as f:
        pickle.dump(X, f)
    print(f"[+] Aggregated {len(X)} samples to {outfile}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--signal', type=str)
    parser.add_argument('--area', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--outfile', type=str)
    args = parser.parse_args()

    if args.collect:
        aggregate_optim(args.outdir, args.outfile)
    else:
        run_optimization(args.signal, args.area, args.config, args.outdir)

if __name__ == "__main__":
    main()