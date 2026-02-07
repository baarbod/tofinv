import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from omegaconf import OmegaConf
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_evaluations
import tofinv.evaluation as eval
import tofinv.utils as utils

# --- HELPER FUNCTIONS ---

def transform(vbase, params, phase_input):
    v_scale, fg_end, mu, alpha, voffset = params
    v = vbase * v_scale
    V = np.abs(np.fft.rfft(v))
    gain_curve = np.linspace(1.0, fg_end, len(V)) ** mu
    mag_norm = V / (np.max(V) + 1e-9)
    weight = mag_norm ** alpha
    weighted_gain = 1 + (gain_curve - 1) * weight
    V_scaled = V * weighted_gain
    return utils.define_velocity_fourier(V_scaled, vbase.size, phase_input, voffset)

def rfft_freqs_and_spectrum(x, tr=0.378, demean=True):
    if demean: x = x - x.mean()
    freqs = np.fft.rfftfreq(len(x), d=tr)
    mag = np.abs(np.fft.rfft(x, axis=0))
    return freqs, mag

def compute_err(x1, x2, rms):
    eps = 1e-12
    min_len = min(len(x1), len(x2))
    
    log_x1 = np.log(x1[:min_len, :] + eps)
    log_x2 = np.log(x2[:min_len, :] + eps)
    
    total_loss = 0
    for i in range(3):
        diff = np.abs(log_x1[:, i] - log_x2[:, i])
        chan_loss = np.sum(diff)
        total_loss += chan_loss * (1 / (rms[i] + 1e-6))
        
    return total_loss

def get_steady_state_constant(sp):
    """Computes the theoretical steady state signal (mT_ss)."""
    fa_rad = np.radians(sp.flip_angle)
    exp_tr_t1 = np.exp(-sp.repetition_time / sp.t1_time)
    exp_te_t2 = np.exp(-sp.echo_time / sp.t2_time)
    return np.sin(fa_rad) * exp_te_t2 * (1 - exp_tr_t1) / (1 - exp_tr_t1 * np.cos(fa_rad))

# --- CORE OPTIMIZATION LOGIC ---

def run_optimization(signal_path, area_path, config_path, baseline_path, outdir):
    param = OmegaConf.load(config_path)
    num_offset = param.scan_param.num_pulse_baseline_offset
    nslice = param.nslice_to_use
    
    # steady state magnetization constant 
    mT_ss = get_steady_state_constant(param.scan_param)
    tr = param.scan_param.repetition_time
    
    outdir = Path(outdir)
    plots_dir = outdir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    s_raw_full, xarea, area = eval.load_data(signal_path, area_path, param)
    
    window_size = param.synthetic.num_time_point
    nwindows = np.min([s_raw_full.shape[0] // window_size, 3]) 
    all_v_bases, all_opt_params = [], []

    # load baseline_ref
    baseline_ref = np.loadtxt(baseline_path)
    
    for i in range(nwindows):
        print(f"[*] Optimizing Window {i}...")
        s_raw = s_raw_full[i*window_size : (i+1)*window_size, :]
        v_base = (s_raw[:, 0] - s_raw[:, 0].mean()) / (s_raw[:, 0].max() + 1e-9)
        phase = np.angle(np.fft.rfft(v_base))
        
        s_target = s_raw / baseline_ref
        
        psd_measured = np.abs(np.fft.rfft(s_target, axis=0))
        rms = np.sqrt(np.mean(psd_measured**2, axis=0))

        mean_measured = np.mean(s_target, axis=0)
        
        def objective(p):
            v_opt = transform(v_base, p, phase)
            ssim = eval.run_forward_model(v_opt, xarea, area, param)[num_offset:, :]
            ssim_scaled = ssim / mT_ss
            
            # --- Frequency Domain Error ---
            psd_sim = np.abs(np.fft.rfft(ssim_scaled, axis=0))
            if np.any(np.isnan(psd_sim)):
                return 1e6
            
            freq_err = compute_err(psd_sim, psd_measured, rms)
            
            # --- Mean Difference Error (Time Domain) ---
            mu_sim = np.mean(ssim_scaled, axis=0)
            mean_err = np.sum(np.abs(mu_sim - mean_measured))
            
            # --- Combined Loss ---
            return freq_err + (1.0 * mean_err)

        # NEED TO FIX HARD CODED BOUNDS
        res = gp_minimize(objective, [(1e-7, 2.0), (1.0, 2.0), (0.25, 14.0), (1e-7, 3.0), (-0.3, 0.3)], 
                          n_calls=60, n_initial_points=30, acq_func="gp_hedge", initial_point_generator='lhs')
        all_v_bases.append(v_base)
        all_opt_params.append(res.x)

        # ---------------- DIAGNOSTIC FIGURES ----------------
        # Generate the optimized velocity and model prediction
        v_opt = transform(v_base, res.x, phase)
        
        # Run forward model
        ssim_opt_full = eval.run_forward_model(v_opt, xarea, area, param)
        ssim_opt = ssim_opt_full[num_offset:, :]
        ssim_opt_plot = ssim_opt / mT_ss
        
        # 2. Time-Domain Figure
        fig_td, axes = plt.subplots(2, 2, figsize=(8, 7), sharey='row')
        axes[0, 0].plot(v_base, color='gray')
        axes[0, 0].set_title('Base-v (Input)')
        
        axes[0, 1].plot(v_opt, color='black')
        axes[0, 1].set_title('Opt-v (Transformed)')
        
        for ch in range(nslice):
            axes[1, 0].plot(s_target[:, ch], alpha=0.7)
        axes[1, 0].set_title('Measured Signal (S)')
        
        for ch in range(nslice):
            axes[1, 1].plot(ssim_opt_plot[:, ch], alpha=0.7)
        axes[1, 1].set_title('Simulated (Opt)')
        
        for ax in axes.flatten(): ax.set_xlabel('Frames')
        plt.tight_layout()
        fig_td.savefig(plots_dir / f'Results_TD_win{i}.png')
        plt.close(fig_td)
        
        # 3. Frequency-Domain Figure
        fig_fd, axes = plt.subplots(2, 2, figsize=(8, 7), sharey='row')
        freqs, m_base = rfft_freqs_and_spectrum(v_base, tr=tr)
        _, m_opt = rfft_freqs_and_spectrum(v_opt, tr=tr)
        
        axes[0, 0].plot(freqs, m_base, color='gray')
        axes[0, 0].set_title('Base PSD')
        
        axes[0, 1].plot(freqs, m_opt, color='black')
        axes[0, 1].set_title('Opt PSD')
        
        for ch in range(nslice):
            f, m = rfft_freqs_and_spectrum(s_target[:, ch], tr=tr)
            axes[1, 0].plot(f, m, alpha=0.7)
        axes[1, 0].set_title('Measured PSD')
        
        for ch in range(nslice):
            f, m = rfft_freqs_and_spectrum(ssim_opt_plot[:, ch], tr=tr)
            axes[1, 1].plot(f, m, alpha=0.7)
        axes[1, 1].set_title('Simulated PSD')
        
        for ax in axes.flatten(): ax.set_xlabel('Freq (Hz)')
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
            mag = np.fft.rfft(v_opt, axis=0)
            if mag.size == 151: amps.append(mag) # HARDCODED, NEED TO FIX

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
    parser.add_argument('--baseline', type=str)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--outfile', type=str)
    args = parser.parse_args()

    if args.collect:
        aggregate_optim(args.outdir, args.outfile)
    else:
        run_optimization(args.signal, args.area, args.config, args.baseline, args.outdir)

if __name__ == "__main__":
    main()