import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from omegaconf import OmegaConf
from skopt import gp_minimize, dump
from skopt.plots import plot_convergence, plot_evaluations
import tofinv.evaluation as eval
import tofinv.utils as utils
import skopt
from scipy.signal.windows import tukey
from scipy.signal import find_peaks # Added for adaptive peak detection

def transform(vbase, params, phase_input):
    v_scale, beta, voffset = params
    voffset = 0
    v = vbase * v_scale
    V = np.fft.rfft(v) 
    V_mag = np.abs(V)
    freq_norm = np.linspace(0, 1, len(V_mag))
    gain_curve = np.exp(beta * freq_norm)
    weighted_gain = 1 + (gain_curve - 1)
    V_scaled = V_mag * weighted_gain
    vout = utils.define_velocity_fourier(V_scaled, vbase.size, phase_input, voffset)
    return make_periodic(vout)

def rfft_freqs_and_spectrum(x, tr=0.378, demean=True):
    if demean: x = x - x.mean(axis=0)
    freqs = np.fft.rfftfreq(len(x), d=tr)
    mag = np.abs(np.fft.rfft(x, axis=0))
    return freqs, mag

def compute_err(x1, x2, rms):
    eps = 1e-12  
    nslice = len(rms)
    log_x1 = np.log(x1 + eps)
    log_x2 = np.log(x2 + eps)
    total_loss = 0
    for i in range(nslice):
        diff = np.abs(log_x1[:, i] - log_x2[:, i])
        chan_loss = np.sum(diff)
        total_loss += chan_loss
    return total_loss

def make_periodic(signal, alpha=0.1):
    dc_offset = np.mean(signal)
    ac_signal = signal - dc_offset
    window = tukey(len(signal), alpha=alpha)
    ac_windowed = ac_signal * window
    return ac_windowed + dc_offset

def run_optimization(signal_path, area_path, config_path, outdir):
    param = OmegaConf.load(config_path)

    optim_kwargs = OmegaConf.to_container(param.optim, resolve=True)
    optim_kwargs["dimensions"] = [tuple(dim) for dim in optim_kwargs["dimensions"]]    
    
    nslice = param.nslice_to_use
    tr = param.scan_param.repetition_time
    
    outdir = Path(outdir)
    plots_dir = outdir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    s_raw_full, xarea, area = eval.load_data(signal_path, area_path, param)
    
    if param.synthetic.areamode == 'straight_tube':
        xarea = np.linspace(-3, 3, param.scan_param.num_pulse)
        area = np.ones_like(xarea)
    
    window_size = param.scan_param.num_pulse
    nwindows = np.min([s_raw_full.shape[0] // window_size, 3]) 
    all_v_bases, all_opt_params = [], []
    from scipy.ndimage import gaussian_filter1d
    for i in range(nwindows):
        print(f"[*] Optimizing Window {i}...")
        s_raw = s_raw_full[i*window_size : (i+1)*window_size, :]
        
        raw_sig = s_raw[:, 0] - np.mean(s_raw[:, 0])
        V_fft = np.fft.rfft(raw_sig)
        V_mag, V_phase = np.abs(V_fft), np.angle(V_fft)
        V_mag_smooth = gaussian_filter1d(V_mag, sigma=1.5)
        power_factor = 3.0 
        V_mag_selective = V_mag_smooth ** power_factor
        
        fig_dbg, ax_dbg = plt.subplots(figsize=(10, 5))
        plot_scale = np.max(V_mag_smooth) / np.max(V_mag_selective)
        ax_dbg.plot(V_mag, color='black', alpha=0.3, label='Original FFT Mag')
        ax_dbg.plot(V_mag_smooth, color='blue', alpha=0.4, linewidth=2, label='Standard Gaussian')
        ax_dbg.plot(V_mag_selective * plot_scale, color='red', linewidth=2, label=f'Power Law (x^{power_factor})')
        
        ax_dbg.set_title(f'Window {i} - Power Law Attenuation (Method 1)')
        ax_dbg.legend()
        ax_dbg.grid(True, alpha=0.3)
        
        debug_plot_path = plots_dir / f'vbase_smoothing_win{i}.png'
        fig_dbg.savefig(debug_plot_path)
        plt.close(fig_dbg)
        
        V_reconstructed = V_mag_selective * np.exp(1j * V_phase)
        v_base_time = np.fft.irfft(V_reconstructed, n=len(raw_sig))
        v_base = v_base_time / (np.max(np.abs(v_base_time)) + 1e-9)
        v_base = make_periodic(v_base)
        phase = np.angle(np.fft.rfft(v_base))
        
        s_target = utils.scale_data(s_raw)
        
        psd_measured = np.abs(np.fft.rfft(s_target, axis=0))
        psd_measured = psd_measured[1:, :]
        
        rms = np.sqrt(np.mean(psd_measured**2, axis=0))
        rms = rms / np.sum(rms)
        
        def objective(p):
            v_opt = transform(v_base, p, phase)
            
            ssim = eval.run_forward_model(v_opt, xarea, area, param, ncpu=-1)
            ssim_scaled = utils.scale_data(ssim)
            psd_sim = np.abs(np.fft.rfft(ssim_scaled, axis=0))
            if np.any(np.isnan(psd_sim)):
                return 1e6
            
            psd_sim = psd_sim[1:, :]
            freq_err = compute_err(psd_sim, psd_measured, rms)
            return freq_err

        res = gp_minimize(objective, **optim_kwargs)

        all_v_bases.append(v_base)
        all_opt_params.append(res.x)

        if 'callbacks' in res.specs:
            del res.specs['callbacks']
        res.specs['args']['func'] = None 

        dump(res, outdir / f'optim_result_win{i}.pkl')
        
        # ---------------- DIAGNOSTIC FIGURES ----------------
        v_opt = transform(v_base, res.x, phase)
        
        ssim_opt = eval.run_forward_model(v_opt, xarea, area, param, ncpu=24)        
        ssim_opt_plot = utils.scale_data(ssim_opt)
        
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
        
        axes[0, 0].plot(freqs[1:], m_base[1:], color='gray')
        axes[0, 0].set_title('Base PSD')
        
        axes[0, 1].plot(freqs[1:], m_opt[1:], color='black')
        axes[0, 1].set_title('Opt PSD')
        
        for ch in range(nslice):
            f, m = rfft_freqs_and_spectrum(s_target[:, ch], tr=tr)
            axes[1, 0].plot(f[1:], m[1:], alpha=0.7)
        axes[1, 0].set_title('Measured PSD')
        
        for ch in range(nslice):
            f, m = rfft_freqs_and_spectrum(ssim_opt_plot[:, ch], tr=tr)
            axes[1, 1].plot(f[1:], m[1:], alpha=0.7)
        axes[1, 1].set_title('Simulated PSD')
        
        for ax in axes.flatten(): ax.set_xlabel('Freq (Hz)')
        plt.tight_layout()
        fig_fd.savefig(plots_dir / f'Results_FD_win{i}.png')
        plt.close(fig_fd)
        
        # 4. Bayesian Optimization Plots
        conv_ax = plot_convergence(res)
        conv_ax.set_title(f"Convergence (Window {i})")
        conv_ax.get_figure().savefig(plots_dir / f'convergence_win{i}.png')
        
        eval_axes = plot_evaluations(res)
        eval_axes.get_figure().savefig(plots_dir / f'evaluations_win{i}.png')
        
        plt.close('all')

        np.savetxt(outdir / 'base_velocity.txt', np.column_stack(all_v_bases))
        np.savetxt(outdir / 'optim_param.txt', np.array(all_opt_params))

# --- AGGREGATION LOGIC ---
def aggregate_optim(search_dir, outfile):
    search_path = Path(search_dir)
    data_triplets = [] 
    result_files = sorted(list(search_path.rglob("optim_result_win*.pkl")))
    
    for res_file in result_files:
        parts = res_file.parts
        try:
            sub_idx = parts.index("subjects")
            subject_name = parts[sub_idx + 1]
        except (ValueError, IndexError):
            subject_name = res_file.parents[3].name if len(res_file.parents) > 3 else "unknown"

        res = skopt.load(res_file)
        
        v_base_path = res_file.parent / "base_velocity.txt"
        
        if v_base_path.exists():
            v_bases = np.loadtxt(v_base_path)
            try:
                win_idx = int(res_file.stem.split('win')[-1])
                if v_bases.ndim > 1:
                    v_base = v_bases[:, win_idx]
                else:
                    v_base = v_bases
                
                data_triplets.append((v_base, res.x, subject_name))
                
            except Exception as e:
                print(f"Error matching {res_file.name} to v_base: {e}")

    with open(outfile, "wb") as f:
        pickle.dump(data_triplets, f)
        
    print(f"[+] Aggregated {len(data_triplets)} (v_base, res, subject) triplets to {outfile}")

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