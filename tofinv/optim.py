import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import logging
from pathlib import Path
from omegaconf import OmegaConf
from skopt import gp_minimize, dump
from skopt.plots import plot_convergence, plot_evaluations
import tofinv.evaluation as eval
import tofinv.utils as utils
import skopt
from scipy.signal.windows import tukey
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def transform(vbase, params, phase_input):
    v_scale, beta = params
    v = vbase * v_scale
    V = np.fft.rfft(v) 
    V_mag = np.abs(V)
    freq_norm = np.linspace(0, 1, len(V_mag))
    gain_curve = np.exp(beta * freq_norm)
    V_scaled = V_mag * gain_curve
    voffset = 0
    vout = utils.define_velocity_fourier(V_scaled, vbase.size, phase_input, voffset)
    return make_periodic(vout)

def rfft_freqs_and_spectrum(x, tr=0.378, demean=True):
    if demean: x = x - x.mean(axis=0)
    freqs = np.fft.rfftfreq(len(x), d=tr)
    mag = np.abs(np.fft.rfft(x, axis=0))
    return freqs, mag

def compute_err(x1, x2, rms):
    return float(np.sum(np.abs(x1 - x2) / rms))

def make_periodic(signal, alpha=0.1):
    dc_offset = np.mean(signal)
    ac_signal = signal - dc_offset
    window = tukey(len(signal), alpha=alpha)
    ac_windowed = ac_signal * window
    return ac_windowed + dc_offset

def run_optimization(signal_path, area_path, config_path, outdir):
    logger.info(f"Loading configuration from {config_path}")
    param = OmegaConf.load(config_path)
    raw_alpha = param.scan_param.alpha_list
    if isinstance(raw_alpha, str):
        logger.warning("Detected alpha_list parsed as string. Cleaning and ensuring positive values...")
        parts = raw_alpha.replace('--', '-').split()
        param.scan_param.alpha_list = [abs(float(p)) for p in parts]
    elif isinstance(raw_alpha, (list, getattr(OmegaConf, 'ListConfig', list))):
        param.scan_param.alpha_list = [
            abs(float(str(x).replace('--', '-'))) if isinstance(x, str) else abs(float(x))
            for x in raw_alpha
        ]
    optim_kwargs = OmegaConf.to_container(param.optim_gp_minimize, resolve=True)
    optim_kwargs["dimensions"] = [tuple(dim) for dim in optim_kwargs["dimensions"]]    
    optim_kwargs["random_state"] = param.global_seed
    min_num_windows = param.optim.get("min_num_windows", 3)
    power_factor = param.optim.get("power_factor", 2.0)
    v_base_multiplier = param.optim.get("v_base_multiplier", 0.5)
    nslice = param.nslice_to_use
    tr = param.scan_param.repetition_time
    outdir = Path(outdir)
    plots_dir = outdir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading input signal and area data...")
    s_raw_full, xarea, area = eval.load_data(signal_path, area_path, param)
    logger.info(f"Loaded signal shape: {s_raw_full.shape}")
    if param.synthetic.areamode == 'straight_tube':
        logger.warning("Using 'straight_tube' mode: overriding area profile with constant values.")
        xarea = np.linspace(-3, 3, param.scan_param.num_pulse)
        area = np.ones_like(xarea)
    window_size = param.scan_param.num_pulse
    nwindows = np.min([s_raw_full.shape[0] // window_size, min_num_windows]) 
    logger.info(f"Optimizing {nwindows} windows (Window Size: {window_size})")
    all_v_bases, all_opt_params = [], []
    for i in range(nwindows):
        logger.info(f"--- Starting Optimization: Window {i} ---")
        s_raw = s_raw_full[i*window_size : (i+1)*window_size, :]
        raw_sig = s_raw[:, 0] - np.mean(s_raw[:, 0])
        V_fft = np.fft.rfft(raw_sig)
        V_mag, V_phase = np.abs(V_fft), np.angle(V_fft)
        V_mag_smooth = gaussian_filter1d(V_mag, sigma=1.5)
        V_mag_selective = V_mag_smooth ** power_factor
        logger.info(f"Window {i}: Base velocity reconstructed using power-law attenuation (p={power_factor})")
        fig_dbg, ax_dbg = plt.subplots(figsize=(10, 5))
        plot_scale = np.max(V_mag_smooth) / (np.max(V_mag_selective) + 1e-9)
        ax_dbg.plot(V_mag, color='black', alpha=0.3, label='Original FFT Mag')
        ax_dbg.plot(V_mag_smooth, color='blue', alpha=0.4, linewidth=2, label='Standard Gaussian')
        ax_dbg.plot(V_mag_selective * plot_scale, color='red', linewidth=2, label=f'Power Law (x^{power_factor})')
        ax_dbg.set_title(f'Window {i} - Power Law Attenuation')
        ax_dbg.legend()
        ax_dbg.grid(True, alpha=0.3)
        fig_dbg.savefig(plots_dir / f'vbase_smoothing_win{i}.png')
        plt.close(fig_dbg)
        V_reconstructed = V_mag_selective * np.exp(1j * V_phase)
        v_base_time = np.fft.irfft(V_reconstructed, n=len(raw_sig))
        v_base = v_base_multiplier * v_base_time / (np.max(np.abs(v_base_time)) + 1e-9)
        v_base = make_periodic(v_base)
        phase = np.angle(np.fft.rfft(v_base))
        s_target = utils.scale_data(s_raw)
        psd_measured = np.abs(np.fft.rfft(s_target, axis=0))[1:, :]
        rms = np.sqrt(np.mean(psd_measured**2, axis=0))
        rms = rms / (np.sum(rms) + 1e-9)
        
        def objective(p):
            v_opt = transform(v_base, p, phase)
            ssim = eval.run_forward_model(v_opt, xarea, area, param, ncpu=-1, enable_logging=False)
            ssim_scaled = utils.scale_data(ssim)
            psd_sim = np.abs(np.fft.rfft(ssim_scaled, axis=0))
            if np.any(np.isnan(psd_sim)):
                return 1e6
            psd_sim = psd_sim[1:, :]
            return compute_err(psd_sim, psd_measured, rms)

        logger.info(f"Window {i}: Launching gp_minimize...")
        res = gp_minimize(objective, **optim_kwargs)
        logger.info(f"Window {i}: Best Params found: {res.x} | Best Loss: {res.fun:.4f}")
        all_v_bases.append(v_base)
        all_opt_params.append(res.x)
        if 'callbacks' in res.specs: del res.specs['callbacks']
        res.specs['args']['func'] = None 
        dump(res, outdir / f'optim_result_win{i}.pkl')
        logger.info(f"Window {i}: Generating diagnostic figures...")
        v_opt = transform(v_base, res.x, phase)
        ssim_opt = eval.run_forward_model(v_opt, xarea, area, param, ncpu=1)        
        ssim_opt_plot = utils.scale_data(ssim_opt)
        fig_td, axes = plt.subplots(2, 2, figsize=(8, 7), sharey='row')
        axes[0, 0].plot(v_base, color='gray')
        axes[0, 1].plot(v_opt, color='black')
        for ch in range(nslice):
            axes[1, 0].plot(s_target[:, ch], alpha=0.7)
            axes[1, 1].plot(ssim_opt_plot[:, ch], alpha=0.7)
        plt.tight_layout()
        fig_td.savefig(plots_dir / f'Results_TD_win{i}.png')
        plt.close(fig_td)
        conv_ax = plot_convergence(res)
        conv_ax.get_figure().savefig(plots_dir / f'convergence_win{i}.png')
        eval_ax = plot_evaluations(res)
        eval_ax.get_figure().savefig(plots_dir / f'evaluation_win{i}.png')
        plt.close('all')
        
    np.savetxt(outdir / 'base_velocity.txt', np.column_stack(all_v_bases))
    np.savetxt(outdir / 'optim_param.txt', np.array(all_opt_params))
    logger.info(f"Saved optimized velocities and parameters to {outdir}")

def aggregate_optim(search_dir, outfile):
    logger.info(f"Aggregating optimization results from {search_dir}")
    search_path = Path(search_dir)
    data_triplets = [] 
    result_files = sorted(list(search_path.rglob("optim_result_win*.pkl")))
    logger.info(f"Found {len(result_files)} result files.")
    for res_file in result_files:
        try:
            parts = res_file.parts
            sub_idx = parts.index("subjects") if "subjects" in parts else -1
            subject_name = parts[sub_idx + 1] if sub_idx != -1 else res_file.parents[3].name 
            res = skopt.load(res_file)
            v_base_path = res_file.parent / "base_velocity.txt"
            if v_base_path.exists():
                v_bases = np.loadtxt(v_base_path)
                win_idx = int(res_file.stem.split('win')[-1])
                v_base = v_bases[:, win_idx] if v_bases.ndim > 1 else v_bases
                data_triplets.append((v_base, res.x, subject_name))
            else:
                logger.warning(f"Base velocity file missing for {res_file}")
        except Exception as e:
            logger.error(f"Error processing {res_file.name}: {e}")
    with open(outfile, "wb") as f:
        pickle.dump(data_triplets, f)
    logger.info(f"[+] Aggregated {len(data_triplets)} triplets to {outfile}")

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