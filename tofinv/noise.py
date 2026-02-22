import os
import argparse
import numpy as np
import nibabel as nib
import pickle
import surfa as sf
from pathlib import Path
import matplotlib.pyplot as plt

def extract_noise(func_file, synthseg_file, sbref_file, outdir, num_end_slices=2):
    """Extracts noise timeseries from the top slices of the CSF mask."""
    for f in [func_file, synthseg_file, sbref_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing input: {f}")
    
    image_synthseg = sf.load_volume(synthseg_file)
    image_target = sf.load_volume(sbref_file)
    resampled = image_synthseg.resample_like(image_target, method='nearest')
    
    mask = resampled == 15
    mask_sum_slicewise = np.sum(mask, axis=(0, 1))
    noise_slices = np.where(mask_sum_slicewise > 0)[0][-num_end_slices:]
    
    noise_mask = np.zeros_like(mask)
    noise_mask[:, :, noise_slices] = mask[:, :, noise_slices]
    
    func_data = nib.load(func_file).get_fdata()
    func_ts = func_data[noise_mask]
    
    mean_per_voxel = np.mean(func_ts, axis=1, keepdims=True)
    noise_signal_scaled = (func_ts - mean_per_voxel) / (mean_per_voxel + 1e-9)
    
    os.makedirs(outdir, exist_ok=True)
    out_file = os.path.join(outdir, "noise.txt")
    np.savetxt(out_file, noise_signal_scaled, fmt="%.6f")
    
    plt.figure(figsize=(10, 4))
    plt.plot(noise_signal_scaled.T, color='gray', alpha=0.3, lw=0.5)
    plt.plot(np.mean(noise_signal_scaled, axis=0), color='black', lw=2, label='Mean Noise Signal')
    plt.title(f"CSF Noise Signal (Top {num_end_slices} Slices)")
    plt.xlabel("Timepoints (TR)")
    plt.ylabel("fMRI signal scaled")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_file = os.path.join(outdir, "noise_plot.png")
    plt.savefig(plot_file, dpi=150)
    plt.close()
    
    print(f"[+] Saved noise signal {noise_signal_scaled.shape} to {out_file}")
    print(f"[+] Saved plot to {plot_file}")

def aggregate_noise(search_dir, outfile, desired_sample_length=300):
    """Finds all noise.txt files in a directory and pickles them into a bank."""
    search_path = Path(search_dir)
    files = list(search_path.rglob("noise.txt"))
    print(f"[*] Found {len(files)} noise files. Aggregating...")

    all_noise_reshaped = []
    for path in files:
        ts = np.loadtxt(path)
        if ts.ndim == 1:
            ts = ts[None, :]
        
        nvoxel, ntime = ts.shape
        nbatch = int(ntime / desired_sample_length)
        if nbatch == 0: continue
            
        ts_trimmed = ts[:, :nbatch * desired_sample_length]
        ts_rs = ts_trimmed.reshape((nbatch * nvoxel, desired_sample_length))
        all_noise_reshaped.append(ts_rs)

    combined = np.vstack(all_noise_reshaped)
    with open(outfile, 'wb') as f:
        pickle.dump(combined, f)
    print(f"[+] Combined noise array saved to {outfile}, shape: {combined.shape}")

def main():
    parser = argparse.ArgumentParser(description="tofinv noise module")
    parser.add_argument("--collect", action="store_true", help="Aggregate mode")
    
    parser.add_argument("--func", type=str)
    parser.add_argument("--synthseg", type=str)
    parser.add_argument("--sbref", type=str)
    
    parser.add_argument("--outdir", type=str, help="Search dir for collect, or output dir for extraction")
    parser.add_argument("--outfile", type=str, help="Path for the final .pkl bank")
    parser.add_argument("--desired_sample_length", type=int)
    
    args = parser.parse_args()

    if args.collect:
        if not args.outdir or not args.outfile or not args.desired_sample_length:
            raise ValueError("Aggregation requires --outdir, --outfile, and --desired_sample_length")
        aggregate_noise(args.outdir, args.outfile, args.desired_sample_length)
    else:
        if not args.func or not args.outdir:
            raise ValueError("Extraction requires --func and --outdir")
        extract_noise(args.func, args.synthseg, args.sbref, args.outdir)

if __name__ == "__main__":
    main()