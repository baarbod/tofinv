import os
import argparse
import numpy as np
import nibabel as nib
import pickle
import surfa as sf
from pathlib import Path
import matplotlib.pyplot as plt
import logging

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def extract_noise(func_file, synthseg_file, sbref_file, outdir, num_end_slices=2):
    """Extracts noise timeseries from the top slices of the CSF mask."""
    logger.info(f"Starting noise extraction for: {func_file}")
    
    for f in [func_file, synthseg_file, sbref_file]:
        if not os.path.exists(f):
            logger.error(f"Missing input file: {f}")
            raise FileNotFoundError(f"Missing input: {f}")
    
    logger.info("Resampling synthseg to SBRef space...")
    image_synthseg = sf.load_volume(synthseg_file)
    image_target = sf.load_volume(sbref_file)
    resampled = image_synthseg.resample_like(image_target, method='nearest')
    
    mask = resampled == 15
    mask_sum_slicewise = np.sum(mask, axis=(0, 1))
    noise_slices = np.where(mask_sum_slicewise > 0)[0][-num_end_slices:]
    
    if len(noise_slices) == 0:
        logger.error("No CSF (Label 15) voxels found in the volume mask.")
        raise ValueError("Empty noise mask.")

    logger.info(f"Extracting noise from slices: {noise_slices}")
    noise_mask = np.zeros_like(mask)
    noise_mask[:, :, noise_slices] = mask[:, :, noise_slices]
    
    func_data = nib.load(func_file).get_fdata()
    func_ts = func_data[noise_mask]
    
    mean_per_voxel = np.mean(func_ts, axis=1, keepdims=True)
    noise_signal_scaled = (func_ts - mean_per_voxel) / (mean_per_voxel + 1e-9)
    
    os.makedirs(outdir, exist_ok=True)
    out_file = os.path.join(outdir, "noise.txt")
    np.savetxt(out_file, noise_signal_scaled, fmt="%.6f")
    
    # Plotting
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
    
    logger.info(f"Saved noise signal {noise_signal_scaled.shape} to {out_file}")
    logger.info(f"Saved plot to {plot_file}")

def aggregate_noise(search_dir, outfile, desired_sample_length=300):
    """Finds all noise.txt files in a directory and pickles them into a bank."""
    search_path = Path(search_dir)
    files = list(search_path.rglob("noise.txt"))
    logger.info(f"Found {len(files)} noise files in {search_dir}. Aggregating...")

    all_noise_reshaped = []
    for path in files:
        ts = np.loadtxt(path)
        if ts.ndim == 1:
            ts = ts[None, :]
        
        nvoxel, ntime = ts.shape
        nbatch = int(ntime / desired_sample_length)
        
        if nbatch == 0:
            logger.warning(f"Skipping {path}: timeseries length {ntime} < {desired_sample_length}")
            continue
            
        ts_trimmed = ts[:, :nbatch * desired_sample_length]
        ts_rs = ts_trimmed.reshape((nbatch * nvoxel, desired_sample_length))
        all_noise_reshaped.append(ts_rs)
        logger.info(f"Processed {path.parent.name}: added {nbatch * nvoxel} samples.")

    if not all_noise_reshaped:
        logger.error("No valid noise samples found for aggregation!")
        return

    combined = np.vstack(all_noise_reshaped)
    with open(outfile, 'wb') as f:
        pickle.dump(combined, f)
    logger.info(f"[+] Combined noise array saved to {outfile}, final shape: {combined.shape}")

def main():
    parser = argparse.ArgumentParser(description="tofinv noise module")
    parser.add_argument("--collect", action="store_true", help="Aggregate mode")
    parser.add_argument("--func", type=str)
    parser.add_argument("--synthseg", type=str)
    parser.add_argument("--sbref", type=str)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--outfile", type=str)
    parser.add_argument("--desired_sample_length", type=int)
    
    args = parser.parse_args()

    if args.collect:
        if not args.outdir or not args.outfile or not args.desired_sample_length:
            logger.error("Aggregation requires --outdir, --outfile, and --desired_sample_length")
            raise ValueError("Missing arguments for --collect")
        aggregate_noise(args.outdir, args.outfile, args.desired_sample_length)
    else:
        if not args.func or not args.outdir:
            logger.error("Extraction requires --func and --outdir")
            raise ValueError("Missing arguments for extraction")
        extract_noise(args.func, args.synthseg, args.sbref, args.outdir)

if __name__ == "__main__":
    main()