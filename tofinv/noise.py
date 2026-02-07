# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import nibabel as nib
import pickle
import surfa as sf
from pathlib import Path

def extract_noise(func_file, synthseg_file, sbref_file, outdir, num_end_slices=2):
    """Extracts noise timeseries from the top slices of the CSF mask."""
    for f in [func_file, synthseg_file, sbref_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing input: {f}")
    
    # Load and resample mask
    image_synthseg = sf.load_volume(synthseg_file)
    image_target = sf.load_volume(sbref_file)
    resampled = image_synthseg.resample_like(image_target, method='nearest')
    
    # Define noise mask (top N slices of CSF label 15)
    mask = resampled == 15
    mask_sum_slicewise = np.sum(mask, axis=(0, 1))
    noise_slices = np.where(mask_sum_slicewise > 0)[0][-num_end_slices:]
    
    noise_mask = np.zeros_like(mask)
    noise_mask[:, :, noise_slices] = mask[:, :, noise_slices]
    
    # Load functional data and compute PSC
    func_data = nib.load(func_file).get_fdata()
    func_ts = func_data[noise_mask]
    
    # extract reference baseline value
    baseline_ref = np.expand_dims(np.mean(func_ts), axis=0)
    
    mean_per_voxel = np.mean(func_ts, axis=1, keepdims=True)
    psc = (func_ts - mean_per_voxel) / (mean_per_voxel + 1e-9) * 100
    
    os.makedirs(outdir, exist_ok=True)
    out_file = os.path.join(outdir, "noise.txt")
    np.savetxt(out_file, psc, fmt="%.6f")
    print(f"[+] Saved noise signal {psc.shape} to {out_file}")
    
    out_file = os.path.join(outdir, "baseline_ref.txt")
    np.savetxt(out_file, baseline_ref, fmt="%.6f")
    print(f"[+] Saved baseline reference signal to {out_file}")

def aggregate_noise(search_dir, outfile, desired_sample_length=300):
    """Finds all noise.txt files in a directory and pickles them into a bank."""
    search_path = Path(search_dir)
    files = list(search_path.rglob("noise.txt"))
    print(f"[*] Found {len(files)} noise files. Aggregating...")

    all_psc_reshaped = []
    for path in files:
        ts = np.loadtxt(path)
        if ts.ndim == 1:
            ts = ts[None, :]
        
        nvoxel, ntime = ts.shape
        nbatch = int(ntime / desired_sample_length)
        if nbatch == 0: continue
            
        ts_trimmed = ts[:, :nbatch * desired_sample_length]
        ts_rs = ts_trimmed.reshape((nbatch * nvoxel, desired_sample_length))
        all_psc_reshaped.append(ts_rs)

    combined = np.vstack(all_psc_reshaped)
    with open(outfile, 'wb') as f:
        pickle.dump(combined, f)
    print(f"[+] Combined noise array saved to {outfile}, shape: {combined.shape}")

def main():
    parser = argparse.ArgumentParser(description="tofinv noise module")
    parser.add_argument("--collect", action="store_true", help="Aggregate mode")
    
    # Extraction Args
    parser.add_argument("--func", type=str)
    parser.add_argument("--synthseg", type=str)
    parser.add_argument("--sbref", type=str)
    
    # Common/Agg Args
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