# -*- coding: utf-8 -*-
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import center_of_mass
import argparse
import pickle

def add_boundary(A, depthfromslc1):
    # adds a wide area at each end of the vector to mimic the 
    # large CSF spaces above/below the 4th ventricle
    clip_min = 20
    area_clipped = np.clip(A, clip_min, None)

    # Ramp 1
    ramp_start, ramp_end, ramp_target = -50, -60, 120
    area_clipped = area_clipped.copy()
    mask_ramp = (depthfromslc1 <= ramp_start) & (depthfromslc1 >= ramp_end)
    area_clipped[mask_ramp] = clip_min + (ramp_target - clip_min) * (ramp_start - depthfromslc1[mask_ramp]) / (ramp_start - ramp_end)
    area_clipped[depthfromslc1 < ramp_end] = ramp_target

    # Ramp 2
    ramp_start, ramp_end, ramp_target = 50, 60, 120
    mask_ramp = (depthfromslc1 >= ramp_start) & (depthfromslc1 <= ramp_end)
    area_clipped[mask_ramp] = clip_min + (ramp_target - clip_min) * (ramp_start - depthfromslc1[mask_ramp]) / (ramp_start - ramp_end)
    area_clipped[depthfromslc1 > ramp_end] = ramp_target
    return area_clipped

def compute_area(func_path, anat_path, aseg_path, reg_path, output_path):
    func_img, anat_img, aseg_img = nib.load(func_path), nib.load(anat_path), nib.load(aseg_path)
    func_data, anat_data, aseg_data = func_img.get_fdata(), anat_img.get_fdata(), aseg_img.get_fdata()
    
    anatVOX2RAS = np.array([[-1, 0, 0, 0.5*aseg_data.shape[0]], [0, 0, 1, -0.5*aseg_data.shape[2]], [0, -1, 0, 0.5*aseg_data.shape[1]], [0, 0, 0, 1]])
    funcVOX2RAS = np.array([[-2.5, 0, 0, 1.25*func_data.shape[0]], [0, 0, 2.5, -1.25*func_data.shape[2]], [0, -2.5, 0, 1.25*func_data.shape[1]], [0, 0, 0, 1]])
    R = np.loadtxt(reg_path, skiprows=4, max_rows=4) 

    aseg_data[aseg_data != 15] = 0
    centroid_3d = center_of_mass(aseg_data)
    init_funcCRS = np.linalg.inv(funcVOX2RAS) @ R @ anatVOX2RAS @ np.array([*centroid_3d, 1]).T
    funcCRS0 = np.array([int(np.floor(init_funcCRS[0])), int(np.floor(init_funcCRS[1])), 0, 1])

    incr, lenx, leny = 0.2, 2.5, 2.5
    voxel_area = (incr * lenx) * (incr * leny)
    deltax = deltay = np.arange(-8, 8 + incr, incr)
    deltaz = np.arange(-25, 25 + incr, incr)
    
    area_contribution = np.zeros((deltax.size, deltay.size, deltaz.size))
    T1 = np.zeros((deltax.size, deltay.size, deltaz.size))

    # Grid search
    for xind, dx in enumerate(deltax):
        for yind, dy in enumerate(deltay):
            for zind, dz in enumerate(deltaz):
                ifuncCRS = funcCRS0 + np.array([dx, dy, dz, 0])
                anat_crs = np.linalg.inv(anatVOX2RAS) @ np.linalg.inv(R) @ funcVOX2RAS @ ifuncCRS
                crs_int = np.floor(anat_crs[:3]).astype(int)
                if np.all((crs_int >= 0) & (crs_int < aseg_data.shape)):
                    if aseg_data[crs_int[0], crs_int[1], crs_int[2]] == 15:
                        area_contribution[xind, yind, zind] = voxel_area
                        T1[xind, yind, zind] = anat_data[crs_int[0], crs_int[1], crs_int[2]]

    A = np.zeros(len(deltaz))
    for i in range(len(deltaz)):
        Aslice, T1slice = area_contribution[:,:,i], T1[:,:,i].copy()
        T1slice[T1slice == 0] = np.nan
        if not np.all(np.isnan(T1slice)):
            ref = np.nanmin(T1slice)
            area_scaled = Aslice * (ref / T1slice)
            A[i] = np.nansum(area_scaled)

    depthfromslc1 = deltaz * 2.5 
    A = add_boundary(A, depthfromslc1)
    
    # Save results (converting to cm/cm2)
    depth_cm, A_cm2 = depthfromslc1 * 0.1, A * 0.01
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "area.txt", np.column_stack([depth_cm, A_cm2]), fmt="%.6f")

    # Plot
    plt.figure()
    plt.plot(depth_cm, A_cm2, marker='.')
    plt.xlabel("Distance from 1st fMRI slice (cm)")
    plt.ylabel("Cross-sectional Area (cm²)")
    plt.title("4th ventricle area-depth profile")
    plt.savefig(out_dir / "area_vs_depth.png")
    plt.close()

# def aggregate_area(input_manifest, outdir):
#     area_col = Path(outdir) / "area_collection"
#     area_col.mkdir(parents=True, exist_ok=True)
    
#     with open(input_manifest, 'r') as f:
#         for line in f:
#             if not line.strip(): continue
#             file_path, sub, ses = line.strip().split('\t')
#             src = Path(file_path)
#             if src.exists():
#                 data = np.loadtxt(src) # Files are already scaled by compute_area
#                 np.savetxt(area_col / f"{sub}_area.txt", data, fmt="%.6f")
#                 print(f"[+] Collected: {sub}")

def aggregate_area(input_manifest, outdir):
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    
    xarea_all = []
    area_all = []
    
    with open(input_manifest, 'r') as f:
        for line in f:
            if not line.strip(): continue
            file_path, sub, ses = line.strip().split('\t')
            src = Path(file_path)
            
            if src.exists():
                data = np.loadtxt(src) 
                xarea_all.append(data[:, 0])
                area_all.append(data[:, 1])
                print(f"[+] Collected: {sub}")

    # Define the output pickle path
    pkl_path = outdir_path / "area_collection.pkl"
    
    # Save the variables as a tuple to match your loading logic
    with open(pkl_path, 'wb') as f:
        pickle.dump((xarea_all, area_all), f)
        
    print(f"\n[!] Successfully saved aggregated areas to: {pkl_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--func'); parser.add_argument('--anat')
    parser.add_argument('--aseg'); parser.add_argument('--reg')
    parser.add_argument('--outdir'); parser.add_argument('--input_manifest')
    args = parser.parse_args()

    if args.collect:
        aggregate_area(args.input_manifest, args.outdir)
    else:
        compute_area(args.func, args.anat, args.aseg, args.reg, args.outdir)

if __name__ == "__main__":
    main()
