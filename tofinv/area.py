# -*- coding: utf-8 -*-
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from pathlib import Path
from scipy.ndimage import center_of_mass
import argparse
import pickle
import logging

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def add_boundary(A, depthfromslc1):
    logger.info("Applying artificial ramp boundaries to area vector to mimic CSF spaces around the 4th ventricle.")
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

def create_slice_montage(anat_slices, mask_slices, aseg_slices, depth_values, output_path):
    active_indices = [i for i in range(len(depth_values)) 
                      if np.any(mask_slices[..., i] > 0) or np.any(aseg_slices[..., i] == 15)]
    
    if not active_indices:
        logger.warning(f"No active slices found for the 4th ventricle. Skipping montage: {output_path}")
        return

    logger.info(f"Generating slice montage with {len(active_indices)} active slices.")
    start_idx = max(0, active_indices[0] - 1)
    end_idx = min(len(depth_values) - 1, active_indices[-1] + 1)
    plot_indices = list(range(start_idx, end_idx + 1))
    num_slices = len(plot_indices)
    cols = 5
    rows = int(np.ceil(num_slices / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.atleast_1d(axes).flatten()
    BRIGHT_RED = '#FF0000'
    BRIGHT_LIME = '#00FF00'
    bright_red_cmap = ListedColormap([BRIGHT_RED])
    overlay_alpha = 0.7 
    
    plot_pos = 0
    for plot_pos, i in enumerate(plot_indices):
        ax = axes[plot_pos]
        ax.imshow(anat_slices[..., i].T, cmap='gray', origin='lower', interpolation='none')
        aseg_layer = (aseg_slices[..., i] == 15).astype(float)
        if np.any(aseg_layer):
            ax.contour(aseg_layer.T, colors=BRIGHT_LIME, levels=[0.5], linewidths=1.5)
        active_mask = (mask_slices[..., i] > 0).astype(float)
        if np.any(active_mask):
            masked_data = np.ma.masked_where(active_mask == 0, active_mask)
            ax.imshow(masked_data.T, cmap=bright_red_cmap, alpha=overlay_alpha, origin='lower', interpolation='none')
        ax.set_title(f"Z: {depth_values[i]:.2f}cm", fontsize=10, color='white', backgroundcolor='black')
        ax.axis('off')
    
    for j in range(plot_pos + 1, len(axes)):
        fig.delaxes(axes[j])
        
    legend_elements = [
        Line2D([0], [0], color=BRIGHT_LIME, lw=2, label='Aseg Label (15) Target'),
        mpatches.Patch(color=BRIGHT_RED, alpha=overlay_alpha, label='Voxels Included in Sum')
    ]
    legend = fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize='large', frameon=True)
    legend.get_frame().set_facecolor('black')
    legend.get_frame().set_edgecolor('none')
    for text in legend.get_texts():
        text.set_color('white')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()
    logger.info(f"Successfully saved montage to {output_path}")
    
def compute_area(func_path, anat_path, aseg_path, reg_path, output_path, func_vox=2.5, anat_vox=1.0):
    logger.info(f"Starting area computation for {func_path}")
    
    try:
        func_img, anat_img, aseg_img = nib.load(func_path), nib.load(anat_path), nib.load(aseg_path)
        func_data, anat_data, aseg_data = func_img.get_fdata(), anat_img.get_fdata(), aseg_img.get_fdata()
        logger.info(f"NIfTI volumes loaded. Func shape: {func_data.shape}, Anat shape: {anat_data.shape}")
    except Exception as e:
        logger.error(f"Failed to load NIfTI files: {e}")
        raise

    # Parameterized Affine Matrices
    anat_half = anat_vox / 2.0
    func_half = func_vox / 2.0

    anatVOX2RAS = np.array([
        [-anat_vox, 0, 0, anat_half * aseg_data.shape[0]], 
        [0, 0, anat_vox, -anat_half * aseg_data.shape[2]], 
        [0, -anat_vox, 0, anat_half * aseg_data.shape[1]], 
        [0, 0, 0, 1]
    ])
    
    funcVOX2RAS = np.array([
        [-func_vox, 0, 0, func_half * func_data.shape[0]], 
        [0, 0, func_vox, -func_half * func_data.shape[2]], 
        [0, -func_vox, 0, func_half * func_data.shape[1]], 
        [0, 0, 0, 1]
    ])
    
    logger.info(f"Reading registration matrix from {reg_path}")
    R = np.loadtxt(reg_path, skiprows=4, max_rows=4) 

    # Filter for 4th ventricle (Label 15)
    mask_15 = (aseg_data == 15)
    if not np.any(mask_15):
        logger.error("Label 15 (4th ventricle) not found in aseg volume!")
        raise ValueError("Empty aseg label 15")
    
    aseg_data[~mask_15] = 0
    centroid_3d = center_of_mass(aseg_data)
    logger.info(f"Calculated 4th ventricle centroid in anat CRS: {centroid_3d}")

    init_funcCRS = np.linalg.inv(funcVOX2RAS) @ R @ anatVOX2RAS @ np.array([*centroid_3d, 1]).T
    funcCRS0 = np.array([int(np.floor(init_funcCRS[0])), int(np.floor(init_funcCRS[1])), 0, 1])

    incr = 0.2
    lenx = leny = func_vox
    voxel_area = (incr * lenx) * (incr * leny)
    
    deltax = deltay = np.arange(-8, 8 + incr, incr)
    deltaz = np.arange(-25, 25 + incr, incr)
    
    logger.info(f"Starting grid search. Grid size: {deltax.size}x{deltay.size}x{deltaz.size}")
    
    area_contribution = np.zeros((deltax.size, deltay.size, deltaz.size))
    T1 = np.zeros((deltax.size, deltay.size, deltaz.size))
    viz_anat = np.zeros((deltax.size, deltay.size, deltaz.size))
    viz_aseg = np.zeros((deltax.size, deltay.size, deltaz.size))

    # Grid search
    inv_anat = np.linalg.inv(anatVOX2RAS)
    inv_R = np.linalg.inv(R)
    
    for xind, dx in enumerate(deltax):
        for yind, dy in enumerate(deltay):
            for zind, dz in enumerate(deltaz):
                ifuncCRS = funcCRS0 + np.array([dx, dy, dz, 0])
                anat_crs = inv_anat @ inv_R @ funcVOX2RAS @ ifuncCRS
                crs_int = np.floor(anat_crs[:3]).astype(int)
                
                if np.all((crs_int >= 0) & (crs_int < aseg_data.shape)):
                    val_aseg = aseg_data[crs_int[0], crs_int[1], crs_int[2]]
                    val_anat = anat_data[crs_int[0], crs_int[1], crs_int[2]]
                    
                    viz_anat[xind, yind, zind] = val_anat
                    viz_aseg[xind, yind, zind] = val_aseg
                    
                    if val_aseg == 15:
                        area_contribution[xind, yind, zind] = voxel_area
                        T1[xind, yind, zind] = val_anat

    logger.info("Grid search complete. Calculating area profile.")
    A = np.zeros(len(deltaz))
    for i in range(len(deltaz)):
        Aslice, T1slice = area_contribution[:,:,i], T1[:,:,i].copy()
        T1slice[T1slice == 0] = np.nan
        if not np.all(np.isnan(T1slice)):
            ref = np.nanmin(T1slice)
            area_scaled = Aslice * (ref / T1slice)
            A[i] = np.nansum(area_scaled)

    # Convert functional voxel depth to mm
    depthfromslc1 = deltaz * func_vox 
    A = add_boundary(A, depthfromslc1)
    
    depth_cm = depthfromslc1 * 0.1
    A_cm2 = A * 0.01
    
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "area.txt", np.column_stack([depth_cm, A_cm2]), fmt="%.6f")
    logger.info(f"Saved area.txt to {out_dir}")

    # Plot
    plt.figure()
    plt.plot(depth_cm, A_cm2, marker='.')
    plt.xlabel("Distance from 1st fMRI slice (cm)")
    plt.ylabel("Cross-sectional Area (cm²)")
    plt.title("4th ventricle area-depth profile")
    plt.savefig(out_dir / "area_vs_depth.png")
    plt.close()
    logger.info("Saved area_vs_depth.png")

    create_slice_montage(
        viz_anat, 
        area_contribution, 
        viz_aseg, 
        depth_cm, 
        out_dir / "slice_inspection.png"
    )

def aggregate_area(search_dir, outfile):
    logger.info(f"Searching for area.txt files in {search_dir}")
    search_path = Path(search_dir)
    area_files = sorted(list(search_path.rglob("area.txt")))
    
    logger.info(f"Found {len(area_files)} files to aggregate.")
    collection = []

    for f in area_files:
        parts = f.parts
        try:
            area_idx = parts.index("area")
            sub = parts[area_idx - 2]
            ses = parts[area_idx - 1]
            data = np.loadtxt(f) 
            collection.append((data[:, 0], data[:, 1], sub))
            logger.info(f"Collected: {sub} | {ses}")

        except (ValueError, IndexError) as e:
            logger.warning(f"Skipping problematic path: {f}")

    with open(outfile, "wb") as f_out:
        pickle.dump(collection, f_out)
    
    logger.info(f"[+] Successfully aggregated {len(collection)} area records to {outfile}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect', action='store_true', help="Aggregate existing area files")
    parser.add_argument('--func', type=str)
    parser.add_argument('--anat', type=str)
    parser.add_argument('--aseg', type=str)
    parser.add_argument('--reg', type=str)
    parser.add_argument('--func_vox', type=float, default=2.5)
    parser.add_argument('--anat_vox', type=float, default=1.0)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--outfile', type=str)
    
    args = parser.parse_args()

    if args.collect:
        aggregate_area(args.outdir, args.outfile)
    else:
        compute_area(args.func, args.anat, args.aseg, args.reg, args.outdir, 
                     func_vox=args.func_vox, anat_vox=args.anat_vox)

if __name__ == "__main__":
    main()