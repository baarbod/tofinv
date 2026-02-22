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

def create_slice_montage(anat_slices, mask_slices, aseg_slices, depth_values, output_path):
    active_indices = [i for i in range(len(depth_values)) 
                      if np.any(mask_slices[..., i] > 0) or np.any(aseg_slices[..., i] == 15)]
    if not active_indices:
        print("[-] No active slices found for the 4th ventricle. Skipping montage.")
        return
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
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black') # Black background for final image
    plt.close()
    
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

    viz_anat = np.zeros((deltax.size, deltay.size, deltaz.size))
    viz_aseg = np.zeros((deltax.size, deltay.size, deltaz.size))

    # Grid search
    for xind, dx in enumerate(deltax):
        for yind, dy in enumerate(deltay):
            for zind, dz in enumerate(deltaz):
                ifuncCRS = funcCRS0 + np.array([dx, dy, dz, 0])
                anat_crs = np.linalg.inv(anatVOX2RAS) @ np.linalg.inv(R) @ funcVOX2RAS @ ifuncCRS
                crs_int = np.floor(anat_crs[:3]).astype(int)
                if np.all((crs_int >= 0) & (crs_int < aseg_data.shape)):
                    
                    viz_anat[xind, yind, zind] = anat_data[crs_int[0], crs_int[1], crs_int[2]]
                    viz_aseg[xind, yind, zind] = aseg_data[crs_int[0], crs_int[1], crs_int[2]]
                    
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

    out_dir = Path(output_path)
    depth_cm = deltaz * 2.5 * 0.1
    create_slice_montage(
        viz_anat, 
        area_contribution, 
        viz_aseg, 
        depth_cm, 
        out_dir / "slice_inspection.png"
    )

def aggregate_area(search_dir, outfile):
    search_path = Path(search_dir)
    area_files = sorted(list(search_path.rglob("area.txt")))
    
    collection = []

    for f in area_files:
        parts = f.parts
        try:
            area_idx = parts.index("area")
            sub = parts[area_idx - 2]
            ses = parts[area_idx - 1]
            data = np.loadtxt(f) 
            collection.append((data[:, 0], data[:, 1], sub))

        except (ValueError, IndexError) as e:
            print(f"Skipping malformed path: {f}")

    with open(outfile, "wb") as f_out:
        pickle.dump(collection, f_out)
    
    print(f"[+] Aggregated {len(collection)} area records to {outfile}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect', action='store_true', help="Aggregate existing area files")
    
    # Standard processing args
    parser.add_argument('--func', type=str)
    parser.add_argument('--anat', type=str)
    parser.add_argument('--aseg', type=str)
    parser.add_argument('--reg', type=str)
    
    # Shared/Collection args
    parser.add_argument('--outdir', type=str, help="Output dir for compute, or Search dir for collect")
    parser.add_argument('--outfile', type=str, help="Path for the final pickle (use with --collect)")
    
    args = parser.parse_args()

    if args.collect:
        aggregate_area(args.outdir, args.outfile)
    else:
        compute_area(args.func, args.anat, args.aseg, args.reg, args.outdir)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
