import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
from omegaconf import OmegaConf
import tofmodel.inverse.utils as utils
from tofmodel.forward import posfunclib as pfl
from tofmodel.forward import simulate as tm
from functools import partial

# Load config once
cfg = OmegaConf.load('config/config_dummy.yml')
p = cfg.scan_param

# ==========================================
# Configuration & Parameters
# ==========================================
N_samples = 3  # Number of dummy samples to generate
time_points = 350
noise_std = 0.01

# Spatial Resolutions (mm)
anat_voxel_size = 1.0 # [mm] CHANGE THIS IF NEEDED 
func_voxel_size = p.slice_width # [mm]  

# Physical Dimensions of our dummy brain (mm)
phys_xy = 80.0
phys_z = 100.0

# Calculate grid dimensions based on voxel size
anat_grid_xy = int(phys_xy / anat_voxel_size)  # 80x80
anat_z = int(phys_z / anat_voxel_size)         # 100 slices

func_grid_xy = int(phys_xy / func_voxel_size)  # 32x32
func_z_full = int(phys_z / func_voxel_size)    # 40 slices

# Ventricle Geometry in Physical Space (mm)
base_diameter = 6.0
widest_diameter = 6.0
z_start_ramp = 25.0
z_widest = 50.0
z_end_ramp = 75.0

# ==========================================
# 1. Define Anatomical Space Geometry (Static)
# ==========================================
x_anat, y_anat, z_anat_grid = np.mgrid[0:anat_grid_xy, 0:anat_grid_xy, 0:anat_z]
center_x_anat, center_y_anat = anat_grid_xy // 2, anat_grid_xy // 2

z_phys_anat = np.arange(anat_z) * anat_voxel_size
R_anat = np.ones(anat_z) * base_diameter

up_mask_a = (z_phys_anat >= z_start_ramp) & (z_phys_anat < z_widest)
R_anat[up_mask_a] = np.interp(z_phys_anat[up_mask_a], [z_start_ramp, z_widest], [base_diameter, widest_diameter])

down_mask_a = (z_phys_anat >= z_widest) & (z_phys_anat < z_end_ramp)
R_anat[down_mask_a] = np.interp(z_phys_anat[down_mask_a], [z_widest, z_end_ramp], [widest_diameter, base_diameter])

x_dist_anat_mm = (x_anat - center_x_anat) * anat_voxel_size
y_dist_anat_mm = (y_anat - center_y_anat) * anat_voxel_size
ventricle_mask_anat = x_dist_anat_mm**2 + y_dist_anat_mm**2 <= R_anat[z_anat_grid]**2

anat_data = np.full((anat_grid_xy, anat_grid_xy, anat_z), 0.0)
anat_data[15:65, 15:65, 15:85] = 100 
anat_data[ventricle_mask_anat] = 20  

aseg_data = np.full((anat_grid_xy, anat_grid_xy, anat_z), 0, dtype=np.int16)
aseg_data[15:65, 15:65, 15:85] = 2
aseg_data[ventricle_mask_anat] = 15

# ==========================================
# 2. Define Functional Space Geometry (Static)
# ==========================================
z_start_func_idx = int(z_widest / func_voxel_size) 
func_z_cropped = func_z_full - z_start_func_idx    

x_func, y_func, z_func_grid = np.mgrid[0:func_grid_xy, 0:func_grid_xy, 0:func_z_cropped]
center_x_func, center_y_func = func_grid_xy // 2, func_grid_xy // 2

z_phys_func = (np.arange(func_z_cropped) + z_start_func_idx) * func_voxel_size
R_func = np.ones(func_z_cropped) * base_diameter

up_mask_f = (z_phys_func >= z_start_ramp) & (z_phys_func < z_widest)
R_func[up_mask_f] = np.interp(z_phys_func[up_mask_f], [z_start_ramp, z_widest], [base_diameter, widest_diameter])

down_mask_f = (z_phys_func >= z_widest) & (z_phys_func < z_end_ramp)
R_func[down_mask_f] = np.interp(z_phys_func[down_mask_f], [z_widest, z_end_ramp], [widest_diameter, base_diameter])

x_dist_func_mm = (x_func - center_x_func) * func_voxel_size
y_dist_func_mm = (y_func - center_y_func) * func_voxel_size
ventricle_mask_func = x_dist_func_mm**2 + y_dist_func_mm**2 <= R_func[z_func_grid]**2

sbref_data = np.full((func_grid_xy, func_grid_xy, func_z_cropped), 0.0)
sbref_data[6:26, 6:26, :] = 500 
sbref_data[ventricle_mask_func] = 1500 

synthseg_aseg_data = np.full((func_grid_xy, func_grid_xy, func_z_cropped), 0, dtype=np.int16)
synthseg_aseg_data[6:26, 6:26, :] = 2
synthseg_aseg_data[ventricle_mask_func] = 15

anat_affine = np.diag([anat_voxel_size, anat_voxel_size, anat_voxel_size, 1.0])
func_affine = np.diag([func_voxel_size, func_voxel_size, func_voxel_size, 1.0])
z_trans_mm = z_start_func_idx * func_voxel_size 
func_affine[2, 3] = z_trans_mm 

# ==========================================
# LOOP: Generate N Samples
# ==========================================
t_arr = p.repetition_time*np.arange(time_points)
xarea = np.linspace(-3, 3, time_points)
area = np.ones_like(xarea)

# Frequencies for the velocity composite signal
target_freqs = [0.05, 0.15, 1.0]

for n in range(1, N_samples + 1):
    print(f"Generating Sample {n}/{N_samples}...")
    
    out_dir = f'dummy_data/dummy{n}'
    os.makedirs(out_dir, exist_ok=True)

    # 3. Simulate Inflow Signal (Dynamic per sample)
    # Generate random amplitudes between 0.1 and 0.6 for each frequency
    random_amps = np.random.uniform(0.1, 0.6, size=len(target_freqs))
    
    # Generate random phases between 0 and 2*pi for each frequency
    random_phases = np.random.uniform(0, 2 * np.pi, size=len(target_freqs))
    
    # Create composite velocity signal
    v_orig = np.zeros_like(t_arr, dtype=float)
    for freq, amp, phase in zip(target_freqs, random_amps, random_phases):
        v_orig += amp * np.sin(2 * np.pi * freq * t_arr + phase)

    v_up = utils.upsample(v_orig, v_orig.size*100+1, p.repetition_time).flatten()
    t = np.arange(0, p.repetition_time*time_points, p.repetition_time/100)

    t_base, v_base = utils.add_baseline_period(t, v_up, p.repetition_time*p.num_pulse_baseline_offset)
    x_func_sim = partial(pfl.compute_position_numeric_spatial, 
                         tr_vect=t_base, vts=v_base, 
                         xarea=xarea, area=area)

    s_raw = tm.simulate_inflow(p.repetition_time, p.echo_time, time_points+p.num_pulse_baseline_offset, 
                                p.slice_width, p.flip_angle, p.t1_time, p.t2_time, p.num_slice, 
                                p.alpha_list, p.MBF, x_func_sim, ncpu=-1, varysliceprofile=True, dx=0.005, offset_fact=0, enable_logging=False)

    s_raw = s_raw[p.num_pulse_baseline_offset:, :4]

    # 4. Generate 4D Functional Data (Dynamic per sample)
    noise_shape = (func_grid_xy, func_grid_xy, func_z_cropped, time_points)
    func_data = np.random.normal(0, noise_std, noise_shape)

    for func_slice in range(min(4, func_z_cropped)):
        slice_mask = ventricle_mask_func[:, :, func_slice]
        signal = s_raw[:, func_slice]
        func_data[slice_mask, func_slice, :] += signal

    # 5. Save Data and Affines
    nib.save(nib.Nifti1Image(anat_data, anat_affine), os.path.join(out_dir, 'anat.nii.gz'))
    nib.save(nib.Nifti1Image(aseg_data, anat_affine), os.path.join(out_dir, 'aseg.nii.gz'))
    nib.save(nib.Nifti1Image(sbref_data, func_affine), os.path.join(out_dir, 'sbref.nii.gz'))
    nib.save(nib.Nifti1Image(synthseg_aseg_data, func_affine), os.path.join(out_dir, 'aseg_sbref_space.nii.gz'))
    nib.save(nib.Nifti1Image(func_data, func_affine), os.path.join(out_dir, 'func.nii.gz'))

    anat_center_z_mm = (anat_z * anat_voxel_size) / 2.0
    func_center_z_mm = (z_start_func_idx * func_voxel_size) + ((func_z_cropped * func_voxel_size) / 2.0)
    tkr_y_trans = anat_center_z_mm - func_center_z_mm

    reg_content = f"""dummy_subject
{func_voxel_size:.6f}
{func_voxel_size:.6f}
{func_voxel_size:.6f}
1.000000 0.000000 0.000000 0.000000
0.000000 1.000000 0.000000 {tkr_y_trans:.6f}
0.000000 0.000000 1.000000 0.000000
0 0 0 1
round
"""
    with open(os.path.join(out_dir, 'reg.dat'), 'w') as f:
        f.write(reg_content.strip() + '\n')

    # 6. Visualization (Separated Velocity and fMRI signal)
    fig = plt.figure(figsize=(20, 6)) # Widened to accommodate 3 plots

    # Plot 1: 3D Ventricle Space
    ax1 = fig.add_subplot(131, projection='3d')
    a_x, a_y, a_z = np.where(ventricle_mask_anat)
    ax1.scatter(a_x * anat_voxel_size, a_y * anat_voxel_size, a_z * anat_voxel_size, 
                c='cyan', alpha=0.05, label='Anat Ventricle (1mm)')

    f_x, f_y, f_z = np.where(ventricle_mask_func)
    f_z_phys = (f_z + z_start_func_idx) * func_voxel_size
    ax1.scatter(f_x * func_voxel_size, f_y * func_voxel_size, f_z_phys, 
                c='red', alpha=0.2, label='Cropped fMRI Ventricle (2.5mm)')

    ax1.set_title(f"3D Ventricle Space (Sample {n})")
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")
    ax1.legend()

    # Plot 2: fMRI Timeseries
    ax2 = fig.add_subplot(132)
    for func_slice in range(min(4, func_z_cropped)):
        slice_mask = ventricle_mask_func[:, :, func_slice]
        if slice_mask.any():
            timeseries = func_data[slice_mask, func_slice, :].mean(axis=0)
            ax2.plot(t_arr, timeseries, label=f"Slice {func_slice}")
    
    ax2.set_title(f"fMRI Timeseries (Sample {n})")
    ax2.set_xlabel("Time Points")
    ax2.set_ylabel("Mean Signal Amplitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Input Velocity
    ax3 = fig.add_subplot(133)
    ax3.plot(t_arr, v_orig, 'k-', label='Composite Velocity')
    ax3.set_title(f"Simulated Inflow Velocity (Sample {n})")
    ax3.set_xlabel("Time Points")
    ax3.set_ylabel("Velocity Amplitude")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'summary_plot_sample_{n}.png'))
    plt.close(fig) 

print("All samples generated successfully!")