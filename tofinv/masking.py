import argparse
import pathlib
import subprocess
import nibabel as nib
import numpy as np
from scipy import ndimage
import surfa as sf
from autocsfmask.automask import run_automask


def fix_sbref(infile, outfile):
    img = nib.load(infile)
    data = img.get_fdata()
    if data.ndim == 3:
        nib.save(img, outfile)
        return
    # Pick the brightest volume (usually the best reference)
    idx = int(np.argmax([data[..., i].mean() for i in range(data.shape[-1])]))
    out_img = nib.Nifti1Image(data[..., idx], img.affine, img.header)
    out_img.set_data_dtype(np.float32)
    nib.save(out_img, outfile)

def main():
    parser = argparse.ArgumentParser(description="Run automask with CLI inputs.")
    parser.add_argument("--func", required=True)
    parser.add_argument("--sbref", required=True)
    parser.add_argument("--nslice_to_keep", required=True, type=int)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--container", required=True, help="Path to freesurfer container file")
    parser.add_argument("--container_bind", required=True, help="Paths to bind for apptainer")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sbref_fixed = outdir / "SBRef_fixed.nii.gz"
    fix_sbref(args.sbref, sbref_fixed)

    # 1. Run SynthSeg via Apptainer
    container = args.container
    print(f"[*] Running SynthSeg via {args.container}")
    subprocess.run([
        "apptainer", "exec", "-B", args.container_bind,
        container, "mri_synthseg", "--i", str(sbref_fixed), "--o", str(outdir),
    ], check=True)

    # 2. Mask Dilated Post-processing
    synthseg_file = outdir / "SBRef_fixed_synthseg.nii.gz"
    resampled = sf.load_volume(synthseg_file).resample_like(sf.load_volume(sbref_fixed), method='nearest')
    mask = resampled == 15  # CSF label
    
    # Dilate mask to ensure coverage
    dilated = ndimage.binary_dilation(mask[:, :, :args.nslice_to_keep], structure=ndimage.generate_binary_structure(3, 4), iterations=1)
    
    # Handle empty slices by propagating neighboring slice masks
    for islice in range(args.nslice_to_keep - 1):
        if np.sum(dilated[:, :, islice]) == 0:
            dilated[:, :, islice] = dilated[:, :, islice + 1]
    if np.sum(dilated[:, :, -1]) == 0:
        dilated[:, :, -1] = dilated[:, :, -2]
        
    mask_path = outdir / "dilated_mask.npy"
    np.save(mask_path, dilated)

    print(f"[*] Running automask for {args.func}")
    run_automask(
        func=args.func,
        sbref=str(sbref_fixed),
        synthseg=str(mask_path),
        outdir=str(outdir),
        metrics_list_names=['sbref']
    )

if __name__ == "__main__":
    main()