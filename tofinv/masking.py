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
    parser.add_argument("--container", help="Path to freesurfer container file")
    parser.add_argument("--container_bind", help="Paths to bind for apptainer")
    parser.add_argument("--dummy_run", action="store_true", help="flag when running with dummy data, which will skip running SynthSeg and use a pre-generated mask instead")
    
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sbref_fixed = outdir / "SBRef_fixed.nii.gz"
    fix_sbref(args.sbref, sbref_fixed)

    container = args.container
    if args.dummy_run:
        print("[*] Running in dummy mode, skipping SynthSeg and using pre-generated mask.")
        func_dir = pathlib.Path(args.func).parent
        dummy_mask_path = func_dir / "aseg_sbref_space.nii.gz"
        subprocess.run(["cp", str(dummy_mask_path), str(outdir / "SBRef_fixed_synthseg.nii.gz")])
    else:
        cmd = ["mri_synthseg", "--i", str(sbref_fixed), "--o", str(outdir)]
        use_container = args.container and args.container.lower() != "none"
        if use_container:
            print(f"[*] Running SynthSeg via Container: {args.container}")
            # Only add bind if it's also not "None"
            if args.container_bind and args.container_bind.lower() != "none":
                bind_args = ["-B", args.container_bind]
            else:
                bind_args = []
            cmd = ["apptainer", "exec"] + bind_args + [args.container] + cmd
        else:
            print("[*] Running SynthSeg via local FreeSurfer module")
        subprocess.run(cmd, check=True)

    
    synthseg_file = outdir / "SBRef_fixed_synthseg.nii.gz"
    resampled = sf.load_volume(synthseg_file).resample_like(sf.load_volume(sbref_fixed), method='nearest')
    mask = resampled == 15  # CSF label
    
    dilated = ndimage.binary_dilation(mask[:, :, :args.nslice_to_keep+1], structure=ndimage.generate_binary_structure(3, 4), iterations=1)
    
    for islice in range(args.nslice_to_keep):
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
        boundmask=str(mask_path),
        outdir=str(outdir),
        metrics_list_names=['skew', 'sbref']
    )

if __name__ == "__main__":
    main()