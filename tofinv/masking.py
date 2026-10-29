import argparse
import pathlib

from autocsfmask.automask import run_autocsfmask


def main():
    parser = argparse.ArgumentParser(description="Run AutoCSFMask from ToFInv.")
    parser.add_argument("--func", required=True)
    parser.add_argument("--sbref", required=True)
    parser.add_argument("--nslice_to_keep", required=True, type=int)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--container", help="Path to FreeSurfer container file")
    parser.add_argument("--container_bind", help="Paths to bind for Apptainer")
    parser.add_argument("--global_seed", default=42, type=int, help="Seed for reproducibility")
    parser.add_argument("--csf_label", default=15, type=int, help="FreeSurfer label for CSF; default is 15")
    parser.add_argument("--dummy_run", action="store_true", help="Use ToFInv dummy-data segmentation.")
    parser.add_argument("--metrics_list", nargs="+", default=["sd"], help="List of metrics for AutoCSFMask.")
    args = parser.parse_args()

    synthseg_file = None

    if args.dummy_run:
        print(
            "[*] Running in dummy mode, skipping SynthSeg and using "
            "ToFInv's pre-generated segmentation."
        )
        synthseg_file = (pathlib.Path(args.func).parent / "aseg_sbref_space.nii.gz")

        if not synthseg_file.exists():
            raise FileNotFoundError(
                f"Dummy SynthSeg file not found: {synthseg_file}"
            )

    run_autocsfmask(
        func=args.func,
        sbref=args.sbref,
        nslice_to_keep=args.nslice_to_keep,
        outdir=args.outdir,
        container=args.container,
        container_bind=args.container_bind,
        global_seed=args.global_seed,
        csf_label=args.csf_label,
        metrics_list=args.metrics_list,
        synthseg_file=str(synthseg_file) if synthseg_file else None,
    )


if __name__ == "__main__":
    main()