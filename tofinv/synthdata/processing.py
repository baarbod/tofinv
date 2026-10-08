import os
import pickle
import logging
import argparse
import numpy as np
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from tofinv.utils import available_cpu_count

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def sort_inputs(input_dir, output_dir, batch_size):
    logger.info(f"--- Starting Input Sorting from {input_dir} ---")
    inputs_all_samples = []
    nproton_list = []

    files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
    if not files:
        logger.error(f"No .pkl files found in {input_dir}")
        raise FileNotFoundError(f"No .pkl files found in {input_dir}")

    logger.info(f"Loading {len(files)} batch files for redistribution...")
    for batch_name in files:
        path = os.path.join(input_dir, batch_name)
        try:
            with open(path, "rb") as f:
                batch_inputs = pickle.load(f)
            for sample_input in batch_inputs:
                inputs_all_samples.append(sample_input)
                nproton_list.append(sample_input['x0_array'].shape[0])
        except Exception as e:
            raise RuntimeError(f"Failed to load sampling batch {batch_name}: {e}") from e

    total_samples = len(inputs_all_samples)
    if total_samples == 0:
        raise RuntimeError("No sampling inputs were loaded.")
    logger.info(f"Total samples collected: {total_samples}")

    logger.info("Sorting samples by proxy for workload (particle count)...")
    sort_indices = np.argsort(nproton_list)[::-1]
    inputs_all_samples_sorted = [inputs_all_samples[i] for i in sort_indices]

    if os.path.exists(output_dir):
        logger.info(f"Cleaning existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    batches = [inputs_all_samples_sorted[x:x+batch_size] 
               for x in range(0, len(inputs_all_samples_sorted), batch_size)]
    
    logger.info(f"Writing {len(batches)} redistributed batches (Batch Size: {batch_size})...")
    for i, batch in enumerate(batches):
        new_task_id = i + 1
        for sample in batch:
            sample['input_data']['task_id'] = new_task_id 
        
        out_path = os.path.join(output_dir, f"task{new_task_id:03}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(batch, f)
    
    logger.info(f"Successfully redistributed {total_samples} samples into {len(batches)} batches.")

def _process_sample(sample_path):
    try:
        with open(sample_path, "rb") as f:
            data = pickle.load(f)
        xx = np.column_stack((data['X'], data['xarea'], data['area']))
        xx = np.swapaxes(xx, -1, -2)
        yy = np.expand_dims(data['v'], axis=0)
        if np.isnan(xx).any() or np.isinf(xx).any():
            return "invalid"
        return xx, yy
    except Exception:
        return "error"

def combine_simulations(sim_dir, output_dir, workers=None, expected_samples=None):
    logger.info(f"--- Starting Simulation Dataset Combination in {sim_dir} ---")
    all_paths = []
    batch_dirs = sorted([d for d in os.listdir(sim_dir) if d.startswith("batch_")])
    for batch_name in batch_dirs:
        batch_subdir = os.path.join(sim_dir, batch_name)
        all_paths.extend([
            os.path.join(batch_subdir, f) 
            for f in os.listdir(batch_subdir) if f.endswith(".pkl")
        ])
    num_files = len(all_paths)
    if not all_paths:
        logger.error("No simulation result files (.pkl) found in subdirectories.")
        raise FileNotFoundError("No simulation result files (.pkl) found in subdirectories.")

    n_workers = min(workers or available_cpu_count(), available_cpu_count())
    X_list, y_list = [], []
    invalid_count = 0
    error_count = 0
    logger.info(f"Processing {num_files} samples using {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_process_sample, all_paths))

    for res in results:
        if isinstance(res, tuple):
            X_list.append(res[0])
            y_list.append(res[1])
        elif res == "invalid":
            invalid_count += 1
        elif res == "error":
            error_count += 1

    if invalid_count > 0:
        logger.warning(f"Discarded {invalid_count} samples containing NaNs or Infs.")
    if error_count > 0:
        raise RuntimeError(f"Failed to process {error_count} simulation samples due to I/O or pickle errors.")

    if not X_list:
        logger.error("No valid samples collected. Dataset construction aborted.")
        raise RuntimeError("No valid samples collected; dataset construction aborted.")
    if expected_samples is not None and len(X_list) != expected_samples:
        raise RuntimeError(f"Expected {expected_samples} valid samples, found {len(X_list)}.")

    logger.info("Stacking data into final tensors...")
    X_final = np.stack(X_list, axis=0)
    y_final = np.stack(y_list, axis=0)

    os.makedirs(output_dir, exist_ok=True)
    master_file = os.path.join(output_dir, "dataset.pkl")
    atomic_pickle_dump([X_final, y_final], master_file)
        
    logger.info(f"Saved master dataset to {master_file}")
    logger.info(f"Final Tensor Shapes -> X: {X_final.shape}, y: {y_final.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input sorting and dataset combination tools")
    subparsers = parser.add_subparsers(dest="action", required=True)

    sort_parser = subparsers.add_parser("sort", help="Sort and redistribute input batches")
    sort_parser.add_argument("--input_dir", required=True)
    sort_parser.add_argument("--output_dir", required=True)
    sort_parser.add_argument("--batch_size", type=int, required=True)

    combine_parser = subparsers.add_parser("combine", help="Aggregate simulation results into a master dataset")
    combine_parser.add_argument("--sim_dir", required=True)
    combine_parser.add_argument("--output_dir", required=True)
    combine_parser.add_argument("--workers", type=int, default=None)
    combine_parser.add_argument("--expected_samples", type=int, default=None)

    args = parser.parse_args()

    if args.action == "sort":
        sort_inputs(args.input_dir, args.output_dir, args.batch_size)
    elif args.action == "combine":
        combine_simulations(args.sim_dir, args.output_dir, args.workers, args.expected_samples)
