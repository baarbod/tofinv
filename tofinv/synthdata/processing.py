# tofinv/synthdata/processing.py
import os
import pickle
import logging
import argparse
import numpy as np
import shutil
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- SORTING LOGIC ---

def sort_inputs(input_dir, output_dir, batch_size):
    """Load all input batches, sort them by proton count, and redistribute."""
    inputs_all_samples = []
    nproton_list = []

    files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {input_dir}")

    logger.info(f"Loading {len(files)} batches for sorting...")
    for batch_name in files:
        path = os.path.join(input_dir, batch_name)
        with open(path, "rb") as f:
            batch_inputs = pickle.load(f)
        for sample_input in batch_inputs:
            inputs_all_samples.append(sample_input)
            # x0_array length determines simulation time (proton count)
            nproton_list.append(sample_input['x0_array'].shape[0])

    # Sort descending (largest simulations first)
    sort_indices = np.argsort(nproton_list)[::-1]
    inputs_all_samples_sorted = [inputs_all_samples[i] for i in sort_indices]

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    batches = [inputs_all_samples_sorted[x:x+batch_size] 
               for x in range(0, len(inputs_all_samples_sorted), batch_size)]
    
    for i, batch in enumerate(batches):
        new_task_id = i + 1
        for sample in batch:
            sample['input_data']['task_id'] = new_task_id 
        
        out_path = os.path.join(output_dir, f"task{new_task_id:03}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(batch, f)
    
    logger.info(f"Redistributed into {len(batches)} sorted batches.")

# --- COMBINING LOGIC ---

def _process_sample(sample_path):
    """
    Worker function to handle individual file I/O and transformations.
    """
    try:
        with open(sample_path, "rb") as f:
            data = pickle.load(f)
        
        # Combine features
        xx = np.column_stack((
            data['X'], 
            data['xarea'], 
            data['area']
        ))
        
        # Transform dimensions
        xx = np.swapaxes(xx, -1, -2)
        yy = np.expand_dims(data['v'], axis=0)

        # Filter out bad data
        if np.isnan(xx).any() or np.isinf(xx).any():
            return None
            
        return xx, yy
    except Exception:
        # Avoid logging inside the worker to prevent I/O blocking; 
        # handle errors in the main loop if necessary.
        return None

def combine_simulations(sim_dir, output_dir):
    """Parallelized gathering and transformation of simulation samples."""
    
    # 1. Gather all file paths first
    all_paths = []
    batch_dirs = sorted([d for d in os.listdir(sim_dir) if d.startswith("batch_")])
    for batch_name in batch_dirs:
        batch_subdir = os.path.join(sim_dir, batch_name)
        all_paths.extend([
            os.path.join(batch_subdir, f) 
            for f in os.listdir(batch_subdir) if f.endswith(".pkl")
        ])

    if not all_paths:
        logging.error("No .pkl files found.")
        return

    # 2. Process in parallel using all available CPUs
    n_workers = len(os.sched_getaffinity(0))
    X_list, y_list = [], []
    logging.info(f"Processing {len(all_paths)} samples using {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_process_sample, all_paths))

    # 3. Filter and Stack
    for res in results:
        if res is not None:
            X_list.append(res[0])
            y_list.append(res[1])

    if not X_list:
        logging.error("No valid samples collected after processing.")
        return

    X_final = np.stack(X_list, axis=0)
    y_final = np.stack(y_list, axis=0)

    # 4. Save results
    master_file = os.path.join(output_dir, "dataset.pkl")
    with open(master_file, "wb") as f:
        pickle.dump([X_final, y_final], f)
        
    logging.info(f"Final Dataset: X {X_final.shape}, y {y_final.shape}")

# --- CLI DISPATCHER ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    # Sort sub-command
    sort_parser = subparsers.add_parser("sort")
    sort_parser.add_argument("--input_dir", required=True)
    sort_parser.add_argument("--output_dir", required=True)
    sort_parser.add_argument("--batch_size", type=int, required=True)

    # Combine sub-command
    combine_parser = subparsers.add_parser("combine")
    combine_parser.add_argument("--sim_dir", required=True)
    combine_parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    if args.action == "sort":
        sort_inputs(args.input_dir, args.output_dir, args.batch_size)
    elif args.action == "combine":
        combine_simulations(args.sim_dir, args.output_dir)