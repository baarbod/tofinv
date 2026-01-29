import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

def load_txt(path):
    return np.loadtxt(path)

def calculate_rmse(real, sim):
    # Assuming real and sim are same shape (N_timepoints, 3_slices)
    return np.sqrt(np.mean((real - sim)**2))

def generate_report(input_dirs, output_pdf):
    results = []

    print(input_dirs)
    # 1. Collect data and calculate metrics
    for d in input_dirs:
        # Path parsing logic based on this structure: .../{sub}/{ses}/{run}/evaluation/dataset/{exp}
        parts = d.split(os.sep)
        exp = parts[-1]
        run = parts[-4]
        ses = parts[-5]
        sub = parts[-6]

        try:
            real_sig = load_txt(os.path.join(d, "signal_data.txt"))
            sim_sig = load_txt(os.path.join(d, "signal_simulation.txt"))
            vel = load_txt(os.path.join(d, "velocity_predicted.txt"))
            
            rmse = calculate_rmse(real_sig, sim_sig)
            
            results.append({
                'sub': sub, 'ses': ses, 'run': run, 'exp': exp,
                'rmse': rmse, 'path': d, 'real': real_sig, 'sim': sim_sig, 'vel': vel
            })
        except Exception as e:
            print(f"Skipping {d} due to error: {e}")

    df = pd.DataFrame(results)
    
    with PdfPages(output_pdf) as pdf:
        # --- PAGE 1: GLOBAL STATISTICS ---
        plt.figure(figsize=(11, 8.5))
        plt.suptitle("TofInv Pipeline Summary Report", fontsize=16, fontweight='bold')
        
        # Plot RMSE distribution by Experiment
        plt.subplot(2, 1, 1)
        sns.boxplot(data=df, x='exp', y='rmse', palette='Set2')
        sns.stripplot(data=df, x='exp', y='rmse', color='black', alpha=0.3)
        plt.title("Model Performance by Experiment Type")
        plt.ylabel("RMSE")

        # Summary Table
        plt.subplot(2, 1, 2)
        plt.axis('off')
        summary_stats = df.groupby('exp')['rmse'].describe().round(4).reset_index()
        table = plt.table(cellText=summary_stats.values, colLabels=summary_stats.columns, 
                          loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        
        pdf.savefig()
        plt.close()

        # --- PAGE 2: TOP 3 & BOTTOM 3 COMPARISON ---
        # Pick the extremes based on RMSE
        df_sorted = df.sort_values('rmse')
        top_3 = df_sorted.head(3)
        bottom_3 = df_sorted.tail(3)
        cases_to_plot = pd.concat([top_3, bottom_3])
        labels = ["BEST (Top 3)"] * 3 + ["WORST (Bottom 3)"] * 3

        fig, axes = plt.subplots(6, 2, figsize=(12, 18))
        plt.subplots_adjust(hspace=0.6, wspace=0.3)

        for i, (idx, row) in enumerate(cases_to_plot.iterrows()):
            # Column 1: Signal Fit (Data vs Simulation)
            ax_sig = axes[i, 0]
            # Plot only the first slice for clarity, or mean of 3
            ax_sig.plot(row['real'][:, 0], 'k-', alpha=0.8, label='Data (Slice 1)')
            ax_sig.plot(row['sim'][:, 0], 'r--', alpha=0.8, label='Sim (Slice 1)')
            ax_sig.set_title(f"{labels[i]}: {row['sub']} {row['run']} ({row['exp']})\nRMSE: {row['rmse']:.4f}")
            ax_sig.legend(fontsize=8)

            # Column 2: Predicted Velocity
            ax_vel = axes[i, 1]
            ax_vel.plot(row['vel'], color='blue')
            ax_vel.set_title(f"Predicted Velocity")
            ax_vel.set_ylabel("cm/s")

        pdf.savefig()
        plt.close()

if __name__ == "__main__":
    import sys
    # Expects: python summary.py output.pdf dir1 dir2 dir3 ...
    generate_report(sys.argv[2:], sys.argv[1])