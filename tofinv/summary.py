import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.stats import pearsonr
from scipy import signal

# --- CONFIGURATION ---
sns.set_theme(style="white", context="paper")
# Tolerance for the "Residuals" plot (Green Zone)
RESIDUAL_TOLERANCE = 0.05 

def load_txt(path):
    """Safely load text file, reshaping to 2D array if needed."""
    try:
        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        return data
    except Exception:
        return None

def calculate_metrics(real, sim):
    """
    Computes metrics for a SINGLE case (all slices).
    """
    # 1. RMSE (Global)
    rmse = np.sqrt(np.mean((real - sim)**2))
    
    # 2. Amplitude Ratio (Energy Check)
    # > 1.0 = Simulation is too strong (Overshoot)
    # < 1.0 = Simulation is too weak (Damped)
    range_real = np.ptp(real, axis=0) # Peak-to-Peak
    range_sim = np.ptp(sim, axis=0)
    
    # Handle divide by zero safety
    with np.errstate(divide='ignore', invalid='ignore'):
        amp_ratios = np.divide(range_sim, range_real)
        amp_ratios[range_real == 0] = 0 # If real signal is flat, ratio is 0
    
    avg_amp_ratio = np.mean(amp_ratios)

    # 3. Correlation & Lag (Shape & Timing)
    corrs = []
    lags = []
    
    for i in range(real.shape[1]):
        r_col = real[:, i]
        s_col = sim[:, i]
        
        if np.std(r_col) == 0 or np.std(s_col) == 0:
            corrs.append(0)
            lags.append(0)
        else:
            # Pearson Correlation (Shape)
            r, _ = pearsonr(r_col, s_col)
            corrs.append(r)
            
            # Cross-Correlation (Lag)
            # Detrend first to match shapes
            r_dm = r_col - np.mean(r_col)
            s_dm = s_col - np.mean(s_col)
            
            xcorr = signal.correlate(r_dm, s_dm, mode='full')
            lags_axis = signal.correlation_lags(len(r_dm), len(s_dm), mode='full')
            
            # Find lag at max correlation
            best_lag = lags_axis[np.argmax(xcorr)]
            lags.append(best_lag)

    return rmse, np.mean(corrs), np.mean(lags), avg_amp_ratio

def compute_fft(signal_data, fs=1.0):
    n = len(signal_data)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_vals = np.abs(np.fft.rfft(signal_data, axis=0))
    return freqs, fft_vals

def plot_residuals_panel(ax, real, sim, colors):
    """
    Plots (Real - Sim) with a green 'safe zone' band.
    """
    diff = real - sim
    
    # Define Safe Zone based on signal magnitude
    signal_range = np.max(real) - np.min(real)
    safe_band = signal_range * RESIDUAL_TOLERANCE
    
    # Green Band = Acceptable Error
    ax.axhspan(-safe_band, safe_band, color='green', alpha=0.1, label=f'±{int(RESIDUAL_TOLERANCE*100)}% Tol')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    for i in range(diff.shape[1]):
        c = colors[i % len(colors)]
        ax.plot(diff[:, i], color=c, alpha=0.7, linewidth=1)
        
    ax.set_title("Residuals (Difference: Real - Sim)", fontsize=10, fontweight='bold', pad=3)
    ax.grid(True, alpha=0.3)
    
    # Symmetric Y-limits to make positive/negative error equally visible
    max_err = np.max(np.abs(diff)) * 1.1
    if max_err == 0: max_err = 1.0
    ax.set_ylim(-max_err, max_err)

def plot_single_case_page(pdf, row, rank_label, title_color):
    """
    Creates a detailed 1-page report for a specific case.
    """
    fig = plt.figure(figsize=(11, 14)) 
    
    # Grid: 
    # Row 0: Velocity
    # Row 1: Time Domain
    # Row 2: Residuals
    # Row 3: Frequency
    gs = gridspec.GridSpec(4, 2, height_ratios=[0.5, 1.2, 0.8, 1], hspace=0.35, wspace=0.2)
    
    # --- HEADER ---
    plt.suptitle(f"{rank_label}: {row['sub']} | {row['exp']}", fontsize=16, fontweight='bold', color=title_color)
    
    # Interpret Stats
    amp_status = "Good"
    if row['amp_ratio'] > 1.1: amp_status = "Overshoot"
    elif row['amp_ratio'] < 0.9: amp_status = "Damped"
    
    lag_status = "Ok"
    if row['lag'] > 2: lag_status = "Late"
    elif row['lag'] < -2: lag_status = "Early"

    stats_box = (f"RMSE: {row['rmse']:.4f}\n"
                 f"Corr: {row['corr']:.3f}\n"
                 f"Lag: {row['lag']:.1f} ({lag_status})\n"
                 f"Amp: {row['amp_ratio']:.2f} ({amp_status})")
    
    fig.text(0.88, 0.95, stats_box, ha='right', va='top', fontsize=11, family='monospace',
             bbox=dict(facecolor='white', edgecolor='#ccc', boxstyle='round,pad=0.5'))

    # --- ROW 0: VELOCITY ---
    ax_vel = fig.add_subplot(gs[0, :])
    ax_vel.plot(row['vel'], color='#34495e', linewidth=2)
    ax_vel.set_title("Input Velocity Profile", fontsize=10, fontweight='bold', loc='left')
    ax_vel.margins(x=0)
    ax_vel.set_xticks([]) 
    ax_vel.grid(True, alpha=0.3)

    real = row['real']
    sim = row['sim']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    
    # Unified Y-Limits for Time
    y_min = min(real.min(), sim.min())
    y_max = max(real.max(), sim.max())
    margin = (y_max - y_min) * 0.1
    ylims = (y_min - margin, y_max + margin)

    # --- ROW 1: TIME DOMAIN (Real vs Sim) ---
    ax_real = fig.add_subplot(gs[1, 0])
    ax_sim = fig.add_subplot(gs[1, 1], sharey=ax_real, sharex=ax_real)
    
    for i in range(real.shape[1]):
        c = colors[i % len(colors)]
        ax_real.plot(real[:, i], color=c, alpha=0.8, linewidth=1.5, label=f'Ch{i}')
        ax_sim.plot(sim[:, i], color=c, alpha=0.8, linewidth=1.5)

    ax_real.set_title("Measured (Real)", fontsize=11, fontweight='bold')
    ax_sim.set_title("Simulated (Model)", fontsize=11, fontweight='bold')
    ax_real.set_ylim(ylims)
    ax_real.grid(True, alpha=0.3)
    ax_sim.grid(True, alpha=0.3)
    ax_real.legend(loc='upper right', fontsize=8)

    # --- ROW 2: RESIDUALS ---
    ax_res = fig.add_subplot(gs[2, :], sharex=ax_real)
    plot_residuals_panel(ax_res, real, sim, colors)

    # --- ROW 3: FREQUENCY ---
    ax_freq_r = fig.add_subplot(gs[3, 0])
    ax_freq_s = fig.add_subplot(gs[3, 1], sharey=ax_freq_r)
    
    freqs, fft_r = compute_fft(real)
    _, fft_s = compute_fft(sim)
    fft_max = max(fft_r.max(), fft_s.max()) * 1.1
    
    for i in range(real.shape[1]):
        c = colors[i % len(colors)]
        ax_freq_r.plot(freqs, fft_r[:, i], color=c, alpha=0.8)
        ax_freq_s.plot(freqs, fft_s[:, i], color=c, alpha=0.8)

    ax_freq_r.set_title("Frequency (Real)", fontsize=11)
    ax_freq_s.set_title("Frequency (Sim)", fontsize=11)
    ax_freq_r.set_ylim(0, fft_max)
    ax_freq_r.set_xlabel("Hz")
    ax_freq_s.set_xlabel("Hz")
    ax_freq_r.grid(True, alpha=0.3)
    ax_freq_s.grid(True, alpha=0.3)

    pdf.savefig(fig)
    plt.close()

def generate_report(input_dirs, output_pdf):
    results = []
    print(f"Scanning {len(input_dirs)} input directories...")

    # 1. COLLECT DATA FROM ALL DIRECTORIES
    for i, d in enumerate(input_dirs):
        if i % 10 == 0: print(f"Processing {i}/{len(input_dirs)}...")
        
        d = os.path.normpath(d)
        parts = d.split(os.sep)
        if not parts[-1]: parts.pop()
        
        try:
            # Parse Path (Adjust indices if your path structure differs)
            exp = parts[-1]
            run = parts[-4]
            ses = parts[-5]
            sub = parts[-6]

            # Load Files
            real = load_txt(os.path.join(d, "signal_data.txt"))
            sim = load_txt(os.path.join(d, "signal_simulation.txt"))
            vel = load_txt(os.path.join(d, "velocity_predicted.txt"))
            
            if real is None or sim is None: continue

            # Compute Metrics
            rmse, corr, lag, amp_ratio = calculate_metrics(real, sim)
            
            results.append({
                'sub': sub, 'ses': ses, 'run': run, 'exp': exp,
                'rmse': rmse, 'corr': corr, 'lag': lag, 'amp_ratio': amp_ratio,
                'real': real, 'sim': sim, 'vel': vel
            })
        except Exception as e:
            # print(f"Skipping {d}: {e}")
            pass

    if not results:
        print("No valid data found.")
        return

    # Create Master DataFrame containing ALL cases
    df = pd.DataFrame(results)
    print(f"Successfully processed {len(df)} cases.")

    with PdfPages(output_pdf) as pdf:
        
        # =========================================================
        # PAGE 1: GLOBAL STATISTICS (USES ALL DATA)
        # =========================================================
        fig = plt.figure(figsize=(11, 8.5))
        plt.suptitle(f"Global Performance Report (N={len(df)})", fontsize=18, fontweight='bold')
        
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)
        
        # Plot 1: Correlation vs Lag (Scatter)
        # Helps identify if lags are causing poor correlation
        ax1 = fig.add_subplot(gs[0, 0])
        sns.scatterplot(data=df, x='lag', y='corr', hue='exp', style='exp', palette='deep', ax=ax1, s=60)
        ax1.axvline(0, color='red', linestyle='--', alpha=0.4)
        ax1.set_title("Synchronization: Lag vs Correlation")
        ax1.set_xlabel("Lag (Samples)")
        ax1.set_ylabel("Pearson Correlation")
        
        # Plot 2: Amplitude Fidelity (Histogram)
        # --- NEW WAY (Datapoints) ---
        ax2 = fig.add_subplot(gs[0, 1])

        # 1. Draw the Box (Summary)
        sns.boxplot(data=df, x='amp_ratio', color='white', showfliers=False, ax=ax2)

        # 2. Draw the Points (Raw Data) - "The Bunch of Datapoints"
        # alpha=0.5 makes them semi-transparent so you can see density
        sns.stripplot(data=df, x='amp_ratio', color='green', alpha=0.5, size=4, jitter=True, ax=ax2)

        ax2.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='Ideal')
        ax2.set_title("Amplitude Ratio (Individual Cases)")
        ax2.set_xlabel("Ratio < 1.0 (Damped) | Ratio > 1.0 (Overshoot)")

        # Plot 3: Global RMSE Boxplot
        ax3 = fig.add_subplot(gs[1, :])
        sns.boxplot(data=df, x='exp', y='rmse', hue='exp', palette='Set2', ax=ax3, dodge=False)
        # Jitter plot to see density
        sns.stripplot(data=df, x='exp', y='rmse', color='black', alpha=0.3, size=3, ax=ax3)
        ax3.set_title("Error Distribution by Experiment Type")
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # =========================================================
        # PAGE 2+: DETAILED CASES (TOP 2 / BOTTOM 2)
        # =========================================================
        
        # Sort to find extremes
        df_sorted = df.sort_values('rmse')
        
        # Top 2 Best
        for i, (idx, row) in enumerate(df_sorted.head(2).iterrows()):
            plot_single_case_page(pdf, row, f"BEST CASE #{i+1}", "#2ca02c") # Green Title

        # Bottom 2 Worst
        for i, (idx, row) in enumerate(df_sorted.tail(2).iterrows()):
            plot_single_case_page(pdf, row, f"WORST CASE #{i+1}", "#d62728") # Red Title

    print(f"Report saved to: {output_pdf}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <output_pdf> <input_dir1> [input_dir2 ...]")
    else:
        generate_report(sys.argv[2:], sys.argv[1])