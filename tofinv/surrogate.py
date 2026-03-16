import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from pathlib import Path

from tofinv.utils import scale_data 

def rfft_freqs_and_spectrum(x, tr):
    N = len(x)
    freqs = np.fft.rfftfreq(N, d=tr)
    spectrum = np.abs(np.fft.rfft(x)) / N 
    return freqs, spectrum

def scale_area(xarea, area):
    middle_index = xarea.shape[0] // 2
    area_scaled = area / (area[middle_index] + 1e-6)
    return area_scaled

class SurrogateConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding='same'),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, padding='same'),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=5, padding='same'),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, out_channels, kernel_size=3, padding='same')
        )

    def forward(self, x):
        return self.network(x)

def main():
    parser = argparse.ArgumentParser(description="Train Surrogate Model")
    parser.add_argument("--dataset", required=True, help="Path to dataset.pkl")
    parser.add_argument("--out_weights", required=True, help="Path to save surrogate model weights")
    parser.add_argument("--outdir", required=True, help="Directory to save plots")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tr", type=float, default=0.378)
    parser.add_argument("--nslice_to_use", type=int, default=3)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    plots_dir = Path(args.outdir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    with open(args.dataset, "rb") as f:
        X_data, y_data = pickle.load(f)

    print("Scaling signal and area data...")
    pos_idx = args.nslice_to_use
    area_idx = args.nslice_to_use + 1

    for i in range(X_data.shape[0]):
        # Scale flow
        to_scale = X_data[i, :args.nslice_to_use, :].T  
        scaled = scale_data(to_scale).T            
        X_data[i, :args.nslice_to_use, :] = scaled      
        
        # Scale area
        xarea_single = X_data[i, pos_idx, :]
        area_single = X_data[i, area_idx, :]
        X_data[i, area_idx, :] = scale_area(xarea_single, area_single)

    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)

    V = y_tensor  
    
    # Extract only the scaled Area [Batch, 1, 300]
    A = X_tensor[:, area_idx : area_idx + 1, :]  
    
    # Inputs: 1 Vel + 1 Area = 2 channels
    surrogate_inputs = torch.cat([V, A], dim=1) 
    surrogate_targets = X_tensor[:, :args.nslice_to_use, :] 

    dataset = TensorDataset(surrogate_inputs, surrogate_targets)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    in_channels = surrogate_inputs.shape[1] # Should be 2
    out_channels = surrogate_targets.shape[1]

    model = SurrogateConv1D(in_channels, out_channels).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        model.train()
        batch_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            batch_train_loss += loss.item()
            
        model.eval()
        batch_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                batch_val_loss += loss.item()
                
        avg_train_loss = batch_train_loss / len(train_loader)
        avg_val_loss = batch_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    print("Training complete! Saving model weights...")
    torch.save(model.state_dict(), args.out_weights)

    model.eval()
    with torch.no_grad():
        idx = np.random.randint(len(val_dataset))
        sample_input, true_target = val_dataset[idx]
        sample_input_batch = sample_input.unsqueeze(0).to(DEVICE)
        pred_target = model(sample_input_batch).squeeze(0).cpu()

    # Input 0 is Velocity
    v_input = sample_input[0, :].numpy()  
    s_target = true_target.T.numpy()     
    ssim_opt_plot = pred_target.T.numpy() 
    nslice = s_target.shape[1]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    freqs, m_v = rfft_freqs_and_spectrum(v_input, tr=args.tr)
    axes[0, 0].plot(freqs[1:], m_v[1:], color='gray')
    axes[0, 0].set_title('Velocity PSD')
    axes[0, 0].set_ylabel('Magnitude')

    for ch in range(nslice):
        f, m = rfft_freqs_and_spectrum(s_target[:, ch], tr=args.tr)
        axes[0, 1].plot(f[1:], m[1:], alpha=0.7)
    axes[0, 1].set_title('Measured Signal PSD')

    for ch in range(nslice):
        f, m = rfft_freqs_and_spectrum(ssim_opt_plot[:, ch], tr=args.tr)
        axes[0, 2].plot(f[1:], m[1:], alpha=0.7)
    axes[0, 2].set_title('Predicted Signal PSD')

    axes[1, 0].plot(v_input, color='gray')
    axes[1, 0].set_title('Input Velocity ($V$)')
    axes[1, 0].set_ylabel('Amplitude')

    for ch in range(nslice):
        axes[1, 1].plot(s_target[:, ch], alpha=0.7)
    axes[1, 1].set_title('Measured Signal ($S_{true}$)')

    for ch in range(nslice):
        axes[1, 2].plot(ssim_opt_plot[:, ch], alpha=0.7)
    axes[1, 2].set_title('Surrogate Predicted ($S_{pred}$)')

    for i in range(3):
        axes[0, i].set_xlabel('Freq (Hz)')
        axes[1, i].set_xlabel('Frames')

    for row in range(2):
        ax_measured = axes[row, 1]
        ax_predicted = axes[row, 2]
        y_min = min(ax_measured.get_ylim()[0], ax_predicted.get_ylim()[0])
        y_max = max(ax_measured.get_ylim()[1], ax_predicted.get_ylim()[1])
        ax_measured.set_ylim(y_min, y_max)
        ax_predicted.set_ylim(y_min, y_max)
        ax_predicted.tick_params(labelleft=False)

    plt.tight_layout()
    fig.savefig(plots_dir / f'Surrogate_Combined_sample{idx}.png')
    plt.close(fig)

    fig_loss, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train Loss', color='royalblue', alpha=0.8)
    ax1.plot(val_losses, label='Val Loss', color='darkorange', linestyle='--', alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_title('Optimization History (Log Scale)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Mean Squared Error')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()

    last_pct = int(args.epochs * 0.2)
    epochs_range = range(args.epochs - last_pct, args.epochs)
    ax2.plot(epochs_range, train_losses[-last_pct:], color='royalblue', label='Train')
    ax2.plot(epochs_range, val_losses[-last_pct:], color='darkorange', linestyle='--', label='Val')
    ax2.set_title(f'Final Convergence (Last {last_pct} Epochs)')
    ax2.set_xlabel('Epochs')
    ax2.grid(True, alpha=0.2)
    ax2.legend()

    plt.tight_layout()
    fig_loss.savefig(plots_dir / 'loss_curves.png')
    plt.close(fig_loss)

if __name__ == "__main__":
    main()
    