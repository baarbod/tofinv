import os
import argparse
import pickle
import logging
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from pathlib import Path

from tofinv.utils import scale_data 

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

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
    parser.add_argument("--out_weights", required=True, help="Path to save weights")
    parser.add_argument("--outdir", required=True, help="Directory for plots")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tr", type=float, default=0.378)
    parser.add_argument("--nslice_to_use", type=int, required=True)
    parser.add_argument("--global_seed", type=int, default=42)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.global_seed)
    torch.manual_seed(args.global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.global_seed)
    logger.info(f"Using device: {DEVICE}")

    plots_dir = Path(args.outdir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset: {args.dataset}")
    with open(args.dataset, "rb") as f:
        X_data, y_data = pickle.load(f)
    X_data = np.asarray(X_data, dtype=np.float64).copy()
    y_data = np.asarray(y_data, dtype=np.float64)
    if X_data.ndim != 3 or y_data.ndim != 3 or X_data.shape[1] != args.nslice_to_use + 2:
        raise ValueError("Dataset channels do not match --nslice_to_use.")
    if X_data.shape[0] < 2 or X_data.shape[2] != y_data.shape[2]:
        raise ValueError("Dataset must contain at least two samples with matching time dimensions.")

    logger.info(f"Processing and scaling {X_data.shape[0]} samples...")
    pos_idx = args.nslice_to_use
    area_idx = args.nslice_to_use + 1

    for i in range(X_data.shape[0]):
        to_scale = X_data[i, :args.nslice_to_use, :].T  
        X_data[i, :args.nslice_to_use, :] = scale_data(to_scale).T 
        xarea_single = X_data[i, pos_idx, :]
        area_single = X_data[i, area_idx, :]
        X_data[i, area_idx, :] = scale_area(xarea_single, area_single)

    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)
    V = y_tensor  
    A = X_tensor[:, area_idx : area_idx + 1, :]  
    surrogate_inputs = torch.cat([V, A], dim=1) 
    surrogate_targets = X_tensor[:, :args.nslice_to_use, :] 
    dataset = TensorDataset(surrogate_inputs, surrogate_targets)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    split_generator = torch.Generator().manual_seed(args.global_seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_generator)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.global_seed))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    in_channels = surrogate_inputs.shape[1] 
    out_channels = surrogate_targets.shape[1]
    model = SurrogateConv1D(in_channels, out_channels).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)

    logger.info("Starting training loop...")
    train_losses, val_losses = [], []

    best_loss = float("inf")
    best_state = None
    patience, no_improvement = 100, 0
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
        
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            no_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            no_improvement += 1
            if no_improvement >= patience:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:04d} | Train MSE: {avg_train_loss:.7f} | Val MSE: {avg_val_loss:.7f}")

    logger.info(f"Saving model to {args.out_weights}")
    if best_state is None:
        raise RuntimeError("Surrogate training did not produce a valid checkpoint.")
    torch.save({
        "schema_version": 1,
        "model_state_dict": best_state,
        "architecture": {"in_channels": in_channels, "out_channels": out_channels},
        "training": {"global_seed": args.global_seed, "best_validation_mse": best_loss},
    }, args.out_weights)
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        idx = np.random.randint(len(val_dataset))
        sample_input, true_target = val_dataset[idx]
        pred_target = model(sample_input.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()

    logger.info("Generating diagnostic plots...")
    v_input = sample_input[0, :].numpy() 
    s_target = true_target.T.numpy() 
    s_pred = pred_target.T.numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    f_v, m_v = rfft_freqs_and_spectrum(v_input, tr=args.tr)
    axes[0, 0].plot(f_v[1:], m_v[1:], color='black', lw=1.5)
    axes[0, 0].set_title('Input Velocity PSD')
    
    for ch in range(out_channels):
        f, m = rfft_freqs_and_spectrum(s_target[:, ch], tr=args.tr)
        axes[0, 1].plot(f[1:], m[1:], alpha=0.6, label=f'Ch{ch}')
        f_p, m_p = rfft_freqs_and_spectrum(s_pred[:, ch], tr=args.tr)
        axes[0, 2].plot(f_p[1:], m_p[1:], alpha=0.6)

    axes[0, 1].set_title('Physics-Model PSD (Ground Truth)')
    axes[0, 2].set_title('Surrogate PSD (Prediction)')

    axes[1, 0].plot(v_input, color='black', lw=1.5)
    axes[1, 0].set_title('Input Velocity Profile')
    
    for ch in range(out_channels):
        axes[1, 1].plot(s_target[:, ch], alpha=0.6)
        axes[1, 2].plot(s_pred[:, ch], alpha=0.6)

    axes[1, 1].set_title('Physics-Model Signal')
    axes[1, 2].set_title('Surrogate Prediction')

    for ax in axes.flatten(): 
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(plots_dir / f'Surrogate_Validation_sample{idx}.png', dpi=150)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train MSE')
    plt.plot(val_losses, label='Val MSE', alpha=0.7)
    plt.yscale('log')
    plt.title('Surrogate Model Learning History')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, which='both', alpha=0.2)
    plt.savefig(plots_dir / 'training_history.png')
    
    logger.info("Training and validation summary complete.")

if __name__ == "__main__":
    main()
