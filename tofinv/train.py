# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import pickle
import time 
import argparse
import json

from tofinv.nn_models import TOFinverse
import tofinv.utils as utils
import tofinv.noise_injection as noise

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


def load_dataset(dataset, noisedir=None, noise_method='none', gauss_low=0.01, gauss_high=0.1, scalemax=0.1):
    with open(dataset, "rb") as f:
        X, y = pickle.load(f)
    
    nslice_to_use = X.shape[1] - 2
    pos_idx = nslice_to_use
    area_idx = nslice_to_use + 1
    
    if noise_method == 'gaussian':
        print("Adding Gaussian noise...")
        X = noise.add_gaussian_noise(X, gauss_low=gauss_low, gauss_high=gauss_high)
        print("Gaussian noise injection complete.")

    elif noise_method == 'pca':
        if noisedir is None:
            raise ValueError("noisedir must be provided when using PCA-based noise.")
        path_to_noise_data = os.path.join(noisedir, 'noise_data.pkl')
        path_to_pca_model = os.path.join(noisedir, 'pca_model.pkl')
        print("Adding PCA-based noise...")

        if os.path.exists(path_to_pca_model):
            model = noise.load_pca_model(path_to_pca_model)
        else:
            if not os.path.exists(path_to_noise_data):
                raise FileNotFoundError("Noise data not found.")
            noise_data = noise.load_noise_data(path_to_noise_data)
            model = noise.define_pca_model(noise_data)
            noise.save_pca_model(model, path_to_pca_model)
        X = noise.add_pca_noise(X, model, scalemax=scalemax)

    # Scale signals and area
    for i in range(X.shape[0]):
        to_scale = X[i, :nslice_to_use, :].T 
        scaled = utils.scale_data(to_scale).T
        X[i, :nslice_to_use, :] = scaled 
        
        xarea_single = X[i, pos_idx, :]
        area_single = X[i, area_idx, :]
        X[i, area_idx, :] = utils.scale_area(xarea_single, area_single)
        
    # Split the inputs
    flow_x = X[:, :nslice_to_use, :]
    area_x = X[:, area_idx : area_idx + 1, :]
    
    indices = np.arange(X.shape[0])
    idx_train, idx_test = train_test_split(indices, test_size=0.1)

    flow_train, flow_test = flow_x[idx_train], flow_x[idx_test]
    area_train, area_test = area_x[idx_train], area_x[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    if torch.cuda.is_available():
        flow_train = torch.tensor(flow_train, dtype=torch.float32).cuda()
        area_train = torch.tensor(area_train, dtype=torch.float32).cuda()
        y_train = torch.tensor(y_train, dtype=torch.float32).cuda()
        flow_test = torch.tensor(flow_test, dtype=torch.float32).cuda()
        area_test = torch.tensor(area_test, dtype=torch.float32).cuda()
        y_test = torch.tensor(y_test, dtype=torch.float32).cuda()
    else:
        flow_train = torch.tensor(flow_train, dtype=torch.float32)
        area_train = torch.tensor(area_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        flow_test = torch.tensor(flow_test, dtype=torch.float32)
        area_test = torch.tensor(area_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

    return flow_train, area_train, flow_test, area_test, y_train, y_test


def train(loader, model, surrogate_model, criterion, optimizer, lambda_phys):
    model.train()
    epoch_loss, epoch_v_loss, epoch_phys_loss = 0.0, 0.0, 0.0
    
    for flow_inputs, area_inputs, labels in loader:
        optimizer.zero_grad()
        
        # Predict Velocity
        pred_v = model(flow_inputs, area_inputs) 
        loss_v = criterion(pred_v, labels) 
        
        # Physics Loss
        if lambda_phys > 0:
            surrogate_in = torch.cat([pred_v, area_inputs], dim=1)
            pred_signal = surrogate_model(surrogate_in)
            loss_physics = criterion(pred_signal, flow_inputs)
            total_loss = loss_v + (lambda_phys * loss_physics)
        else:
            loss_physics = torch.tensor(0.0)
            total_loss = loss_v

        total_loss.backward()
        optimizer.step()
        
        epoch_loss += total_loss.item() * flow_inputs.size(0)
        epoch_v_loss += loss_v.item() * flow_inputs.size(0)
        epoch_phys_loss += loss_physics.item() * flow_inputs.size(0)
        
    n = len(loader.dataset)
    return epoch_loss / n, epoch_v_loss / n, epoch_phys_loss / n


def test(loader, model, surrogate_model, criterion, lambda_phys):
    model.eval()
    epoch_loss, epoch_v_loss, epoch_phys_loss = 0.0, 0.0, 0.0
    with torch.no_grad():
        for flow_inputs, area_inputs, labels in loader:
            pred_v = model(flow_inputs, area_inputs)
            loss_v = criterion(pred_v, labels)
            
            if lambda_phys > 0:
                surrogate_in = torch.cat([pred_v, area_inputs], dim=1)
                pred_signal = surrogate_model(surrogate_in)
                loss_physics = criterion(pred_signal, flow_inputs)
                total_loss = loss_v + (lambda_phys * loss_physics)
            else:
                loss_physics = torch.tensor(0.0)
                total_loss = loss_v
                
            epoch_loss += total_loss.item() * flow_inputs.size(0)
            epoch_v_loss += loss_v.item() * flow_inputs.size(0)
            epoch_phys_loss += loss_physics.item() * flow_inputs.size(0)
            
    n = len(loader.dataset)
    return epoch_loss / n, epoch_v_loss / n, epoch_phys_loss / n


def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    flow_train, area_train, flow_test, area_test, y_train, y_test = load_dataset(
        args.dataset, noisedir=args.noisedir, noise_method=args.noise_method, 
        gauss_low=args.gauss_low, gauss_high=args.gauss_high, scalemax=args.noise_scale
    )
    
    train_dataset = torch.utils.data.TensorDataset(flow_train, area_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(flow_test, area_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    
    nflow_in = flow_train.shape[1]
    num_out_features = y_train.shape[1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running training using {device}")
    
    model = TOFinverse(nflow_in=nflow_in, nfeature_out=num_out_features)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
    )

    criterion = nn.MSELoss()
    
    # Load Surrogate (1 V + 1 Area = 2 in_channels)
    print(f"Loading surrogate model from {args.surrogate_path}...")
    surrogate_model = SurrogateConv1D(in_channels=2, out_channels=nflow_in)
    surrogate_model.load_state_dict(torch.load(args.surrogate_path, map_location=device, weights_only=True))
    surrogate_model.to(device)
    surrogate_model.eval()
    for param in surrogate_model.parameters():
        param.requires_grad = False
    print("Surrogate model loaded and frozen.")
    
    config = vars(args)
    with open(os.path.join(args.outdir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
        
    train_losses, test_losses = [], []
    best_loss = float('inf')
    patience_counter = 0
    time_start = time.time()
    
    for epoch in range(args.epochs):
        train_loss, train_v, train_phys = train(train_loader, model, surrogate_model, criterion, optimizer, args.lambda_phys)
        test_loss, test_v, test_phys = test(test_loader, model, surrogate_model, criterion, args.lambda_phys)
        
        scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{args.epochs}] | LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} (V: {train_v:.4f}, Phys: {train_phys:.4f}) | Test Loss: {test_loss:.4f} (V: {test_v:.4f}, Phys: {test_phys:.4f})')
        
        if epoch >= args.warmup_epochs:
            if test_loss < best_loss - args.min_delta:
                best_loss = test_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': test_loss
                }, args.out_weights)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                    
    time_end = time.time()
    print(f"Total training time: {time_end - time_start:.2f} seconds")

    history = {'train_losses': train_losses, 'test_losses': test_losses}
    np.savez(os.path.join(args.outdir, 'training_history.npz'), **history)
    
    fig, ax= plt.subplots()
    ax.plot(train_losses, label='Train Loss')
    ax.plot(test_losses, label='Test Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.legend()
    ax.grid()
    
    plot_path = os.path.join(args.outdir, 'loss_vs_epochs.png')
    fig.savefig(plot_path)
    print(f"Saved loss plot to {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train TOFinverse model")
    
    # Core IO paths
    parser.add_argument("--dataset", type=str, required=True, help="Path to the input dataset pickle file")
    parser.add_argument("--surrogate_path", type=str, required=True, help="Path to the pretrained surrogate model weights")
    parser.add_argument("--outdir", type=str, required=True, help="Directory to save logs and plots")
    parser.add_argument("--out_weights", type=str, required=True, help="Path to save the best model weights")
    
    # Noise arguments
    parser.add_argument("--noisedir", type=str, default=None, help="Directory containing PCA noise data")
    parser.add_argument("--noise_method", type=str, default='none', choices=['none', 'gaussian', 'pca'])
    parser.add_argument("--gauss_low", type=float, default=0.01)
    parser.add_argument("--gauss_high", type=float, default=0.1)
    parser.add_argument("--noise_scale", type=float, default=0.5)
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda_phys", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=10)

    args = parser.parse_args()
    main(args)