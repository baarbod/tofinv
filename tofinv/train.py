# tofinv/train_inverse.py

import os
import pickle
import logging
import argparse
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tofinv.nn_models import TOFinverse
import tofinv.utils as utils
import tofinv.noise_injection as noise

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class SurrogateConv1D(nn.Module):
    """Re-definition of the Surrogate architecture for loading weights."""
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
    def forward(self, x): return self.network(x)

def load_dataset(dataset_path, noisedir=None, noise_method='none', gauss_low=0.01, gauss_high=0.1, scalemax=0.1):
    """Loads, noises, scales, and splits the dataset."""
    logger.info(f"Loading raw dataset from {dataset_path}")
    with open(dataset_path, "rb") as f:
        X, y = pickle.load(f)
    
    n_samples = X.shape[0]
    nslice_to_use = X.shape[1] - 2
    area_idx = nslice_to_use + 1
    
    # Noise Injection
    if noise_method == 'gaussian':
        logger.info(f"Injecting Gaussian noise (range: {gauss_low}-{gauss_high})")
        X = noise.add_gaussian_noise(X, gauss_low=gauss_low, gauss_high=gauss_high)
    elif noise_method == 'pca':
        logger.info(f"Injecting PCA-based noise from {noisedir}")
        path_to_pca_model = os.path.join(noisedir, 'pca_model.pkl')
        model = noise.load_pca_model(path_to_pca_model)
        X = noise.add_pca_noise(X, model, scalemax=scalemax)

    # Scaling and Normalization
    logger.info("Applying signal scaling and area normalization...")
    for i in range(n_samples):
        # Scale TOF signals
        to_scale = X[i, :nslice_to_use, :].T 
        X[i, :nslice_to_use, :] = utils.scale_data(to_scale).T 
        # Normalize Area
        X[i, area_idx, :] = utils.scale_area(X[i, nslice_to_use, :], X[i, area_idx, :])
        
    flow_x = X[:, :nslice_to_use, :]
    area_x = X[:, area_idx : area_idx + 1, :]
    
    # Train/Test Split
    idx_train, idx_test = train_test_split(np.arange(n_samples), test_size=0.1, random_state=42)
    
    def to_torch(arr): return torch.tensor(arr, dtype=torch.float32)

    return (to_torch(flow_x[idx_train]), to_torch(area_x[idx_train]), to_torch(y[idx_train]),
            to_torch(flow_x[idx_test]), to_torch(area_x[idx_test]), to_torch(y[idx_test]))

def run_epoch(loader, model, surrogate, criterion, optimizer=None, lambda_phys=1.0, device='cpu'):
    """Handles both training and evaluation logic."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss, total_v_loss, total_p_loss = 0.0, 0.0, 0.0
    
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for flow_in, area_in, v_true in loader:
            flow_in, area_in, v_true = flow_in.to(device), area_in.to(device), v_true.to(device)
            
            if is_train: optimizer.zero_grad()
            
            # Predict Velocity
            v_pred = model(flow_in, area_in) 
            loss_v = criterion(v_pred, v_true) 
            
            # Physics Regularization (Surrogate Feedback)
            loss_phys = torch.tensor(0.0, device=device)
            if lambda_phys > 0:
                # Concatenate predicted velocity and area for surrogate input
                surr_in = torch.cat([v_pred, area_in], dim=1)
                signal_pred = surrogate(surr_in)
                loss_phys = criterion(signal_pred, flow_in)
            
            loss_total = loss_v + (lambda_phys * loss_phys)

            if is_train:
                loss_total.backward()
                optimizer.step()
            
            total_loss += loss_total.item() * flow_in.size(0)
            total_v_loss += loss_v.item() * flow_in.size(0)
            total_p_loss += loss_phys.item() * flow_in.size(0)
            
    n = len(loader.dataset)
    return total_loss/n, total_v_loss/n, total_p_loss/n

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing TOFinverse Training on {device}")

    # 1. Load Data
    f_tr, a_tr, y_tr, f_te, a_te, y_te = load_dataset(
        args.dataset, args.noisedir, args.noise_method, 
        args.gauss_low, args.gauss_high, args.noise_scale
    )
    
    train_loader = DataLoader(TensorDataset(f_tr, a_tr, y_tr), batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(TensorDataset(f_te, a_te, y_te), batch_size=args.batch)

    # 2. Model Setup
    model = TOFinverse(nflow_in=f_tr.shape[1], nfeature_out=y_tr.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # 3. Load Frozen Surrogate
    logger.info(f"Loading pre-trained surrogate: {args.surrogate_path}")
    surrogate = SurrogateConv1D(in_channels=2, out_channels=f_tr.shape[1]).to(device)
    surrogate.load_state_dict(torch.load(args.surrogate_path, map_location=device, weights_only=True))
    surrogate.eval()
    for p in surrogate.parameters(): p.requires_grad = False

    # 4. Training Loop
    best_loss = float('inf')
    patience_cnt = 0
    history = {'train': [], 'test': []}

    for epoch in range(args.epochs):
        tr_l, tr_v, tr_p = run_epoch(train_loader, model, surrogate, nn.MSELoss(), optimizer, args.lambda_phys, device)
        te_l, te_v, te_p = run_epoch(test_loader, model, surrogate, nn.MSELoss(), None, args.lambda_phys, device)
        
        scheduler.step(te_l)
        history['train'].append(tr_l); history['test'].append(te_l)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:03d} | Train: {tr_l:.5f} (Phys: {tr_p:.5f}) | Test: {te_l:.5f} (Phys: {te_p:.5f})")

        # Early Stopping & Checkpointing
        if te_l < best_loss - args.min_delta:
            best_loss = te_l
            patience_cnt = 0
            torch.save(model.state_dict(), args.out_weights)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    # 5. Finalize
    np.savez(os.path.join(args.outdir, 'history.npz'), **history)
    logger.info(f"Training Complete. Best Validation Loss: {best_loss:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--surrogate_path", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--out_weights", required=True)
    parser.add_argument("--noise_method", default='pca', choices=['none', 'gaussian', 'pca'])
    parser.add_argument("--noisedir", default=None)
    parser.add_argument("--noise_scale", type=float, default=0.2)
    parser.add_argument("--gauss_low", type=float, default=0.01)
    parser.add_argument("--gauss_high", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_phys", type=float, default=0.5, help="Weight of the physics loss")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=1e-5)
    main(parser.parse_args())