"""Training Script for Iceberg Drift LSTM Model.

Loads all 365 daily Antarctic iceberg drift history files, formats sliding-window
sequences, trains the IcebergDriftLSTM neural network, and saves the checkpoint.
"""

import os
import glob
import json
import time
import math
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from iceberg_lstm import IcebergDriftLSTM, extract_step_features

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainLSTM")


def load_dataset(data_dir: Path, window_size: int = 6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and process sliding windows from all drift history JSON files."""
    json_files = sorted(glob.glob(str(data_dir / "drift_history_2026-*.json")))
    if not json_files:
        raise FileNotFoundError(f"No drift history JSON files found in {data_dir}")

    logger.info(f"Ingesting {len(json_files)} daily drift history files...")

    X_list = []
    Y_list = []

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        steps = data.get("steps", [])
        if len(steps) < window_size + 2:
            continue

        ref_lat = steps[0]["lat"]
        ref_lon = steps[0]["lon"]

        features = [extract_step_features(s, ref_lat, ref_lon) for s in steps]

        # Extract sliding windows
        for i in range(len(features) - window_size):
            x_window = features[i : i + window_size]
            target_step = steps[i + window_size]
            prev_step = steps[i + window_size - 1]

            # Delta lat, delta lon, next u, next v
            delta_lat = target_step["lat"] - prev_step["lat"]
            delta_lon = target_step["lon"] - prev_step["lon"]
            u_next = target_step.get("u", 0.0)
            v_next = target_step.get("v", 0.0)

            y_target = [delta_lat, delta_lon, u_next, v_next]

            X_list.append(x_window)
            Y_list.append(y_target)

    X_arr = np.array(X_list, dtype=np.float32)
    Y_arr = np.array(Y_list, dtype=np.float32)

    logger.info(f"Dataset generated: {len(X_arr):,} sequence samples. Shape: X={X_arr.shape}, Y={Y_arr.shape}")
    return torch.tensor(X_arr), torch.tensor(Y_arr)


def train_lstm(
    data_dir: Path,
    checkpoints_dir: Path,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.2,
):
    """Execute complete training loop for IcebergDriftLSTM."""
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    X, Y = load_dataset(data_dir)

    # Train / Val Split
    n_samples = len(X)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    indices = np.random.permutation(n_samples)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_dataset = TensorDataset(X[train_idx], Y[train_idx])
    val_dataset = TensorDataset(X[val_idx], Y[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = IcebergDriftLSTM(input_dim=6, hidden_dim=64, num_layers=2, output_dim=4, dropout=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    history = []

    logger.info(f"Starting LSTM training for {epochs} epochs on CPU...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds, sigma = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_x)

        train_loss /= n_train
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds, sigma = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * len(batch_x)

        val_loss /= n_val

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": round(scheduler.get_last_lr()[0], 6),
        })

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoints_dir / "iceberg_lstm.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "input_dim": 6,
                "hidden_dim": 64,
                "num_layers": 2,
                "output_dim": 4,
            }, checkpoint_path)
            logger.info(f"⭐ Saved new best checkpoint to {checkpoint_path}")

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f}s. Best Val Loss: {best_val_loss:.6f}")

    # Save history
    with open(checkpoints_dir / "lstm_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return model, history


if __name__ == "__main__":
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    checkpoints_dir = project_root / "checkpoints_local"

    train_lstm(data_dir=data_dir, checkpoints_dir=checkpoints_dir, epochs=15)
