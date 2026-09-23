"""Iceberg Drift LSTM Neural Network Architecture.

Provides PyTorch IcebergDriftLSTM model for forecasting spatial iceberg drift trajectories,
drift velocities (u, v), and position uncertainty cones over multi-hour horizons.
"""

import math
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np


class IcebergDriftLSTM(nn.Module):
    """Deep LSTM model for spatio-temporal iceberg trajectory and drift forecasting."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Number of input features per timestep [d_lat, d_lon, u, v, speed, heading_rad].
            hidden_dim: Number of hidden units in LSTM layers.
            num_layers: Number of stacked LSTM layers.
            output_dim: Number of predicted features [delta_lat, delta_lon, u_next, v_next].
            dropout: Dropout probability between recurrent layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

        # Uncertainty variance head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus(),  # guarantee positive uncertainty sigma
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, input_dim)
        Returns:
            predictions: (Batch, output_dim) [delta_lat, delta_lon, u, v]
            uncertainty: (Batch, 1) sigma
        """
        lstm_out, (hn, cn) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # (Batch, hidden_dim)

        preds = self.head(last_hidden)
        sigma = self.uncertainty_head(last_hidden)
        return preds, sigma


def extract_step_features(step: Dict[str, Any], ref_lat: float, ref_lon: float) -> List[float]:
    """Convert a raw drift step into normalized relative input features."""
    d_lat = step.get("lat", 0.0) - ref_lat
    d_lon = step.get("lon", 0.0) - ref_lon
    u = step.get("u", 0.0)
    v = step.get("v", 0.0)
    speed = step.get("speed", math.sqrt(u**2 + v**2))
    heading_deg = step.get("heading", 0.0)
    heading_rad = math.radians(heading_deg)

    return [d_lat, d_lon, u, v, speed, heading_rad]


def predict_trajectory(
    model: IcebergDriftLSTM,
    history_steps: List[Dict[str, Any]],
    forecast_hours: int = 24,
    dt_hours: float = 1.0,
) -> List[Dict[str, Any]]:
    """Generate recursive multi-step forward drift trajectory predictions.

    Args:
        model: Trained IcebergDriftLSTM instance.
        history_steps: List of chronological past steps (min length 4).
        forecast_hours: Forecast horizon in hours.
        dt_hours: Interval step between predictions.

    Returns:
        List of forecasted steps with [lat, lon, u, v, speed, pos_uncertainty_km, elapsed_h].
    """
    model.eval()
    if not history_steps:
        return []

    # Reference origin
    ref_lat = history_steps[0]["lat"]
    ref_lon = history_steps[0]["lon"]

    seq_features = [extract_step_features(s, ref_lat, ref_lon) for s in history_steps[-8:]]
    curr_lat = history_steps[-1]["lat"]
    curr_lon = history_steps[-1]["lon"]
    curr_t = history_steps[-1].get("t", 0.0)
    curr_elapsed = history_steps[-1].get("elapsed_h", 0.0)

    future_steps = []
    num_steps = int(forecast_hours / dt_hours)

    with torch.no_grad():
        current_seq = torch.tensor([seq_features], dtype=torch.float32)

        for step_idx in range(1, num_steps + 1):
            preds, sigma = model(current_seq)
            pred_vals = preds[0].numpy()
            d_lat_step, d_lon_step, u_next, v_next = pred_vals
            sigma_val = float(sigma[0, 0].item())

            # Accumulate spatial progression
            curr_lat += float(d_lat_step)
            curr_lon += float(d_lon_step)
            curr_elapsed += dt_hours
            curr_t += dt_hours * 3600.0

            speed_next = math.sqrt(u_next**2 + v_next**2)
            heading_deg = (math.degrees(math.atan2(v_next, u_next)) + 360.0) % 360.0

            # Progressive uncertainty growth (m)
            uncertainty_m = max(50.0, sigma_val * 1000.0 * math.sqrt(step_idx))

            step_data = {
                "step": step_idx,
                "elapsed_h": round(curr_elapsed, 2),
                "lat": round(curr_lat, 5),
                "lon": round(curr_lon, 5),
                "u": round(float(u_next), 4),
                "v": round(float(v_next), 4),
                "speed": round(float(speed_next), 4),
                "heading": round(float(heading_deg), 1),
                "pos_uncertainty_m": round(uncertainty_m, 1),
                "uncertainty_radius_km": round(uncertainty_m / 1000.0, 2),
            }
            future_steps.append(step_data)

            # Recursive input window update
            next_feat = extract_step_features(step_data, ref_lat, ref_lon)
            next_feat_tensor = torch.tensor([[next_feat]], dtype=torch.float32)
            current_seq = torch.cat([current_seq[:, 1:, :], next_feat_tensor], dim=1)

    return future_steps
