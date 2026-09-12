"""1D-CNN IDS oracle over the flow vector."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


class CNNIDS(nn.Module):
    def __init__(
        self,
        n_features: int,
        conv_channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        dropout: float = 0.3,
        dense_hidden: int = 128,
    ) -> None:
        super().__init__()
        conv_channels = conv_channels or [32, 64, 128]
        kernel_sizes = kernel_sizes or [5, 5, 3]

        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, k in zip(conv_channels, kernel_sizes, strict=True):
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # Compute flattened size after conv/pool
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_features)
            flat = self.conv(dummy).view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) -> (batch, 1, n_features)
        x = x.unsqueeze(1)
        return self.head(self.conv(x)).squeeze(-1)


def train_ids_cnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    device: torch.device | None = None,
) -> CNNIDS:
    """Train a small 1D-CNN IDS on tabular flow data."""
    params = params or {}
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = x_train.shape[1]

    model = CNNIDS(
        n_features=n_features,
        conv_channels=list(params.get("conv_channels", [32, 64, 128])),
        kernel_sizes=list(params.get("kernel_sizes", [5, 5, 3])),
        dropout=float(params.get("dropout", 0.3)),
        dense_hidden=int(params.get("dense_hidden", 128)),
    ).to(device)

    epochs = int(params.get("epochs", 30))
    batch_size = int(params.get("batch_size", 512))
    lr = float(params.get("lr", 1e-3))
    weight_decay = float(params.get("weight_decay", 1e-4))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    return model
