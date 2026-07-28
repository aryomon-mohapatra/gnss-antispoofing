"""
lstm_autoencoder.py
-------------------
LSTM Autoencoder for temporal anomaly detection.
Trained ONLY on genuine GNSS signals.
Spoofed signals produce high reconstruction error (MSE).
"""

import torch
import torch.nn as nn
import numpy as np


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len, features)
        _, (hidden, _) = self.lstm(x)
        # Use last layer's hidden state
        latent = self.fc(hidden[-1])
        return latent


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_size, output_size, seq_len, num_layers, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_fc = nn.Linear(hidden_size, output_size)

    def forward(self, latent):
        # Expand latent to sequence
        x = self.fc(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(x)
        return self.output_fc(out)  # (batch, seq_len, features)


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder.
    Reconstruction MSE per epoch is the anomaly score.
    """
    def __init__(self, input_size=18, hidden_size=128, latent_dim=64,
                 num_layers=2, dropout=0.2, seq_len=30):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = LSTMEncoder(input_size, hidden_size, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_size, input_size, seq_len, num_layers, dropout)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def reconstruction_error(self, x):
        """Per-sample MSE between input and reconstruction."""
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            # MSE per sample (mean over seq and features)
            mse = ((x - recon) ** 2).mean(dim=(1, 2))
        return mse.cpu().numpy()

    def anomaly_score(self, x, threshold):
        """Returns normalized score in [0, 1] relative to threshold."""
        mse = self.reconstruction_error(x)
        return np.clip(mse / (threshold * 2), 0, 1)


def build_lstm_model(cfg):
    model = LSTMAutoencoder(
        input_size=18,
        hidden_size=cfg["lstm_autoencoder"]["hidden_size"],
        latent_dim=cfg["lstm_autoencoder"]["latent_dim"],
        num_layers=cfg["lstm_autoencoder"]["num_layers"],
        dropout=cfg["lstm_autoencoder"]["dropout"],
        seq_len=cfg["data"]["window_size"],
    )
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[LSTM] Parameters: {total:.2f}M")
    return model
