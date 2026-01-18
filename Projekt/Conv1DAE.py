import torch
from torch import nn
from torch import optim
import numpy as np


class Conv1DAE(nn.Module):

    def __init__(self, input_dim: int = 1, latent_dim: int = 8, device='cuda', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, latent_dim, 11, stride=1, padding=1, device=device),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2, ceil_mode=False),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, input_dim, 11, stride=2, padding=1, output_padding=1, device=device)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DetailPreservingLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        mse = torch.mean((y_pred - y_true) ** 2, dim=-1)
        mae = torch.mean(torch.abs(y_pred - y_true), dim=-1)

        grad_true = y_true[..., 1:] - y_true[..., :-1]
        grad_pred = y_pred[..., 1:] - y_pred[..., :-1]

        grad_loss = torch.mean((grad_true - grad_pred) ** 2, dim=-1)

        return self.alpha * mse + self.beta * mae + self.gamma * grad_loss


def conv1dae_train(
        model: Conv1DAE,
        data: torch.Tensor,
        epochs: int = 50,
        lr: float = 1e-3,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        device: str = 'cuda'
):
    model.train()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = DetailPreservingLoss(alpha=alpha, beta=beta, gamma=gamma)

    train_losses = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        n_batches = data.shape[1]

        for batch in data:
            reconstruction = model(batch)
            loss = torch.mean(criterion(reconstruction, batch))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / n_batches)

    return train_losses


def detect_anomalies_conv1dae(
        model: Conv1DAE,
        data: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        threshold_percentile: int = 95,
        device='cuda'
):
    model.eval()
    model = model.to(device)

    reconstruction_errors, reconstruction = [], []
    loss = DetailPreservingLoss(alpha=alpha, beta=beta, gamma=gamma)

    with torch.no_grad():
        for sequence in data:
            seq_reconstruction = model(sequence)
            validation_loss = loss(seq_reconstruction, sequence)
            seq_errors = validation_loss.cpu().numpy()

            reconstruction_errors.append(seq_errors)
            reconstruction.append(torch.squeeze(seq_reconstruction, 1).cpu().numpy())

    reconstruction_errors = np.asarray(reconstruction_errors)
    reconstruction = np.asarray(reconstruction)
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    predictions = np.where(reconstruction_errors > threshold, -1, 1)

    return predictions.squeeze(), reconstruction_errors.squeeze(), threshold, reconstruction