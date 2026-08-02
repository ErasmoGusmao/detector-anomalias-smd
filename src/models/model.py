"""Modelo de detecção de anomalias do SMD (autoencoder em PyTorch).

Abordagem não-supervisionada por **erro de reconstrução**: o autoencoder é
treinado apenas com dados normais (o conjunto de treino do SMD não contém
anomalias) e aprende a reconstruir janelas temporais das métricas. Em
inferência, janelas com erro de reconstrução alto são candidatas a anomalia.

O modelo opera sobre **janelas deslizantes** de tamanho ``WINDOW_SIZE``: cada
amostra tem shape ``(window_size, n_features)`` e o ``forward`` devolve a
reconstrução com o mesmo shape.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

from src.utils import config


class Autoencoder(nn.Module):
    """Autoencoder para reconstrução de janelas temporais do SMD.

    O encoder comprime uma janela ``(window_size, input_dim)`` até um espaço
    latente de dimensão ``latent_dim``; o decoder reconstrói a janela original.
    A arquitetura interna (MLP sobre a janela achatada, LSTM, conv. 1-D, etc.)
    fica a critério da implementação, desde que ``forward`` preserve o shape.
    """

    def __init__(
        self,
        input_dim: int,
        window_size: int = config.WINDOW_SIZE,
        hidden_dims: tuple[int, ...] = config.HIDDEN_DIMS,
        latent_dim: int = config.LATENT_DIM,
    ) -> None:
        """Constrói as camadas do encoder e do decoder.

        Args:
            input_dim: Número de features por timestep (colunas do SMD).
            window_size: Tamanho da janela temporal.
            hidden_dims: Dimensões das camadas ocultas do encoder (o decoder
                as espelha na ordem inversa).
            latent_dim: Dimensão do espaço latente (gargalo).
        """
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim deve ser maior que zero")
        if window_size <= 0:
            raise ValueError("window_size deve ser maior que zero")
        if latent_dim <= 0:
            raise ValueError("latent_dim deve ser maior que zero")
        if any(hidden_dim <= 0 for hidden_dim in hidden_dims):
            raise ValueError("hidden_dims deve conter apenas valores maiores que zero")

        self.input_dim = input_dim
        self.window_size = window_size
        self.flat_dim = input_dim * window_size

        encoder_dims = (self.flat_dim, *hidden_dims, latent_dim)
        decoder_dims = (latent_dim, *reversed(hidden_dims), self.flat_dim)

        self.encoder = self._build_mlp(encoder_dims)
        self.decoder = self._build_mlp(decoder_dims)

    @staticmethod
    def _build_mlp(dimensions: tuple[int, ...]) -> nn.Sequential:
        """Monta um MLP com ReLU entre as camadas lineares."""
        layers: list[nn.Module] = []
        for index, (in_features, out_features) in enumerate(
            zip(dimensions, dimensions[1:])
        ):
            layers.append(nn.Linear(in_features, out_features))
            if index < len(dimensions) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstrói o lote de janelas informado.

        Args:
            x: Tensor de janelas com shape ``(batch, window_size, input_dim)``.

        Returns:
            Reconstrução com o mesmo shape de ``x``.
        """
        if x.ndim != 3:
            raise ValueError(
                "x deve ter shape (batch, window_size, input_dim); "
                f"recebido tensor com {x.ndim} dimensões"
            )
        if x.shape[1] != self.window_size or x.shape[2] != self.input_dim:
            raise ValueError(
                "shape incompatível: esperado "
                f"(batch, {self.window_size}, {self.input_dim}), "
                f"recebido {tuple(x.shape)}"
            )

        flattened = x.reshape(x.shape[0], self.flat_dim)
        latent = self.encoder(flattened)
        reconstructed = self.decoder(latent)
        return reconstructed.reshape(x.shape[0], self.window_size, self.input_dim)


def create_model(
    input_dim: int,
    window_size: int = config.WINDOW_SIZE,
    hidden_dims: tuple[int, ...] = config.HIDDEN_DIMS,
    latent_dim: int = config.LATENT_DIM,
    device: str = config.DEVICE,
) -> Autoencoder:
    """Instancia o autoencoder e o move para o device de execução.

    Args:
        input_dim: Número de features por timestep (``X.shape[1]``).
        window_size: Tamanho da janela temporal.
        hidden_dims: Dimensões das camadas ocultas do encoder.
        latent_dim: Dimensão do espaço latente.
        device: Device de destino ("cpu" ou "cuda").

    Returns:
        Instância de ``Autoencoder`` pronta para treino, já em ``device``.
    """
    target_device = torch.device(device)
    model = Autoencoder(
        input_dim=input_dim,
        window_size=window_size,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
    )
    return model.to(target_device)


def reconstruction_error(
    model: Autoencoder,
    X: np.ndarray,
    window_size: int = config.WINDOW_SIZE,
    device: str = config.DEVICE,
) -> np.ndarray:
    """Calcula o erro de reconstrução (MSE) por janela.

    Executa o modelo em modo de avaliação (sem gradiente) sobre as janelas
    deslizantes extraídas de ``X`` e devolve o erro médio de reconstrução das
    features no último timestep de cada janela.

    Args:
        model: Autoencoder treinado.
        X: Matriz de métricas padronizada, shape ``(n_amostras, n_features)``.
        window_size: Tamanho da janela temporal (deve casar com o do treino).
        device: Device de execução.

    Returns:
        Array 1-D com o erro de reconstrução do último timestep de cada janela,
        shape ``(n_amostras - window_size + 1,)``.

    Nota:
        Cada erro é associado ao último timestep de sua janela. Portanto, os
        rótulos correspondentes começam em ``y_test[window_size - 1:]``.
    """
    X_array = np.asarray(X, dtype=np.float32)

    if X_array.ndim != 2:
        raise ValueError("X deve ter shape (n_amostras, n_features)")
    if window_size <= 0:
        raise ValueError("window_size deve ser maior que zero")
    if window_size != model.window_size:
        raise ValueError(
            f"window_size={window_size} difere do usado pelo modelo "
            f"({model.window_size})"
        )
    if X_array.shape[1] != model.input_dim:
        raise ValueError(
            f"X possui {X_array.shape[1]} features, mas o modelo espera "
            f"{model.input_dim}"
        )
    if X_array.shape[0] < window_size:
        raise ValueError(
            "X deve conter ao menos window_size amostras para formar uma janela"
        )
    if not np.isfinite(X_array).all():
        raise ValueError("X deve conter apenas valores finitos")

    target_device = torch.device(device)
    model.to(target_device)
    model.eval()

    number_of_windows = X_array.shape[0] - window_size + 1
    batch_size = max(1, config.BATCH_SIZE)
    error_batches: list[np.ndarray] = []

    with torch.inference_mode():
        for batch_start in range(0, number_of_windows, batch_size):
            batch_end = min(batch_start + batch_size, number_of_windows)
            windows = sliding_window_view(X_array[batch_start : batch_end + window_size - 1],
                                          window_shape=window_size,
                                          axis=0,
                                         )
            windows = np.moveaxis(windows, -1, 1)
            batch = torch.from_numpy(np.ascontiguousarray(windows)).to(target_device)
            reconstructed = model(batch)
            errors = torch.mean(
                (reconstructed[:, -1, :] - batch[:, -1, :]) ** 2,
                dim=1,
            )
            error_batches.append(errors.cpu().numpy())

    return np.concatenate(error_batches)
