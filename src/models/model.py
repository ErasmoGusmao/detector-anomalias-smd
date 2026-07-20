"""Modelo de detecção de anomalias do SMD (autoencoder em PyTorch).

Abordagem não-supervisionada por **erro de reconstrução**: o autoencoder é
treinado apenas com dados normais (o conjunto de treino do SMD não contém
anomalias) e aprende a reconstruir janelas temporais das métricas. Em
inferência, janelas com erro de reconstrução alto são candidatas a anomalia.

O modelo opera sobre **janelas deslizantes** de tamanho ``WINDOW_SIZE``: cada
amostra tem shape ``(window_size, n_features)`` e o ``forward`` devolve a
reconstrução com o mesmo shape.

ESQUELETO (Entrega 3): assinaturas e contratos definidos; a implementação do
corpo é tarefa da equipe (ver divisão de PRs no plano da Entrega 3).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

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
        # TODO(equipe): definir self.encoder e self.decoder (nn.Sequential/
        # nn.Module) a partir de input_dim, window_size, hidden_dims e
        # latent_dim. Guardar input_dim/window_size se forem úteis no forward.
        raise NotImplementedError("TODO(equipe): construir encoder e decoder")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstrói o lote de janelas informado.

        Args:
            x: Tensor de janelas com shape ``(batch, window_size, input_dim)``.

        Returns:
            Reconstrução com o mesmo shape de ``x``.
        """
        # TODO(equipe): x -> encoder -> latente -> decoder -> reconstrução.
        raise NotImplementedError("TODO(equipe): implementar o forward")


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
    # TODO(equipe): instanciar Autoencoder(...) e chamar .to(device).
    raise NotImplementedError("TODO(equipe): instanciar e mover o modelo para o device")


def reconstruction_error(
    model: Autoencoder,
    X: np.ndarray,
    window_size: int = config.WINDOW_SIZE,
    device: str = config.DEVICE,
) -> np.ndarray:
    """Calcula o erro de reconstrução (MSE) por janela.

    Executa o modelo em modo de avaliação (sem gradiente) sobre as janelas
    deslizantes extraídas de ``X`` e devolve o erro médio de reconstrução de
    cada janela.

    Args:
        model: Autoencoder treinado.
        X: Matriz de métricas padronizada, shape ``(n_amostras, n_features)``.
        window_size: Tamanho da janela temporal (deve casar com o do treino).
        device: Device de execução.

    Returns:
        Array 1-D com o erro de reconstrução por janela, shape
        ``(n_amostras - window_size + 1,)``.

    Nota:
        O número de janelas é menor que ``n_amostras``. O alinhamento entre o
        erro por janela e o rótulo por timestep (``y_test``) para a avaliação
        binária é responsabilidade de ``evaluation.metrics.predict_anomalies``.
    """
    # TODO(equipe): model.eval(); with torch.no_grad(): reconstruir janelas e
    # calcular o MSE médio por janela; retornar np.ndarray 1-D.
    raise NotImplementedError("TODO(equipe): calcular o erro de reconstrução por janela")
