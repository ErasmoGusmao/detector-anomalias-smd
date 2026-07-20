"""Treinamento do detector de anomalias (autoencoder em PyTorch).

Orquestra a preparação dos dados em tensores/janelas, o laço de treino e a
avaliação de perda. O treino é não-supervisionado: o alvo de reconstrução é a
própria janela de entrada (``X -> X``).

ESQUELETO (Entrega 3): assinaturas e contratos definidos; a implementação do
corpo é tarefa da equipe.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.model import Autoencoder
from src.utils import config


def build_dataloader(
    X: np.ndarray,
    window_size: int = config.WINDOW_SIZE,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
    device: str = config.DEVICE,
) -> DataLoader:
    """Cria um ``DataLoader`` de janelas deslizantes para reconstrução.

    Extrai janelas de tamanho ``window_size`` de ``X`` e monta um
    ``TensorDataset`` em que entrada e alvo são a mesma janela (autoencoder).

    Args:
        X: Matriz padronizada, shape ``(n_amostras, n_features)``. Vem do
            pré-processamento NumPy (dtype float64) e deve ser convertida para
            ``float32`` antes de virar tensor.
        window_size: Tamanho da janela temporal.
        batch_size: Tamanho do lote.
        shuffle: Embaralhar as janelas (``True`` no treino, ``False`` em
            validação/teste para preservar a ordem temporal).
        device: Device dos tensores.

    Returns:
        ``DataLoader`` cujos lotes são pares ``(janela, janela)`` de shape
        ``(batch, window_size, n_features)``, em ``float32``.
    """
    # TODO(equipe): gerar janelas deslizantes (ex.: np.lib.stride_tricks.
    # sliding_window_view), converter para torch.float32, montar
    # TensorDataset(janelas, janelas) e retornar DataLoader(...).
    raise NotImplementedError("TODO(equipe): construir o DataLoader de janelas")


def train_one_epoch(
    model: Autoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str = config.DEVICE,
) -> float:
    """Executa uma época de treino.

    Args:
        model: Autoencoder em modo de treino.
        loader: ``DataLoader`` de treino (pares janela/janela).
        optimizer: Otimizador (ex.: ``torch.optim.Adam``).
        loss_fn: Função de perda de reconstrução (ex.: ``nn.MSELoss``).
        device: Device de execução.

    Returns:
        Perda média de reconstrução sobre a época.
    """
    # TODO(equipe): model.train(); para cada lote: zero_grad, forward, loss,
    # backward, step; acumular a perda e retornar a média.
    raise NotImplementedError("TODO(equipe): implementar uma época de treino")


def evaluate_loss(
    model: Autoencoder,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str = config.DEVICE,
) -> float:
    """Calcula a perda média de reconstrução sem atualizar os pesos.

    Usada para a perda de validação e para a perda de teste.

    Args:
        model: Autoencoder a avaliar.
        loader: ``DataLoader`` de validação ou teste.
        loss_fn: Mesma função de perda usada no treino.
        device: Device de execução.

    Returns:
        Perda média de reconstrução sobre o conjunto.
    """
    # TODO(equipe): model.eval(); with torch.no_grad(): acumular a perda por
    # lote e retornar a média.
    raise NotImplementedError("TODO(equipe): calcular a perda média sem gradiente")


def train_model(
    model: Autoencoder,
    X_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    window_size: int = config.WINDOW_SIZE,
    device: str = config.DEVICE,
) -> dict[str, list[float]]:
    """Executa o laço completo de treino/validação do autoencoder.

    Monta os ``DataLoader`` de treino e validação, o otimizador e a perda,
    itera por ``epochs`` épocas e **imprime o erro de treino e de validação a
    cada época** (requisito da Entrega 3).

    Args:
        model: Autoencoder criado por ``create_model``.
        X_train: Matriz de treino padronizada.
        X_val: Matriz de validação padronizada (estatísticas do treino).
        epochs: Número de épocas.
        batch_size: Tamanho do lote.
        learning_rate: Taxa de aprendizado do otimizador.
        window_size: Tamanho da janela temporal.
        device: Device de execução.

    Returns:
        Histórico das perdas: ``{"train_loss": [...], "val_loss": [...]}`` com
        uma entrada por época.
    """
    # TODO(equipe): build_dataloader(X_train, shuffle=True) e
    # build_dataloader(X_val, shuffle=False); Adam(model.parameters(), lr=...);
    # nn.MSELoss(); laço por época chamando train_one_epoch/evaluate_loss,
    # imprimindo as perdas e acumulando no histórico.
    raise NotImplementedError("TODO(equipe): implementar o laço de treino/validação")
