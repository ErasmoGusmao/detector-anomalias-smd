"""Treinamento do detector de anomalias (autoencoder em PyTorch).

Orquestra a preparação dos dados em tensores/janelas, o laço de treino e a
avaliação de perda. O treino é não-supervisionado: o alvo de reconstrução é a
própria janela de entrada (``X -> X``).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.model import Autoencoder
from src.utils import config


def build_dataloader(
    X: np.ndarray,
    window_size: int = config.WINDOW_SIZE,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
) -> DataLoader:
    """Cria um ``DataLoader`` de janelas deslizantes para reconstrução.

    Extrai janelas de tamanho ``window_size`` de ``X`` e monta um
    ``TensorDataset`` em que entrada e alvo são a mesma janela (autoencoder).
    Os tensores permanecem em CPU; os lotes são movidos para ``device`` nos
    laços de treino e avaliação.

    Args:
        X: Matriz padronizada, shape ``(n_amostras, n_features)``.
        window_size: Tamanho da janela temporal.
        batch_size: Tamanho do lote.
        shuffle: Embaralhar as janelas (``True`` no treino, ``False`` em
            validação/teste para preservar a ordem temporal).

    Returns:
        ``DataLoader`` cujos lotes são pares ``(janela, janela)`` de shape
        ``(batch, window_size, n_features)``, em ``float32``.
    """
    X_array = np.asarray(X, dtype=np.float32)

    if X_array.ndim != 2:
        raise ValueError("X deve ter shape (n_amostras, n_features)")
    if window_size <= 0:
        raise ValueError("window_size deve ser maior que zero")
    if batch_size <= 0:
        raise ValueError("batch_size deve ser maior que zero")
    if X_array.shape[0] < window_size:
        raise ValueError(
            "X deve conter ao menos window_size amostras para formar uma janela"
        )
    if not np.isfinite(X_array).all():
        raise ValueError("X deve conter apenas valores finitos")

    windows = sliding_window_view(X_array, window_shape=window_size, axis=0)
    windows = np.moveaxis(windows, -1, 1)
    windows_tensor = torch.from_numpy(np.ascontiguousarray(windows)).to(dtype=torch.float32)
    dataset = TensorDataset(windows_tensor, windows_tensor)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: Autoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str = config.DEVICE,
    max_grad_norm: float = config.MAX_GRAD_NORM,
) -> float:
    """Executa uma época de treino.

    Args:
        model: Autoencoder em modo de treino.
        loader: ``DataLoader`` de treino (pares janela/janela).
        optimizer: Otimizador (ex.: ``torch.optim.Adam``).
        loss_fn: Função de perda de reconstrução (ex.: ``nn.MSELoss``).
        device: Device de execução.
        max_grad_norm: Norma maxima para gradient clipping.

    Returns:
        Perda média de reconstrução sobre a época.
    """
    target_device = torch.device(device)
    model.train()

    total_loss = 0.0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(target_device)
        targets = targets.to(target_device)
        batch_size_atual = inputs.shape[0]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * batch_size_atual
        total_samples += batch_size_atual

    if total_samples == 0:
        raise ValueError("DataLoader de treino não contém amostras")

    return total_loss / total_samples


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
    target_device = torch.device(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(target_device)
            targets = targets.to(target_device)
            batch_size_atual = inputs.shape[0]

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            total_loss += loss.item() * batch_size_atual
            total_samples += batch_size_atual

    if total_samples == 0:
        raise ValueError("DataLoader de avaliação não contém amostras")

    return total_loss / total_samples


def train_model(
    model: Autoencoder,
    X_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    window_size: int = config.WINDOW_SIZE,
    device: str = config.DEVICE,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    scheduler_factor: float = config.SCHEDULER_FACTOR,
    scheduler_patience: int = config.SCHEDULER_PATIENCE,
) -> dict[str, list[float]]:
    """Executa o laço completo de treino/validação do autoencoder.

    Monta os ``DataLoader`` de treino e validação, o otimizador e a perda,
    itera por ``epochs`` épocas e imprime o erro de treino e de validação a
    cada época. Inclui early stopping e learning rate scheduling.

    Args:
        model: Autoencoder criado por ``create_model``.
        X_train: Matriz de treino padronizada.
        X_val: Matriz de validação padronizada (estatísticas do treino).
        epochs: Número maximo de épocas.
        batch_size: Tamanho do lote.
        learning_rate: Taxa de aprendizado do otimizador.
        window_size: Tamanho da janela temporal.
        device: Device de execução.
        patience: Epocas sem melhoras antes de early stopping.
        scheduler_factor: Fator de reducao do LR no scheduler.
        scheduler_patience: Epocas sem melhora antes de reduzir o LR.

    Returns:
        Histórico das perdas: ``{"train_loss": [...], "val_loss": [...]}`` com
        uma entrada por época.
    """
    if epochs <= 0:
        raise ValueError("epochs deve ser maior que zero")
    if learning_rate <= 0:
        raise ValueError("learning_rate deve ser maior que zero")

    target_device = torch.device(device)
    model.to(target_device)

    train_loader = build_dataloader(
        X_train,
        window_size=window_size,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = build_dataloader(
        X_val,
        window_size=window_size,
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_cheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
    )
    loss_fn = nn.MSELoss()
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = model.state_dict()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device=device,
        )
        val_loss = evaluate_loss(model, val_loader, loss_fn, device=device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Época {epoch:03d}/{epochs:03d} | "
            f"Erro de treino: {train_loss:.6f} | "
            f"Erro de validação: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping na epoca {epoch} (sem melhora por {patience} epocas)")
                model.load_state_dict(best_state)
                break

    return history
