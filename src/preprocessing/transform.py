"""Pré-processamento dos dados com NumPy (Entrega 1).

Assinaturas e contratos das transformações. A **implementação** será feita na
tarefa de funções do código (#5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def standardize(X: np.ndarray) -> np.ndarray:
    """Padroniza os atributos para média 0 e desvio-padrão 1 (z-score).
    
    Args:
        X: Matriz de atributos.

    Returns:
        Matriz padronizada.
    """
    X_float = np.asarray(X, dtype=float)
    mean = np.mean(X_float, axis=0)
    std = np.std(X_float, axis=0)
    safe_std = np.where(std == 0, 1.0, std)

    return (X_float - mean) / safe_std


def split_features_target(
    data: pd.DataFrame,
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Separa o DataFrame em matriz de atributos (X) e alvo (y).

    Args:
        data: Dataframe limpo com coluna alvo presente.

    Returns:
        Tupla (X, y).
    """
    if target_column not in data.columns:
        raise ValueError(f"Coluna alvo nao encontrada: {target_column}")

    X = data.drop(columns=[target_column]).to_numpy()
    y = data[target_column].to_numpy()

    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reserva o trecho final da serie para teste, em ordem temporal.
    
    Args:
        X: Matriz de atributos.
        y: Array de rotulos.
        test_size: Fracao dos dados para teste.

    Returns:
        Tupla (X_train, X_test, y_train, y_test).
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size deve estar entre 0 e 1.")

    if len(X) != len(y):
        raise ValueError("X e y devem ter a mesma quantidade de amostras.")

    split_index = int(len(X) * (1 - test_size))
    if split_index == 0 or split_index == len(X):
        raise ValueError("A divisao deve produzir conjuntos de treino e teste.")

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test
