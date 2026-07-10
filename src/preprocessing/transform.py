"""Pré-processamento dos dados com NumPy (Entrega 1).

Assinaturas e contratos das transformações. A **implementação** será feita na
tarefa de funções do código (#5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def standardize(
    X: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Padroniza os atributos para média 0 e desvio-padrão 1 (z-score).

    Se *mean* e *std* não forem fornecidos, são calculados a partir de *X*
    (modo fit+transform). Caso contrário, os parâmetros fornecidos são
    aplicados diretamente (modo transform).

    Args:
        X: Matriz de atributos.
        mean: Média por feature (opcional). Se None, calculada a partir de X.
        std: Desvio-padrão por feature (opcional). Se None, calculado a partir de X.

    Returns:
        Tupla (X_padronizado, mean, std).
    """
    X_float = np.asarray(X, dtype=float)

    if mean is None or std is None:
        mean = np.mean(X_float, axis=0)
        std = np.std(X_float, axis=0)

    safe_std = np.where(std == 0, 1.0, std)

    return (X_float - mean) / safe_std, mean, std


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
    """Divide os dados em dois conjuntos consecutivos, preservando a ordem temporal.
       O trecho inicial fica no primeiro conjunto e o trecho final no segundo.
       Util para criar splits treino/validacao em series temporais,
    
    Args:
        X: Matriz de atributos.
        y: Array auxiliar com o mesmo comprimento de X (ex: indices temporais).
        test_size: Fracao dos dados reservada para o segundo conjunto.

    Returns:
        Tupla (X_first, X_second, y_first, y_second):
            - X_first: trecho inicial da matriz de atributos.
            - X_second: trecho final da matriz de atributos.
            - y_first: trecho inicial do array auxiliar.
            - y_second: trecho final do array auxiliar.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size deve estar entre 0 e 1.")

    if len(X) != len(y):
        raise ValueError("X e y devem ter a mesma quantidade de amostras.")

    split_index = int(len(X) * (1 - test_size))
    if split_index == 0 or split_index == len(X):
        raise ValueError("A divisao deve produzir dois conjuntos nao vazios.")

    X_first = X[:split_index]
    X_test = X[split_index:]
    y_first = y[:split_index]
    y_second = y[split_index:]

    return X_first, X_test, y_train, y_test
