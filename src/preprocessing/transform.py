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
    pass


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
    pass


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
    pass
