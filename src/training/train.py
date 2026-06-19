"""Treinamento do detector de anomalias.

Este modulo deve orquestrar a preparacao dos dados, a instanciacao do modelo
e o ciclo de treino.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: Optional[np.ndarray] = None,
) -> object:
    """Executa o treinamento do modelo informado.
    
    Args:
        model: Instancia de um modelo.
        X_train: Matriz de treino.
        y_train: Rotulos de treino.

    Returns:
        Modelo treinado.
    """
    pass
