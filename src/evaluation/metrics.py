"""Metricas de avaliacao do detector de anomalias."""

from __future__ import annotations

import numpy as np


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula metricas de avaliacao para as predicoes do modelo.

    Args:
        y_true: Array binario de rotulos.
        y_pred: Array binario de predicoes.

    Returns:
        Dicionario com chaves e valores de metricas (precision, recall, f1 score, accuracy)
    """
    pass

