"""Métricas de avaliação do detector de anomalias do SMD."""

from __future__ import annotations

import numpy as np


def _prepare_binary_arrays(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Converte e valida os arrays de entrada antes do cálculo de métricas."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true e y_pred devem ter o mesmo formato")

    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError("y_true e y_pred devem ser arrays unidimensionais")

    if y_true_arr.size == 0:
        raise ValueError("y_true e y_pred não podem ser vazios")

    # Conversão para bool para usar & e ~ nas contagens de
    # TP/FP/FN/TN. Em bool, & e ~ funcionam como esperado
    # (E lógico e negação).
    return y_true_arr != 0, y_pred_arr != 0


def _confusion_matrix_counts(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[int, int, int, int]:
    """Calcula TP, FP, FN, TN uma única vez para reuso pelas métricas."""
    y_true_bin, y_pred_bin = _prepare_binary_arrays(y_true, y_pred)
    tp = int(np.sum(y_true_bin & y_pred_bin))
    fp = int(np.sum(~y_true_bin & y_pred_bin))
    fn = int(np.sum(y_true_bin & ~y_pred_bin))
    tn = int(np.sum(~y_true_bin & ~y_pred_bin))
    return tp, fp, fn, tn


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula a precisão do detector de anomalias.

    Fórmula: precision = TP / (TP + FP). Retorna 0.0 se não houver
    nenhuma predição positiva.

    Args:
        y_true: Array binário de rótulos do SMD (0/1), onde 1 indica
            anomalia.
        y_pred: Array binário de predições do modelo (0/1), onde 1 indica
            anomalia.

    Returns:
        Valor da precisão entre 0.0 e 1.0.
    """
    tp, fp, _, _ = _confusion_matrix_counts(y_true, y_pred)
    predicted_positives = tp + fp
    if predicted_positives == 0:
        return 0.0
    return tp / predicted_positives


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula o recall do detector de anomalias.

    Fórmula: recall = TP / (TP + FN). Retorna 0.0 se não houver
    nenhuma anomalia real.

    Args:
        y_true: Array binário de rótulos do SMD (0/1), onde 1 indica
            anomalia.
        y_pred: Array binário de predições do modelo (0/1), onde 1 indica
            anomalia.

    Returns:
        Valor do recall entre 0.0 e 1.0.
    """
    tp, _, fn, _ = _confusion_matrix_counts(y_true, y_pred)
    actual_positives = tp + fn
    if actual_positives == 0:
        return 0.0
    return tp / actual_positives


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula o F1-score do detector de anomalias.

    Fórmula: F1 = 2 * precision * recall / (precision + recall).

    Args:
        y_true: Array binário de rótulos do SMD (0/1), onde 1 indica
            anomalia.
        y_pred: Array binário de predições do modelo (0/1), onde 1 indica
            anomalia.

    Returns:
        Valor do F1-score entre 0.0 e 1.0.
    """
    tp, fp, fn, _ = _confusion_matrix_counts(y_true, y_pred)
    predicted_positives = tp + fp
    actual_positives = tp + fn
    prec = 0.0 if predicted_positives == 0 else tp / predicted_positives
    rec = 0.0 if actual_positives == 0 else tp / actual_positives
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula a acurácia do detector de anomalias.

    Fórmula: accuracy = (TP + TN) / (TP + TN + FP + FN).

    Args:
        y_true: Array binário de rótulos do SMD (0/1), onde 1 indica
            anomalia.
        y_pred: Array binário de predições do modelo (0/1), onde 1 indica
            anomalia.

    Returns:
        Valor da acurácia entre 0.0 e 1.0.
    """
    tp, fp, fn, tn = _confusion_matrix_counts(y_true, y_pred)
    total = tp + fp + fn + tn
    return (tp + tn) / total


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Calcula métricas de avaliação para as predições do modelo.

    Args:
        y_true: Array binário de rótulos do SMD (0/1), onde 1 indica
            anomalia.
        y_pred: Array binário de predições do modelo (0/1), onde 1 indica
            anomalia.

    Returns:
        Dicionário com chaves e valores de métricas.

        Chaves presentes:
            - precision
            - recall
            - f1_score
            - accuracy
    """
    tp, fp, fn, tn = _confusion_matrix_counts(y_true, y_pred)
    predicted_positives = tp + fp
    actual_positives = tp + fn
    total = tp + fp + fn + tn

    prec = 0.0 if predicted_positives == 0 else tp / predicted_positives
    rec = 0.0 if actual_positives == 0 else tp / actual_positives
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    acc = (tp + tn) / total

    return {
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "accuracy": acc,
    }
