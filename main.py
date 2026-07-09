"""Ponto de entrada do pipeline (Entrega 2).

Orquestra o carregamento, a limpeza e o pré-processamento em NumPy.
"""

from __future__ import annotations

import logging

import numpy as np

from src.data.loader import clean_aligned, clean_data, load_data
from src.preprocessing.transform import split_data, standardize
from src.utils import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")

def main() -> None:
    """Executa o pipeline de carregamento e pré-processamento."""

    # Carregamento
    train_raw = load_data(config.TRAIN_PATH)
    test_raw = load_data(config.TEST_PATH)
    test_label_raw = load_data(config.TEST_LABEL_PATH)

    # Limpeza
    train_clean = clean_data(train_raw)
    test_clean, test_label_clean = clean_aligned(test_raw, test_label_raw)

    # Conversão para NumPy.
    X_train_full = train_clean.to_numpy()
    X_test = test_clean.to_numpy()
    y_test = test_label_clean.to_numpy().ravel()

    # Split treino/validação, preservando a ordem temporal da série.
    time_index = np.arange(len(X_train_full))
    X_train, X_val, idx_train, idx_val = split_data(
        X_train_full,
        time_index,
        test_size=config.VALIDATION_SIZE,
    )

    # Padronização (fit no treino, aplicar em validação e teste)
    X_train, train_mean, train_std = standardize(X_train)
    X_val, _, _ = standardize(X_val, mean=train_mean, std=train_std)
    X_test, _, _ = standardize(X_test, mean=train_mean, std=train_std)

    # Análise simples das dimensões dos dados.
    print()
    print("== Pipeline de dados (Entrega 2) ==")
    print(f"Máquina: {config.FILE_NAME}")
    print(f"Dimensões do arquivo de treino (após limpeza):")
    print(f"  -> treino completo:  {X_train_full.shape}")
    print(f"  -> treino efetivo:   {X_train.shape} (indeces {idx_train[0]}-{idx_train[-1]})")
    print(f"  -> validação:        {X_val.shape} (indeces {idx_val[0]}-{idx_val[-1]})")
    print(f"Dimensões do arquivo de teste (após limpeza):")
    print(f"  -> teste:            {X_test.shape}")
    print(f"Dimensões do arquivo de rótulos de teste (após limpeza):")
    print(f"  -> rótulos de teste: {y_test.shape}, anômalos: {int(y_test.sum())}")
    print()


if __name__ == "__main__":
    main()
