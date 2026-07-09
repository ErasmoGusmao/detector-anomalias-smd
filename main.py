"""Ponto de entrada do pipeline (Entrega 2).

Orquestra o carregamento, a limpeza e o pré-processamento em NumPy.
"""

from __future__ import annotations

import numpy as np

from src.data.loader import clean_data, load_data
from src.preprocessing.transform import split_data, standardize
from src.utils import config


def main() -> None:
    """Executa o pipeline de carregamento e pré-processamento."""

    train_raw = load_data(config.TRAIN_PATH)
    test_raw = load_data(config.TEST_PATH)
    test_label_raw = load_data(config.TEST_LABEL_PATH)

    train_clean = clean_data(train_raw)
    test_clean = clean_data(test_raw)

    # Conversão para NumPy.
    X_train_full = train_clean.to_numpy()
    X_test = test_clean.to_numpy()
    y_test = test_label_raw.to_numpy().ravel()

    # Padronização
    X_train_full = standardize(X_train_full)
    X_test = standardize(X_test)

    # Split treino/validação, preservando a ordem temporal da série. 
    time_index = np.arange(len(X_train_full))
    X_train, X_val, y_train, y_val = split_data(
        X_train_full,
        time_index,
        test_size=config.VALIDATION_SIZE,
    )

    # Análise simples das dimensões dos dados.
    print("== Pipeline de dados (Entrega 2) ==")
    print(f"Máquina: {config.FILE_NAME}")
    print(f"Treino (após limpeza e padronização): {X_train_full.shape}")
    print(f"  -> treino efetivo:  {X_train.shape} (instantes {y_train[0]}-{y_train[-1]})")
    print(f"  -> validação:       {X_val.shape} (instantes {y_val[0]}-{y_val[-1]})")
    print(f"Teste (após limpeza e padronização):   {X_test.shape}")
    print(f"Rótulos de teste:   {y_test.shape}, anômalos: {int(y_test.sum())}")


if __name__ == "__main__":
    main()
