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

    # =======================================================================
    # == Treino e avaliação (Entrega 3 — PyTorch) ==
    # =======================================================================
    # ESQUELETO: a costura abaixo consome X_train, X_val, X_test e y_test já
    # produzidos acima. Está comentada porque os módulos de modelo/treino
    # ainda são stubs; a equipe descomenta conforme cada PR fica pronto.
    #
    # Imports necessários (mover para o topo ao ativar):
    #   from src.models.model import create_model, reconstruction_error
    #   from src.models.persistence import save_model
    #   from src.training.train import build_dataloader, evaluate_loss, train_model
    #   from src.evaluation.metrics import (
    #       calculate_metrics, predict_anomalies, reconstruction_threshold,
    #   )
    #   from src.utils.torch_utils import get_device, set_seed
    #   import torch.nn as nn
    #
    # set_seed()
    # device = get_device(prefer_cuda=False)
    #
    # # Modelo e treino (imprime erro de treino e validação por época).
    # model = create_model(input_dim=X_train.shape[1], device=device)
    # history = train_model(model, X_train, X_val, device=device)
    #
    # # Erro de teste (requisito: imprimir o erro de teste).
    # test_loader = build_dataloader(X_test, shuffle=False, device=device)
    # test_loss = evaluate_loss(model, test_loader, nn.MSELoss(), device=device)
    # print(f"Erro de reconstrução no teste: {test_loss:.6f}")
    #
    # # Persistência (requisito: salvar o modelo).
    # save_model(model)
    #
    # # Avaliação de anomalias (fecha o ciclo com as métricas da Entrega 4).
    # err_train = reconstruction_error(model, X_train, device=device)
    # threshold = reconstruction_threshold(err_train, config.ANOMALY_PERCENTILE)
    # err_test = reconstruction_error(model, X_test, device=device)
    # y_pred = predict_anomalies(err_test, threshold)
    # print(calculate_metrics(y_test, y_pred))


if __name__ == "__main__":
    main()
