"""Ponto de entrada do pipeline do detector de anomalias (SMD).

Orquestra o pipeline completo: carregamento e limpeza dos dados, pré-processamento
em NumPy, treino do autoencoder (PyTorch) com erro de treino/validação por época,
erro de reconstrução no teste, salvamento do modelo e avaliação das anomalias.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import torch.nn as nn

from src.data.loader import clean_aligned, clean_data, load_data
from src.evaluation.metrics import (
    calculate_metrics,
    predict_anomalies,
    reconstruction_threshold,
)
from src.models.model import create_model, reconstruction_error
from src.models.persistence import save_model
from src.preprocessing.transform import split_data, standardize
from src.training.train import build_dataloader, evaluate_loss, train_model
from src.utils import config
from src.utils.torch_utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")

def main() -> None:
    """Executa o pipeline completo de dados, treino e avaliação."""

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
    print("== Treino e avaliação (Entrega 3) ==")
    set_seed()
    device = get_device(prefer_cuda=False)

    # Modelo e treino (imprime o erro de treino e de validação por época).
    model = create_model(input_dim=X_train.shape[1], device=device)
    history = train_model(model, X_train, X_val, device=device)

    # Persistencia do historico de treino (erro por epoca)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    history_path = config.MODEL_DIR / "training_history.json"
    with open(history, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Historico de treino salvo em: {history_path}")
    

    # Erro de reconstrução no teste (requisito: imprimir o erro de teste).
    test_loader = build_dataloader(X_test, shuffle=False)
    test_loss = evaluate_loss(model, test_loader, nn.MSELoss(), device=device)
    print(f"Erro de reconstrução no teste: {test_loss:.6f}")

    # Persistência do modelo treinado (requisito: salvar o modelo).
    save_model(model)
    print(f"Modelo salvo em: {config.MODEL_PATH}")

    # Avaliação de anomalias: limiar pelo erro do treino, predição no teste.
    # O limiar é o percentil ANOMALY_PERCENTILE dos erros de reconstrução do
    # treino (100% normal no SMD). Como o modelo foi treinado nesses dados, os
    # erros são otimisticamente baixos — escolha padrão do método não-supervisionado.
    err_train = reconstruction_error(model, X_train, device=device)
    threshold = reconstruction_threshold(err_train, config.ANOMALY_PERCENTILE)
    err_test = reconstruction_error(model, X_test, device=device)
    y_pred = predict_anomalies(err_test, threshold)

    # Alinha janela → timestep: cada erro corresponde ao último timestep de sua
    # janela, então os rótulos de teste começam em WINDOW_SIZE - 1.
    y_test_aligned = y_test[config.WINDOW_SIZE - 1:]
    metrics = calculate_metrics(y_test_aligned, y_pred)
    print(f"Limiar de anomalia (p{config.ANOMALY_PERCENTILE:g}): {threshold:.6f}")
    print(f"Métricas de detecção: {metrics}")

    # Persistencia dos resultados de avaliacao
results = {
    "test_loss": test_loss,
    "threshold": threshold,
    **metrics,
}
results_path = config.MODEL_DIR / "results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
    print(f"Resultados salvos em: {results_path}")


if __name__ == "__main__":
    main()
