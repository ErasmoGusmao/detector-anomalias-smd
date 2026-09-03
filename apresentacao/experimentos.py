"""Compara configuracoes do autoencoder para o slide 11 da apresentacao.

Roda fora de artifacts/: NAO sobrescreve o modelo nem o results.json do repo.
Para cada configuracao: treina, mede erro de treino/validacao/teste, deriva o
limiar em tres percentis e calcula precisao/recall/F1 contra os rotulos reais.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.loader import clean_aligned, clean_data, load_data  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    calculate_metrics,
    predict_anomalies,
    reconstruction_threshold,
)
from src.models.model import create_model, reconstruction_error  # noqa: E402
from src.preprocessing.transform import split_data, standardize  # noqa: E402
from src.training.train import build_dataloader, evaluate_loss, train_model  # noqa: E402
from src.utils import config  # noqa: E402
from src.utils.torch_utils import get_device, set_seed  # noqa: E402

OUT = Path(__file__).resolve().parent / "experimentos"
OUT.mkdir(parents=True, exist_ok=True)

PERCENTIS = [99.0, 99.5, 99.9]

CONFIGS = [
    {
        "id": "A",
        "nome": "Baseline",
        "descricao": "janela 50 - latente 16 - encoder (64,32)",
        "window_size": 50,
        "hidden_dims": (64, 32),
        "latent_dim": 16,
    },
    {
        "id": "B",
        "nome": "Gargalo estreito",
        "descricao": "janela 50 - latente 8 - encoder (64,32)",
        "window_size": 50,
        "hidden_dims": (64, 32),
        "latent_dim": 8,
    },
    {
        "id": "C",
        "nome": "Janela longa",
        "descricao": "janela 100 - latente 16 - encoder (64,32)",
        "window_size": 100,
        "hidden_dims": (64, 32),
        "latent_dim": 16,
    },
    {
        "id": "D",
        "nome": "Maior capacidade",
        "descricao": "janela 50 - latente 32 - encoder (128,64)",
        "window_size": 50,
        "hidden_dims": (128, 64),
        "latent_dim": 32,
    },
]


def preparar_dados():
    train_raw = load_data(config.TRAIN_PATH)
    test_raw = load_data(config.TEST_PATH)
    label_raw = load_data(config.TEST_LABEL_PATH)

    train_clean = clean_data(train_raw)
    test_clean, label_clean = clean_aligned(test_raw, label_raw)

    X_train_full = train_clean.to_numpy()
    X_test = test_clean.to_numpy()
    y_test = label_clean.to_numpy().ravel()

    idx = np.arange(len(X_train_full))
    X_train, X_val, _, _ = split_data(X_train_full, idx, test_size=config.VALIDATION_SIZE)

    X_train, mean, std = standardize(X_train)
    X_val, _, _ = standardize(X_val, mean=mean, std=std)
    X_test, _, _ = standardize(X_test, mean=mean, std=std)

    return X_train, X_val, X_test, y_test


def rodar(cfg: dict, X_train, X_val, X_test, y_test) -> dict:
    print(f"\n===== CONFIG {cfg['id']} - {cfg['nome']} ({cfg['descricao']}) =====", flush=True)
    set_seed()
    device = get_device(prefer_cuda=False)
    window = cfg["window_size"]

    model = create_model(
        input_dim=X_train.shape[1],
        window_size=window,
        hidden_dims=cfg["hidden_dims"],
        latent_dim=cfg["latent_dim"],
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters())

    inicio = time.perf_counter()
    resultado = train_model(model, X_train, X_val, window_size=window, device=device)
    duracao = time.perf_counter() - inicio

    test_loader = build_dataloader(X_test, window_size=window, shuffle=False)
    test_loss = evaluate_loss(model, test_loader, nn.MSELoss(), device=device)

    err_train = reconstruction_error(model, X_train, window_size=window, device=device)
    err_test = reconstruction_error(model, X_test, window_size=window, device=device)
    y_aligned = y_test[window - 1:]

    por_percentil = {}
    for p in PERCENTIS:
        limiar = reconstruction_threshold(err_train, p)
        y_pred = predict_anomalies(err_test, limiar)
        met = calculate_metrics(y_aligned, y_pred)
        met["threshold"] = limiar
        met["pct_janelas_sinalizadas"] = float(y_pred.mean())
        por_percentil[str(p)] = met
        print(
            f"  p{p}: limiar={limiar:.4f} prec={met['precision']:.4f} "
            f"rec={met['recall']:.4f} f1={met['f1_score']:.4f} "
            f"sinalizadas={met['pct_janelas_sinalizadas']:.4f}",
            flush=True,
        )

    saida = {
        **{k: v for k, v in cfg.items() if k != "hidden_dims"},
        "hidden_dims": list(cfg["hidden_dims"]),
        "n_parametros": n_params,
        "epocas_executadas": len(resultado.history["val_loss"]),
        "melhor_epoca": resultado.best_epoch,
        "melhor_val_loss": resultado.best_val_loss,
        "train_loss_final": resultado.history["train_loss"][-1],
        "test_loss_janela": test_loss,
        "tempo_treino_s": duracao,
        "taxa_base_anomalias": float(y_aligned.mean()),
        "por_percentil": por_percentil,
        "history": resultado.history,
    }
    (OUT / f"config_{cfg['id']}.json").write_text(
        json.dumps(saida, indent=2), encoding="utf-8"
    )
    print(
        f"  -> {n_params} parametros, {saida['epocas_executadas']} epocas, "
        f"{duracao:.1f}s, melhor val={resultado.best_val_loss:.6f}",
        flush=True,
    )
    return saida


def main() -> None:
    X_train, X_val, X_test, y_test = preparar_dados()
    print(
        f"treino={X_train.shape} validacao={X_val.shape} teste={X_test.shape} "
        f"rotulos={y_test.shape} anomalos={int(y_test.sum())}",
        flush=True,
    )

    todos = [rodar(cfg, X_train, X_val, X_test, y_test) for cfg in CONFIGS]
    (OUT / "consolidado.json").write_text(json.dumps(todos, indent=2), encoding="utf-8")
    print("\nCONCLUIDO. Resultados em", OUT, flush=True)


if __name__ == "__main__":
    main()
