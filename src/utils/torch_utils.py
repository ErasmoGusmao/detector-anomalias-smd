"""Utilitários de PyTorch: reprodutibilidade e resolução de device.

Concentra ganchos transversais ao pipeline de treino — fixar sementes (para
reprodutibilidade, RNF a ser cobrado na Entrega 5) e escolher o device.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from src.utils import config


def get_device(
    prefer_cuda: bool = False,
    prefer_mps: bool = False,
) -> str:
    """Resolve o device de execução.

    Args:
        prefer_cuda: Se ``True``, usa "cuda" quando houver GPU disponível;
            tem prioridade sobre MPS quando ambos forem solicitados.
        prefer_mps: Se ``True`` e CUDA não tiver sido selecionado, usa "mps"
            quando uma GPU Apple Silicon estiver disponível.

    Returns:
        "cuda" ou "mps" quando solicitado e disponível; senão, "cpu".
    """
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    if prefer_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Fixa as sementes de aleatoriedade para reprodutibilidade.

    Deve cobrir ``random``, ``numpy`` e ``torch`` (CPU e CUDA).

    Args:
        seed: Semente a aplicar.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
