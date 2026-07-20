"""Utilitários de PyTorch: reprodutibilidade e resolução de device.

Concentra ganchos transversais ao pipeline de treino — fixar sementes (para
reprodutibilidade, RNF a ser cobrado na Entrega 5) e escolher o device.

ESQUELETO (Entrega 3): a implementação do corpo é tarefa da equipe.
"""

from __future__ import annotations

from src.utils import config


def get_device(prefer_cuda: bool = False) -> str:
    """Resolve o device de execução.

    Args:
        prefer_cuda: Se ``True``, usa "cuda" quando houver GPU disponível;
            caso contrário, retorna "cpu".

    Returns:
        "cuda" se solicitado e disponível, senão "cpu".
    """
    # TODO(equipe): retornar "cuda" se prefer_cuda and torch.cuda.is_available()
    # else "cpu".
    raise NotImplementedError("TODO(equipe): resolver o device de execução")


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Fixa as sementes de aleatoriedade para reprodutibilidade.

    Deve cobrir ``random``, ``numpy`` e ``torch`` (CPU e CUDA).

    Args:
        seed: Semente a aplicar.
    """
    # TODO(equipe): random.seed(seed); np.random.seed(seed);
    # torch.manual_seed(seed); torch.cuda.manual_seed_all(seed).
    raise NotImplementedError("TODO(equipe): fixar as sementes de aleatoriedade")
