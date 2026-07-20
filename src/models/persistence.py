"""Persistência do modelo treinado (salvar/carregar).

Isola a serialização do autoencoder da definição da arquitetura (``model.py``)
e do laço de treino (``training/train.py``). Usa o ``state_dict`` do PyTorch:
salva apenas os pesos, então o carregamento exige uma instância com a **mesma
arquitetura**.

ESQUELETO (Entrega 3): a implementação do corpo é tarefa da equipe.
"""

from __future__ import annotations

from pathlib import Path

from src.models.model import Autoencoder
from src.utils import config


def save_model(model: Autoencoder, path: Path = config.MODEL_PATH) -> None:
    """Salva os pesos (``state_dict``) do modelo em disco.

    Cria o diretório de destino se necessário.

    Args:
        model: Autoencoder treinado.
        path: Caminho do arquivo ``.pt`` de destino.
    """
    # TODO(equipe): path.parent.mkdir(parents=True, exist_ok=True);
    # torch.save(model.state_dict(), path).
    raise NotImplementedError("TODO(equipe): salvar o state_dict do modelo")


def load_model(
    model: Autoencoder,
    path: Path = config.MODEL_PATH,
    device: str = config.DEVICE,
) -> Autoencoder:
    """Carrega os pesos salvos em uma instância de modelo já criada.

    Args:
        model: Instância de ``Autoencoder`` com a mesma arquitetura usada no
            treino (criar via ``create_model`` antes de chamar esta função).
        path: Caminho do arquivo ``.pt`` com os pesos.
        device: Device de destino para o mapeamento dos tensores.

    Returns:
        O mesmo ``model``, com os pesos carregados e em modo de avaliação.
    """
    # TODO(equipe): model.load_state_dict(torch.load(path, map_location=device));
    # model.to(device); model.eval(); retornar model.
    raise NotImplementedError("TODO(equipe): carregar o state_dict no modelo")
