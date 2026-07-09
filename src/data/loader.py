"""Carregamento e limpeza dos dados.

Funções iniciais do sistema, responsáveis por ler os arquivos do Server
Machine Dataset (SMD) e entregar um DataFrame pronto para o pré-processamento
em NumPy.
"""

from __future__ import annotations
from pathlib import Path
from typing import Union
import pandas as pd


def load_data(path: Union[str, Path]) -> pd.DataFrame:
    """Carrega a base de dados a partir de um arquivo de métricas do SMD.

    Os arquivos do Server Machine Dataset são texto com valores separados por
    vírgula, **sem cabeçalho** e já normalizados no intervalo [0, 1]. Cada
    coluna representa uma métrica operacional do servidor.

    Args:
        path: Caminho para o arquivo de métricas (ex.: ``train/machine-1-1.txt``).

    Returns:
        Dataframe com os dados brutos, uma coluna por métrica.

    Raises:
        FileNotFoundError: se o arquivo não existir no caminho informado.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Arquivo de dados nao encontrado: {file_path}")

    return pd.read_csv(file_path, header=None)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicatas e linhas com valores ausentes.

    Args:
        data: Dataframe com dados brutos.

    Returns:
        Dataframe limpo, sem valores nulos, com o índice
        reindexado de forma contígua.
    """
    cleaned = data.dropna()
    return cleaned.reset_index(drop=True)
