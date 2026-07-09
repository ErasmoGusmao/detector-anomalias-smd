"""Carregamento e limpeza dos dados.

Funções iniciais do sistema, responsáveis por ler os arquivos do Server
Machine Dataset (SMD) e entregar um DataFrame pronto para o pré-processamento
em NumPy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)


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
    """Remove linhas com valores ausentes ou não numéricos.

    Valores não numéricos são coercidos para NaN antes da remoção.
    O índice original é preservado para manter a referência temporal.

    Args:
        data: Dataframe com dados brutos.

    Returns:
        Dataframe limpo, sem valores nulos nem não numéricos,
        com o índice original preservado.
    """
    n_before = len(data)
    cleaned = data.apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.dropna()
    n_dropped = n_before - len(cleaned)
    if n_dropped > 0:
        logger.info("Removidas %d linhas com valores ausentes ou nao numericos (de %d).", n_dropped, n_before)
    return cleaned


def clean_aligned(*dataframes: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Remove linhas com valores ausentes ou não numéricos, mantendo alinhamento.

    Coerce valores não numéricos para NaN, então identifica linhas onde
    **qualquer** dos DataFrames possui NaN e remove essas linhas de todos
    simultaneamente. O índice original é preservado.

    Args:
        *dataframes: Dois ou mais DataFrames com o mesmo número de linhas.

    Returns:
        Tupla de DataFrames limpos com índice original preservado.

    Raises:
        ValueError: se os DataFrames tiverem comprimentos diferentes.
    """
    if len(dataframes) < 2:
        raise ValueError("clean_aligned requer ao menos dois DataFrames.")

    n_rows = len(dataframes[0])
    for df in dataframes[1:]:
        if len(df) != n_rows:
            raise ValueError("Todos os DataFrames devem ter o mesmo numero de linhas.")

    coerced = [df.apply(pd.to_numeric, errors="coerce") for df in dataframes]

    valid_mask = pd.Series(True, index=coerced[0].index)
    for df in coerced:
        valid_mask &= ~df.isna().any(axis=1)

    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        logger.info(
            "clean_aligned: removidas %d linhas com valores ausentes ou nao numericos (de %d).",
            n_dropped,
            n_rows,
        )

    return tuple(df.loc[valid_mask] for df in coerced)
