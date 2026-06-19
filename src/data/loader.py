"""Carregamento e limpeza dos dados (Entrega 1).

Assinaturas e contratos das funções iniciais do sistema. A **implementação**
será feita na tarefa de funções do código (#5).
"""

from __future__ import annotations
from pathlib import Path
from typing import Union
import pandas as pd


def load_data(path: Union[str, Path]) -> pd.DataFrame:
    """Carrega a base de dados a partir de um arquivo CSV.
    
    Args:
        path: Caminho para o arquivo csv.

    Returns:
        Dataframe com os dados brutos.
    """
    pass


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicatas e linhas com valores ausentes.
    
    Args:
        data: Dataframe com dados brutos.

    Returns:
        Dataframe limpo, sem duplicatas nem valores nulos.
    """
    pass
