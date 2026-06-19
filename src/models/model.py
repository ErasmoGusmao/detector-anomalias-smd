"""Funcoes do modelo de deteccao de anomalias.

Este modulo concentra as funcoes relacionadas ao modelo. A implementacao sera
preenchida nas proximas entregas do projeto.
"""

from __future__ import annotations
from typing import Any
import numpy as np


def create_model() -> Any:
    """Cria e configura o modelo de deteccao de anomalias.
    
    Returns:
        Instancia do modelo configurado.
    """
    pass


def predict(model: Any, X: np.ndarray) -> np.ndarray:
    """Retorna as predicoes de anomalia para os dados informados.
    
    Args:
        model: Modelo treinado.
        X: Matriz de atributos.

    Returns:
        Array binario que preve possiveis anomalias.
    """
    pass
