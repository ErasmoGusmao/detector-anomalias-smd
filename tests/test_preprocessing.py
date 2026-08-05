"""Testes unitários para o módulo de pre-processamento (src/preprocessing/transform.py).

Verifica que as transformações de padronização (z-score) produzem saídas com o shape correto e propriedades estatísticas esperadas (média ~ 0, esvio ~ 1)."""

import unittest
import numpy as np
from src.preprocessing.transform import standards, split_data

class TestPreprocessing(unittest.TestCase):
    """Testes de pre-processamento dos dados."""
