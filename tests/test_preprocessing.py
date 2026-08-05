"""Testes unitários para o módulo de pre-processamento (src/preprocessing/transform.py).

Verifica que as transformações de padronização (z-score) produzem saídas com o shape correto e propriedades estatísticas esperadas (média ~ 0, esvio ~ 1)."""

import unittest
import numpy as np
from src.preprocessing.transform import standards, split_data

class TestPreprocessing(unittest.TestCase):
    """Testes de pre-processamento dos dados."""

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(200,38).astype(np.float64)
        self.X = self.X * 5 + 10 # média 10, std 5
        
    def test_standardize_shape(self):
        """Verifica que standardize preserva as dimensoes originais da matriz."""
        X_std, mean, std = standardize(self.X)
        self.assertEqual(X_std.shape, self.X.shape,"As dimensoes de X e X_std devem ser iguais.")
        self.assertEqual(mean.shape, (38, ), "Devem haver uma média por feature.")
        self assertEqual(std.shape, (38,), "Devem haver um desvio padrão por feature.")

    def test_standardize_mean(self):
        """Verifica que a média das features padronizadas é aproximadamente zero."
        X_std, _, _ = standardize(self.X)
        feature_means = np.mean(X_std, axis=0)
        np.testing.assert_allclose(
            feature_means, np.zeros(38), atol=1e-10,
            err_msg = "Média por feature de X_std deve ser aprox. zero.")

    def
