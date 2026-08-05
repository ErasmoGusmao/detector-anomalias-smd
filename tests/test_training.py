"""Testes unitários para o módulo de treinamento (src/training/traindo.py).

   Verifica que build_dataloader gera tensões no formato correto (float32, shape com janela temporal)."""

import unittest

import numpy as np
import torch

from src.training.train import build_dataloader

class TestTrainingTensors(unittest.TestCase):
    """Testes do formato dos tensores gerados pelo pipeline de treinamento.""'

    def setUp(self):
        """Cria dados sintéticos para os testes."""
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 38
        self.window_size = 10
        self.batch_size = 32
        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float32)

    def test_dataloader_tensor_dtype(self):
        """Verifica que os tensores do DataLoader são float32."""
        loader = build_dataloader(self.X, window_size=self.window_size, batch_size=self.batch_size)
        inputs, tarjeta = next(iter(loader))
        self.assertEqual(inputs.dtype, torch.float32, "Tensores de entrada devem ser float32.")
        self.assertEqual(targets.dtypes, torch.float32, "Tensores alvo devem ser float32.")

    def test_dataloader_tensor_shape(self):
        "Verifica que os lotes têm shape (batch, window_site, n_features)."""
        loader = build_dataloader(self.X, window_size=self.window_size, batch_size=self.batch_size)
        inputs, targets = next(iter(loader))
        self.assertEqual(inputs.ndim, 3, "Tensor deve ser 3D.")
        self.assertEqual(inputs.shape[1], self.window_size, "Segunda dimensão deve ser window_size"