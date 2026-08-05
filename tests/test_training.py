"""Testes unitários para o módulo de treinamento (src/training/train.py).
   Verifica que build_dataloader gera tensores no formato correto (float32, shape com janela temporal)."""
 
import unittest
 
import numpy as np
import torch
 
from src.training.train import build_dataloader
 
class TestTrainingTensors(unittest.TestCase):
    """Testes do formato dos tensores gerados pelo pipeline de treinamento."""
 
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
        inputs, targets = next(iter(loader))
        self.assertEqual(inputs.dtype, torch.float32, "Tensores de entrada devem ser float32.")
        self.assertEqual(targets.dtype, torch.float32, "Tensores alvo devem ser float32.")
 
    def test_dataloader_tensor_shape(self):
        """Verifica as dimensões dos tensores (batch, window_size, n_features)."""
        loader = build_dataloader(self.X, window_size=self.window_size, batch_size=self.batch_size)
        inputs, targets = next(iter(loader))
        self.assertEqual(inputs.ndim, 3, "Tensor deve ser 3D.")
        self.assertEqual(inputs.shape[1], self.window_size, "A segunda dimensão deve ser window_size.")
        self.assertEqual(inputs.shape[2], self.n_features, "A terceira dimensão deve ser n_features.")
        self.assertEqual(inputs.shape, targets.shape, "Entrada e alvo devem ter as mesmas dimensões.")
 
if __name__ == "__main__":
    unittest.main()
