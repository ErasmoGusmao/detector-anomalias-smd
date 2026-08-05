"""Testes unitários para o modelo (src/models/model.py e persistence.py).

   Verifica que o modelo produz saídas com as dimensoes corretas, e que a
   serialização (salvar/carregar) preserva os pesos do modelo.
"""
 
import unittest
import tempfile
from pathlib import Path
 
import numpy as np
import torch
 
from src.models.model import Autoencoder, create_model
from src.models.persistence import save_model, load_model
 
 
class TestModelOutput(unittest.TestCase):
    """Testes da saída do modelo autoencoder."""
 
    def setUp(self):
        """Cria modelo e dados sintéticos para os testes."""
        self.input_dim = 38
        self.window_size = 10
        self.batch_size = 8
        self.model = create_model(
            input_dim=self.input_dim,
            window_size=self.window_size,
            hidden_dims=(32, 16),
            latent_dim=8,
            device="cpu",
        )
        self.x = torch.randn(self.batch_size, self.window_size, self.input_dim)
 
    def test_model_output_shape(self):
        """Verifica que a saída do modelo tem o mesmo shape da entrada (reconstrução)."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.x)
        self.assertEqual(
            output.shape, self.x.shape,
            "Saída do autoencoder deve ter shape (batch, window_size, input_dim)"
        )
 
    def test_model_output_is_finite(self):
        """Verifica que a saída do modelo não contém NaN ou Inf."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.x)
        self.assertTrue(
            torch.isfinite(output).all(),
            "Saída do modelo deve conter apenas valores finitos"
        )
 
 
class TestModelSaving(unittest.TestCase):
    """Testes de salvamento do modelo."""
 
    def setUp(self):
        """Cria modelo para teste de persistência."""
        self.input_dim = 38
        self.window_size = 10
        self.model = create_model(
            input_dim=self.input_dim,
            window_size=self.window_size,
            hidden_dims=(32, 16),
            latent_dim=8,
            device="cpu",
        )
 
    def test_save_model_creates_file(self):
        """Verifica que save_model cria o arquivo .pt no caminho especificado."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(self.model, path=path)
            self.assertTrue(path.exists(), "Arquivo do modelo deve ser criado em disco")
            self.assertGreater(path.stat().st_size, 0, "Arquivo não deve estar vazio")
 
 
class TestModelLoading(unittest.TestCase):
    """Testes de carregamento do modelo."""
 
    def setUp(self):
        """Cria e salva modelo para teste de carregamento."""
        self.input_dim = 38
        self.window_size = 10
        self.model = create_model(
            input_dim=self.input_dim,
            window_size=self.window_size,
            hidden_dims=(32, 16),
            latent_dim=8,
            device="cpu",
        )
 
    def test_load_model_restores_weights(self):
        """Verifica que load_model restaura os pesos idênticos ao modelo salvo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_model(self.model, path=path)
 
            # Cria nova instância com a mesma arquitetura
            new_model = create_model(
                input_dim=self.input_dim,
                window_size=self.window_size,
                hidden_dims=(32, 16),
                latent_dim=8,
                device="cpu",
            )
            loaded_model = load_model(new_model, path=path, device="cpu")
 
            # Compara pesos do modelo original e do carregado
            for (name1, param1), (name2, param2) in zip(
                self.model.state_dict().items(),
                loaded_model.state_dict().items(),
            ):
                self.assertEqual(name1, name2)
                self.assertTrue(
                    torch.equal(param1, param2),
                    f"Pesos de '{name1}' devem ser idênticos após carregamento"
                )
 
if __name__ == "__main__":
    unittest.main()
