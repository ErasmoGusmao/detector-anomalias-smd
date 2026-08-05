"""Testes unitários para o módulo de carregamento de dados (src/data/loader.py).

   Verifica que o pipeline de leitura carrega corretamente os arquivos, produz data frames com o formato esperado e trata erros de caminho inexistente."""

import unittest
from pathlib import Path

import pandas as pd

from src.data.loader import load_data, clean_data
from src.utils import config

class TestDataLoading(unittest.TestCase):
 """Testes de carregamento dos dados."""

    def test_load_data_returns_dataframe(self):
        """Verifica que load_data retorna um DataFrame não vazio com 38 colunas."""
        df = load_data(config.TRAIN_PATH)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0, "DataFrame não deve estar vazio.")
       self.assertEqual(df.shape[1], 38, "SMD deve conter 38 features (colunas).")

 def test_load_data_file_not_found(self):
 """Verifica que load_data levanta FileNotFoundError para caminho inexistente."""
 with self.assertRaises(FileNotFoundError):
 load_data(Path("caminho/inexistente/arquivo.txt"))

 def test_clean_data_removes_nan(self):
 """Verifica que clean_data remove linhas com NaN sem alterar colunas."""
 df = pd.DataFrame({0: [1., None, 3.], 1: [4., 5., 6.]})
 cleaned = clean_data(df)
 self.assertFalse(cleaned.isnull().any().any(), "Não deve haver NaN após limpeza.")
 self.assertEqual(cleaned.shape[1], 2, "Número de colunas deve ser preservado.")
 self.assertEqual(len(cleaned), 2, "Linha com NaN deve ser removida.")

if __name__ == "__main__":
 unittest.main()