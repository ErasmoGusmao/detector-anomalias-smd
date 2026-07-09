"""Configuração central do projeto.

Reúne as constantes do pipeline: caminhos dos dados, nome do arquivo e os parâmetros do
split treino/validação.
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = BASE_DIR / "data" / "ServerMachineDataset"

FILE_NAME: str = "machine-1-1.txt"

TRAIN_PATH: Path = DATA_DIR / "train" / FILE_NAME
TEST_PATH: Path = DATA_DIR / "test" / FILE_NAME
TEST_LABEL_PATH: Path = DATA_DIR / "test_label" / FILE_NAME
INTERPRETATION_LABEL_PATH: Path = DATA_DIR / "interpretation_label" / FILE_NAME

VALIDATION_SIZE: float = 0.2
