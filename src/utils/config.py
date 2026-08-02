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

# ---------------------------------------------------------------------------
# Hiperparâmetros do modelo (Entrega 3 — PyTorch)
# ---------------------------------------------------------------------------
# Valores default de partida. A equipe pode ajustá-los ao calibrar o modelo;
# são configuração, não lógica de negócio.

RANDOM_SEED: int = 42

# Device de execução. "cpu" por padrão; usar torch_utils.get_device(prefer_cuda=True)
# para resolver dinamicamente quando houver GPU disponível.
DEVICE: str = "cpu"

# Janela temporal deslizante (autoencoder sobre sequências do SMD).
WINDOW_SIZE: int = 50

# Laço de treino.
EPOCHS: int = 100
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3

# Early stopping: interrompe o treino se a validação não melhorar por N épocas
EARLY_STOPPING_PATIENCE: int = 10

# Learning rate scheduler (ReduceLROnPlateau).
SCHEDULER_FACTOR: float = 0.5
SCHEDULER_PATIENCE: int = 5

# Gradient clipping: norma máxima dos gradientes.
MAX_GRAD_NORM: float = 1.0

# Arquitetura do autoencoder (dimensões do encoder; o decoder é espelhado).
HIDDEN_DIMS: tuple[int, ...] = (64, 32)
LATENT_DIM: int = 16

# Limiar de anomalia: percentil do erro de reconstrução do treino acima do
# qual uma amostra é marcada como anômala.
ANOMALY_PERCENTILE: float = 99.5

# Persistência do modelo treinado.
MODEL_DIR: Path = BASE_DIR / "artifacts"
MODEL_PATH: Path = MODEL_DIR / "autoencoder.pt"
