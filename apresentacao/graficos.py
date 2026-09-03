"""Figuras da apresentacao, geradas a partir dos resultados reais do projeto.

Nenhum numero e digitado a mao: as curvas vem de ``artifacts/training_history.json``
e das execucoes comparativas gravadas em ``apresentacao/experimentos/``. Rode este
modulo antes de ``gerar_slides.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parent
FIGURAS = AQUI / "figuras"
EXPERIMENTOS = AQUI / "experimentos"

# Paleta categorica validada para daltonismo (deutan/protan/tritan).
SERIE_1 = "#2a78d6"  # azul
SERIE_2 = "#eb6834"  # laranja
SERIE_3 = "#1baf7a"  # verde-agua
SERIE_4 = "#eda100"  # amarelo
CORES_CONFIG = [SERIE_1, SERIE_2, SERIE_3, SERIE_4]

TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
MUDO = "#898781"
GRADE = "#e1e0d9"
SUPERFICIE = "#fcfcfb"
CRITICO = "#d03b3b"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": TINTA_2,
        "text.color": TINTA,
        "xtick.color": MUDO,
        "ytick.color": MUDO,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRADE,
        "grid.linewidth": 0.8,
        "savefig.facecolor": SUPERFICIE,
        "axes.axisbelow": True,
    }
)


def num_br(valor: float, casas: int = 4) -> str:
    """Formata numero no padrao pt-BR (virgula como separador decimal)."""
    return f"{valor:.{casas}f}".replace(".", ",")


def _salvar(fig, nome: str) -> Path:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    caminho = FIGURAS / nome
    fig.savefig(caminho, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  figura: {caminho.name}")
    return caminho


def _aplicar_base(base: float) -> None:
    """Ajusta a tipografia dos eixos ao tamanho final da figura."""
    plt.rcParams.update(
        {
            "font.size": base,
            "axes.labelsize": base * 0.95,
            "xtick.labelsize": base * 0.92,
            "ytick.labelsize": base * 0.92,
        }
    )


def _carregar_configs() -> list[dict]:
    caminho = EXPERIMENTOS / "consolidado.json"
    if not caminho.is_file():
        raise FileNotFoundError(
            "Rode os experimentos antes: consolidado.json nao encontrado em "
            f"{EXPERIMENTOS}"
        )
    return json.loads(caminho.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Curva de treino x validacao
# ---------------------------------------------------------------------------
def curva_treino(figsize=(7.4, 4.0), base=11.0, nome="curva_treino.png") -> Path:
    """Erro de reconstrucao por epoca, com a melhor epoca destacada.

    ``figsize`` deve ser o tamanho final da figura no slide, em polegadas:
    gerar grande e encolher na montagem deixa a tipografia ilegivel.
    """
    historico = json.loads(
        (REPO / "artifacts" / "training_history.json").read_text(encoding="utf-8")
    )
    treino = historico["train_loss"]
    validacao = historico["val_loss"]
    epocas = np.arange(1, len(treino) + 1)
    melhor = int(np.argmin(validacao))

    _aplicar_base(base)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epocas, treino, color=SERIE_1, linewidth=2, label="Treino")
    ax.plot(epocas, validacao, color=SERIE_2, linewidth=2, label="Validação")

    ax.scatter(
        [epocas[melhor]],
        [validacao[melhor]],
        s=70,
        color=SERIE_2,
        edgecolor=SUPERFICIE,
        linewidth=2,
        zorder=5,
    )
    ax.annotate(
        f"melhor época: {epocas[melhor]}\nerro de validação {num_br(validacao[melhor])}",
        xy=(epocas[melhor], validacao[melhor]),
        xytext=(epocas[melhor] - 26, validacao[melhor] + 0.045),
        fontsize=base * 0.91,
        color=TINTA_2,
        arrowprops={"arrowstyle": "-", "color": MUDO, "linewidth": 1},
    )
    ax.axvspan(
        epocas[melhor], epocas[-1], color=MUDO, alpha=0.10, linewidth=0
    )
    ax.text(
        epocas[melhor] + (len(treino) - melhor) / 2,
        max(max(treino), max(validacao)) * 0.93,
        "early stopping\n(10 épocas sem melhora)",
        fontsize=base * 0.82,
        color=TINTA_2,
        ha="center",
    )

    ax.text(
        epocas[-1] + 0.9, treino[-1], "Treino", color=SERIE_1, fontsize=base * 0.95,
        va="center", fontweight="bold",
    )
    ax.text(
        epocas[-1] + 0.9, validacao[-1], "Validação", color=SERIE_2, fontsize=base * 0.95,
        va="center", fontweight="bold",
    )

    ax.set_xlabel("Época")
    ax.set_ylabel("Erro de reconstrução (MSE)")
    ax.set_xlim(0, len(treino) + 19)
    ax.grid(axis="x", visible=False)
    return _salvar(fig, nome)


# ---------------------------------------------------------------------------
# 2. Erro de reconstrucao no teste, limiar e anomalias reais
# ---------------------------------------------------------------------------
def erro_no_teste(figsize=(9.6, 3.9), base=11.0, nome="erro_no_teste.png") -> Path:
    """Serie do erro no conjunto de teste com o limiar e as anomalias rotuladas."""
    import sys

    sys.path.insert(0, str(REPO))
    import numpy as np
    from src.data.loader import clean_aligned, clean_data, load_data
    from src.evaluation.metrics import reconstruction_threshold
    from src.models.model import create_model, reconstruction_error
    from src.models.persistence import load_model
    from src.preprocessing.transform import split_data, standardize
    from src.utils import config

    train_clean = clean_data(load_data(config.TRAIN_PATH))
    test_clean, label_clean = clean_aligned(
        load_data(config.TEST_PATH), load_data(config.TEST_LABEL_PATH)
    )
    X_full = train_clean.to_numpy()
    X_train, _, _, _ = split_data(
        X_full, np.arange(len(X_full)), test_size=config.VALIDATION_SIZE
    )
    X_train, media, desvio = standardize(X_train)
    X_test, _, _ = standardize(test_clean.to_numpy(), mean=media, std=desvio)
    y_test = label_clean.to_numpy().ravel()[config.WINDOW_SIZE - 1:]

    modelo = load_model(create_model(input_dim=X_train.shape[1]))
    erro_treino = reconstruction_error(modelo, X_train)
    erro_teste = reconstruction_error(modelo, X_test)
    limiar = reconstruction_threshold(erro_treino, config.ANOMALY_PERCENTILE)

    _aplicar_base(base)
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(erro_teste))

    dentro = False
    for i, rotulo in enumerate(y_test):
        if rotulo and not dentro:
            inicio, dentro = i, True
        elif not rotulo and dentro:
            ax.axvspan(inicio, i, color=CRITICO, alpha=0.16, linewidth=0)
            dentro = False
    if dentro:
        ax.axvspan(inicio, len(y_test), color=CRITICO, alpha=0.16, linewidth=0)

    ax.plot(x, erro_teste, color=SERIE_1, linewidth=0.7)
    ax.axhline(limiar, color=SERIE_2, linewidth=1.8, linestyle="--")
    ax.annotate(
        f"limiar = {num_br(limiar, 2)}\n(percentil "
        f"{num_br(config.ANOMALY_PERCENTILE, 1)} do erro de treino)",
        xy=(len(x) * 0.30, limiar),
        xytext=(len(x) * 0.05, limiar * 900),
        color=SERIE_2,
        fontsize=base * 0.91,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": SERIE_2, "linewidth": 1},
    )

    ax.set_yscale("log")
    ax.set_xlabel("Janela do conjunto de teste (ordem temporal)")
    ax.set_ylabel("Erro de reconstrução\n(escala log)")
    ax.set_xlim(0, len(x))
    ax.grid(axis="x", visible=False)
    ax.legend(
        handles=[
            plt.Line2D([], [], color=SERIE_1, linewidth=1.6, label="Erro por janela"),
            plt.Line2D([], [], color=SERIE_2, linewidth=1.8, linestyle="--", label="Limiar de anomalia"),
            Patch(facecolor=CRITICO, alpha=0.16, label="Anomalia real (test_label)"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005, 1.0, 0.1),
        mode="expand",
        frameon=False,
        fontsize=base * 0.86,
        ncol=3,
    )
    return _salvar(fig, nome)


# ---------------------------------------------------------------------------
# 3. Comparacao das configuracoes treinadas
# ---------------------------------------------------------------------------
def comparacao_configs(percentil: str = "99.5", figsize=(7.6, 4.0), base=11.0,
                       nome="comparacao_configs.png") -> Path:
    """Precisao, recall e F1 das quatro configuracoes, no mesmo limiar percentil."""
    configs = _carregar_configs()
    metricas = [("precision", "Precisão"), ("recall", "Recall"), ("f1_score", "F1")]

    _aplicar_base(base)
    fig, ax = plt.subplots(figsize=figsize)
    largura = 0.19
    posicoes = np.arange(len(metricas))

    for indice, cfg in enumerate(configs):
        valores = [cfg["por_percentil"][percentil][chave] for chave, _ in metricas]
        deslocamento = (indice - (len(configs) - 1) / 2) * largura
        barras = ax.bar(
            posicoes + deslocamento,
            valores,
            width=largura * 0.80,
            color=CORES_CONFIG[indice],
            label=f"{cfg['id']} - {cfg['nome']}",
        )
        for barra, valor in zip(barras, valores):
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                valor + 0.018,
                num_br(valor, 2),
                ha="center",
                fontsize=base * 0.78,
                color=TINTA_2,
            )

    ax.set_xticks(posicoes)
    ax.set_xticklabels([rotulo for _, rotulo in metricas])
    ax.set_ylabel(f"Valor no conjunto de teste (limiar p{percentil})")
    ax.set_ylim(0, 1.04)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=base * 0.86, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.16))
    return _salvar(fig, nome)


# ---------------------------------------------------------------------------
# 4. Trade-off do limiar
# ---------------------------------------------------------------------------
def tradeoff_limiar(config_id: str = "A", figsize=(7.0, 3.9), base=11.0,
                    nome="tradeoff_limiar.png") -> Path:
    """Precisao x recall x F1 conforme o percentil do limiar, na configuracao base."""
    configs = {cfg["id"]: cfg for cfg in _carregar_configs()}
    cfg = configs[config_id]
    percentis = sorted(cfg["por_percentil"], key=float)
    x = np.arange(len(percentis))

    series = [
        ("precision", "Precisão", SERIE_1),
        ("recall", "Recall", SERIE_2),
        ("f1_score", "F1", SERIE_3),
    ]

    _aplicar_base(base)
    fig, ax = plt.subplots(figsize=figsize)
    for chave, rotulo, cor in series:
        valores = [cfg["por_percentil"][p][chave] for p in percentis]
        ax.plot(x, valores, color=cor, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SUPERFICIE, markeredgewidth=1.6)
        ax.text(x[-1] + 0.07, valores[-1], rotulo, color=cor, fontsize=base * 0.95,
                va="center", fontweight="bold")
        for xi, valor in zip(x, valores):
            ax.text(xi, valor + 0.035, num_br(valor, 2), ha="center", fontsize=base * 0.85,
                    color=TINTA_2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"p{p}".replace(".", ",") for p in percentis])
    ax.set_xlabel("Percentil do erro de treino usado como limiar")
    ax.set_ylabel("Valor no conjunto de teste")
    ax.set_xlim(-0.25, len(percentis) - 0.02)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="x", visible=False)
    return _salvar(fig, nome)


def main() -> None:
    """Gera as figuras ja no tamanho em que aparecem em cada slide.

    Cada figura e desenhada com a largura e a altura finais (em polegadas) do
    espaco que ocupa no slide, e com a tipografia proporcional a esse tamanho.
    Assim nada e reduzido na montagem — o que preservaria o desenho mas
    deixaria eixos e rotulos ilegiveis.
    """
    print("Gerando figuras da apresentacao...")
    # Slide 13 - dois graficos lado a lado
    curva_treino(figsize=(4.35, 2.18), base=8.0, nome="curva_treino.png")
    erro_no_teste(figsize=(6.95, 2.18), base=8.0, nome="erro_no_teste.png")
    # Slide 11 - trade-off do limiar
    tradeoff_limiar(figsize=(3.95, 2.15), base=8.0, nome="tradeoff_limiar.png")
    # Material de apoio do repositorio (nao vai para os slides)
    comparacao_configs(figsize=(7.6, 4.0), base=11.0, nome="comparacao_configs.png")
    print("Figuras em", FIGURAS)


if __name__ == "__main__":
    main()
