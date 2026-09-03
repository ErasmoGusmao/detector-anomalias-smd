# Apresentação final — 13 slides

Material da apresentação do projeto (Grupo 12). O arquivo entregue é o PDF; o `.pptx`
fica junto para quem for apresentar poder ajustar e usar o modo apresentador, onde
aparecem as **anotações de estudo** de cada slide.

| Arquivo | O que é |
|---|---|
| `Grupo12-Detector-de-Anomalias-SMD.pptx` | Apresentação editável, com anotações do apresentador |
| `Grupo12-Detector-de-Anomalias-SMD.pdf` | Versão para envio no Classroom |
| `tema.py` | Paleta, tipografia e blocos de layout |
| `graficos.py` | Figuras, geradas a partir dos resultados reais |
| `gerar_slides.py` | Conteúdo dos 13 slides e das anotações |
| `experimentos.py` | Comparação das configurações do modelo |
| `experimentos/` | Resultados das execuções, em JSON |
| `figuras/` | PNGs gerados por `graficos.py` |

## Como regerar

Os slides são montados por código para que **nenhum número seja digitado à mão**: as
métricas vêm de `artifacts/results.json` e de `apresentacao/experimentos/`. Se o modelo
for retreinado, basta rodar de novo e os slides passam a refletir os novos valores.

```bash
pip install python-pptx matplotlib          # além das dependências do projeto

python apresentacao/experimentos.py         # (opcional) refaz a comparação de configurações
python apresentacao/graficos.py             # gera as figuras
python apresentacao/gerar_slides.py         # monta o .pptx
```

O PDF é exportado a partir do `.pptx` (PowerPoint ou Google Apresentações → *Baixar → PDF*).

## Comparação de configurações

`experimentos.py` treina quatro configurações do autoencoder, mantendo semente, dados e
pré-processamento fixos, e avalia cada uma em três percentis de limiar. É a origem do
slide 11.

| Config | O que muda | Erro de validação | Precisão (p99,5) | Recall (p99,5) | F1 (p99,5) |
|---|---|---|---|---|---|
| A — Baseline | janela 50 · latente 16 · encoder (64,32) | 0,1896 | 0,253 | 0,880 | 0,393 |
| B — Gargalo estreito | janela 50 · latente 8 · encoder (64,32) | 0,2090 | 0,236 | 0,897 | 0,374 |
| C — Janela longa | janela 100 · latente 16 · encoder (64,32) | 0,2048 | 0,275 | 0,964 | **0,427** |
| D — Maior capacidade | janela 50 · latente 32 · encoder (128,64) | **0,1572** | 0,241 | 0,928 | 0,382 |

Dois resultados que orientam o próximo ciclo:

1. **Reconstruir melhor não é detectar melhor.** A configuração D obteve o menor erro de
   validação e não lidera a detecção; a C, com erro de validação pior, tem o melhor F1.
   Capacidade em excesso reconstrói bem também o que é anômalo, e isso reduz a separação
   entre normal e anômalo.
2. **O limiar tem mais efeito que a arquitetura.** Entre as quatro configurações o F1
   varia 0,054; variando apenas o percentil do limiar, na mesma configuração, varia 0,177.
   No percentil 99,9 a precisão chega a 83,3%, ao custo de o recall cair para 43,3%.

A execução de referência do repositório continua sendo a configuração A, registrada em
`artifacts/results.json`.
