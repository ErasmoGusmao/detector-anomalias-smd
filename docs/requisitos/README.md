# Documento de requisitos — organização dos arquivos

Os requisitos do sistema são elaborados com o **GR4ML** (*Goal-Oriented Requirements
Engineering for Machine Learning*), que organiza a elicitação em três visões
complementares. Cada visão é escrita em um arquivo próprio para que as frentes avancem
em paralelo, sem conflito de edição. A consolidação final acontece em `docs/requisitos.md`.

## Arquivos

| Arquivo | Visão | Conteúdo | Requisitos que origina |
|---------|-------|----------|------------------------|
| `01-business-view.md` | Business View | Atores, objetivos (*Goals*), indicadores, decisões (*Decision Goals*) e perguntas de negócio (*Question Goals*) | Requisitos Funcionais (**RF**) |
| `02-analytics-view.md` | Analytics Design View | Tipo de análise, tarefa de aprendizado de máquina, algoritmos candidatos, qualidades esperadas (*Softgoals*) e métricas associadas | Requisitos Não Funcionais (**RNF**) |
| `03-data-view.md` | Data Preparation View | Fontes e entidades de dados do SMD, transformações aplicadas e composição do conjunto final | Requisitos de Dados (**RD**) |

Cada arquivo entrega **a sua visão e também os requisitos que nascem dela** — assim cada
requisito já fica rastreável até o elemento do modelo que o originou, sem depender da
conclusão das demais visões.

## Como redigir os requisitos

Modelos de frase adotados:

- **RF** — "O sistema deve *[ação/cálculo]*, a partir de *[dados de entrada]*, para permitir
  que *[ator]* *[decisão apoiada]*."
- **RNF** — "O sistema deve garantir *[qualidade]*, medida por *[métrica]*, com
  *[meta/limite]*."
- **RD** — "O sistema deve integrar/coletar/tratar *[dado]*, proveniente de *[fonte]*,
  aplicando *[transformação]*, para alimentar *[RF correspondente]*."

Regras de escrita:

1. Usar sempre "o sistema deve" — nunca "seria bom se" ou "o ideal seria".
2. Um requisito por frase; não combinar duas exigências na mesma sentença.
3. O requisito precisa ser **testável**: alguém de fora deve conseguir dizer, olhando o
   sistema pronto, se ele foi atendido.
4. O requisito precisa ser **rastreável**: indicar de qual elemento da visão ele se originou.

## Numeração

Os identificadores definitivos (`RF01`, `RNF01`, `RD01`, ...) são atribuídos na
consolidação, para evitar colisão entre as frentes. Nos arquivos de visão, basta listar os
requisitos na ordem em que aparecem.
