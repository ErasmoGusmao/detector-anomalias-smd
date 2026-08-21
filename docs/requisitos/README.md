# Documento de requisitos — GR4ML

Os requisitos do sistema foram elaborados com o **GR4ML** (*Goal-Oriented Requirements
Engineering for Machine Learning*), que organiza a elicitação em três visões complementares
e só então traduz os modelos em requisitos redigidos.

O documento completo é o caderno da aula prática preenchido pelo grupo:

📄 **[`gr4ml-caderno-grupo12.pdf`](gr4ml-caderno-grupo12.pdf)** — 21 páginas, com os dois
diagramas (Business View e pipeline de dados).

## O que cada fase produziu

| Fase | Visão | Conteúdo | Requisitos que origina |
|------|-------|----------|------------------------|
| 0 | Alinhamento | Stakeholder principal e dor de negócio que motiva o detector | — |
| 1 | Business View | Atores, objetivos (*Goals*), indicadores, decisões (*Decision Goals*) e perguntas de negócio (*Question Goals*), com diagrama | Requisitos Funcionais (**RF**) |
| 2 | Analytics Design View | Tipo de análise, tarefa de aprendizado de máquina, algoritmos candidatos, qualidades esperadas (*Softgoals*) e métricas associadas | Requisitos Não Funcionais (**RNF**) |
| 3 | Data Preparation View | Fontes e entidades do SMD, transformações aplicadas e diagrama do pipeline até o conjunto final | Requisitos de Dados (**RD**) |
| 4 | Dos modelos aos requisitos | Tradução das três visões em requisitos redigidos + critério de aceitação | — |

## Requisitos consolidados

A Fase 4 do documento traz **7 Requisitos Funcionais** (`RF-01` a `RF-07`), **4 Requisitos
Não Funcionais** (`RNF-01` a `RNF-04`) e **5 Requisitos de Dados** (`RD01` a `RD05`), além do
critério de aceitação do conjunto.

Cada requisito cita o elemento do modelo que o originou — Decision Goal e Question Goal para
os RF, Softgoal e métrica para os RNF, entidades e transformações para os RD. É essa
rastreabilidade que liga o documento de volta às três visões.

## Como os requisitos foram redigidos

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

## Escopo do protótipo

O documento descreve o sistema-alvo. A validação da entrega atual é offline e restrita à
`machine-1-1`, com janelas de 50 amostras — a nota de escopo da Fase 1 e o próprio `RF-03`
registram o que ainda não está implementado no código deste repositório.
