# Detector de Anomalias em Métricas de Servidores (SMD)

Projeto Integrador da disciplina **Engenharia de Software para IA e Frameworks Profundos** (pós-graduação, CIn) — **Grupo 12**.

> ⚠️ **Nome provisório** (`detector-anomalias-smd`), sujeito a confirmação pela equipe.
> Tema proposto por Leonardo Magalhães e Breno Santos.

## Problema

Sistemas computacionais modernos geram continuamente métricas operacionais, como uso
de CPU, memória, disco, rede e outros indicadores de funcionamento. Em ambientes reais,
mudanças inesperadas nessas métricas podem indicar falhas, degradação de desempenho,
sobrecarga, mau funcionamento de serviços ou outros eventos que exigem investigação.

O problema abordado neste projeto é a **detecção de comportamentos anômalos em séries
temporais multivariadas de servidores**. A proposta é construir um sistema de Inteligência
Artificial capaz de carregar métricas operacionais, pré-processar os dados e, nas próximas
entregas, treinar e avaliar um modelo para identificar pontos ou períodos com comportamento
incomum.

Esse tipo de solução é relevante porque a identificação manual de anomalias em grandes
volumes de métricas é custosa, sujeita a atrasos e dependente da experiência de quem
monitora o sistema. Um detector automatizado pode apoiar equipes de operação, engenharia
e observabilidade, apontando situações suspeitas que merecem análise.

Nesta etapa inicial, o foco do projeto ainda não é entregar um modelo treinado, mas sim
definir o problema, organizar a estrutura do repositório, documentar a base de dados
pretendida e preparar as primeiras funções do pipeline.

## Base de dados: SMD (Server Machine Dataset)

A base de dados escolhida pelo grupo é o **SMD — Server Machine Dataset**, associado ao
trabalho **OmniAnomaly**. O dataset está disponível publicamente no Kaggle em
[SMD_OnmiAD](https://www.kaggle.com/datasets/mgusat/smd-onmiad) e também aparece na
referência original do projeto
[NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly).

O SMD reúne métricas coletadas de servidores ao longo do tempo. Segundo a documentação
do OmniAnomaly, o dataset possui dados de **28 máquinas**, organizadas em grupos de
entidades nomeadas no formato `machine-<grupo>-<indice>`. Cada máquina contém séries
temporais multivariadas com **38 dimensões** de métricas. A base foi construída para o
problema de detecção de anomalias em dados operacionais de servidores.

A organização da base inclui:

- `train`: primeira metade da série temporal de cada máquina, usada para treinamento;
- `test`: segunda metade da série temporal de cada máquina, usada para avaliação;
- `test_label`: rótulos que indicam se cada ponto do conjunto de teste é normal ou anômalo;
- `interpretation_label`: indicação das dimensões associadas às anomalias.

Para manter o escopo viável durante a disciplina, a primeira versão do projeto deve trabalhar
com um recorte controlado da base, por exemplo uma única máquina, antes de expandir para
as 28 máquinas. Essa decisão reduz a complexidade inicial sem descaracterizar o problema,
pois cada máquina do SMD já representa uma série temporal multivariada completa.

O SMD é adequado ao escopo do projeto porque permite aplicar diretamente os requisitos da
disciplina:

- carregamento de dados a partir de arquivos;
- limpeza e pré-processamento de séries temporais;
- uso de NumPy para normalização, divisão e manipulação matricial;
- uso futuro de PyTorch para treinamento de um modelo de detecção de anomalias;
- avaliação experimental com métricas como precisão, revocação e F1-score;
- modularização do pipeline em carregamento, pré-processamento, modelo, treinamento,
  avaliação e inferência.

Como limitação inicial, as métricas do SMD são anonimizadas. Isso significa que o projeto
consegue estudar o comportamento numérico das séries e detectar anomalias, mas não deve
prometer diagnósticos operacionais específicos, como identificar exatamente qual componente
real do servidor falhou.

> 📌 **Nota:** a escolha do **dataset SMD** e o recorte do problema foram **ratificados
> pelo grupo** na reunião de alinhamento da Entrega 1; o **nome** do projeto segue como
> denominação de trabalho, podendo ser ajustado. Esta versão entrega a **estrutura
> organizada e tipada** do projeto; as funções estão com assinatura e contrato definidos,
> e a implementação será preenchida na sequência.

## Estrutura do projeto (Entrega 1)

```
.
├── data/              # base de dados — SMD
├── src/
│   ├── data/          # carregamento e limpeza (load_data, clean_data)
│   ├── preprocessing/ # transformações NumPy (standardize, split_data)
│   └── utils/         # configuração (constantes do pipeline)
├── main.py            # ponto de entrada — orquestra o pipeline
├── requirements.txt
└── README.md
```

A estrutura segue a ideia de **separação de responsabilidades** e usa **type hints**
nas funções (assinaturas e contratos definidos). A implementação das funções e os
módulos de modelo, treino, avaliação e testes entram nas próximas entregas.

## Como executar

```bash
# 1. criar e ativar um ambiente virtual
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate

# 2. instalar dependências
pip install -r requirements.txt

# 3. ponto de entrada do pipeline (implementação das funções em andamento)
python main.py
```

## Funções iniciais

| Função | Módulo | Responsabilidade |
|--------|--------|------------------|
| `load_data(path)` | `src/data/loader.py` | Carrega a base de dados (CSV). |
| `clean_data(data)` | `src/data/loader.py` | Remove duplicatas e valores ausentes. |
| `standardize(X)` | `src/preprocessing/transform.py` | Padroniza atributos (z-score). |
| `split_data(X, y)` | `src/preprocessing/transform.py` | Divide em treino/teste. |
| `main()` | `main.py` | Orquestra o pipeline. |

## Status das etapas

| Entrega | Conteúdo | Status |
|---------|----------|--------|
| 1 | Descrição/contextualização do problema | ✅ Concluído |
| 1 | Documentação da base de dados SMD | ✅ Concluído |
| 1 | Funções iniciais | 🟡 Em andamento |
| 1 | Modularização e organização do código | ✅ Concluído |
| 1 | Tipagem (type hints) | ✅ Concluído |
| 2 | Implementação em PyTorch (parte 1) + NumPy | ⬜ Pendente |
| 3 | Implementação em PyTorch (parte 2) | ⬜ Pendente |
| 4 | Testes automatizados (unittest) | ⬜ Pendente |
| 5 | Requisitos | ⬜ Pendente |
| 6 | Design/arquitetura + Git e colaboração | ⬜ Pendente |
| Final | Apresentação | ⬜ Pendente |

> Pendências da Entrega 1: **confirmar o nome** (hoje provisório), definir o **recorte
> inicial do SMD** que será usado no primeiro experimento e implementar as **funções
> iniciais**.

## Equipe

**Grupo 12:**

- Leonardo Magalhães
- Breno Santos
- Erasmo Gusmão
- Gabriel Santana
- João Mateus
- João Pedro
- Orlando

> Tema proposto por Leonardo Magalhães e Breno Santos.
