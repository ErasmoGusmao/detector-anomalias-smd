# Detector de Anomalias em Métricas de Servidores (SMD)

Projeto Integrador da disciplina **Engenharia de Software para IA e Frameworks Profundos** (pós-graduação, CIn) — **Grupo 12**.

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
> pelo grupo** na reunião de alinhamento da Entrega 1. Esta versão entrega a **estrutura
> organizada e tipada** do projeto; as funções estão com assinatura e contrato definidos,
> e a implementação será preenchida na sequência.

## Entendendo os dados e a estratégia de detecção

Esta seção explica, de forma didática, **como os dados estão organizados** e **por que** o
conjunto de treino não contém anomalias enquanto o de teste contém — duas dúvidas naturais
sobre o SMD.

### 1. O formato: cada coluna é uma métrica (feature)

Cada arquivo `.txt` é a série temporal de **uma máquina** e **não possui cabeçalho**. Ainda
assim, ele tem features: são as **38 colunas**, cada uma representando uma métrica
operacional do servidor (CPU, memória, rede, disco, etc.) acompanhada ao longo do tempo.

```
                       38 colunas = 38 métricas (features)
                  ┌───────┬───────┬───────┬─────┬────────┐
   instante 0     │  m0   │  m1   │  m2   │ ... │  m37   │ ┐
   instante 1     │  ...  │       │       │     │        │ │
   instante 2     │  ...  │       │       │     │        │ │  N linhas
      ...         │       │       │       │     │        │ │  = instantes de tempo
   instante N-1   │  ...  │       │       │     │        │ ┘
                  └───────┴───────┴───────┴─────┴────────┘
    • cada LINHA  = uma "foto" da máquina num instante de tempo
    • cada COLUNA = uma métrica acompanhada ao longo do tempo
```

**Por que as colunas não têm nome?** As métricas do SMD são **anonimizadas** pela fonte
original (por privacidade). Sabemos que são 38 sinais operacionais, mas não qual é CPU, qual
é memória, e assim por diante. Para a detecção isso não é problema: o modelo aprende o
*padrão conjunto* das 38 métricas, sem precisar do significado de cada uma. Como os valores
já vêm **normalizados na faixa [0, 1]**, também não há escalas diferentes atrapalhando.

### 2. Por que o treino é só "normal" e o teste tem anomalias

A ideia central é que o modelo **aprende apenas o comportamento normal** e detecta anomalias
por contraste:

- **`train`** — contém **só comportamento normal** (sem anomalias). É com ele que o modelo
  aprende "como é o normal".
- **`test`** — contém trechos normais **e anômalos**, para avaliar se o detector consegue
  separá-los.
- **`test_label`** — vetor de **0/1** (1 = instante anômalo), usado **apenas na avaliação**,
  nunca no treino.

> Isto **não** é um classificador "normal vs. anomalia". O modelo nunca vê exemplos de
> anomalia: ele aprende a reproduzir o normal e estranha o que foge desse padrão.

Uma analogia: um operador que passou meses vendo **só** o servidor saudável. No dia em que
algo sai do padrão, ele estranha na hora — não porque conhece aquela falha específica, mas
porque *aquilo não é o normal que ele conhece*.

### 3. Como a anomalia é detectada: erro de reconstrução + linha de corte

A abordagem pretendida para as próximas entregas usa o **erro de reconstrução**: o modelo
tenta reproduzir cada instante a partir do que aprendeu sobre o comportamento normal.

- Instante **normal** → o modelo reconstrói bem → **erro baixo**.
- Instante **anômalo** → o modelo nunca viu algo parecido → reconstrói mal → **erro alto**.

Define-se então uma **linha de corte (limiar / *threshold*)**: todo instante com erro acima
dela é marcado como anomalia.

```
  erro de
  reconstrução
     ▲
     │                                   ╭╮    ← pico = anomalia (erro alto)
     │                                   ││
  ───┼──── limiar ─────────────────────────────────────  ← linha de corte
     │    ╱╲     ╱╲     ╱╲     ╱╲     ╱╲ ││ ╱╲
     │  ╱╲╱ ╲╱╲╱  ╲╱╲╱   ╲╱╲╱   ╲╱╲ ╱╲╱ ╰╯  ╲╱╲   ← normal (erro baixo)
     └──────────────────────────────────────────────────►  tempo
```

### 4. Paradigma: não-supervisionado no treino, supervisionado na avaliação

| Etapa | Dados usados | Usa os rótulos? | Natureza |
|-------|--------------|-----------------|----------|
| **Treino** | `train` (38 features, tudo normal) | ❌ Não (não há rótulo no treino) | **Não-supervisionado** — aprende a normalidade |
| **Detecção** | `test` (38 features) → erro → limiar | ❌ Não (o modelo decide sozinho) | — |
| **Avaliação** | previsão do modelo × `test_label` | ✅ Sim (somente aqui) | **Supervisionada** — mede precisão/revocação/F1 |

Esse desenho — treinar só na classe normal e avaliar com rótulos — é chamado de **detecção
de anomalias semi-supervisionada** (ou *modelagem da normalidade*). Ele faz sentido para o
SMD porque anomalias são **raras e imprevisíveis**: é inviável coletar exemplos de todas as
falhas possíveis, mas o comportamento normal é abundante e fácil de aprender.

> ⚠️ **Impacto no pipeline:** como o `train` é só features (sem coluna-alvo embutida), a
> função `split_features_target` **não se aplica diretamente a ele** — os rótulos vivem
> separados, em `test_label`, e entram apenas na etapa de avaliação.

### 5. O `interpretation_label`: quais métricas explicam cada anomalia

Enquanto o `test_label` diz **quando** existe anomalia (um 0/1 por instante), o
`interpretation_label` diz **por que / onde** ela acontece — isto é, **quais das 38 métricas**
são as responsáveis por cada anomalia. Cada linha descreve um **evento de anomalia** (um
trecho contínuo no tempo) no formato:

```
<instante_início>-<instante_fim>:<dim1>,<dim2>,...

Exemplo:  15849-16368:1,9,10,12,13,14,15
          └─────┬────┘ └──────┬────────┘
            quando         quais métricas explicam a anomalia
        (do instante       (as dimensões 1, 9, 10, 12, 13, 14 e 15
         15849 ao 16368)    estavam fora do padrão nesse trecho)
```

Os eventos variam de **amplos** (afetando quase todas as 38 métricas, como um incidente
generalizado) a **curtos e localizados** (poucas métricas, como uma falha pontual e
recorrente).

| Arquivo | Responde à pergunta | Granularidade |
|---------|---------------------|---------------|
| `test_label` | **QUANDO** há anomalia? | por instante (0/1) |
| `interpretation_label` | **POR QUE / ONDE** está a anomalia? | quais métricas a causam |

A utilidade tem duas frentes: **interpretabilidade** (apontar ao operador quais sinais
investigar, em vez de varrer as 38 manualmente) e **avaliação de uma tarefa mais avançada**
(medir não só se a anomalia foi detectada, mas se as **dimensões responsáveis** foram
corretamente localizadas).

> Para a **detecção básica** (descobrir *quando* há anomalia) bastam `train` + `test` +
> `test_label`. O `interpretation_label` apoia o passo **mais avançado** de localizar a
> causa, e é **opcional** no escopo mínimo. Detalhe técnico: as dimensões são numeradas de
> **1 a 38** (1-indexed); na matriz NumPy (0-indexed), a "dimensão 1" é a **coluna 0** —
> atenção ao *off-by-one* ao cruzá-las com os dados.

## Como obter a base de dados

Os arquivos do SMD **não são versionados** neste repositório — datasets ficam fora do Git
por boa prática (a pasta `data/` é ignorada pelo `.gitignore`, exceto o `.gitkeep`). Cada
integrante precisa baixar a base e colocá-la em `data/` localmente. Há duas formas:

**Opção A — Kaggle**

```bash
# requer a Kaggle CLI autenticada (pip install kaggle + token de API)
kaggle datasets download -d mgusat/smd-onmiad -p data/ --unzip
```

Alternativamente, baixe pelo navegador em
[SMD_OnmiAD](https://www.kaggle.com/datasets/mgusat/smd-onmiad) e extraia o conteúdo em `data/`.

**Opção B — repositório original (OmniAnomaly)**

```bash
git clone https://github.com/NetManAIOps/OmniAnomaly.git
# copie a pasta ServerMachineDataset/ para data/
```

Organização típica dos arquivos após o download — um `.txt` por máquina, nomeado no
formato `machine-<grupo>-<indice>`:

```
data/
└── ServerMachineDataset/
    ├── train/                 # séries de treino
    ├── test/                  # séries de teste
    ├── test_label/            # rótulos normal/anômalo do conjunto de teste
    └── interpretation_label/  # dimensões associadas às anomalias
```

> Confira a estrutura após extrair (ela pode variar conforme a fonte) e ajuste o caminho
> de leitura usado pelo pipeline. Os arquivos baixados ficam **fora do controle de versão**
> (cobertos pelo `.gitignore`), mantendo o repositório leve e reprodutível.

## Estrutura do projeto (Entrega 1)

```
.
├── data/                  # base de dados - SMD
├── notebooks/             # experimentos e exploracao
├── src/
│   ├── data/
│   │   └── loader.py      # carregamento e limpeza dos dados
│   ├── preprocessing/
│   │   └── transform.py   # transformacoes e split dos dados
│   ├── models/
│   │   └── model.py       # definicao do modelo
│   ├── training/
│   │   └── train.py       # rotina de treinamento
│   ├── evaluation/
│   │   └── metrics.py     # metricas de avaliacao
│   └── utils/
│       └── config.py      # configuracoes do pipeline
├── main.py                # ponto de entrada do pipeline
├── requirements.txt
└── README.md
```

A estrutura segue a ideia de **separação de responsabilidades** e usa **type hints**
nas funções (assinaturas e contratos definidos). A implementação das funções e os
módulos de testes entram nas próximas entregas.

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
| `split_features_target(data, target_column)` | `src/preprocessing/transform.py` | Separa atributos e variavel alvo. |
| `split_data(X, y)` | `src/preprocessing/transform.py` | Reserva o trecho final da serie para teste, em ordem temporal. |
| `create_model()` | `src/models/model.py` | Cria e configura o modelo. |
| `predict(model, X)` | `src/models/model.py` | Gera predicoes com o modelo treinado. |
| `train_model(model, X_train, y_train)` | `src/training/train.py` | Executa a rotina de treinamento. |
| `calculate_metrics(y_true, y_pred)` | `src/evaluation/metrics.py` | Calcula metricas de avaliacao. |
| `main()` | `main.py` | Orquestra o pipeline. |

## Status das etapas

| Entrega | Conteúdo | Status |
|---------|----------|--------|
| 1 | Descrição/contextualização do problema | ✅ Concluído |
| 1 | Documentação da base de dados SMD | ✅ Concluído |
| 1 | Funções iniciais | ✅ Concluído |
| 1 | Modularização e organização do código | ✅ Concluído |
| 1 | Tipagem (type hints) | ✅ Concluído |
| 2 | Uso adequado de NumPy | 🔄 Em andamento |
| 3 | Implementação em PyTorch (partes 1 e 2) | ⬜ Pendente |
| 4 | Testes automatizados (unittest) | ⬜ Pendente |
| 5 | Requisitos | ⬜ Pendente |
| 6 | Design/arquitetura + Git e colaboração | ⬜ Pendente |
| Final | Apresentação | ⬜ Pendente |

> Entrega 2 (em andamento): o pré-processamento com **NumPy** em
> `src/preprocessing/transform.py` (padronização, separação atributos/alvo e divisão
> treino/teste) já está implementado. Faltam o carregamento e a limpeza dos dados
> (`load_data`/`clean_data`) e a orquestração do `main.py`, além de definir o **recorte
> inicial do SMD** (começar por uma única máquina) para o primeiro experimento.

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
