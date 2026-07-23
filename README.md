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

O projeto evoluiu de forma incremental: a estrutura do repositório foi definida, o pipeline
de dados com NumPy (carregamento, limpeza e pré-processamento) está implementado e o
modelo de detecção (autoencoder em PyTorch) está em integração.

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
> pelo grupo** na reunião de alinhamento da Entrega 1. O projeto já conta com a **estrutura
> organizada e tipada**, com o **pipeline de dados implementado** (carregamento → limpeza →
> NumPy → split treino/validação → padronização) e com o **laço de treino do autoencoder
> (PyTorch) implementado** (arquitetura, utilitários e `train_model` prontos). Restam como
> pendência da Entrega 3 a persistência do modelo e a binarização por erro de reconstrução.

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

### 6. Métricas de avaliação: por que acurácia não é a métrica mais adequada

O conjunto de dados SMD é desbalanceado, pois o número de instantes normais é significativamente maior do que o de instantes anômalos. Nesse contexto, a **acurácia (accuracy)** não é a métrica mais adequada para avaliar o desempenho do modelo, uma vez que um classificador trivial (*dummy*), que sempre indica a classe normal, ainda obteria uma acurácia elevada. Por esse motivo, este projeto utiliza como principais métricas de avaliação a **precisão (*precision*)**, a **sensibilidade (*recall*)** e a **medida F1 (*F1-score*)**, deixando a **acurácia** apenas como uma métrica complementar.

### 7. O que esperar na saída do `python main.py`: duas bases de erro

Ao rodar o pipeline completo da Entrega 3, a saída imprime três tipos de valor numérico
relacionados ao erro de reconstrução. Eles **não são comparáveis diretamente** porque
medem coisas diferentes sobre conjuntos diferentes:

| Saída impressa | Calculado por | O que mede | Sobre qual conjunto |
|----------------|--------------|------------|---------------------|
| `Erro de treino` / `Erro de validação` (por época) | `evaluate_loss` via `nn.MSELoss` | MSE médio da **janela inteira** (todos os `window_size` timesteps × features) | Treino / validação — dados 100% normais, padronizados com a própria média/desvio |
| `Erro de reconstrução no teste` | `evaluate_loss` via `nn.MSELoss` | MSE médio da **janela inteira** (mesma fórmula acima) | Teste — dados normais **e** anômalos, padronizados com as estatísticas do **treino** |
| Erros que alimentam o **limiar e a predição** | `reconstruction_error` | MSE apenas do **último timestep** de cada janela | Treino (para calcular o limiar) e teste (para classificar) |

**Por que o erro de teste pode ficar ordens de magnitude acima do erro de treino/validação:**
o conjunto de teste do SMD é padronizado com a média e o desvio padrão calculados sobre
o treino (que é 100% normal). Janelas anômalas do teste produzem z-scores muito altos
nessa escala, elevando o MSE da janela inteira. Isso é **comportamento esperado do
método**, não um defeito do modelo ou do pipeline.

**Sobre o limiar de anomalia (`ANOMALY_PERCENTILE = 99.0`):**
o limiar é definido como o percentil 99 dos erros de reconstrução *do conjunto de treino*
(calculados por `reconstruction_error`, que usa apenas o último timestep de cada janela).
Como o modelo foi treinado nesses mesmos dados, esses erros são otimisticamente baixos —
escolha padrão e conhecida do método de detecção não-supervisionada por reconstrução.

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

## Estrutura do projeto

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
│   │   ├── model.py       # definicao do autoencoder (PyTorch)
│   │   └── persistence.py # salvamento e carregamento do modelo
│   ├── training/
│   │   └── train.py       # rotina de treinamento
│   ├── evaluation/
│   │   └── metrics.py     # metricas de avaliacao
│   └── utils/
│       ├── config.py      # configuracoes do pipeline
│       └── torch_utils.py # device e reprodutibilidade (PyTorch)
├── main.py                # ponto de entrada do pipeline
├── requirements.txt
└── README.md
```

A estrutura segue a ideia de **separação de responsabilidades** e usa **type hints**
em todas as funções. O pipeline de dados (carregamento, limpeza e pré-processamento
NumPy) está implementado; o módulo de modelo (`model.py`) e o laço de treino
(`train.py`) estão implementados. A persistência do modelo e a binarização por erro de
reconstrução (`persistence.py`, `metrics.py` — funções de threshold/predict) ainda são
stubs pendentes para conclusão da Entrega 3.
Os testes automatizados (unittest) entram na Entrega 4.

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

## Funções do pipeline

As tabelas abaixo listam as funções públicas de cada módulo com seus contratos
de entrada/saída. As marcadas como **(stub)** têm assinatura e docstring definidas,
mas a implementação do corpo está pendente para a Entrega 3.

### Carregamento e limpeza — `src/data/loader.py`

| Função | Responsabilidade |
|--------|-----------------|
| `load_data(path)` | Lê um arquivo `.txt` do SMD (sem cabeçalho, separado por vírgula) e devolve um DataFrame, uma coluna por métrica. |
| `clean_data(data)` | Remove linhas com valores ausentes ou não numéricos; preserva o índice original. |
| `clean_aligned(*dataframes)` | Limpa dois ou mais DataFrames simultaneamente, removendo as mesmas linhas de todos para manter o alinhamento temporal (ex.: `test` + `test_label`). |

### Pré-processamento NumPy — `src/preprocessing/transform.py`

| Função | Responsabilidade |
|--------|-----------------|
| `standardize(X, mean=None, std=None)` | Padroniza a matriz por z-score. Sem `mean`/`std` faz fit+transform; com eles, só transform (aplica parâmetros do treino a validação/teste). Retorna `(X_pad, mean, std)`. |
| `split_features_target(data, target_column)` | Separa um DataFrame em matriz de atributos `X` e array alvo `y` a partir do nome da coluna alvo. |
| `split_data(X, y, test_size=0.2)` | Divide `X` e `y` em dois trechos consecutivos, preservando a ordem temporal. Retorna `(X_first, X_second, y_first, y_second)`. |

### Modelo — `src/models/model.py`

| Função / Classe | Responsabilidade |
|----------------|-----------------|
| `Autoencoder` | Classe `nn.Module`. Autoencoder MLP que recebe janelas `(batch, window_size, input_dim)`, achata, passa pelo encoder e decoder e devolve a reconstrução com o mesmo shape. |
| `create_model(input_dim, window_size, hidden_dims, latent_dim, device)` | Instancia o `Autoencoder` com os hiperparâmetros informados e o move para o device de execução. Retorna o modelo pronto para treino. |
| `reconstruction_error(model, X, window_size, device)` | Executa o modelo em modo de avaliação sobre janelas deslizantes de `X` e devolve o MSE do último timestep de cada janela (array 1-D). |

### Persistência — `src/models/persistence.py` (stub)

| Função | Responsabilidade |
|--------|-----------------|
| `save_model(model, path)` | Salva o `state_dict` do autoencoder em disco, criando o diretório de destino se necessário. |
| `load_model(model, path, device)` | Carrega os pesos salvos em uma instância já criada, move para o device e coloca em modo de avaliação. |

### Treinamento — `src/training/train.py`

| Função | Responsabilidade |
|--------|-----------------|
| `build_dataloader(X, window_size, batch_size, shuffle, device)` | Extrai janelas deslizantes de `X`, monta um `TensorDataset(janela, janela)` e devolve um `DataLoader` em `float32`. |
| `train_one_epoch(model, loader, optimizer, loss_fn, device)` | Executa uma época de treino e devolve a perda média de reconstrução. |
| `evaluate_loss(model, loader, loss_fn, device)` | Calcula a perda média de reconstrução sem atualizar pesos (usado em validação e teste). |
| `train_model(model, X_train, X_val, epochs, batch_size, learning_rate, window_size, device)` | Laço completo de treino/validação: monta os DataLoaders, itera por `epochs` épocas imprimindo o erro de treino e validação a cada uma, e devolve o histórico `{"train_loss": [...], "val_loss": [...]}`. |

### Avaliação — `src/evaluation/metrics.py`

| Função | Responsabilidade |
|--------|-----------------|
| `calculate_metrics(y_true, y_pred)` | Calcula precisão, recall, F1-score e acurácia a partir de arrays binários 0/1; devolve dicionário. |
| `precision_score(y_true, y_pred)` | Precisão: TP / (TP + FP). |
| `recall_score(y_true, y_pred)` | Recall: TP / (TP + FN). |
| `f1_score(y_true, y_pred)` | F1: média harmônica de precisão e recall. |
| `accuracy_score(y_true, y_pred)` | Acurácia: (TP + TN) / total. |
| `reconstruction_threshold(errors_train, percentile)` **(stub)** | Define o limiar de anomalia pelo percentil dos erros de reconstrução do treino. |
| `predict_anomalies(errors, threshold)` **(stub)** | Binariza os erros de reconstrução em rótulos 0/1 aplicando o limiar. |

### Utilitários PyTorch — `src/utils/torch_utils.py`

| Função | Responsabilidade |
|--------|-----------------|
| `get_device(prefer_cuda, prefer_mps)` | Resolve o device de execução: retorna `"cuda"` ou `"mps"` quando solicitado e disponível; caso contrário, `"cpu"`. |
| `set_seed(seed)` | Fixa as sementes de `random`, NumPy e PyTorch (incluindo CUDA) para reprodutibilidade. |

### Orquestração — `main.py`

| Função | Responsabilidade |
|--------|-----------------|
| `main()` | Orquestra o pipeline: carrega os três arquivos do SMD, limpa, converte para NumPy, faz o split treino/validação, padroniza e imprime as dimensões dos conjuntos. O bloco de treino/avaliação PyTorch está preparado como esqueleto comentado, a ser ativado conforme os módulos de treinamento ficarem prontos. |

## Status das etapas

| Entrega | Conteúdo | Status |
|---------|----------|--------|
| 1 | Descrição/contextualização do problema | ✅ Concluído |
| 1 | Documentação da base de dados SMD | ✅ Concluído |
| 1 | Funções iniciais | ✅ Concluído |
| 1 | Modularização e organização do código | ✅ Concluído |
| 1 | Tipagem (type hints) | ✅ Concluído |
| 2 | Uso adequado de NumPy | ✅ Concluído |
| 3 | Implementação em PyTorch (partes 1 e 2) | 🔄 Em andamento |
| 4 | Testes automatizados (unittest) | ⬜ Pendente |
| 5 | Requisitos | ⬜ Pendente |
| 6 | Design/arquitetura + Git e colaboração | ⬜ Pendente |
| Final | Apresentação | ⬜ Pendente |

> **Entrega 2 concluída:** pipeline de dados NumPy funcional de ponta a ponta —
> carregamento (`load_data`, `clean_data`, `clean_aligned`), conversão para NumPy,
> split treino/validação temporal e padronização z-score. O `python main.py` roda o
> pipeline completo e imprime as dimensões dos conjuntos. **Entrega 3 em andamento:**
> a arquitetura do autoencoder (`Autoencoder`, `create_model`, `reconstruction_error`),
> os utilitários de PyTorch (`get_device`, `set_seed`) e o laço completo de treino
> (`build_dataloader`, `train_one_epoch`, `evaluate_loss`, `train_model` em
> `src/training/train.py`) já estão implementados. Faltam a persistência do modelo
> (`save_model`/`load_model` em `src/models/persistence.py`) e a binarização por erro
> de reconstrução (`reconstruction_threshold`/`predict_anomalies` em
> `src/evaluation/metrics.py`).

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
