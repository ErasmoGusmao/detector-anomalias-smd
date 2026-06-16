# Detector de Anomalias em Métricas de Servidores (SMD)

Projeto Integrador da disciplina **Engenharia de Software para IA e Frameworks Profundos** (pós-graduação, CIn) — **Grupo 12**.

> ⚠️ **Nome provisório** (`detector-anomalias-smd`), sujeito a confirmação pela equipe.
> Tema proposto por Leonardo Magalhães e Breno Santos.

## Problema

Detectar **comportamentos anômalos em execuções de sistemas computacionais a partir de
métricas operacionais** (CPU, memória, latência etc.). A base de dados de referência é o
**SMD — Server Machine Dataset** (OmniAnomaly). O objetivo é construir, com boas práticas
de engenharia de software, um sistema de IA que carregue os dados, pré-processe e (nas
próximas entregas) treine e avalie um modelo de detecção de anomalias.

> 📌 **Provisório:** tanto o **nome** do projeto quanto a escolha do **dataset SMD** e o
> recorte do problema são uma proposta do grupo, **a ratificar na reunião de alinhamento**.
> Esta versão entrega a **estrutura organizada e tipada** do projeto; as funções estão com
> assinatura e contrato definidos, e a implementação será preenchida na sequência.

## Estrutura do projeto (Entrega 1)

```
.
├── data/              # base de dados — SMD (a confirmar pela equipe)
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
| 1 | Escolha/contextualização do problema + funções iniciais | 🟡 Em andamento |
| 1 | Modularização e organização do código | ✅ Concluído |
| 1 | Tipagem (type hints) | ✅ Concluído |
| 2 | Implementação em PyTorch (parte 1) + NumPy | ⬜ Pendente |
| 3 | Implementação em PyTorch (parte 2) | ⬜ Pendente |
| 4 | Testes automatizados (unittest) | ⬜ Pendente |
| 5 | Requisitos | ⬜ Pendente |
| 6 | Design/arquitetura + Git e colaboração | ⬜ Pendente |
| Final | Apresentação | ⬜ Pendente |

> Pendências da Entrega 1: **confirmar o nome** (hoje provisório), **ratificar o dataset
> SMD** e implementar as **funções iniciais**. Repositório no GitHub em criação.

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
