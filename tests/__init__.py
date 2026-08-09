"""Suíte de testes automatizados do detector de anomalias.

Os testes são escritos com ``unittest`` (biblioteca padrão). Cada módulo do
pacote cobre um módulo correspondente de ``src``:

- ``test_data.py``          -> ``src/data/loader.py``
- ``test_preprocessing.py`` -> ``src/preprocessing/transform.py``
- ``test_model.py``         -> ``src/models/model.py`` e ``persistence.py``
- ``test_training.py``      -> ``src/training/train.py``

Execução a partir da raiz do projeto::

    python -m unittest discover -s tests    # suíte completa
    python -m unittest tests.test_model     # um único módulo
"""
