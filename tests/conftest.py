"""Fixtures compartilhadas pelos testes de interface do Streamlit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import streamlit as st


CATEGORIAS = (
    "Normal",
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
    "Reconnaissance",
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms",
)


def construir_dataset_interface(quantidade: int = 300) -> pd.DataFrame:
    """Cria uma base pequena, variada e suficiente para todos os módulos."""
    linhas = []

    for indice in range(quantidade):
        categoria = CATEGORIAS[indice % len(CATEGORIAS)]
        spkts = 1 + indice % 40
        deslocamento = CATEGORIAS.index(categoria)
        dpkts = max(0, round(1.4 * spkts + deslocamento + indice % 3))

        linhas.append(
            {
                "id": indice + 1,
                "dur": 0.001 * (1 + indice % 25) + (indice // 100) * 0.2,
                "proto": "udp" if indice % 7 == 0 else "tcp",
                "service": ("http", "dns", "ftp")[indice % 3],
                "state": ("FIN", "CON", "INT")[indice % 3],
                "spkts": spkts,
                "dpkts": dpkts,
                "sbytes": 120 + 17 * spkts + indice % 11,
                "dbytes": 80 + 13 * dpkts + indice % 7,
                "rate": 5.0 + (indice % 60) * 0.75,
                "sttl": (31, 62, 127, 254)[indice % 4],
                "dttl": (29, 60, 125, 252)[indice % 4],
                "attack_cat": categoria,
                "label": 0 if categoria == "Normal" else 1,
            }
        )

    return pd.DataFrame(linhas)


@pytest.fixture
def dados_interface_reduzidos(monkeypatch):
    """Evita renderizar 175 mil linhas em cada teste visual independente."""
    dados = construir_dataset_interface()
    leitura_original = pd.read_csv

    def ler_csv_controlado(caminho, *args, **kwargs):
        if Path(caminho).name == "UNSW_NB15_training-set.csv":
            return dados.copy(deep=True)
        return leitura_original(caminho, *args, **kwargs)

    st.cache_data.clear()
    monkeypatch.setattr(pd, "read_csv", ler_csv_controlado)
    yield dados
    st.cache_data.clear()
