import os

import matplotlib
import pandas as pd
import streamlit as st
matplotlib.use("Agg")

from interface.pagina_inicial import renderizar_pagina_inicial
from interface.pagina_validacao import renderizar_pagina_validacao
from interface.modulo_2 import renderizar_modulo_2
from interface.modulo_3 import renderizar_modulo_3
from interface.modulo_4 import renderizar_modulo_4
from interface.modulo_5 import renderizar_modulo_5
from interface.modulo_6 import renderizar_modulo_6

# Configuração da página
st.set_page_config(page_title="Laboratório Estatístico - Redes", layout="wide")

@st.cache_data
def carregar_dados():
    # Descobre o caminho absoluto da pasta onde o script está rodando
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    
    # Constrói o caminho correto voltando uma pasta e entrando em 'data'
    caminho_arquivo = os.path.join(
        diretorio_atual,
        "..",
        "data",
        "UNSW_NB15_training-set.csv",
    )
    
    df = pd.read_csv(caminho_arquivo)
    return df

df = carregar_dados()

st.sidebar.header("Navegação do Laboratório")
modulo = st.sidebar.radio(
    "Selecione a Análise",
    [
        "Início: Apresentação do Projeto",
        "Validação: Núcleo Estatístico",
        "Módulo 2: Estatística Descritiva",
        "Módulo 3: Probabilidade e Simulação",
        "Módulo 4: Distribuições Teóricas",
        "Módulo 5: Correlação e Regressão Linear",
        "Módulo 6: Relatório de Descobertas",
    ],
    key="navegacao_modulo",
)

if modulo == "Início: Apresentação do Projeto":
    renderizar_pagina_inicial(df)

elif modulo == "Validação: Núcleo Estatístico":
    renderizar_pagina_validacao(df)

elif modulo == "Módulo 2: Estatística Descritiva":
    renderizar_modulo_2(df)

elif modulo == "Módulo 3: Probabilidade e Simulação":
    renderizar_modulo_3(df)
elif modulo == "Módulo 4: Distribuições Teóricas":
    renderizar_modulo_4(df)

elif modulo == "Módulo 5: Correlação e Regressão Linear":
    renderizar_modulo_5(df)

elif modulo == "Módulo 6: Relatório de Descobertas":
    renderizar_modulo_6(df)
