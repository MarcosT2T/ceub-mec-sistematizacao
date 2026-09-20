from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from interface import pagina_inicial


def test_metadados_academicos_estao_documentados():
    assert pagina_inicial.INSTITUICAO == (
        "Centro Universitário de Brasília (CEUB)"
    )
    assert pagina_inicial.CURSO == "Análise e Desenvolvimento de Sistemas"
    assert pagina_inicial.DISCIPLINA == (
        "Matemática e Estatística para Computação"
    )
    assert pagina_inicial.PROFESSOR_ORIENTADOR == "Prof. Romes Heriberto"
    assert len(pagina_inicial.INTEGRANTES) == 3


def test_identificar_particoes_oficiais_por_cardinalidade_e_classes():
    assert pagina_inicial.identificar_particao(
        175_341,
        56_000,
        119_341,
    ) == "Treinamento"
    assert pagina_inicial.identificar_particao(
        82_332,
        37_000,
        45_332,
    ) == "Teste"
    assert pagina_inicial.identificar_particao(100, 50, 50) == (
        "Não identificada"
    )


def test_resumo_inicial_conta_classes_ausentes_e_repeticoes():
    dados = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "attack_cat": ["Normal", "Normal", "Exploits", "Exploits"],
            "dur": [1.0, 1.0, 2.0, None],
        }
    )

    resumo = pagina_inicial.calcular_resumo_inicial(dados)

    assert resumo["registros"] == 4
    assert resumo["normal"] == 2
    assert resumo["ataques"] == 2
    assert resumo["categorias_ataque"] == 1
    assert resumo["ausentes"] == 1
    assert resumo["vetores_repetidos"] == 1
    assert resumo["percentual_repetidos"] == 25.0
    assert resumo["particao"] == "Não identificada"


def test_csv_oficial_tem_cardinalidade_e_classes_esperadas():
    caminho = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "UNSW_NB15_training-set.csv"
    )
    cabecalho = pd.read_csv(caminho, nrows=0)
    classes = pd.read_csv(caminho, usecols=["attack_cat"])
    contagens = classes["attack_cat"].value_counts()

    assert len(cabecalho.columns) == 45
    assert len(classes) == 175_341
    assert contagens["Normal"] == 56_000
    assert contagens.drop("Normal").sum() == 119_341
    assert len(contagens.drop("Normal")) == 9


def test_interface_inicia_na_apresentacao_do_projeto(
    dados_interface_reduzidos,
):
    caminho_app = Path(__file__).parents[1] / "src" / "app.py"
    app = AppTest.from_file(caminho_app, default_timeout=60).run()

    assert not app.exception
    assert app.radio(key="navegacao_modulo").value == (
        "Início: Apresentação do Projeto"
    )
    assert len(app.metric) == 5
    assert len(app.dataframe) == 3
