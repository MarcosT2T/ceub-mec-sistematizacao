from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from nucleo import minhastats
from interface import modulo_2


def test_calcular_resumo_numerico_usa_medidas_populacionais():
    resumo = modulo_2.calcular_resumo_numerico([1, 2, 3, 4])

    assert resumo["n"] == 4
    assert resumo["media"] == pytest.approx(2.5)
    assert resumo["mediana"] == pytest.approx(2.5)
    assert resumo["variancia"] == pytest.approx(1.25)
    assert resumo["desvio_padrao"] == pytest.approx(1.25 ** 0.5)
    assert resumo["q1"] == pytest.approx(1.75)
    assert resumo["q3"] == pytest.approx(3.25)
    assert resumo["iqr"] == pytest.approx(1.5)
    assert resumo["quantidade_outliers"] == 0


def test_tabela_frequencias_discreta_preserva_todos_os_registros():
    tabela = modulo_2.tabela_frequencias_numerica(
        [0, 0, 1, 1, 1, 2],
        numero_classes=5,
    )

    assert list(tabela["Classe ou valor"]) == ["0", "1", "2"]
    assert tabela["Frequência absoluta"].sum() == 6
    assert tabela["Frequência relativa (%)"].sum() == pytest.approx(100.0)
    assert tabela.iloc[-1]["Frequência acumulada (%)"] == pytest.approx(100.0)


def test_tabela_frequencias_numerica_mantem_valor_extremo():
    dados = list(range(1, 31)) + [1_000]

    tabela = modulo_2.tabela_frequencias_numerica(
        dados,
        numero_classes=5,
    )

    linha_extrema = tabela[tabela["Tipo"] == "Além da cerca superior do IQR"]

    assert len(linha_extrema) == 1
    assert linha_extrema.iloc[0]["Frequência absoluta"] == 1
    assert tabela["Frequência absoluta"].sum() == len(dados)
    assert tabela["Frequência relativa (%)"].sum() == pytest.approx(100.0)


def test_tabela_categorica_ordena_e_grafico_agrega_cauda():
    tabela = modulo_2.tabela_frequencias_categorica(
        ["tcp", "udp", "tcp", "icmp", "tcp", "udp", "arp"],
    )
    grafico = modulo_2.dados_grafico_categorias(tabela, limite=2)

    assert list(tabela["Categoria"]) == ["tcp", "udp", "arp", "icmp"]
    assert tabela["Frequência absoluta"].sum() == 7
    assert list(grafico["Categoria"]) == [
        "tcp",
        "udp",
        "Outras categorias",
    ]
    assert grafico.iloc[-1]["Frequência absoluta"] == 2


def test_colunas_ausentes_identifica_dataset_incompleto():
    dados = pd.DataFrame({"dur": [1.0], "proto": ["tcp"]})

    faltantes = modulo_2.colunas_ausentes(dados)

    assert "dur" not in faltantes
    assert "proto" not in faltantes
    assert "attack_cat" in faltantes
    assert "sttl" in faltantes


@pytest.mark.parametrize(
    ("dados", "trecho_esperado"),
    [
        ([1, 2, 20], "assimetria à direita"),
        ([-20, -2, -1], "assimetria à esquerda"),
        ([-1, 0, 1], "Média e mediana estão próximas"),
    ],
)
def test_interpretar_assimetria_nao_atribui_causa_nao_demonstrada(
    dados,
    trecho_esperado,
):
    interpretacao = minhastats.interpretar_assimetria(dados)

    assert trecho_esperado in interpretacao
    assert "Há outliers" not in interpretacao


def test_interface_do_modulo_2_carrega_sem_excecoes(
    dados_interface_reduzidos,
):
    caminho_app = Path(__file__).parents[1] / "src" / "app.py"
    app = AppTest.from_file(caminho_app, default_timeout=30).run()

    assert not app.exception
    app.radio(key="navegacao_modulo").set_value(
        "Módulo 2: Estatística Descritiva"
    ).run(timeout=30)

    assert not app.exception
    assert app.selectbox(key="modulo2_variavel").value == "dur"
    assert len(app.metric) == 12

    app.selectbox(key="modulo2_variavel").select("sttl").run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 12

    app.selectbox(key="modulo2_variavel").select("attack_cat").run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 4
    assert app.dataframe[0].value["Frequência absoluta"].sum() == len(
        dados_interface_reduzidos
    )
