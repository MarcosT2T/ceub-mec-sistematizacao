from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from nucleo import minhastats
from interface import modulo_6


def test_calcular_composicao_agrega_nove_tipos_de_ataque():
    dados = pd.DataFrame(
        {
            "attack_cat": [
                "Normal",
                "Normal",
                "Exploits",
                "Fuzzers",
                "Generic",
            ]
        }
    )

    tabela, resumo = modulo_6.calcular_composicao(dados)

    assert resumo["total"] == 5
    assert resumo["normal"] == 2
    assert resumo["ataques"] == 3
    assert resumo["percentual_normal"] == pytest.approx(40.0)
    assert resumo["percentual_ataques"] == pytest.approx(60.0)
    assert resumo["razao_ataque_normal"] == pytest.approx(1.5)
    assert tabela["Registros"].sum() == 5


def test_calcular_comparacao_tcp_filtra_protocolo_e_preserva_grupos():
    dados = pd.DataFrame(
        {
            "attack_cat": [
                "Normal", "Normal", "Normal",
                "Exploits", "Exploits", "Exploits",
                "Fuzzers", "Fuzzers", "Fuzzers",
                "Normal",
            ],
            "proto": ["tcp"] * 9 + ["udp"],
            "spkts": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1000],
            "dpkts": [2, 4, 6, 4, 7, 10, 3, 2, 1, 0],
        }
    )

    tabela, pares = modulo_6.calcular_comparacao_tcp(dados)
    por_grupo = tabela.set_index("Grupo")

    assert por_grupo.loc["Normal", "Pares"] == 3
    assert por_grupo.loc["Normal", "Inclinação"] == pytest.approx(2.0)
    assert por_grupo.loc["Normal", "Intercepto"] == pytest.approx(0.0)
    assert por_grupo.loc["Normal", "r"] == pytest.approx(1.0)
    assert por_grupo.loc["Normal", "R²"] == pytest.approx(1.0)

    assert por_grupo.loc["Exploits", "Inclinação"] == pytest.approx(3.0)
    assert por_grupo.loc["Exploits", "Intercepto"] == pytest.approx(1.0)
    assert por_grupo.loc["Fuzzers", "Inclinação"] == pytest.approx(-1.0)
    assert por_grupo.loc["Fuzzers", "Intercepto"] == pytest.approx(4.0)
    assert len(pares["Normal"]) == 3
    assert 1000 not in pares["Normal"]


def test_calcular_qualidade_quantifica_vetores_repetidos_sem_id():
    dados = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "dur": [1.0, 1.0, 2.0, 2.0],
            "attack_cat": ["Normal", "Normal", "Exploits", "Exploits"],
        }
    )

    resumo = modulo_6.calcular_qualidade_dados(dados)

    assert resumo["registros"] == 4
    assert resumo["vetores_repetidos"] == 2
    assert resumo["percentual_vetores_repetidos"] == pytest.approx(50.0)
    assert resumo["ausentes"] == 0


def test_resumo_duracao_usa_desvio_populacional_e_compara_classes():
    dados = pd.DataFrame(
        {
            "dur": [0.0, 1.0, 2.0, 3.0, 100.0],
            "attack_cat": [
                "Normal",
                "Normal",
                "Exploits",
                "Exploits",
                "Exploits",
            ],
        }
    )

    resumo = modulo_6.calcular_resumo_duracao(dados)

    assert resumo["desvio"] == pytest.approx(
        minhastats.desvio_padrao(dados["dur"].tolist(), amostral=False)
    )
    assert resumo["coeficiente_variacao"] == pytest.approx(
        minhastats.coeficiente_variacao(
            dados["dur"].tolist(),
            amostral=False,
        )
    )
    assert resumo["outliers_por_classe"]["Normal"]["outliers"] == 0
    assert resumo["outliers_por_classe"]["Ataques"]["outliers"] == 1


def test_sensibilidade_p99_revela_influencia_de_valor_extremo():
    linhas = []

    for grupo in modulo_6.GRUPOS_COMPARACAO:
        for x in range(1, 101):
            if grupo == "Normal":
                y = 2 * x
            elif grupo == "Exploits":
                y = 10_000 if x == 100 else x
            else:
                y = 3 * x + 1

            linhas.append(
                {
                    "attack_cat": grupo,
                    "proto": "tcp",
                    "spkts": x,
                    "dpkts": y,
                }
            )

    dados = pd.DataFrame(linhas)
    tabela, pares = modulo_6.calcular_comparacao_tcp(dados)
    sensibilidade = modulo_6.calcular_sensibilidade_p99(tabela, pares)
    por_grupo = sensibilidade.set_index("Grupo")

    assert por_grupo.loc["Normal", "r completo"] == pytest.approx(1.0)
    assert por_grupo.loc["Normal", "r até P99"] == pytest.approx(1.0)
    assert por_grupo.loc["Normal", "Inclinação até P99"] == pytest.approx(2.0)
    assert por_grupo.loc["Fuzzers", "Inclinação até P99"] == pytest.approx(3.0)
    assert por_grupo.loc["Exploits", "Pares até P99"] == 99
    assert por_grupo.loc["Exploits", "r até P99"] == pytest.approx(1.0)
    assert por_grupo.loc[
        "Exploits", "Inclinação completa"
    ] != pytest.approx(
        por_grupo.loc["Exploits", "Inclinação até P99"]
    )


def test_colunas_ausentes_identifica_dataset_incompleto():
    dados = pd.DataFrame(
        {
            "attack_cat": ["Normal"],
            "dur": [1.0],
        }
    )

    faltantes = modulo_6.colunas_ausentes(dados)

    assert "attack_cat" not in faltantes
    assert "dur" not in faltantes
    assert "proto" in faltantes
    assert "spkts" in faltantes
    assert "dpkts" in faltantes


def test_interface_do_modulo_6_renderiza_relatorio_revisado(
    dados_interface_reduzidos,
):
    caminho_app = Path(__file__).parents[1] / "src" / "app.py"
    app = AppTest.from_file(caminho_app, default_timeout=60).run()

    app.radio(key="navegacao_modulo").set_value(
        "Módulo 6: Relatório de Descobertas"
    ).run(timeout=60)

    assert not app.exception
    assert app.get_by_key("m6_escala_dispersao").value == "Faixa completa"
    assert app.get_by_key("m6_mostrar_residuos").value is False
    assert len(app.metric) == 8
    assert len(app.dataframe) >= 4
