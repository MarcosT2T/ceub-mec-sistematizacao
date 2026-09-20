from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from interface import modulo_5


def test_selecionar_grupo_separa_normal_ataques_e_categoria():
    dados = pd.DataFrame(
        {
            "attack_cat": ["Normal", "Exploits", "Fuzzers", "Normal"],
            "label": [0, 1, 1, 0],
            "valor": [1, 2, 3, 4],
        }
    )

    todos = modulo_5.selecionar_grupo_m5(dados, "Todos os registros")
    ataques = modulo_5.selecionar_grupo_m5(dados, "Todos os ataques")
    normal = modulo_5.selecionar_grupo_m5(dados, "Normal")
    exploits = modulo_5.selecionar_grupo_m5(dados, "Exploits")

    assert todos is dados
    assert ataques["valor"].tolist() == [2, 3]
    assert normal["valor"].tolist() == [1, 4]
    assert exploits["valor"].tolist() == [2]


def test_filtrar_protocolo_preserva_todos_ou_seleciona_um():
    dados = pd.DataFrame(
        {
            "proto": ["tcp", "udp", "tcp"],
            "valor": [1, 2, 3],
        }
    )

    todos = modulo_5.filtrar_protocolo_m5(
        dados,
        "Todos os protocolos",
    )
    tcp = modulo_5.filtrar_protocolo_m5(dados, "tcp")

    assert todos is dados
    assert tcp["valor"].tolist() == [1, 3]


def test_preparar_pares_remove_apenas_ausentes_e_infinitos():
    dados = pd.DataFrame(
        {
            "x": [1.0, 2.0, float("inf"), 4.0, 5.0],
            "y": [10.0, float("nan"), 30.0, float("-inf"), 50.0],
            "outra": ["a", "b", "c", "d", "e"],
        }
    )
    original = dados.copy(deep=True)

    pares = modulo_5.preparar_pares_m5(dados, "x", "y")

    assert pares.to_dict("records") == [
        {"x": 1.0, "y": 10.0},
        {"x": 5.0, "y": 50.0},
    ]
    pd.testing.assert_frame_equal(dados, original)


def test_calcular_resumo_reproduz_reta_perfeita():
    pares = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [3.0, 5.0, 7.0, 9.0],
        }
    )

    resumo = modulo_5.calcular_resumo_m5(pares, "x", "y")

    assert resumo["erro"] is None
    assert resumo["n"] == 4
    assert resumo["inclinacao"] == pytest.approx(2.0)
    assert resumo["intercepto"] == pytest.approx(1.0)
    assert resumo["r"] == pytest.approx(1.0)
    assert resumo["r2"] == pytest.approx(1.0)
    assert resumo["y_estimado"] == pytest.approx([3.0, 5.0, 7.0, 9.0])
    assert resumo["x_reta"] == [1.0, 4.0]
    assert resumo["y_reta"] == pytest.approx([3.0, 9.0])


def test_calcular_resumo_trata_y_constante_sem_inventar_correlacao():
    pares = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [5.0, 5.0, 5.0, 5.0],
        }
    )

    resumo = modulo_5.calcular_resumo_m5(pares, "x", "y")

    assert resumo["erro"] is None
    assert resumo["y_constante"] is True
    assert resumo["inclinacao"] == pytest.approx(0.0)
    assert resumo["intercepto"] == pytest.approx(5.0)
    assert resumo["r"] is None
    assert resumo["r2"] is None


def test_calcular_resumo_informa_x_constante():
    pares = pd.DataFrame(
        {
            "x": [2.0, 2.0, 2.0],
            "y": [1.0, 2.0, 3.0],
        }
    )

    resumo = modulo_5.calcular_resumo_m5(pares, "x", "y")

    assert resumo["x_constante"] is True
    assert resumo["inclinacao"] is None
    assert "X é constante" in resumo["erro"]


def test_calcular_resumo_exige_tres_pares_para_analise():
    pares = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 5.0]})

    resumo = modulo_5.calcular_resumo_m5(pares, "x", "y")

    assert resumo["n"] == 2
    assert resumo["inclinacao"] is None
    assert "pelo menos três" in resumo["erro"]


@pytest.mark.parametrize(
    ("resumo", "situacao"),
    [
        (
            {
                "n": 2,
                "x_constante": False,
                "y_constante": False,
                "erro": "poucos pares",
                "r": None,
                "r2": None,
                "inclinacao": None,
                "intercepto": None,
            },
            "Menos de 3 pares",
        ),
        (
            {
                "n": 3,
                "x_constante": True,
                "y_constante": False,
                "erro": "X constante",
                "r": None,
                "r2": None,
                "inclinacao": None,
                "intercepto": None,
            },
            "X constante",
        ),
        (
            {
                "n": 3,
                "x_constante": False,
                "y_constante": True,
                "erro": None,
                "r": None,
                "r2": None,
                "inclinacao": 0.0,
                "intercepto": 5.0,
            },
            "Y constante",
        ),
        (
            {
                "n": 3,
                "x_constante": False,
                "y_constante": False,
                "erro": None,
                "r": 0.5,
                "r2": 0.25,
                "inclinacao": 1.0,
                "intercepto": 0.0,
            },
            "Calculado",
        ),
    ],
)
def test_linha_comparacao_classifica_situacao(resumo, situacao):
    linha = modulo_5.linha_comparacao_m5("Grupo", resumo)

    assert linha["Grupo"] == "Grupo"
    assert linha["Situação"] == situacao


def test_colunas_ausentes_identifica_dataset_incompleto():
    dados = pd.DataFrame(
        {
            "spkts": [1],
            "dpkts": [2],
            "attack_cat": ["Normal"],
        }
    )

    faltantes = modulo_5.colunas_ausentes_m5(dados)

    assert "spkts" not in faltantes
    assert "dpkts" not in faltantes
    assert "attack_cat" not in faltantes
    assert "dur" in faltantes
    assert "label" in faltantes
    assert "proto" in faltantes


def test_interface_do_modulo_5_executa_ajuste_e_predicao(
    dados_interface_reduzidos,
):
    caminho_app = Path(__file__).parents[1] / "src" / "app.py"
    app = AppTest.from_file(caminho_app, default_timeout=60).run()

    app.radio(key="navegacao_modulo").set_value(
        "Módulo 5: Correlação e Regressão Linear"
    ).run(timeout=60)

    assert not app.exception
    assert app.get_by_key("modulo5_grupo").value == "Todos os registros"
    assert app.get_by_key("modulo5_protocolo").value == (
        "Todos os protocolos"
    )
    assert app.get_by_key("modulo5_x").value == "spkts"
    assert app.get_by_key("modulo5_y").value == "dpkts"
    assert len(app.metric) >= 8
    assert len(app.number_input) == 1

    app.get_by_key("modulo5_mostrar_comparacao").set_value(True).run(
        timeout=60
    )

    assert not app.exception
    assert app.get_by_key("modulo5_mostrar_comparacao").value is True
    assert app.get_by_key("modulo5_ataque_a_comparacao").value == "Exploits"
    assert app.get_by_key("modulo5_ataque_b_comparacao").value == "Fuzzers"
    assert app.get_by_key("modulo5_protocolo_comparacao").value == "tcp"
    assert len(app.dataframe) >= 2
