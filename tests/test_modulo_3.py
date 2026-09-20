from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from nucleo import minhastats
from interface import modulo_3


def test_simular_frequencia_relativa_e_reprodutivel():
    dados = ["Normal", "Ataque", "Normal", "Normal"]
    original = dados.copy()

    primeira = minhastats.simular_frequencia_relativa(
        dados,
        "Normal",
        100,
        semente=42,
    )
    segunda = minhastats.simular_frequencia_relativa(
        dados,
        "Normal",
        100,
        semente=42,
    )

    assert primeira == segunda
    assert len(primeira) == 100
    assert all(0 <= proporcao <= 1 for proporcao in primeira)
    assert dados == original


@pytest.mark.parametrize(
    "numero_sorteios",
    [0, -1, 1.5, True],
)
def test_simular_frequencia_relativa_rejeita_numero_invalido(
    numero_sorteios,
):
    with pytest.raises(ValueError, match="sorteios"):
        minhastats.simular_frequencia_relativa(
            ["Normal"],
            "Normal",
            numero_sorteios,
            semente=42,
        )


def test_simulacao_categoria_certa_produz_trajetoria_deterministica():
    proporcoes = minhastats.simular_frequencia_relativa(
        ["Normal", "Normal"],
        "Normal",
        10,
        semente=1,
    )

    assert proporcoes == [1.0] * 10


def test_gerar_medias_amostrais_e_reprodutivel_e_preserva_entrada():
    dados = [1.0, 2.0, 3.0, 4.0]
    original = dados.copy()

    primeira = minhastats.gerar_medias_amostrais(
        dados,
        3,
        20,
        semente=123,
    )
    segunda = minhastats.gerar_medias_amostrais(
        dados,
        3,
        20,
        semente=123,
    )

    assert primeira == segunda
    assert len(primeira) == 20
    assert dados == original


@pytest.mark.parametrize(
    ("tamanho_amostra", "numero_repeticoes"),
    [
        (0, 10),
        (-1, 10),
        (1.5, 10),
        (True, 10),
        (10, 0),
        (10, -1),
        (10, 1.5),
        (10, True),
    ],
)
def test_gerar_medias_amostrais_rejeita_parametros_invalidos(
    tamanho_amostra,
    numero_repeticoes,
):
    with pytest.raises(ValueError, match="amostra e repetições"):
        minhastats.gerar_medias_amostrais(
            [1.0, 2.0],
            tamanho_amostra,
            numero_repeticoes,
            semente=42,
        )


@pytest.mark.parametrize(
    "valor",
    [float("nan"), float("inf"), float("-inf"), "não numérico"],
)
def test_gerar_medias_amostrais_rejeita_valores_nao_finitos(valor):
    with pytest.raises(ValueError, match="finitos"):
        minhastats.gerar_medias_amostrais(
            [1.0, valor, 3.0],
            2,
            10,
            semente=42,
        )


def test_calcular_resumo_lgn_preserva_referencia_empirica():
    proporcoes, resumo = modulo_3.calcular_resumo_lgn(
        ["A", "A", "A", "B"],
        "A",
        1_000,
        semente=42,
    )

    assert len(proporcoes) == 1_000
    assert resumo["frequencia_particao"] == 3
    assert resumo["referencia"] == pytest.approx(0.75)
    assert resumo["ocorrencias_esperadas"] == pytest.approx(750.0)
    assert resumo["erro_absoluto"] == pytest.approx(
        abs(resumo["proporcao_final"] - 0.75)
    )
    assert resumo["proporcao_final"] == pytest.approx(0.75, abs=0.02)


def test_limites_verticais_lgn_mantem_evento_raro_visivel():
    inferior, superior = modulo_3.limites_verticais_lgn(
        [0.0] * 99 + [0.01],
        referencia=0.001,
    )

    assert inferior == 0.0
    assert superior < 0.02
    assert superior > 0.01


def test_calcular_resumo_tcl_com_dados_constantes():
    medias, resumo = modulo_3.calcular_resumo_tcl(
        [5.0, 5.0, 5.0],
        tamanho_amostra=10,
        numero_repeticoes=50,
        semente=42,
    )

    assert medias == [5.0] * 50
    assert resumo["media_particao"] == pytest.approx(5.0)
    assert resumo["media_das_medias"] == pytest.approx(5.0)
    assert resumo["desvio_das_medias"] == pytest.approx(0.0)
    assert resumo["erro_padrao_teorico"] == pytest.approx(0.0)
    assert resumo["erro_monte_carlo_centro"] == pytest.approx(0.0)
    assert resumo["diferenca_dispersao_percentual"] is None


def test_tcl_reproduz_centro_e_erro_padrao_em_experimento_controlado():
    _, resumo = modulo_3.calcular_resumo_tcl(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        tamanho_amostra=25,
        numero_repeticoes=10_000,
        semente=42,
    )

    assert resumo["media_das_medias"] == pytest.approx(
        resumo["media_particao"],
        abs=0.03,
    )
    assert resumo["desvio_das_medias"] == pytest.approx(
        resumo["erro_padrao_teorico"],
        rel=0.03,
    )
    assert resumo["erro_monte_carlo_centro"] == pytest.approx(
        resumo["erro_padrao_teorico"] / 100,
    )


def test_colunas_ausentes_identifica_dataset_incompleto():
    dados = pd.DataFrame({"dur": [1.0], "attack_cat": ["Normal"]})

    faltantes = modulo_3.colunas_ausentes(dados)

    assert "dur" not in faltantes
    assert "attack_cat" not in faltantes
    assert "proto" in faltantes
    assert "dttl" in faltantes


def test_interface_do_modulo_3_executa_lgn_e_tcl(
    dados_interface_reduzidos,
):
    caminho_app = Path(__file__).parents[1] / "src" / "app.py"
    app = AppTest.from_file(caminho_app, default_timeout=30).run()

    app.radio(key="navegacao_modulo").set_value(
        "Módulo 3: Probabilidade e Simulação"
    ).run(timeout=30)
    assert not app.exception
    assert app.get_by_key("modulo3_experimento").value == (
        "Lei dos Grandes Números"
    )
    assert len(app.metric) == 4

    app.get_by_key("modulo3_experimento").set_value(
        "Teorema Central do Limite"
    ).run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 6
