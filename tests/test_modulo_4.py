from pathlib import Path

import pandas as pd
import pytest
from scipy import stats
from streamlit.testing.v1 import AppTest

from nucleo import minhastats
from interface import modulo_4


TOLERANCIA_RELATIVA = 1e-12
TOLERANCIA_ABSOLUTA = 1e-12


@pytest.mark.parametrize("x", [-2.0, 0.0, 1.25, 7.0])
def test_pdf_e_cdf_normal_comparam_com_scipy(x):
    media = 1.5
    desvio = 2.3

    assert minhastats.pdf_normal(x, media, desvio) == pytest.approx(
        stats.norm.pdf(x, loc=media, scale=desvio),
        rel=TOLERANCIA_RELATIVA,
        abs=TOLERANCIA_ABSOLUTA,
    )
    assert minhastats.cdf_normal(x, media, desvio) == pytest.approx(
        stats.norm.cdf(x, loc=media, scale=desvio),
        rel=TOLERANCIA_RELATIVA,
        abs=TOLERANCIA_ABSOLUTA,
    )


@pytest.mark.parametrize("x", [-1.0, 0.0, 0.5, 3.0, 10.0])
def test_pdf_e_cdf_exponencial_comparam_com_scipy(x):
    taxa = 0.4
    referencia = stats.expon(scale=1 / taxa)

    assert minhastats.pdf_exponencial(x, taxa) == pytest.approx(
        referencia.pdf(x),
        rel=TOLERANCIA_RELATIVA,
        abs=TOLERANCIA_ABSOLUTA,
    )
    assert minhastats.cdf_exponencial(x, taxa) == pytest.approx(
        referencia.cdf(x),
        rel=TOLERANCIA_RELATIVA,
        abs=TOLERANCIA_ABSOLUTA,
    )


@pytest.mark.parametrize("k", [-1, 0, 1, 4, 12])
def test_pmf_e_cdf_poisson_comparam_com_scipy(k):
    taxa = 3.7

    assert minhastats.pmf_poisson(k, taxa) == pytest.approx(
        stats.poisson.pmf(k, mu=taxa),
        rel=TOLERANCIA_RELATIVA,
        abs=TOLERANCIA_ABSOLUTA,
    )
    assert minhastats.cdf_poisson(k, taxa) == pytest.approx(
        stats.poisson.cdf(k, mu=taxa),
        rel=1e-11,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    ("funcao", "argumentos"),
    [
        (minhastats.pdf_normal, (0.0, 0.0, 0.0)),
        (minhastats.cdf_normal, (0.0, 0.0, -1.0)),
        (minhastats.pdf_exponencial, (0.0, 0.0)),
        (minhastats.cdf_exponencial, (0.0, -1.0)),
        (minhastats.pmf_poisson, (1, 0.0)),
        (minhastats.cdf_poisson, (1, -1.0)),
    ],
)
def test_distribuicoes_rejeitam_parametros_nao_positivos(
    funcao,
    argumentos,
):
    with pytest.raises(ValueError, match="positiv"):
        funcao(*argumentos)


@pytest.mark.parametrize("valor", [float("nan"), float("inf"), "texto"])
def test_distribuicoes_rejeitam_parametros_nao_finitos(valor):
    with pytest.raises(ValueError, match="finitos"):
        minhastats.pdf_normal(valor, 0.0, 1.0)


def test_distancia_ks_normal_compara_com_scipy():
    dados = [-1.0, -0.5, 0.0, 0.2, 0.7, 1.4]
    media = 0.1
    desvio = 0.9

    resultado = minhastats.distancia_kolmogorov_smirnov(
        dados,
        lambda x: minhastats.cdf_normal(x, media, desvio),
    )
    referencia = stats.kstest(
        dados,
        stats.norm(loc=media, scale=desvio).cdf,
    ).statistic

    assert resultado == pytest.approx(
        referencia,
        rel=TOLERANCIA_RELATIVA,
        abs=TOLERANCIA_ABSOLUTA,
    )


def test_distancia_ks_exponencial_compara_com_scipy():
    dados = [0.0, 0.2, 0.7, 1.0, 2.5, 4.0]
    taxa = 0.8

    resultado = minhastats.distancia_kolmogorov_smirnov(
        dados,
        lambda x: minhastats.cdf_exponencial(x, taxa),
    )
    referencia = stats.kstest(
        dados,
        stats.expon(scale=1 / taxa).cdf,
    ).statistic

    assert resultado == pytest.approx(
        referencia,
        rel=TOLERANCIA_RELATIVA,
        abs=TOLERANCIA_ABSOLUTA,
    )


def test_distancia_discreta_considera_lados_dos_saltos():
    dados = [0, 0, 1, 2, 2, 2]
    taxa = 1.4

    resultado = minhastats.distancia_kolmogorov_discreta(
        dados,
        lambda x: minhastats.cdf_poisson(x, taxa),
    )

    frequencias = {0: 2, 1: 1, 2: 3}
    acumulada = 0
    distancias = []

    for valor in sorted(frequencias):
        distancias.append(
            abs(
                acumulada / len(dados)
                - stats.poisson.cdf(valor - 1, mu=taxa)
            )
        )
        acumulada += frequencias[valor]
        distancias.append(
            abs(
                acumulada / len(dados)
                - stats.poisson.cdf(valor, mu=taxa)
            )
        )

    assert resultado == pytest.approx(max(distancias))


@pytest.mark.parametrize(
    ("dados", "funcao", "mensagem"),
    [
        ([], lambda x: 0.0, "vazia"),
        ([0, 1.5], lambda x: 0.0, "inteiros"),
        ([0, float("inf")], lambda x: 0.0, "inteiros"),
        ([0, 1], None, "função"),
        ([0, 1], lambda x: -0.1, "zero e um"),
    ],
)
def test_distancia_discreta_rejeita_entradas_invalidas(
    dados,
    funcao,
    mensagem,
):
    with pytest.raises(ValueError, match=mensagem):
        minhastats.distancia_kolmogorov_discreta(dados, funcao)


@pytest.mark.parametrize(
    ("dados", "funcao", "mensagem"),
    [
        ([], lambda x: x, "vazia"),
        ([1.0, float("nan")], lambda x: x, "finitos"),
        ([1.0], None, "função"),
        ([1.0], lambda x: 2.0, "zero e um"),
    ],
)
def test_distancia_ks_rejeita_entradas_invalidas(
    dados,
    funcao,
    mensagem,
):
    with pytest.raises(ValueError, match=mensagem):
        minhastats.distancia_kolmogorov_smirnov(dados, funcao)


def test_resumo_continuo_estima_parametros_com_nucleo_proprio():
    dados = [0.0, 0.5, 1.0, 2.0, 4.0]

    resumo = modulo_4.calcular_resumo_ajuste(dados, "continua")

    assert resumo["n"] == 5
    assert resumo["media"] == pytest.approx(1.5)
    assert resumo["mediana"] == pytest.approx(1.0)
    assert resumo["desvio_padrao"] == pytest.approx(
        minhastats.desvio_padrao(dados, amostral=False)
    )
    assert resumo["parametro_segunda"] == pytest.approx(2 / 3)
    assert resumo["segunda_distribuicao"] == "Exponencial"
    assert resumo["proporcao_zeros"] == pytest.approx(0.2)
    assert 0 <= resumo["distancia_normal"] <= 1
    assert 0 <= resumo["distancia_segunda"] <= 1


def test_resumo_de_contagem_calcula_dispersao_e_poisson():
    dados = [0, 0, 1, 1, 2, 8]

    resumo = modulo_4.calcular_resumo_ajuste(dados, "contagem")

    assert resumo["segunda_distribuicao"] == "Poisson"
    assert resumo["parametro_segunda"] == pytest.approx(
        minhastats.media(dados)
    )
    assert resumo["indice_dispersao"] == pytest.approx(
        minhastats.variancia(dados, amostral=False)
        / minhastats.media(dados)
    )
    assert resumo["proporcao_zeros"] == pytest.approx(2 / 6)


def test_resumo_de_contagem_rejeita_valores_decimais():
    with pytest.raises(ValueError, match="inteiras"):
        modulo_4.calcular_resumo_ajuste(
            [0.0, 1.5, 2.0],
            "contagem",
        )


def test_resumo_rejeita_variavel_constante():
    with pytest.raises(ValueError, match="constante"):
        modulo_4.calcular_resumo_ajuste(
            [2.0, 2.0, 2.0],
            "continua",
        )


def test_filtro_de_grupo_separa_normal_e_ataques():
    dados = pd.DataFrame(
        {
            "attack_cat": ["Normal", "Exploits", "Normal", "Generic"],
            "dur": [1.0, 2.0, 3.0, 4.0],
        }
    )

    normal = modulo_4.filtrar_grupo(dados, "Tráfego normal")
    ataques = modulo_4.filtrar_grupo(dados, "Todos os ataques")

    assert normal["dur"].tolist() == [1.0, 3.0]
    assert ataques["dur"].tolist() == [2.0, 4.0]


def test_zoom_do_histograma_nao_renormaliza_dados_visiveis():
    dados = [0.0, 1.0, 2.0, 100.0]
    minimo, maximo = modulo_4.calcular_intervalo_visual(
        dados,
        "Região central (IQR)",
    )
    centros, densidades, largura, quantidade_visivel = (
        modulo_4.construir_histograma_densidade(
            dados,
            minimo,
            maximo,
            numero_classes=10,
        )
    )

    assert len(centros) == 10
    assert quantidade_visivel == 3
    assert sum(densidades) * largura == pytest.approx(3 / 4)


def test_amplitude_completa_preserva_todos_os_registros_no_histograma():
    dados = [0.0, 1.0, 2.0, 100.0]
    minimo, maximo = modulo_4.calcular_intervalo_visual(
        dados,
        "Amplitude completa",
    )
    _, densidades, largura, quantidade_visivel = (
        modulo_4.construir_histograma_densidade(
            dados,
            minimo,
            maximo,
            numero_classes=10,
        )
    )

    assert quantidade_visivel == len(dados)
    assert sum(densidades) * largura == pytest.approx(1.0)


def test_colunas_ausentes_identifica_dataset_incompleto():
    dados = pd.DataFrame({"dur": [1.0], "attack_cat": ["Normal"]})

    faltantes = modulo_4.colunas_ausentes(dados)

    assert "dur" not in faltantes
    assert "attack_cat" not in faltantes
    assert "rate" in faltantes
    assert "dpkts" in faltantes


def test_interface_do_modulo_4_executa_modelos_continuo_e_discreto(
    dados_interface_reduzidos,
):
    caminho_app = Path(__file__).parents[1] / "src" / "app.py"
    app = AppTest.from_file(caminho_app, default_timeout=40).run()

    app.radio(key="navegacao_modulo").set_value(
        "Módulo 4: Distribuições Teóricas"
    ).run(timeout=40)

    assert not app.exception
    assert app.get_by_key("modulo4_variavel").value == "dur"
    assert len(app.metric) == 5
    assert len(app.dataframe) == 1

    app.get_by_key("modulo4_variavel").set_value("spkts").run(timeout=40)

    assert not app.exception
    assert app.get_by_key("modulo4_variavel").value == "spkts"
    assert len(app.metric) == 5
    assert len(app.dataframe) == 1
