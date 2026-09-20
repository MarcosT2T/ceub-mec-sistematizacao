import ast
from pathlib import Path

from streamlit.testing.v1 import AppTest

from interface import pagina_validacao


def test_catalogo_contem_todas_as_funcoes_do_nucleo():
    caminho_nucleo = Path(__file__).parents[1] / "src" / "nucleo" / "minhastats.py"
    arvore = ast.parse(caminho_nucleo.read_text(encoding="utf-8"))
    funcoes_definidas = {
        no.name for no in arvore.body if isinstance(no, ast.FunctionDef)
    }

    assert len(funcoes_definidas) == 27
    assert set(pagina_validacao.CATALOGO) == funcoes_definidas


def test_catalogo_completo_aprova_todos_os_exemplos_controlados():
    inventario = pagina_validacao.validar_catalogo_completo()

    assert len(inventario) == 27
    assert set(inventario["Estado do exemplo"]) == {"APROVADO"}


def test_validacoes_cobrem_numero_tupla_sequencia_texto_e_distribuicao():
    nomes = [
        "media",
        "regressao_linear",
        "detectar_outliers",
        "interpretar_assimetria",
        "simular_frequencia_relativa",
        "pdf_normal",
        "distancia_kolmogorov_smirnov",
        "distancia_kolmogorov_discreta",
        "_validar_parametros_distribuicao",
    ]

    resultados = [
        pagina_validacao.executar_validacao_funcao(nome) for nome in nomes
    ]

    assert all(resultado.aprovado for resultado in resultados)


def test_codigo_exibido_e_obtido_da_funcao_real():
    codigo = pagina_validacao._codigo_funcao("media")

    assert codigo.startswith("def media(dados):")
    assert "return sum(dados) / len(dados)" in codigo


def test_tolerancia_aceita_ruido_minimo_e_rejeita_diferenca_material():
    assert pagina_validacao.resultados_equivalentes(1.0, 1.0 + 1e-12)
    assert not pagina_validacao.resultados_equivalentes(1.0, 1.001)


def test_interface_exibe_selecao_codigo_e_comparacao_lado_a_lado(
    dados_interface_reduzidos,
):
    caminho_app = Path(__file__).parents[1] / "src" / "app.py"
    app = AppTest.from_file(caminho_app, default_timeout=30).run()

    app.radio(key="navegacao_modulo").set_value(
        "Validação: Núcleo Estatístico"
    ).run(timeout=30)

    assert not app.exception
    assert app.get_by_key("validacao_categoria").value == "Todas"
    assert app.get_by_key("validacao_funcao").value == "media"
    assert app.get_by_key("validacao_origem").value == (
        "Exemplo didático reproduzível"
    )
    assert len(app.code) >= 5
    assert len(app.dataframe) == 1
    assert len(app.metric) == 4
    assert any("[AGUARDANDO]" in bloco.value for bloco in app.code)

    app.button(key="editor_executar_media").click().run(timeout=30)

    assert not app.exception
    assert any("[RESULTADO] APROVADO" in bloco.value for bloco in app.code)
    assert len(app.metric) == 5

    app.text_area(key="editor_json_media").set_value(
        '{"valores": [true, 2]}'
    ).run(timeout=30)
    app.button(key="editor_executar_media").click().run(timeout=30)

    assert not app.exception
    assert any("[ENTRADAS] REPROVADAS" in bloco.value for bloco in app.code)

    app.selectbox(key="validacao_funcao").set_value(
        "regressao_linear"
    ).run(timeout=30)

    assert not app.exception
    assert app.get_by_key("validacao_funcao").value == "regressao_linear"
    assert any(
        "def regressao_linear" in bloco.value for bloco in app.code
    )

    app.selectbox(key="validacao_funcao").set_value("percentil").run(timeout=30)

    assert not app.exception
    assert app.get_by_key("validacao_percentil").value == 75
