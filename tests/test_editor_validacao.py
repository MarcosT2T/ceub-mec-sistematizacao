import pytest

from interface import editor_validacao
from interface import pagina_validacao


def test_todas_as_funcoes_possuem_entrada_padrao_valida_e_execucao_aprovada():
    x = pagina_validacao.DADOS_X_DIDATICOS.copy()
    y = pagina_validacao.DADOS_Y_DIDATICOS.copy()

    for nome in pagina_validacao.CATALOGO:
        entradas = editor_validacao.entradas_padrao(nome, x, y, 75.0)
        resultado = editor_validacao.executar(nome, entradas)

        assert resultado.aprovado, nome


@pytest.mark.parametrize(
    ("texto", "mensagem"),
    [
        ("{", "JSON inválido"),
        ("[1, 2, 3]", "raiz do JSON"),
    ],
)
def test_carregar_json_rejeita_documento_invalido_ou_excessivo(texto, mensagem):
    with pytest.raises(ValueError, match=mensagem):
        editor_validacao.carregar_json_entradas(texto)


def test_carregar_json_rejeita_documento_excessivo():
    with pytest.raises(ValueError, match="excede o limite"):
        editor_validacao.carregar_json_entradas("x" * 100_001)


@pytest.mark.parametrize(
    ("entradas", "mensagem"),
    [
        ({}, "obrigatórios ausentes"),
        ({"valores": [1, 2], "atalho": "np.mean"}, "não reconhecidos"),
        ({"valores": [True, 2]}, "deve ser um número real"),
        ({"valores": [1, float("nan")]}, "deve ser finito"),
        ({"valores": [0] * 10_001}, "limite seguro"),
    ],
)
def test_medidas_univariadas_rejeitam_campos_e_valores_invalidos(
    entradas,
    mensagem,
):
    with pytest.raises(ValueError, match=mensagem):
        editor_validacao.validar_entradas("media", entradas)


@pytest.mark.parametrize(
    ("nome", "entradas", "mensagem"),
    [
        ("variancia", {"valores": [1], "amostral": True}, "pelo menos 2"),
        ("variancia", {"valores": [1, 2], "amostral": 1}, "true ou false"),
        (
            "coeficiente_variacao",
            {"valores": [-1, 1], "amostral": True},
            "média diferente de zero",
        ),
        ("percentil", {"valores": [1, 2], "p": 101}, "entre 0 e 100"),
    ],
)
def test_parametros_descritivos_respeitam_condicoes_matematicas(
    nome,
    entradas,
    mensagem,
):
    with pytest.raises(ValueError, match=mensagem):
        editor_validacao.validar_entradas(nome, entradas)


@pytest.mark.parametrize(
    ("nome", "entradas", "mensagem"),
    [
        ("covariancia", {"x": [1, 2], "y": [1], "amostral": True}, "mesmo número"),
        ("correlacao_pearson", {"x": [1, 1], "y": [2, 3]}, "dois valores distintos"),
        ("correlacao_pearson", {"x": [1, 2], "y": [3, 3]}, "dois valores distintos"),
        ("regressao_linear", {"x": [1, 1], "y": [2, 3]}, "dois valores distintos"),
        (
            "coeficiente_determinacao",
            {"y": [1, 2], "y_estimado": [1]},
            "mesmo tamanho",
        ),
        (
            "coeficiente_determinacao",
            {"y": [2, 2], "y_estimado": [2, 2]},
            "possuir variação",
        ),
    ],
)
def test_relacoes_rejeitam_pares_incompativeis_ou_indefinidos(
    nome,
    entradas,
    mensagem,
):
    with pytest.raises(ValueError, match=mensagem):
        editor_validacao.validar_entradas(nome, entradas)


@pytest.mark.parametrize(
    ("nome", "entradas", "mensagem"),
    [
        ("pdf_normal", {"x": 0, "mu": 0, "sigma": 0}, "positivo"),
        ("cdf_normal", {"x": 0, "mu": 0, "sigma": -1}, "positivo"),
        ("pdf_exponencial", {"x": 1, "lambd": 0}, "positivo"),
        ("cdf_exponencial", {"x": 1, "lambd": -1}, "positivo"),
        ("pmf_poisson", {"k": 2, "lambd": 0}, "positivo"),
        ("cdf_poisson", {"x": 2, "lambd": -1}, "positivo"),
        (
            "distancia_kolmogorov_smirnov",
            {"dados": [1, 2], "distribuicao": "exponencial", "mu": 0, "sigma": 1},
            "deve ser 'normal'",
        ),
        (
            "distancia_kolmogorov_discreta",
            {"dados": [0, 1.5], "lambd": 1},
            "número inteiro",
        ),
    ],
)
def test_distribuicoes_rejeitam_dominio_ou_modelo_invalido(
    nome,
    entradas,
    mensagem,
):
    with pytest.raises(ValueError, match=mensagem):
        editor_validacao.validar_entradas(nome, entradas)


@pytest.mark.parametrize(
    ("nome", "entradas", "mensagem"),
    [
        (
            "simular_frequencia_relativa",
            {
                "dados_categoricos": ["Ataque"],
                "categoria_alvo": "Ataque",
                "n_sorteios": 0,
                "semente": 1,
            },
            "maior ou igual a 1",
        ),
        (
            "simular_frequencia_relativa",
            {
                "dados_categoricos": ["Ataque"],
                "categoria_alvo": "Ataque",
                "n_sorteios": 10,
                "semente": True,
            },
            "número inteiro",
        ),
        (
            "gerar_medias_amostrais",
            {
                "dados": [1, 2],
                "tamanho_amostra": 10_000,
                "n_repeticoes": 101,
                "semente": 1,
            },
            "excede o limite seguro",
        ),
    ],
)
def test_simulacoes_rejeitam_custo_ou_parametro_invalido(
    nome,
    entradas,
    mensagem,
):
    with pytest.raises(ValueError, match=mensagem):
        editor_validacao.validar_entradas(nome, entradas)


def test_terminal_distingue_espera_erro_e_aprovacao():
    aguardando = editor_validacao.montar_terminal("media")
    erro = editor_validacao.montar_terminal("media", erro="JSON inválido")
    execucao = editor_validacao.executar("media", {"valores": [1, 2, 3]})
    aprovado = editor_validacao.montar_terminal("media", execucao=execucao)

    assert "[AGUARDANDO]" in aguardando
    assert "[ENTRADAS] REPROVADAS" in erro
    assert "Execução cancelada" in erro
    assert "[ENTRADAS] APROVADAS" in aprovado
    assert "[RESULTADO] APROVADO" in aprovado
    assert "nenhum código arbitrário" in aprovado
