import statistics
import pytest
import numpy as np
from scipy import stats
from nucleo import minhastats

# Dados simulando características de pacotes de rede (ex: Duração, Tamanho em Bytes)
DADOS_X = [45.2, 52.1, 48.0, 51.5, 47.9, 49.3, 50.1, 46.8]
DADOS_Y = [12.0, 14.5, 11.2, 15.1, 13.0, 12.8, 14.0, 11.9]
DADOS_MODA = [1, 2, 2, 3, 4]


# Tolerâncias numéricas usadas para considerar equivalentes
# os resultados próprios e os das bibliotecas de referência,
# evitando diferenças irrelevantes de ponto flutuante.
TOLERANCIA_RELATIVA = 1e-9
TOLERANCIA_ABSOLUTA = 1e-12

def test_media():
    resultado_nosso = minhastats.media(DADOS_X)
    resultado_np = np.mean(DADOS_X)
    assert resultado_nosso == pytest.approx(resultado_np, rel=TOLERANCIA_RELATIVA,
        abs=TOLERANCIA_ABSOLUTA)

def test_mediana():
    resultado_nosso = minhastats.mediana(DADOS_X)
    resultado_np = np.median(DADOS_X)
    assert resultado_nosso == pytest.approx(resultado_np,rel=TOLERANCIA_RELATIVA,abs=TOLERANCIA_ABSOLUTA)

def test_moda():
    resultado_nosso = minhastats.moda(DADOS_MODA)
    resultado_stats = statistics.mode(DADOS_MODA)
    assert resultado_nosso == resultado_stats

def test_amplitude():
    resultado_nosso = minhastats.amplitude(DADOS_X)
    resultado_np = np.ptp(DADOS_X) # Peak-to-peak (max - min)
    assert resultado_nosso == pytest.approx(resultado_np, rel=TOLERANCIA_RELATIVA,abs=TOLERANCIA_ABSOLUTA)

def test_variancia_amostral():
    resultado_nosso = minhastats.variancia(DADOS_X, amostral=True)
    resultado_np = np.var(DADOS_X, ddof=1) # ddof=1 indica amostral no numpy
    assert resultado_nosso == pytest.approx(resultado_np,  rel=TOLERANCIA_RELATIVA,abs=TOLERANCIA_ABSOLUTA)

def test_variancia_populacional():
    resultado_nosso = minhastats.variancia(DADOS_X, amostral=False)
    resultado_np = np.var(DADOS_X, ddof=0)

    assert resultado_nosso == pytest.approx(resultado_np,rel=TOLERANCIA_RELATIVA,abs=TOLERANCIA_ABSOLUTA)

def test_desvio_padrao_amostral():
    resultado_nosso = minhastats.desvio_padrao(DADOS_X, amostral=True)
    resultado_np = np.std(DADOS_X, ddof=1)
    assert resultado_nosso == pytest.approx(resultado_np,rel=TOLERANCIA_RELATIVA,abs=TOLERANCIA_ABSOLUTA)
    
def test_desvio_padrao_populacional():
    resultado_nosso = minhastats.desvio_padrao(DADOS_X, amostral=False)
    resultado_np = np.std(DADOS_X, ddof=0)

    assert resultado_nosso == pytest.approx(resultado_np,rel=TOLERANCIA_RELATIVA,abs=TOLERANCIA_ABSOLUTA)

def test_coeficiente_variacao():
    resultado_nosso = minhastats.coeficiente_variacao(
        DADOS_X,
        amostral=True
    )

    resultado_np = (
        np.std(DADOS_X, ddof=1)
        / np.mean(DADOS_X)
    ) * 100

    assert resultado_nosso == pytest.approx(resultado_np,rel=TOLERANCIA_RELATIVA,abs=TOLERANCIA_ABSOLUTA)
    
def test_percentil():
    # Testando os quartis (25, 50, 75)
    for p in [25, 50, 75]:
        resultado_nosso = minhastats.percentil(DADOS_X, p)
        resultado_np = np.percentile(DADOS_X, p, method='linear')
        assert resultado_nosso == pytest.approx(resultado_np,rel=TOLERANCIA_RELATIVA,abs=TOLERANCIA_ABSOLUTA)

def test_covariancia():
    resultado_nosso = minhastats.covariancia(DADOS_X, DADOS_Y, amostral=True)
    resultado_np = np.cov(DADOS_X, DADOS_Y, ddof=1)[0][1]
    assert resultado_nosso == pytest.approx(resultado_np,rel=TOLERANCIA_RELATIVA,abs=TOLERANCIA_ABSOLUTA)

def test_correlacao_pearson():
    resultado_nosso = minhastats.correlacao_pearson(DADOS_X, DADOS_Y)
    resultado_scipy = stats.pearsonr(DADOS_X, DADOS_Y)[0]
    assert resultado_nosso == pytest.approx(resultado_scipy,rel=TOLERANCIA_RELATIVA,abs=TOLERANCIA_ABSOLUTA)

@pytest.mark.parametrize(
    "funcao",
    [
        minhastats.media,
        minhastats.mediana,
        minhastats.moda,
        minhastats.amplitude,
        lambda dados: minhastats.percentil(dados, 50),
    ],
    ids=["media", "mediana", "moda", "amplitude", "percentil"],
)
def test_medidas_univariadas_rejeitam_lista_vazia(funcao):
    with pytest.raises(ValueError, match="vazia"):
        funcao([])


def test_variancia_amostral_exige_dois_valores_e_populacional_aceita_um():
    with pytest.raises(ValueError, match="pelo menos 2"):
        minhastats.variancia([5.0], amostral=True)

    assert minhastats.variancia([5.0], amostral=False) == pytest.approx(0.0)


def test_coeficiente_variacao_rejeita_media_zero():
    with pytest.raises(ValueError, match="Média é zero"):
        minhastats.coeficiente_variacao([-1.0, 1.0])


@pytest.mark.parametrize("percentil", [-1, 101])
def test_percentil_rejeita_posicao_fora_do_intervalo(percentil):
    with pytest.raises(ValueError, match="entre 0 e 100"):
        minhastats.percentil([1.0, 2.0, 3.0], percentil)


@pytest.mark.parametrize(
    ("x", "y", "mensagem"),
    [
        ([1.0, 2.0], [1.0], "mesmo tamanho"),
        ([1.0], [2.0], "pelo menos 2"),
    ],
    ids=["tamanhos_diferentes", "um_par"],
)
def test_covariancia_rejeita_entradas_invalidas(x, y, mensagem):
    with pytest.raises(ValueError, match=mensagem):
        minhastats.covariancia(x, y, amostral=True)

@pytest.mark.parametrize(
    "x, y, mensagem",
    [
        ([1, 2], [3], "mesmo tamanho"),
        ([], [], "pelo menos 2"),
        ([1], [2], "pelo menos 2"),
    ],
    ids=["tamanhos_diferentes", "listas_vazias", "um_par"],
)
def test_correlacao_pearson_entradas_invalidas(x, y, mensagem):
    with pytest.raises(ValueError, match=mensagem):
        minhastats.correlacao_pearson(x, y)


@pytest.mark.parametrize(
    "x, y",
    [
        ([5, 5, 5, 5], [10, 20, 30, 40]),
        ([1, 2, 3, 4], [10, 10, 10, 10]),
    ],
    ids=["x_constante", "y_constante"],
)
def test_correlacao_pearson_variavel_constante(x, y):
    with pytest.raises(ValueError, match="indefinida"):
        minhastats.correlacao_pearson(x, y)


@pytest.mark.parametrize(
    "valor",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "infinito_positivo", "infinito_negativo"],
)
@pytest.mark.parametrize("coluna", ["x", "y"])
def test_correlacao_pearson_valores_nao_finitos(valor, coluna):
    x = [1.0, 2.0, 3.0]
    y = [2.0, 4.0, 6.0]

    if coluna == "x":
        x[1] = valor
    else:
        y[1] = valor

    with pytest.raises(ValueError, match="finitos"):
        minhastats.correlacao_pearson(x, y)


def test_correlacao_pearson_nao_altera_entradas():
    x = [3, 1, 4, 2]
    y = [5, 2, 4, 4]
    x_original = x.copy()
    y_original = y.copy()

    minhastats.correlacao_pearson(x, y)

    assert x == x_original
    assert y == y_original

@pytest.mark.parametrize(
    "x, y",
    [
        (DADOS_X, DADOS_Y),
        ([1, 2, 3, 4], [2, 4, 5, 4]),
        ([1, 2, 3, 4], [8, 6, 4, 2]),
        ([3, 1, 4, 2], [5, 2, 4, 4]),
    ],
    ids=["decimais", "com_dispersao", "decrescente", "pares_fora_de_ordem"],
)
def test_regressao_linear_comparacao_scipy(x, y):
    # SciPy funciona como referência somente no arquivo de testes.
    resultado_nosso = minhastats.regressao_linear(x, y)
    referencia = stats.linregress(x, y)

    assert resultado_nosso == pytest.approx(
        (referencia.slope, referencia.intercept),
        rel=TOLERANCIA_RELATIVA,
        abs=TOLERANCIA_ABSOLUTA,
    )


@pytest.mark.parametrize(
    "x, y, esperado",
    [
        ([1, 2, 3], [3, 5, 7], (2.0, 1.0)),
        ([1, 2, 3, 4], [2, 4, 5, 4], (0.7, 2.0)),
        ([1, 3], [4, 8], (2.0, 2.0)),
        ([1, 2, 3], [5, 5, 5], (0.0, 5.0)),
        ([1, 2, 3], [-2, 1, -2], (0.0, -1.0)),
    ],
    ids=[
        "reta_perfeita",
        "exemplo_calculado_a_mao",
        "dois_pares",
        "y_constante",
        "inclinacao_zero_com_y_variavel",
    ],
)
def test_regressao_linear_resultados_conhecidos(x, y, esperado):
    # Exemplos conhecidos verificam os coeficientes e a ordem do retorno.
    resultado = minhastats.regressao_linear(x, y)

    assert resultado == pytest.approx(
        esperado,
        rel=TOLERANCIA_RELATIVA,
        abs=TOLERANCIA_ABSOLUTA,
    )


@pytest.mark.parametrize(
    "x, y, mensagem",
    [
        ([1, 2], [3], "mesmo tamanho"),
        ([], [], "pelo menos 2"),
        ([1], [2], "pelo menos 2"),
        ([2, 2, 2], [1, 2, 3], "X é constante"),
    ],
    ids=["tamanhos_diferentes", "listas_vazias", "um_par", "x_constante"],
)
def test_regressao_linear_entradas_invalidas(x, y, mensagem):
    with pytest.raises(ValueError, match=mensagem):
        minhastats.regressao_linear(x, y)


@pytest.mark.parametrize(
    "valor",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "infinito_positivo", "infinito_negativo"],
)
@pytest.mark.parametrize("coluna", ["x", "y"])
def test_regressao_linear_valores_nao_finitos(valor, coluna):
    x = [1.0, 2.0, 3.0]
    y = [2.0, 4.0, 6.0]

    # Verifica cada valor inválido nas duas variáveis.
    if coluna == "x":
        x[1] = valor
    else:
        y[1] = valor

    with pytest.raises(ValueError, match="finitos"):
        minhastats.regressao_linear(x, y)


def test_regressao_linear_nao_altera_entradas():
    x = [3, 1, 4, 2]
    y = [5, 2, 4, 4]
    x_original = x.copy()
    y_original = y.copy()

    minhastats.regressao_linear(x, y)

    assert x == x_original
    assert y == y_original

@pytest.mark.parametrize(
    "x, inclinacao, intercepto, esperado",
    [
        (4, 0.7, 2.0, 4.8),
        (0, 2.0, 5.0, 5.0),
        (3, -2.0, 10.0, 4.0),
        (100, 0.0, 5.0, 5.0),
        (1, 2.0, -10.0, -8.0),
        (-2, 3.0, 1.0, -5.0),
    ],
    ids=[
        "exemplo_anterior",
        "x_zero_retorna_intercepto",
        "inclinacao_negativa",
        "reta_horizontal",
        "preserva_resultado_negativo",
        "x_negativo",
    ],
)
def test_predicao_linear_resultados_conhecidos(
    x, inclinacao, intercepto, esperado
):
    resultado = minhastats.predicao_linear(x, inclinacao, intercepto)

    assert resultado == pytest.approx(
        esperado,
        rel=TOLERANCIA_RELATIVA,
        abs=TOLERANCIA_ABSOLUTA,
    )


@pytest.mark.parametrize("x_novo", [45.2, 49.5, 52.1])
def test_predicao_linear_integracao_com_regressao(x_novo):
    # Usa as duas funções próprias em sequência.
    inclinacao, intercepto = minhastats.regressao_linear(DADOS_X, DADOS_Y)
    resultado_nosso = minhastats.predicao_linear(
        x_novo, inclinacao, intercepto
    )

    # Referência independente: ajuste com SciPy e avaliação com NumPy.
    referencia = stats.linregress(DADOS_X, DADOS_Y)

    # NumPy recebe os coeficientes em ordem crescente de potência: a + b*x.
    resultado_referencia = np.polynomial.polynomial.polyval(
        x_novo, [referencia.intercept, referencia.slope]
    )

    assert resultado_nosso == pytest.approx(
        resultado_referencia,
        rel=TOLERANCIA_RELATIVA,
        abs=TOLERANCIA_ABSOLUTA,
    )


@pytest.mark.parametrize(
    "valor",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "infinito_positivo", "infinito_negativo"],
)
@pytest.mark.parametrize("parametro", ["x", "inclinacao", "intercepto"])
def test_predicao_linear_entradas_nao_finitas(valor, parametro):
    argumentos = {
        "x": 2.0,
        "inclinacao": 3.0,
        "intercepto": 1.0,
    }
    argumentos[parametro] = valor

    with pytest.raises(ValueError, match="finitos"):
        minhastats.predicao_linear(**argumentos)


def test_predicao_linear_resultado_nao_finito():
    # Cada entrada é finita, mas o produto ultrapassa a faixa de float.
    with pytest.raises(ValueError, match="resultado|resultou"):
        minhastats.predicao_linear(
            x=1e308, inclinacao=1e308, intercepto=0.0)
        
@pytest.mark.parametrize(
    "x, y",
    [
        (DADOS_X, DADOS_Y),
        ([1, 2, 3, 4], [2, 4, 5, 4]),
        ([1, 2, 3, 4], [8, 6, 4, 2]),
        ([1, 2, 3], [-2, 1, -2]),
    ],
    ids=[
        "decimais",
        "com_dispersao",
        "reta_decrescente_perfeita",
        "reta_horizontal_com_y_variavel",
    ],
)
def test_coeficiente_determinacao_comparacao_scipy(x, y):
    # Integra as três funções próprias: ajustar, prever e avaliar.
    inclinacao, intercepto = minhastats.regressao_linear(x, y)
    y_estimado = [
        minhastats.predicao_linear(valor, inclinacao, intercepto)
        for valor in x
    ]

    resultado_nosso = minhastats.coeficiente_determinacao(y, y_estimado)

    # R² = r² neste caso: regressão linear simples com intercepto,
    # avaliada nos mesmos dados usados no ajuste e com Y não constante.
    referencia = stats.linregress(x, y)
    resultado_referencia = referencia.rvalue ** 2

    assert resultado_nosso == pytest.approx(
        resultado_referencia,
        rel=TOLERANCIA_RELATIVA,
        abs=TOLERANCIA_ABSOLUTA,
    )


@pytest.mark.parametrize(
    "y, y_estimado, esperado",
    [
        ([1, 2, 3], [1, 2, 3], 1.0),
        ([1, 2, 3], [2, 2, 2], 0.0),
        ([1, 2, 3], [3, 2, 1], -3.0),
        ([2, 4, 5, 4], [2.7, 3.4, 4.1, 4.8], 49 / 95),
        ([1, 3], [1, 2], 0.5),
    ],
    ids=[
        "predicoes_perfeitas",
        "predicao_pela_media",
        "pior_que_a_media",
        "exemplo_calculado_a_mao",
        "dois_pares",
    ],
)
def test_coeficiente_determinacao_resultados_conhecidos(
    y, y_estimado, esperado
):
    resultado = minhastats.coeficiente_determinacao(y, y_estimado)

    assert resultado == pytest.approx(
        esperado,
        rel=TOLERANCIA_RELATIVA,
        abs=TOLERANCIA_ABSOLUTA,
    )


@pytest.mark.parametrize(
    "y, y_estimado, mensagem",
    [
        ([1, 2], [1], "mesmo tamanho"),
        ([], [], "pelo menos 2"),
        ([1], [1], "pelo menos 2"),
        ([5, 5, 5], [5, 5, 5], "constantes"),
        ([5, 5, 5], [1, 2, 3], "constantes"),
        ([0.1, 0.1, 0.1], [0.1, 0.1, 0.1], "constantes"),
    ],
    ids=[
        "tamanhos_diferentes",
        "listas_vazias",
        "um_par",
        "y_constante_predicao_perfeita",
        "y_constante_predicao_com_erros",
        "y_constante_decimal",
    ],
)
def test_coeficiente_determinacao_entradas_invalidas(
    y, y_estimado, mensagem
):
    with pytest.raises(ValueError, match=mensagem):
        minhastats.coeficiente_determinacao(y, y_estimado)


@pytest.mark.parametrize(
    "valor",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "infinito_positivo", "infinito_negativo"],
)
@pytest.mark.parametrize("coluna", ["y", "y_estimado"])
def test_coeficiente_determinacao_valores_nao_finitos(valor, coluna):
    y = [1.0, 2.0, 3.0]
    y_estimado = [1.1, 1.9, 3.2]

    if coluna == "y":
        y[1] = valor
    else:
        y_estimado[1] = valor

    with pytest.raises(ValueError, match="finitos"):
        minhastats.coeficiente_determinacao(y, y_estimado)


def test_coeficiente_determinacao_nao_altera_entradas():
    y = [2, 4, 5, 4]
    y_estimado = [2.7, 3.4, 4.1, 4.8]
    y_original = y.copy()
    y_estimado_original = y_estimado.copy()

    minhastats.coeficiente_determinacao(y, y_estimado)

    assert y == y_original
    assert y_estimado == y_estimado_original


def test_coeficiente_determinacao_somas_nao_finitas():
    # Os valores de entrada são finitos, mas seus quadrados excedem a faixa numérica de float. A função deve informar o problema.
    with pytest.raises(ValueError, match="não finitas"):
        minhastats.coeficiente_determinacao(
            [-1e308, 1e308], [0.0, 0.0]
        )
