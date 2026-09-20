import math
import random

def media(dados):
    """
    Calcula a média aritmética simples.
    Fórmula: (Σ x_i) / n
    """
    if not dados:
        raise ValueError("A lista de dados não pode estar vazia.")
    return sum(dados) / len(dados)

def mediana(dados):
    """
    Encontra o valor central dos dados ordenados.
    """
    if not dados:
        raise ValueError("A lista de dados não pode estar vazia.")
    ordenados = sorted(dados)
    n = len(ordenados)
    meio = n // 2
    
    if n % 2 == 0:
        return (ordenados[meio - 1] + ordenados[meio]) / 2.0
    return ordenados[meio]

def moda(dados):
    """
    Retorna o valor mais frequente. Em caso de múltiplas modas, retorna a primeira encontrada.
    """
    if not dados:
        raise ValueError("A lista de dados não pode estar vazia.")
    contagem = {}
    for valor in dados:
        contagem[valor] = contagem.get(valor, 0) + 1
    
    maior_frequencia = max(contagem.values())
    modas = [k for k, v in contagem.items() if v == maior_frequencia]
    return modas[0]

def amplitude(dados):
    """
    Calcula a diferença entre o maior e o menor valor.
    """
    if not dados:
        raise ValueError("A lista de dados não pode estar vazia.")
    return max(dados) - min(dados)

def variancia(dados, amostral=True):
    """
    Calcula a variância. 
    Se amostral=True (padrão), divide por (n-1). Se False, divide por n.
    Fórmula (Amostral): Σ (x_i - x̄)² / (n - 1)
    """
    n = len(dados)
    if n < 2 and amostral:
        raise ValueError("A variância amostral requer pelo menos 2 dados.")
    
    m = media(dados)
    soma_quadrados = sum((x - m) ** 2 for x in dados)
    divisor = n - 1 if amostral else n
    
    return soma_quadrados / divisor

def desvio_padrao(dados, amostral=True):
    """
    Calcula o desvio padrão (raiz quadrada da variância).
    """
    return math.sqrt(variancia(dados, amostral))

def coeficiente_variacao(dados, amostral=True):
    """
    Calcula o coeficiente de variação em porcentagem.
    Fórmula: (Desvio Padrão / Média) * 100
    """
    m = media(dados)
    if m == 0:
        raise ValueError("Média é zero, impossível calcular o CV.")
    dp = desvio_padrao(dados, amostral)
    return (dp / m) * 100

def percentil(dados, p):
    """
    Calcula o percentil 'p' (0 a 100) usando interpolação linear.
    """
    if not dados:
        raise ValueError("A lista de dados não pode estar vazia.")
    if not 0 <= p <= 100:
        raise ValueError("O percentil deve estar entre 0 e 100.")
        
    ordenados = sorted(dados)
    n = len(ordenados)
    
    # Índice real 
    k = (n - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    
    if f == c:
        return ordenados[int(k)]
        
    # Interpolação linear
    d0 = ordenados[f] * (c - k)
    d1 = ordenados[c] * (k - f)
    return d0 + d1

def covariancia(x, y, amostral=True):
    """
    Calcula a covariância entre duas listas de variáveis numéricas.
    Fórmula (Amostral): Σ ((x_i - x̄) * (y_i - ȳ)) / (n - 1)
    """
    if len(x) != len(y):
        raise ValueError("As listas X e Y devem ter o mesmo tamanho.")
    n = len(x)
    if n < 2 and amostral:
        raise ValueError("A covariância amostral requer pelo menos 2 pares de dados.")
        
    media_x = media(x)
    media_y = media(y)
    
    soma_produtos = sum((xi - media_x) * (yi - media_y) for xi, yi in zip(x, y))
    divisor = n - 1 if amostral else n
    
    return soma_produtos / divisor

def correlacao_pearson(x, y):
    """
    Calcula o coeficiente de correlação linear de Pearson (r).

    Fórmula:
        r = covariancia_amostral(X, Y) / (desvio_X * desvio_Y)

    Parâmetros:
        x: sequência de números reais finitos.
        y: sequência de números reais finitos, pareada com x.

    Retorna:
        float: coeficiente de correlação de Pearson, sem arredondamento.

    Levanta:
        ValueError: se as sequências tiverem tamanhos diferentes, menos de
            dois pares, algum valor não finito, alguma variável constante
            ou se o resultado do cálculo não for finito.
        TypeError: se algum elemento não for um número real.

    Observações:
        A função usa covariância e desvios padrões amostrais. Os divisores
        n - 1 se cancelam na razão que define Pearson.
        As sequências recebidas não são alteradas.
    """
    if len(x) != len(y):
        raise ValueError("As listas X e Y devem ter o mesmo tamanho.")

    if len(x) < 2:
        raise ValueError(
            "A correlação de Pearson requer pelo menos 2 pares de dados."
        )

    for xi, yi in zip(x, y):
        if not math.isfinite(xi) or not math.isfinite(yi):
            raise ValueError(
                "As listas X e Y devem conter apenas números finitos."
            )

    cov = covariancia(x, y, amostral=True)
    dp_x = desvio_padrao(x, amostral=True)
    dp_y = desvio_padrao(y, amostral=True)

    if dp_x == 0 or dp_y == 0:
        raise ValueError(
            "A correlação de Pearson é indefinida quando uma das "
            "variáveis é constante."
        )

    r = cov / (dp_x * dp_y)

    if not math.isfinite(r):
        raise ValueError(
            "O cálculo da correlação de Pearson resultou em valor não finito."
        )

    return r

def regressao_linear(x, y):
    """
    Calcula os coeficientes da regressão linear simples por mínimos quadrados.

    Modelo:
        y_estimado = intercepto + inclinacao * x

    Parâmetros:
        x: lista de números reais finitos da variável explicativa.
        y: lista de números reais finitos da variável resposta.
           Os valores x[i] e y[i] devem pertencer à mesma observação.

    Fórmulas:
        inclinacao = Σ((x_i - media_x) * (y_i - media_y))
                     / Σ((x_i - media_x) ** 2)
        intercepto = media_y - inclinacao * media_x

    Retorna:
        tuple: (inclinacao, intercepto), sem arredondamento.

    Levanta:
        ValueError: se as listas tiverem tamanhos diferentes, possuírem menos de dois pares, contiverem NaN ou infinito, ou se a variável X for constante.
        TypeError: se houver elementos que não sejam números reais.
        
    Observações:
        Y constante é permitido e resulta em uma reta horizontal.
        A função não altera as listas recebidas.
    """
    # Evita que zip descarte silenciosamente valores da lista mais longa.
    if len(x) != len(y):
        raise ValueError("As listas X e Y devem ter o mesmo tamanho.")

    if len(x) < 2:
        raise ValueError(
            "A regressão linear requer pelo menos 2 pares de dados."
        )

    # NaN e infinito não podem participar de um ajuste numérico válido.
    for xi, yi in zip(x, y):
        if not math.isfinite(xi) or not math.isfinite(yi):
            raise ValueError(
                "As listas X e Y devem conter apenas números finitos."
            )

    # Reutiliza a média implementada e validada pela equipe.
    media_x = media(x)
    media_y = media(y)

    soma_produtos = 0.0
    soma_quadrados_x = 0.0

    # Acumula o numerador e o denominador usando desvios das médias.
    for xi, yi in zip(x, y):
        desvio_x = xi - media_x
        desvio_y = yi - media_y
        soma_produtos += desvio_x * desvio_y
        soma_quadrados_x += desvio_x ** 2

    # Sem variação em X, não é possível determinar uma inclinação única.
    if soma_quadrados_x == 0:
        raise ValueError(
            "A regressão linear é indefinida quando a variável X é constante."
        )

    inclinacao = soma_produtos / soma_quadrados_x
    intercepto = media_y - inclinacao * media_x

    return inclinacao, intercepto

def predicao_linear(x, inclinacao, intercepto):
    """
    Calcula uma predição usando os coeficientes de uma reta já ajustada.

    Fórmula:
        y_estimado = intercepto + inclinacao * x

    Parâmetros:
        x: número real finito para o qual será calculada a predição.
        inclinacao: coeficiente angular da reta, real e finito.
        intercepto: coeficiente linear da reta, real e finito.

    Retorna:
        int ou float: valor estimado de Y, sem arredondamento.

    Levanta:
        ValueError: se alguma entrada ou o resultado não for finito.
        TypeError: se alguma entrada não for um número real.

    Observações:
        A função utiliza os coeficientes recebidos, sem reajustar a reta.
        Valores negativos são preservados.
        A avaliação de extrapolação depende da faixa dos dados originais
        e deve ser feita pela aplicação.
    """
    for valor in (x, inclinacao, intercepto):
        if not math.isfinite(valor):
            raise ValueError(
                "X, inclinação e intercepto devem ser números finitos."
            )

    y_estimado = intercepto + inclinacao * x

    # Entradas finitas muito grandes podem produzir infinito na operação.
    if not math.isfinite(y_estimado):
        raise ValueError(
            "A predição resultou em um valor não finito."
        )

    # O arredondamento e a interpretação pertencem à apresentação.
    return y_estimado

def coeficiente_determinacao(y, y_estimado):
    """
    Calcula o coeficiente de determinação (R²) a partir das predições.

    Fórmula:
        R² = 1 - soma_quadrados_residuos / soma_quadrados_total

    Parâmetros:
        y: sequência de valores observados, com pelo menos 2 elementos.
        y_estimado: sequência de valores estimados, na mesma ordem de y.
        As duas sequências devem ter o mesmo tamanho e números reais finitos.

    Retorna:
        float: valor de R², sem arredondamento.
        Valores negativos são preservados: indicam erros maiores do que
        os obtidos ao prever a média dos valores observados.

    Levanta:
        ValueError: se os tamanhos forem diferentes, houver menos de
            2 pares, valores não finitos, Y constante ou resultados
            intermediários/final não finitos, ou se a variação de Y
            desaparecer por limitações da precisão numérica.
        TypeError: se algum elemento não for um número real.

    Observações:
        Não ajusta uma regressão nem gera predições.
        Para Y constante, R² é indefinido, mesmo com predições perfeitas,
        pois a soma dos quadrados total é zero.
        As sequências recebidas não são alteradas.
    """
    if len(y) != len(y_estimado):
        raise ValueError("Y observado e Y estimado devem ter o mesmo tamanho.")

    if len(y) < 2:
        raise ValueError("São necessários pelo menos 2 pares de valores.")

    for observado, estimado in zip(y, y_estimado):
        if not math.isfinite(observado) or not math.isfinite(estimado):
            raise ValueError("Os valores observados e estimados devem ser finitos.")

    # Verifica os valores diretamente: a média de decimais repetidos
    # pode apresentar uma pequena diferença de arredondamento.
    if all(observado == y[0] for observado in y):
        raise ValueError("O R² é indefinido quando os valores observados de Y são constantes.")

    # Reutiliza a média implementada anteriormente.
    media_y = media(y)
    soma_quadrados_residuos = 0.0
    soma_quadrados_total = 0.0

    for observado, estimado in zip(y, y_estimado):
        # Resíduo: diferença entre o valor real e a predição.
        residuo = observado - estimado

        # Referência: diferença entre o valor real e a média observada.
        desvio = observado - media_y

        soma_quadrados_residuos += residuo * residuo
        soma_quadrados_total += desvio * desvio

    # Entradas finitas ainda podem gerar somas que excedem a faixa de float.
    if (
        not math.isfinite(soma_quadrados_residuos)
        or not math.isfinite(soma_quadrados_total)
    ):
        raise ValueError("O cálculo do R² produziu somas não finitas.")
    if soma_quadrados_total == 0:
        raise ValueError("A variação de Y ficou numericamente nula no cálculo do R².")     
    r_quadrado = 1.0 - soma_quadrados_residuos / soma_quadrados_total

    if not math.isfinite(r_quadrado):
        raise ValueError("O cálculo do R² resultou em um valor não finito.")

    return r_quadrado

def limites_iqr(dados):
    """
    Calcula os limites inferior e superior para detecção de outliers usando o IQR.
    Fórmula: Q1 - 1.5*IQR e Q3 + 1.5*IQR.
    """
    q1 = percentil(dados, 25)
    q3 = percentil(dados, 75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    return limite_inferior, limite_superior

def detectar_outliers(dados):
    """
    Retorna uma lista contendo apenas os valores que são outliers na distribuição.
    """
    inf, sup = limites_iqr(dados)
    return [x for x in dados if x < inf or x > sup]

def interpretar_assimetria(dados):
    """
    Interpreta a assimetria da distribuição comparando a Média e a Mediana.

    A comparação é um diagnóstico descritivo. Ela não demonstra, sozinha,
    simetria completa nem permite atribuir a diferença a erros ou outliers.
    """
    m = media(dados)
    md = mediana(dados)

    # Escala absoluta evita erros quando a média é negativa ou próxima de zero.
    escala = max(abs(m), abs(md), 1e-12)
    diferenca_relativa = abs(m - md) / escala

    if diferenca_relativa < 0.005:
        return (
            "Média e mediana estão próximas, o que é compatível com maior "
            "equilíbrio em torno do centro. Essa comparação isolada não "
            "demonstra simetria completa."
        )
    elif m > md:
        return (
            "A média superior à mediana é um indício de assimetria à "
            "direita. Isso é compatível com concentração em valores menores "
            "e uma cauda formada por valores elevados; não prova que esses "
            "valores sejam erros, outliers pelo IQR ou ataques."
        )
    else:
        return (
            "A média inferior à mediana é um indício de assimetria à "
            "esquerda. Isso é compatível com concentração em valores maiores "
            "e uma cauda formada por valores baixos; não prova que esses "
            "valores sejam erros, outliers pelo IQR ou ataques."
        )

def simular_frequencia_relativa(
    dados_categoricos,
    categoria_alvo,
    n_sorteios,
    semente=None,
):
    """
    Simula inspeções aleatórias (amostragem com reposição) de uma lista de dados reais.
    Demonstra a Lei dos Grandes Números rastreando a frequência da categoria alvo.
    """
    if not dados_categoricos:
        raise ValueError("A lista de dados não pode estar vazia.")
    if (
        isinstance(n_sorteios, bool)
        or not isinstance(n_sorteios, int)
        or n_sorteios <= 0
    ):
        raise ValueError("O número de sorteios deve ser > 0.")

    proporcoes = []
    sucessos = 0
    n_dados = len(dados_categoricos)
    gerador = random.Random(semente)

    for i in range(1, n_sorteios + 1):
        sorteado = dados_categoricos[gerador.randrange(n_dados)]
        if sorteado == categoria_alvo:
            sucessos += 1
        proporcoes.append(sucessos / i)

    return proporcoes

def gerar_medias_amostrais(
    dados,
    tamanho_amostra,
    n_repeticoes,
    semente=None,
):
    """
    Sorteia amostras repetidas de um dataset e calcula a média de cada uma.
    Demonstra o Teorema Central do Limite.
    """
    if not dados:
        raise ValueError("A lista de dados não pode estar vazia.")
    parametros_invalidos = (
        isinstance(tamanho_amostra, bool)
        or not isinstance(tamanho_amostra, int)
        or tamanho_amostra <= 0
        or isinstance(n_repeticoes, bool)
        or not isinstance(n_repeticoes, int)
        or n_repeticoes <= 0
    )
    if parametros_invalidos:
        raise ValueError("Tamanho da amostra e repetições devem ser > 0.")
    try:
        possui_valor_invalido = any(
            not math.isfinite(float(valor))
            for valor in dados
        )
    except (TypeError, ValueError, OverflowError) as erro:
        raise ValueError(
            "Todos os valores devem ser números finitos."
        ) from erro

    if possui_valor_invalido:
        raise ValueError("Todos os valores devem ser números finitos.")

    medias = []
    n_dados = len(dados)
    gerador = random.Random(semente)

    for _ in range(n_repeticoes):
        amostra = [
            dados[gerador.randrange(n_dados)]
            for _ in range(tamanho_amostra)
        ]
        medias.append(media(amostra))

    return medias

def pdf_normal(x, mu, sigma):
    """
    Calcula a densidade de probabilidade da distribuição Normal.
    Fórmula: (1 / (sigma * sqrt(2*pi))) * e^(-0.5 * ((x - mu)/sigma)^2)
    """
    x, mu, sigma = _validar_parametros_distribuicao(x, mu, sigma)

    if sigma <= 0:
        raise ValueError("O desvio padrão da Normal deve ser positivo.")

    coeficiente = 1.0 / (sigma * math.sqrt(2 * math.pi))
    expoente = -0.5 * ((x - mu) / sigma) ** 2
    return coeficiente * math.exp(expoente)


def cdf_normal(x, mu, sigma):
    """Calcula a função de distribuição acumulada da Normal."""
    x, mu, sigma = _validar_parametros_distribuicao(x, mu, sigma)

    if sigma <= 0:
        raise ValueError("O desvio padrão da Normal deve ser positivo.")

    padronizado = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1.0 + math.erf(padronizado))


def pdf_exponencial(x, lambd):
    """
    Calcula a densidade de probabilidade da distribuição Exponencial.
    Fórmula: lambda * e^(-lambda * x) para x >= 0
    """
    x, lambd = _validar_parametros_distribuicao(x, lambd)

    if lambd <= 0:
        raise ValueError("A taxa da Exponencial deve ser positiva.")
    if x < 0:
        return 0.0

    return lambd * math.exp(-lambd * x)


def cdf_exponencial(x, lambd):
    """Calcula a função de distribuição acumulada da Exponencial."""
    x, lambd = _validar_parametros_distribuicao(x, lambd)

    if lambd <= 0:
        raise ValueError("A taxa da Exponencial deve ser positiva.")
    if x < 0:
        return 0.0

    return 1.0 - math.exp(-lambd * x)


def pmf_poisson(k, lambd):
    """
    Calcula a probabilidade de uma contagem na distribuição de Poisson.

    Fórmula: P(X=k) = e^(-lambda) * lambda^k / k!
    """
    k, lambd = _validar_parametros_distribuicao(k, lambd)

    if lambd <= 0:
        raise ValueError("A taxa da Poisson deve ser positiva.")
    if k < 0 or not k.is_integer():
        return 0.0

    return math.exp(
        -lambd
        + k * math.log(lambd)
        - math.lgamma(k + 1.0)
    )


def cdf_poisson(x, lambd):
    """Calcula a probabilidade acumulada P(X <= x) da Poisson."""
    x, lambd = _validar_parametros_distribuicao(x, lambd)

    if lambd <= 0:
        raise ValueError("A taxa da Poisson deve ser positiva.")
    if x < 0:
        return 0.0

    limite = math.floor(x)
    termo = math.exp(-lambd)
    acumulada = termo

    for k in range(1, limite + 1):
        termo *= lambd / k
        acumulada += termo

        if acumulada >= 1.0:
            return 1.0

    return min(acumulada, 1.0)


def distancia_kolmogorov_smirnov(dados, funcao_cdf):
    """
    Calcula a maior distância entre a CDF empírica e uma CDF teórica.

    A função retorna apenas a distância descritiva D. Ela não calcula p-valor,
    pois a interpretação inferencial depende de hipóteses adicionais e muda
    quando os parâmetros do modelo são estimados nos próprios dados.
    """
    if not dados:
        raise ValueError("A lista de dados não pode estar vazia.")
    if not callable(funcao_cdf):
        raise ValueError("A CDF teórica deve ser uma função válida.")

    try:
        ordenados = sorted(float(valor) for valor in dados)
    except (TypeError, ValueError, OverflowError) as erro:
        raise ValueError(
            "Todos os valores devem ser números finitos."
        ) from erro

    if any(not math.isfinite(valor) for valor in ordenados):
        raise ValueError("Todos os valores devem ser números finitos.")

    n = len(ordenados)
    distancia = 0.0

    for indice, valor in enumerate(ordenados, start=1):
        probabilidade = float(funcao_cdf(valor))

        if (
            not math.isfinite(probabilidade)
            or probabilidade < 0
            or probabilidade > 1
        ):
            raise ValueError(
                "A CDF teórica deve retornar valores entre zero e um."
            )

        distancia_superior = indice / n - probabilidade
        distancia_inferior = probabilidade - (indice - 1) / n
        distancia = max(
            distancia,
            distancia_superior,
            distancia_inferior,
        )

    return distancia


def distancia_kolmogorov_discreta(dados, funcao_cdf):
    """
    Calcula a maior distância entre duas CDFs em dados discretos.

    Diferentemente do caso contínuo, a CDF teórica também possui saltos. Por
    isso, a função compara os limites à esquerda e à direita de cada valor.
    """
    if not dados:
        raise ValueError("A lista de dados não pode estar vazia.")
    if not callable(funcao_cdf):
        raise ValueError("A CDF teórica deve ser uma função válida.")

    try:
        valores = [float(valor) for valor in dados]
    except (TypeError, ValueError, OverflowError) as erro:
        raise ValueError(
            "Todos os valores devem ser números finitos."
        ) from erro

    if any(
        not math.isfinite(valor) or not valor.is_integer()
        for valor in valores
    ):
        raise ValueError(
            "A distância discreta requer valores inteiros finitos."
        )

    frequencias = {}

    for valor in valores:
        inteiro = int(valor)
        frequencias[inteiro] = frequencias.get(inteiro, 0) + 1

    n = len(valores)
    acumulada_empirica = 0
    distancia = 0.0

    for valor in sorted(frequencias):
        empirica_esquerda = acumulada_empirica / n
        teorica_esquerda = float(funcao_cdf(valor - 1))
        acumulada_empirica += frequencias[valor]
        empirica_direita = acumulada_empirica / n
        teorica_direita = float(funcao_cdf(valor))

        for probabilidade in (teorica_esquerda, teorica_direita):
            if (
                not math.isfinite(probabilidade)
                or probabilidade < 0
                or probabilidade > 1
            ):
                raise ValueError(
                    "A CDF teórica deve retornar valores entre zero e um."
                )

        distancia = max(
            distancia,
            abs(empirica_esquerda - teorica_esquerda),
            abs(empirica_direita - teorica_direita),
        )

    return distancia


def _validar_parametros_distribuicao(*valores):
    """Converte e valida parâmetros numéricos usados nas distribuições."""
    try:
        convertidos = tuple(float(valor) for valor in valores)
    except (TypeError, ValueError, OverflowError) as erro:
        raise ValueError(
            "Os parâmetros da distribuição devem ser números finitos."
        ) from erro

    if any(not math.isfinite(valor) for valor in convertidos):
        raise ValueError(
            "Os parâmetros da distribuição devem ser números finitos."
        )

    return convertidos

