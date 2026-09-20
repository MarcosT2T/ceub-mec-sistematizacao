"""Página auditável de validação das funções de ``minhastats.py``."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
import statistics
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from interface.editor_validacao import (
    carregar_json_entradas,
    entradas_padrao,
    executar,
    montar_terminal,
    serializar_entradas,
)
from nucleo import minhastats


TOLERANCIA_RELATIVA = 1e-9
TOLERANCIA_ABSOLUTA = 1e-12
DADOS_X_DIDATICOS = [45.2, 52.1, 48.0, 51.5, 47.9, 49.3, 50.1, 49.3]
DADOS_Y_DIDATICOS = [12.0, 14.5, 11.2, 15.1, 13.0, 12.8, 14.0, 11.9]
DADOS_OUTLIERS = [10.0, 11.0, 12.0, 12.0, 13.0, 14.0, 100.0]
DADOS_KS_CONTINUO = [-1.1, -0.4, 0.0, 0.6, 1.2]
DADOS_KS_DISCRETO = [0, 1, 1, 2, 3, 4]


@dataclass(frozen=True)
class DefinicaoFuncao:
    """Explicação e referência de uma função existente no núcleo próprio."""

    nome: str
    titulo: str
    grupo: str
    modulo: str
    finalidade: str
    formula: str
    referencia: str
    metodo: str
    codigo_referencia: str
    testes: str


@dataclass(frozen=True)
class ResultadoValidacao:
    """Evidências produzidas para uma função e um exemplo."""

    proprio: Any
    referencia: Any
    aprovado: bool
    diferenca: float | None
    entradas: str
    criterio: str
    observacao: str = ""


def _d(
    nome: str,
    titulo: str,
    grupo: str,
    modulo: str,
    finalidade: str,
    formula: str,
    referencia: str,
    metodo: str,
    codigo_referencia: str,
    testes: str,
) -> DefinicaoFuncao:
    return DefinicaoFuncao(
        nome,
        titulo,
        grupo,
        modulo,
        finalidade,
        formula,
        referencia,
        metodo,
        codigo_referencia,
        testes,
    )


DEFINICOES = [
    _d(
        "media", "Média aritmética", "Estatística descritiva",
        "Módulos 1, 2, 3, 4 e 6",
        "Resume o centro pela soma dividida pelo número de observações.",
        r"\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i",
        "NumPy — numpy.mean", "Equivalência numérica",
        "resultado_referencia = np.mean(valores)",
        "NumPy, resultado conhecido e lista vazia.",
    ),
    _d(
        "mediana", "Mediana", "Estatística descritiva", "Módulos 1, 2 e 5",
        "Encontra o centro dos dados ordenados e reduz a influência de extremos.",
        r"\tilde{x}=x_{((n+1)/2)}\;\text{ ou }\;\frac{x_{(n/2)}+x_{(n/2+1)}}{2}",
        "NumPy — numpy.median", "Equivalência numérica",
        "resultado_referencia = np.median(valores)",
        "Casos pares, ímpares e lista vazia.",
    ),
    _d(
        "moda", "Moda", "Estatística descritiva", "Módulos 1 e 2",
        "Retorna o primeiro valor entre os que possuem frequência máxima.",
        r"Mo=\operatorname*{primeiro\ arg\,max}_{v}\;f(v)",
        "Python statistics — statistics.mode",
        "Igualdade exata e mesma política de empate",
        "resultado_referencia = statistics.mode(valores)",
        "Repetição real, empate documentado e lista vazia.",
    ),
    _d(
        "amplitude", "Amplitude total", "Estatística descritiva", "Módulos 1 e 2",
        "Mede a distância entre o maior e o menor valor observado.",
        r"A=x_{\max}-x_{\min}", "NumPy — numpy.ptp", "Equivalência numérica",
        "resultado_referencia = np.ptp(valores)",
        "NumPy, resultado conhecido e lista vazia.",
    ),
    _d(
        "variancia", "Variância amostral", "Estatística descritiva",
        "Módulos 1, 2, 4 e 6", "Quantifica a dispersão quadrática em torno da média.",
        r"s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2",
        "NumPy — numpy.var com ddof=1", "Equivalência e mesmo divisor",
        "resultado_referencia = np.var(valores, ddof=1)",
        "Modos amostral/populacional, casos conhecidos e tamanho mínimo.",
    ),
    _d(
        "desvio_padrao", "Desvio-padrão amostral", "Estatística descritiva",
        "Módulos 1, 2, 4 e 6", "Expressa a dispersão na mesma unidade dos dados.",
        r"s=\sqrt{s^2}", "NumPy — numpy.std com ddof=1",
        "Equivalência e mesmo divisor",
        "resultado_referencia = np.std(valores, ddof=1)",
        "Modos amostral/populacional e comparação com NumPy.",
    ),
    _d(
        "coeficiente_variacao", "Coeficiente de variação", "Estatística descritiva",
        "Módulos 1, 2 e 6", "Compara a dispersão relativa em percentual da média.",
        r"CV=\frac{s}{\bar{x}}\times100\%", "SciPy — scipy.stats.variation",
        "Equivalência e mesmo divisor",
        "resultado_referencia = 100 * stats.variation(valores, ddof=1)",
        "Referência externa e rejeição de média zero.",
    ),
    _d(
        "percentil", "Percentil com interpolação linear", "Posição e outliers",
        "Módulos 1 e 2", "Localiza uma posição percentual nos dados ordenados.",
        r"k=(n-1)\frac{p}{100}\quad\text{com interpolação entre }\lfloor k\rfloor\text{ e }\lceil k\rceil",
        "NumPy — numpy.percentile, método linear", "Equivalência e mesmo método",
        "resultado_referencia = np.percentile(valores, p, method='linear')",
        "Percentis de 0 a 100, interpolação e limites inválidos.",
    ),
    _d(
        "covariancia", "Covariância amostral", "Relações e regressão",
        "Módulos 1 e 5", "Mede como duas variáveis variam conjuntamente.",
        r"s_{xy}=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})",
        "NumPy — numpy.cov com ddof=1", "Equivalência e pareamento preservado",
        "resultado_referencia = np.cov(x, y, ddof=1)[0, 1]",
        "Tamanhos, mínimo de pares e referência NumPy.",
    ),
    _d(
        "correlacao_pearson", "Correlação de Pearson", "Relações e regressão",
        "Módulos 1 e 5", "Mede direção e intensidade da associação linear.",
        r"r=\frac{s_{xy}}{s_xs_y}", "SciPy — scipy.stats.pearsonr",
        "Equivalência numérica", "resultado_referencia = stats.pearsonr(x, y).statistic",
        "SciPy, constantes, NaN, infinitos, tamanhos e imutabilidade.",
    ),
    _d(
        "regressao_linear", "Regressão linear simples", "Relações e regressão",
        "Módulos 1 e 5", "Calcula inclinação e intercepto por mínimos quadrados.",
        r"b_1=\frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2},\quad b_0=\bar{y}-b_1\bar{x}",
        "SciPy — scipy.stats.linregress", "Equivalência elemento a elemento",
        "ajuste = stats.linregress(x, y)\nresultado_referencia = (ajuste.slope, ajuste.intercept)",
        "SciPy, cálculos manuais, casos-limite, erros e imutabilidade.",
    ),
    _d(
        "predicao_linear", "Predição linear", "Relações e regressão", "Módulo 5",
        "Aplica a reta ajustada para estimar Y em um valor de X.",
        r"\hat{y}=b_0+b_1x", "NumPy — polynomial.polyval", "Equivalência numérica",
        "resultado_referencia = np.polynomial.polynomial.polyval(x_novo, [intercepto, inclinacao])",
        "Resultados conhecidos, integração, finitude e overflow.",
    ),
    _d(
        "coeficiente_determinacao", "Coeficiente de determinação R²",
        "Relações e regressão", "Módulo 5",
        "Compara o erro da reta com o erro de usar somente a média de Y.",
        r"R^2=1-\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}",
        "NumPy — somas de quadrados", "Equivalência numérica",
        "residuos = np.sum((np.asarray(y) - np.asarray(y_estimado)) ** 2)\ntotal = np.sum((np.asarray(y) - np.mean(y)) ** 2)\nresultado_referencia = 1 - residuos / total",
        "Referência externa, casos conhecidos, Y constante e imutabilidade.",
    ),
    _d(
        "limites_iqr", "Limites de outliers pelo IQR", "Posição e outliers",
        "Módulo 2", "Calcula as cercas inferior e superior da regra de 1,5 IQR.",
        r"IQR=Q_3-Q_1,\quad L_i=Q_1-1{,}5IQR,\quad L_s=Q_3+1{,}5IQR",
        "NumPy — percentile + regra IQR", "Equivalência elemento a elemento",
        "q1, q3 = np.percentile(valores, [25, 75], method='linear')\niqr = q3 - q1\nresultado_referencia = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)",
        "Integração com percentis e valores extremos no Módulo 2.",
    ),
    _d(
        "detectar_outliers", "Detecção de outliers pelo IQR", "Posição e outliers",
        "Módulo 2", "Seleciona valores fora das cercas calculadas pela regra do IQR.",
        r"O=\{x_i\mid x_i<L_i\;\lor\;x_i>L_s\}",
        "NumPy — percentile + máscara", "Igualdade da sequência e da ordem",
        "q1, q3 = np.percentile(valores, [25, 75], method='linear')\nli, ls = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)\nresultado_referencia = [v for v in valores if v < li or v > ls]",
        "Integração com limites IQR e caso com extremo conhecido.",
    ),
    _d(
        "interpretar_assimetria", "Interpretação descritiva da assimetria",
        "Interpretação", "Módulo 2",
        "Produz cautela textual a partir da diferença relativa entre média e mediana.",
        r"d=\frac{|\bar{x}-\tilde{x}|}{\max(|\bar{x}|,|\tilde{x}|,10^{-12})}",
        "Regra especificada, recalculada com NumPy", "Validação por propriedade",
        "media_ref, mediana_ref = np.mean(valores), np.median(valores)\nescala = max(abs(media_ref), abs(mediana_ref), 1e-12)\n# classifica equilíbrio, direita ou esquerda",
        "Três ramos da decisão e cautelas contra conclusões causais.",
    ),
    _d(
        "simular_frequencia_relativa", "Simulação da frequência relativa",
        "Probabilidade e simulação", "Módulo 3 — Lei dos Grandes Números",
        "Amostra com reposição e acompanha a proporção acumulada de um evento.",
        r"\hat{p}_n=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}(X_i=\text{alvo})",
        "Caso controlado com resultado conhecido", "Invariante determinística",
        "dados = ['Ataque'] * 4\nresultado_referencia = [1.0] * 12",
        "Reprodutibilidade por semente, parâmetros e trajetória conhecida.",
    ),
    _d(
        "gerar_medias_amostrais", "Geração de médias amostrais",
        "Probabilidade e simulação", "Módulo 3 — Teorema Central do Limite",
        "Sorteia amostras com reposição e calcula a média de cada repetição.",
        r"\bar{X}_j=\frac{1}{m}\sum_{i=1}^{m}X_{ji}",
        "Caso controlado com população constante", "Invariante determinística",
        "dados = [4.0] * 5\nresultado_referencia = [4.0] * 8",
        "Reprodutibilidade, preservação da entrada, parâmetros e finitude.",
    ),
    _d(
        "pdf_normal", "Densidade da Normal", "Distribuições teóricas", "Módulo 4",
        "Calcula a densidade da distribuição Normal em um ponto.",
        r"f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}",
        "SciPy — scipy.stats.norm.pdf", "Equivalência numérica",
        "resultado_referencia = stats.norm.pdf(x, loc=mu, scale=sigma)",
        "Vários pontos, parâmetros inválidos e valores não finitos.",
    ),
    _d(
        "cdf_normal", "CDF da Normal", "Distribuições teóricas", "Módulo 4",
        "Calcula a probabilidade acumulada da Normal até um ponto.",
        r"F(x)=\frac{1}{2}\left[1+\operatorname{erf}\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)\right]",
        "SciPy — scipy.stats.norm.cdf", "Equivalência numérica",
        "resultado_referencia = stats.norm.cdf(x, loc=mu, scale=sigma)",
        "Vários pontos, parâmetros inválidos e valores não finitos.",
    ),
    _d(
        "pdf_exponencial", "Densidade da Exponencial", "Distribuições teóricas",
        "Módulo 4", "Calcula a densidade exponencial para valores não negativos.",
        r"f(x)=\lambda e^{-\lambda x},\quad x\geq0",
        "SciPy — scipy.stats.expon.pdf", "Equivalência numérica",
        "resultado_referencia = stats.expon.pdf(x, scale=1 / taxa)",
        "Vários pontos, suporte negativo, taxa e finitude.",
    ),
    _d(
        "cdf_exponencial", "CDF da Exponencial", "Distribuições teóricas", "Módulo 4",
        "Calcula a probabilidade acumulada exponencial.",
        r"F(x)=1-e^{-\lambda x},\quad x\geq0",
        "SciPy — scipy.stats.expon.cdf", "Equivalência numérica",
        "resultado_referencia = stats.expon.cdf(x, scale=1 / taxa)",
        "Vários pontos, suporte negativo, taxa e finitude.",
    ),
    _d(
        "pmf_poisson", "Massa de probabilidade de Poisson", "Distribuições teóricas",
        "Módulo 4", "Calcula a probabilidade de uma contagem inteira específica.",
        r"P(X=k)=e^{-\lambda}\frac{\lambda^k}{k!}",
        "SciPy — scipy.stats.poisson.pmf", "Equivalência numérica",
        "resultado_referencia = stats.poisson.pmf(k, mu=taxa)",
        "Contagens, valores não inteiros, taxa e finitude.",
    ),
    _d(
        "cdf_poisson", "CDF de Poisson", "Distribuições teóricas", "Módulo 4",
        "Acumula a probabilidade das contagens até um limite.",
        r"P(X\leq k)=\sum_{j=0}^{\lfloor k\rfloor}e^{-\lambda}\frac{\lambda^j}{j!}",
        "SciPy — scipy.stats.poisson.cdf", "Equivalência numérica",
        "resultado_referencia = stats.poisson.cdf(k, mu=taxa)",
        "Contagens, suporte negativo, taxa e finitude.",
    ),
    _d(
        "distancia_kolmogorov_smirnov", "Distância KS contínua", "Qualidade de ajuste",
        "Módulo 4", "Mede a maior distância entre CDF empírica e CDF contínua.",
        r"D=\sup_x|F_n(x)-F(x)|", "SciPy — scipy.stats.kstest",
        "Equivalência da estatística D",
        "cdf_referencia = lambda v: stats.norm.cdf(v, loc=0.0, scale=1.0)\nresultado_referencia = stats.kstest(valores, cdf_referencia).statistic",
        "SciPy, CDF inválida, vazio e valores não finitos.",
    ),
    _d(
        "distancia_kolmogorov_discreta", "Distância de aderência discreta",
        "Qualidade de ajuste", "Módulo 4",
        "Compara os lados esquerdo e direito dos saltos de duas CDFs discretas.",
        r"D=\max_k\{|F_n(k^-)-F(k^-)|,\;|F_n(k)-F(k)|\}",
        "NumPy + SciPy — CDF empírica e poisson.cdf",
        "Equivalência por cálculo vetorizado",
        "valores, contagens = np.unique(dados, return_counts=True)\nfd = np.cumsum(contagens) / len(dados)\nfe = (np.cumsum(contagens) - contagens) / len(dados)\nresultado_referencia = max(np.max(abs(fe - stats.poisson.cdf(valores-1, taxa))), np.max(abs(fd - stats.poisson.cdf(valores, taxa))))",
        "Saltos à esquerda/direita, inteiros, CDF inválida e vazio.",
    ),
    _d(
        "_validar_parametros_distribuicao", "Validador interno de parâmetros",
        "Infraestrutura interna", "Módulo 4",
        "Converte parâmetros para float e rejeita valores não finitos.",
        r"v_j=\operatorname{float}(a_j)\quad\land\quad v_j\text{ finito}",
        "Python + NumPy — conversão e isfinite", "Equivalência da tupla",
        "convertidos = tuple(float(v) for v in valores)\nassert np.all(np.isfinite(convertidos))\nresultado_referencia = convertidos",
        "Exercitado por todas as distribuições e seus testes de erro.",
    ),
]
CATALOGO = {definicao.nome: definicao for definicao in DEFINICOES}
FUNCOES_COM_DADOS_SELECIONAVEIS = {
    "media",
    "mediana",
    "moda",
    "amplitude",
    "variancia",
    "desvio_padrao",
    "coeficiente_variacao",
    "percentil",
    "covariancia",
    "correlacao_pearson",
    "regressao_linear",
    "predicao_linear",
    "coeficiente_determinacao",
}


def resultados_equivalentes(resultado: float, referencia: float) -> bool:
    """Aplica a tolerância adotada nas comparações de ponto flutuante."""
    return math.isclose(
        float(resultado),
        float(referencia),
        rel_tol=TOLERANCIA_RELATIVA,
        abs_tol=TOLERANCIA_ABSOLUTA,
    )


def _equivalentes(proprio: Any, referencia: Any) -> bool:
    if isinstance(proprio, (list, tuple, np.ndarray)) and isinstance(
        referencia, (list, tuple, np.ndarray)
    ):
        return len(proprio) == len(referencia) and all(
            _equivalentes(a, b) for a, b in zip(proprio, referencia)
        )
    if isinstance(proprio, (int, float, np.number)) and isinstance(
        referencia, (int, float, np.number)
    ):
        return resultados_equivalentes(proprio, referencia)
    return proprio == referencia


def _maior_diferenca(proprio: Any, referencia: Any) -> float | None:
    if isinstance(proprio, (list, tuple, np.ndarray)) and isinstance(
        referencia, (list, tuple, np.ndarray)
    ):
        if len(proprio) != len(referencia):
            return None
        valores = [
            diferenca for a, b in zip(proprio, referencia)
            if (diferenca := _maior_diferenca(a, b)) is not None
        ]
        return max(valores, default=0.0)
    if isinstance(proprio, (int, float, np.number)) and isinstance(
        referencia, (int, float, np.number)
    ):
        return abs(float(proprio) - float(referencia))
    return None


def _resultado(
    proprio: Any,
    referencia: Any,
    entradas: str,
    criterio: str,
    observacao: str = "",
    aprovado: bool | None = None,
) -> ResultadoValidacao:
    return ResultadoValidacao(
        proprio,
        referencia,
        _equivalentes(proprio, referencia) if aprovado is None else aprovado,
        _maior_diferenca(proprio, referencia),
        entradas,
        criterio,
        observacao,
    )


def _classificacao_assimetria(valores: list[float]) -> str:
    media_ref, mediana_ref = float(np.mean(valores)), float(np.median(valores))
    escala = max(abs(media_ref), abs(mediana_ref), 1e-12)
    if abs(media_ref - mediana_ref) / escala < 0.005:
        return "Equilíbrio compatível"
    return "Assimetria à direita" if media_ref > mediana_ref else "Assimetria à esquerda"


def _classificacao_texto(texto: str) -> str:
    if "à direita" in texto:
        return "Assimetria à direita"
    if "à esquerda" in texto:
        return "Assimetria à esquerda"
    return "Equilíbrio compatível"


def _distancia_discreta_referencia(dados: list[int], taxa: float) -> float:
    valores, contagens = np.unique(dados, return_counts=True)
    acumuladas = np.cumsum(contagens)
    esquerda = (acumuladas - contagens) / len(dados)
    direita = acumuladas / len(dados)
    return float(max(
        np.max(np.abs(esquerda - stats.poisson.cdf(valores - 1, mu=taxa))),
        np.max(np.abs(direita - stats.poisson.cdf(valores, mu=taxa))),
    ))


def executar_validacao_funcao(
    nome: str,
    x: list[float] | None = None,
    y: list[float] | None = None,
    percentil_selecionado: float = 75.0,
) -> ResultadoValidacao:
    """Executa a referência correspondente à função selecionada."""
    x = DADOS_X_DIDATICOS.copy() if x is None else [float(v) for v in x]
    y = DADOS_Y_DIDATICOS.copy() if y is None else [float(v) for v in y]
    tol = "math.isclose(rel_tol=1e-9, abs_tol=1e-12)"
    simples = {
        "media": lambda: _resultado(minhastats.media(x), np.mean(x), f"valores = {x}", tol),
        "mediana": lambda: _resultado(minhastats.mediana(x), np.median(x), f"valores = {x}", tol),
        "moda": lambda: _resultado(minhastats.moda(x), statistics.mode(x), f"valores = {x}", "mesmo valor e política de empate"),
        "amplitude": lambda: _resultado(minhastats.amplitude(x), np.ptp(x), f"valores = {x}", tol),
        "variancia": lambda: _resultado(minhastats.variancia(x, True), np.var(x, ddof=1), f"valores = {x}; amostral = True", tol + "; ddof=1"),
        "desvio_padrao": lambda: _resultado(minhastats.desvio_padrao(x, True), np.std(x, ddof=1), f"valores = {x}; amostral = True", tol + "; ddof=1"),
        "coeficiente_variacao": lambda: _resultado(minhastats.coeficiente_variacao(x, True), 100 * stats.variation(x, ddof=1), f"valores = {x}; amostral = True", tol + "; percentual"),
        "percentil": lambda: _resultado(minhastats.percentil(x, percentil_selecionado), np.percentile(x, percentil_selecionado, method="linear"), f"valores = {x}; p = {percentil_selecionado:g}", tol + "; interpolação linear"),
        "covariancia": lambda: _resultado(minhastats.covariancia(x, y, True), np.cov(x, y, ddof=1)[0, 1], f"x = {x}; y = {y}; amostral = True", tol + "; pares e ddof=1"),
        "correlacao_pearson": lambda: _resultado(minhastats.correlacao_pearson(x, y), stats.pearsonr(x, y).statistic, f"x = {x}; y = {y}", tol),
    }
    if nome in simples:
        return simples[nome]()

    if nome in {
        "regressao_linear",
        "predicao_linear",
        "coeficiente_determinacao",
    }:
        inclinacao, intercepto = minhastats.regressao_linear(x, y)
        ajuste = stats.linregress(x, y)
        if nome == "regressao_linear":
            return _resultado((inclinacao, intercepto), (ajuste.slope, ajuste.intercept), f"x = {x}; y = {y}", tol + "; dois coeficientes")
        if nome == "predicao_linear":
            x_novo = minhastats.mediana(x)
            return _resultado(minhastats.predicao_linear(x_novo, inclinacao, intercepto), np.polynomial.polynomial.polyval(x_novo, [ajuste.intercept, ajuste.slope]), f"x_novo = {x_novo}; inclinacao = {inclinacao}; intercepto = {intercepto}", tol)
        estimados = [minhastats.predicao_linear(v, inclinacao, intercepto) for v in x]
        residuos = np.sum((np.asarray(y) - np.asarray(estimados)) ** 2)
        total = np.sum((np.asarray(y) - np.mean(y)) ** 2)
        return _resultado(minhastats.coeficiente_determinacao(y, estimados), 1 - residuos / total, f"y = {y}; y_estimado = {[round(v, 6) for v in estimados]}", tol)
    if nome == "limites_iqr":
        q1, q3 = np.percentile(DADOS_OUTLIERS, [25, 75], method="linear")
        iqr = q3 - q1
        return _resultado(minhastats.limites_iqr(DADOS_OUTLIERS), (q1 - 1.5 * iqr, q3 + 1.5 * iqr), f"valores = {DADOS_OUTLIERS}", tol + "; percentis lineares e fator 1,5")
    if nome == "detectar_outliers":
        q1, q3 = np.percentile(DADOS_OUTLIERS, [25, 75], method="linear")
        li, ls = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        referencia = [v for v in DADOS_OUTLIERS if v < li or v > ls]
        return _resultado(minhastats.detectar_outliers(DADOS_OUTLIERS), referencia, f"valores = {DADOS_OUTLIERS}", "mesmos limites, valores e ordem")
    if nome == "interpretar_assimetria":
        texto = minhastats.interpretar_assimetria(DADOS_OUTLIERS)
        propria, referencia = _classificacao_texto(texto), _classificacao_assimetria(DADOS_OUTLIERS)
        return _resultado(texto, referencia, f"valores = {DADOS_OUTLIERS}", "decisão textual igual à regra recalculada com NumPy", "Valida-se a decisão da heurística; não existe frase canônica em biblioteca.", propria == referencia)
    if nome == "simular_frequencia_relativa":
        proprio = minhastats.simular_frequencia_relativa(["Ataque"] * 4, "Ataque", 12, 42)
        return _resultado(proprio, [1.0] * 12, "dados = ['Ataque'] * 4; alvo = 'Ataque'; sorteios = 12; semente = 42", "se toda observação é alvo, toda proporção vale 1", "O teste automatizado separado verifica reprodutibilidade por semente.")
    if nome == "gerar_medias_amostrais":
        proprio = minhastats.gerar_medias_amostrais([4.0] * 5, 3, 8, 42)
        return _resultado(proprio, [4.0] * 8, "dados = [4.0] * 5; tamanho = 3; repetições = 8; semente = 42", "toda amostra de população constante tem média 4", "O teste automatizado separado verifica semente e preservação da entrada.")

    ponto, mu, sigma, taxa, k = 1.25, 0.5, 2.0, 0.8, 4
    distribuicoes = {
        "pdf_normal": lambda: _resultado(minhastats.pdf_normal(ponto, mu, sigma), stats.norm.pdf(ponto, loc=mu, scale=sigma), "x = 1.25; mu = 0.5; sigma = 2.0", tol),
        "cdf_normal": lambda: _resultado(minhastats.cdf_normal(ponto, mu, sigma), stats.norm.cdf(ponto, loc=mu, scale=sigma), "x = 1.25; mu = 0.5; sigma = 2.0", tol),
        "pdf_exponencial": lambda: _resultado(minhastats.pdf_exponencial(ponto, taxa), stats.expon.pdf(ponto, scale=1 / taxa), "x = 1.25; lambda = 0.8", tol),
        "cdf_exponencial": lambda: _resultado(minhastats.cdf_exponencial(ponto, taxa), stats.expon.cdf(ponto, scale=1 / taxa), "x = 1.25; lambda = 0.8", tol),
        "pmf_poisson": lambda: _resultado(minhastats.pmf_poisson(k, taxa), stats.poisson.pmf(k, mu=taxa), "k = 4; lambda = 0.8", tol),
        "cdf_poisson": lambda: _resultado(minhastats.cdf_poisson(k, taxa), stats.poisson.cdf(k, mu=taxa), "k = 4; lambda = 0.8", tol),
    }
    if nome in distribuicoes:
        return distribuicoes[nome]()
    if nome == "distancia_kolmogorov_smirnov":
        proprio = minhastats.distancia_kolmogorov_smirnov(DADOS_KS_CONTINUO, lambda v: minhastats.cdf_normal(v, 0.0, 1.0))
        referencia = stats.kstest(
            DADOS_KS_CONTINUO,
            lambda valor: stats.norm.cdf(valor, loc=0.0, scale=1.0),
        ).statistic
        return _resultado(proprio, referencia, f"valores = {DADOS_KS_CONTINUO}; CDF = Normal(0, 1)", tol + "; somente a estatística D", "O núcleo não calcula p-valor, pois parâmetros estimados alteram sua interpretação.")
    if nome == "distancia_kolmogorov_discreta":
        proprio = minhastats.distancia_kolmogorov_discreta(DADOS_KS_DISCRETO, lambda v: minhastats.cdf_poisson(v, 1.5))
        referencia = _distancia_discreta_referencia(DADOS_KS_DISCRETO, 1.5)
        return _resultado(proprio, referencia, f"valores = {DADOS_KS_DISCRETO}; CDF = Poisson(lambda=1.5)", tol + "; lados esquerdo e direito", "SciPy não oferece um KS inferencial canônico para distribuição discreta; compara-se a distância descritiva.")
    if nome == "_validar_parametros_distribuicao":
        valores = (1, 2.5, "3.75")
        referencia = tuple(float(v) for v in valores)
        assert np.all(np.isfinite(referencia))
        return _resultado(minhastats._validar_parametros_distribuicao(*valores), referencia, "valores = (1, 2.5, '3.75')", "tupla convertida igual e finita")
    raise KeyError(f"Função não catalogada: {nome}")


def preparar_exemplo_dataset(
    dados: pd.DataFrame,
    variavel_x: str,
    variavel_y: str,
    quantidade: int,
) -> tuple[list[float], list[float]]:
    """Seleciona pares finitos sem quebrar o pareamento X–Y."""
    pares = (
        dados[[variavel_x, variavel_y]]
        .replace([float("inf"), float("-inf")], float("nan"))
        .dropna()
        .head(quantidade)
    )
    if len(pares) < 2:
        raise ValueError("O recorte não contém pares suficientes.")
    return (
        [float(v) for v in pares[variavel_x].tolist()],
        [float(v) for v in pares[variavel_y].tolist()],
    )


def formatar_resultado(valor: Any) -> str:
    """Formata números, textos e sequências para inspeção humana."""
    if isinstance(valor, str):
        return valor
    if isinstance(valor, (list, tuple, np.ndarray)):
        itens = list(valor)
        corpo = ", ".join(formatar_resultado(item) for item in itens[:8])
        if len(itens) > 8:
            corpo += ", …"
        abertura, fechamento = (("(", ")") if isinstance(valor, tuple) else ("[", "]"))
        return f"{abertura}{corpo}{fechamento}"
    numero = float(valor)
    if numero == 0:
        return "0"
    if abs(numero) < 1e-5 or abs(numero) >= 1e6:
        return f"{numero:.12e}"
    return f"{numero:.12f}".rstrip("0").rstrip(".")


def _codigo_funcao(nome: str) -> str:
    return inspect.getsource(getattr(minhastats, nome)).strip()


def _assinatura(nome: str) -> str:
    return f"{nome}{inspect.signature(getattr(minhastats, nome))}"


def validar_catalogo_completo() -> pd.DataFrame:
    """Executa todas as funções catalogadas com exemplos controlados."""
    linhas = []
    for definicao in DEFINICOES:
        try:
            resultado = executar_validacao_funcao(definicao.nome)
            estado = "APROVADO" if resultado.aprovado else "REPROVADO"
        except (TypeError, ValueError, OverflowError) as erro:
            estado = f"ERRO: {erro}"
        linhas.append({
            "Função": definicao.nome,
            "Categoria": definicao.grupo,
            "Uso no projeto": definicao.modulo,
            "Referência ou propriedade": definicao.referencia,
            "Validação": definicao.metodo,
            "Estado do exemplo": estado,
        })
    return pd.DataFrame(linhas)


def _selecionar_dados(
    dados: pd.DataFrame,
    nome_funcao: str,
) -> tuple[list[float], list[float], float]:
    if nome_funcao not in FUNCOES_COM_DADOS_SELECIONAVEIS:
        st.caption(
            "Esta função usa um caso controlado próprio. As entradas exatas serão "
            "mostradas logo abaixo, junto da fórmula."
        )
        return DADOS_X_DIDATICOS.copy(), DADOS_Y_DIDATICOS.copy(), 75.0

    origem = st.radio(
        "Dados usados nas funções descritivas e relacionais",
        ["Exemplo didático reproduzível", "Recorte do dataset carregado"],
        horizontal=True,
        key="validacao_origem",
    )
    p = 75
    if nome_funcao == "percentil":
        p = st.slider(
            "Percentil avaliado",
            0, 100, 75, 5,
            key="validacao_percentil",
        )
    if origem == "Exemplo didático reproduzível":
        st.caption(
            "Funções de simulação, outliers e distribuições usam seus próprios "
            "casos controlados, sempre informados na seção de entradas."
        )
        return DADOS_X_DIDATICOS.copy(), DADOS_Y_DIDATICOS.copy(), float(p)

    numericas = [
        coluna for coluna in dados.select_dtypes(include="number").columns
        if coluna not in {"id", "label"} and dados[coluna].nunique(dropna=True) > 1
    ]
    if len(numericas) < 2:
        raise ValueError("A base precisa conter duas variáveis numéricas não constantes.")
    coluna_x, coluna_y, coluna_n = st.columns([2, 2, 1])
    with coluna_x:
        nome_x = st.selectbox(
            "Variável X", numericas,
            index=numericas.index("spkts") if "spkts" in numericas else 0,
            key="validacao_variavel_x",
        )
    opcoes_y = [coluna for coluna in numericas if coluna != nome_x]
    with coluna_y:
        nome_y = st.selectbox(
            "Variável Y", opcoes_y,
            index=opcoes_y.index("dpkts") if "dpkts" in opcoes_y else 0,
            key="validacao_variavel_y",
        )
    with coluna_n:
        quantidade = st.number_input(
            "Pares", 10, 2_000, 200, 10, key="validacao_quantidade"
        )
    x, y = preparar_exemplo_dataset(dados, str(nome_x), str(nome_y), int(quantidade))
    st.caption(
        "O recorte usa os primeiros pares finitos na ordem da base. Ele confere "
        "implementações e não constitui amostragem inferencial."
    )
    return x, y, float(p)


def _renderizar_resultados(
    definicao: DefinicaoFuncao,
    resultado: ResultadoValidacao,
) -> None:
    propria, referencia = st.columns(2, gap="large")
    with propria:
        with st.container(border=True):
            st.markdown("#### Resultado do núcleo próprio")
            st.caption(f"`minhastats.{definicao.nome}`")
            st.code(formatar_resultado(resultado.proprio), language=None, wrap_lines=True)
    with referencia:
        with st.container(border=True):
            st.markdown("#### Resultado da referência")
            st.caption(definicao.referencia)
            st.code(formatar_resultado(resultado.referencia), language=None, wrap_lines=True)


def renderizar_pagina_validacao(dados: pd.DataFrame) -> None:
    """Apresenta uma função por vez com código, referência e evidências."""
    st.title("Validação do núcleo estatístico", anchor=False)
    st.markdown(
        "Escolha uma função para ver **o código implementado pela equipe**, "
        "**a referência executada com as mesmas entradas** e os resultados lado "
        "a lado. O catálogo cobre todas as funções de `minhastats.py`."
    )
    st.info(
        "NumPy, SciPy e `statistics` são referências somente nesta página. Os "
        "resultados apresentados pelos módulos vêm do núcleo próprio.",
        icon=":material/balance:",
    )

    inventario = validar_catalogo_completo()
    aprovadas = int((inventario["Estado do exemplo"] == "APROVADO").sum())
    metricas = st.columns(4)
    metricas[0].metric("Funções catalogadas", len(DEFINICOES), border=True)
    publicas = sum(not item.nome.startswith("_") for item in DEFINICOES)
    internas = len(DEFINICOES) - publicas
    metricas[1].metric("Funções públicas", publicas, border=True)
    metricas[2].metric("Auxiliar interno", internas, border=True)
    metricas[3].metric("Checks aprovados", f"{aprovadas}/{len(DEFINICOES)}", border=True)

    st.subheader("1. Escolha a função", anchor=False)
    grupo = st.selectbox(
        "Filtrar por grupo",
        ["Todas"] + sorted({item.grupo for item in DEFINICOES}),
        key="validacao_categoria",
    )
    opcoes = [
        item.nome for item in DEFINICOES
        if grupo == "Todas" or item.grupo == grupo
    ]
    nome = st.selectbox(
        "Função do núcleo",
        opcoes,
        format_func=lambda chave: f"{CATALOGO[chave].titulo} — {chave}()",
        key="validacao_funcao",
    )
    definicao = CATALOGO[nome]
    try:
        x, y, p = _selecionar_dados(dados, nome)
        resultado = executar_validacao_funcao(nome, x, y, p)
    except (TypeError, ValueError, OverflowError) as erro:
        st.error(f"O exemplo selecionado não permite a validação: {erro}")
        return

    st.subheader("2. Entenda o cálculo", anchor=False)
    resumo = st.columns([1.1, 1, 1.7])
    with resumo[0]:
        with st.container(border=True):
            st.caption("FUNÇÃO")
            st.markdown(f"**`{nome}()`**")
    with resumo[1]:
        with st.container(border=True):
            st.caption("CHECK CONTROLADO")
            st.markdown(
                "**APROVADO**" if resultado.aprovado else "**REPROVADO**"
            )
    with resumo[2]:
        with st.container(border=True):
            st.caption("USO NO PROJETO")
            st.markdown(f"**{definicao.modulo}**")
    st.markdown(f"**Assinatura real:** `{_assinatura(nome)}`")
    st.markdown(definicao.finalidade)
    st.latex(definicao.formula)
    with st.expander("Entradas exatas do exemplo"):
        st.code(resultado.entradas, language=None, wrap_lines=True)

    st.subheader("3. Compare os códigos lado a lado", anchor=False)
    st.caption(
        "À esquerda está o código real carregado de `minhastats.py`. À direita "
        "está o cálculo independente usado para conferir este exemplo."
    )
    codigo_proprio, codigo_referencia = st.columns(2, gap="large")
    with codigo_proprio:
        st.markdown("#### Implementação da equipe")
        st.caption(f"`minhastats.{nome}`")
        st.code(
            _codigo_funcao(nome), language="python", line_numbers=True,
            wrap_lines=False, height=380,
        )
    with codigo_referencia:
        st.markdown("#### Referência de validação")
        st.caption(definicao.referencia)
        st.code(
            definicao.codigo_referencia, language="python", line_numbers=True,
            wrap_lines=False, height=380,
        )

    st.subheader("4. Execute no minieditor seguro", anchor=False)
    st.markdown(
        "Edite somente as **entradas JSON** e pressione o botão de execução. O "
        "terminal chamará a função selecionada e sua referência com os mesmos "
        "valores. O código-fonte permanece somente para leitura: executar Python "
        "arbitrário em uma aplicação pública permitiria acesso indevido ao servidor."
    )
    entradas_iniciais = entradas_padrao(nome, x, y, p)
    json_inicial = serializar_entradas(entradas_iniciais)
    chave_editor = f"editor_json_{nome}"
    chave_resultado = f"editor_resultado_{nome}"
    chave_erro = f"editor_erro_{nome}"
    if chave_editor not in st.session_state:
        st.session_state[chave_editor] = json_inicial

    botoes = st.columns([1, 1, 3])
    restaurar = botoes[0].button(
        "Restaurar exemplo",
        icon=":material/restart_alt:",
        key=f"editor_restaurar_{nome}",
        width="stretch",
    )
    executar_agora = botoes[1].button(
        "Executar comparação",
        type="primary",
        icon=":material/play_arrow:",
        key=f"editor_executar_{nome}",
        width="stretch",
    )
    if restaurar:
        st.session_state[chave_editor] = json_inicial
        st.session_state.pop(chave_resultado, None)
        st.session_state.pop(chave_erro, None)

    texto_entradas = st.text_area(
        "Entradas JSON",
        key=chave_editor,
        height=250,
        help=(
            "São aceitos somente os campos previstos para a função. Tipos, "
            "finitude, tamanhos, domínios e custo computacional são verificados "
            "antes da execução."
        ),
    )
    if executar_agora:
        try:
            entradas = carregar_json_entradas(texto_entradas)
            st.session_state[chave_resultado] = executar(nome, entradas)
            st.session_state.pop(chave_erro, None)
        except (TypeError, ValueError, OverflowError) as erro:
            st.session_state.pop(chave_resultado, None)
            st.session_state[chave_erro] = str(erro)

    execucao_editor = st.session_state.get(chave_resultado)
    erro_editor = st.session_state.get(chave_erro)
    st.markdown("#### Terminal de validação")
    st.code(
        montar_terminal(nome, execucao_editor, erro_editor),
        language="shell",
        height=340,
        wrap_lines=True,
    )
    if execucao_editor is not None:
        _renderizar_resultados(definicao, execucao_editor)
        diferenca = (
            "não se aplica a este tipo de resultado"
            if execucao_editor.diferenca is None
            else f"{execucao_editor.diferenca:.3e}"
        )
        detalhes = st.columns(3)
        detalhes[0].metric("Maior diferença absoluta", diferenca, border=True)
        with detalhes[1]:
            with st.container(border=True):
                st.caption("MÉTODO DE VALIDAÇÃO")
                st.markdown(f"**{definicao.metodo}**")
        with detalhes[2]:
            with st.container(border=True):
                st.caption("REFERÊNCIA")
                st.markdown(f"**{definicao.referencia}**")
        if execucao_editor.aprovado:
            st.success(
                "A execução solicitada foi aprovada pelo critério documentado.",
                icon=":material/check_circle:",
            )
        else:
            st.error(
                "A execução excedeu o critério e deve ser investigada.",
                icon=":material/error:",
            )
    elif erro_editor is not None:
        st.error(
            "As entradas foram rejeitadas antes da execução. Corrija o JSON "
            "conforme a mensagem do terminal.",
            icon=":material/error:",
        )

    with st.expander("Limites de segurança e validação das entradas"):
        st.markdown(
            "- o conteúdo deve ser um objeto JSON válido;\n"
            "- campos ausentes e campos desconhecidos são rejeitados;\n"
            "- booleanos não são aceitos como números;\n"
            "- NaN e infinito são rejeitados;\n"
            "- listas possuem limite de 10.000 itens;\n"
            "- X e Y precisam manter o mesmo tamanho quando pareados;\n"
            "- domínios e condições matemáticas são verificados conforme a função;\n"
            "- simulações são limitadas a 1.000.000 de sorteios elementares;\n"
            "- nenhuma entrada é enviada para `eval()` ou `exec()`."
        )

    st.subheader("5. O que sustenta este check", anchor=False)
    with st.container(border=True):
        st.markdown(
            f"- **Mesmas entradas:** os dois lados recebem os valores exibidos acima.\n"
            f"- **Critério explícito:** {resultado.criterio}.\n"
            "- **Parâmetros alinhados:** divisor, interpolação, suporte e parâmetros "
            "são igualados quando aplicável.\n"
            f"- **Teste automatizado relacionado:** {definicao.testes}\n"
            "- **Rastreabilidade:** o código da equipe é lido diretamente da função "
            "carregada, evitando uma cópia desatualizada na interface."
        )
        if resultado.observacao:
            st.warning(resultado.observacao, icon=":material/info:")
        st.caption(
            "A aprovação comprova este exemplo. A suíte automatizada amplia a "
            "evidência com outros valores, erros e casos-limite; testes não são uma "
            "prova formal para todo valor matematicamente possível."
        )

    with st.expander("Como funciona a tolerância numérica", icon=":material/rule:"):
        st.code(
            "math.isclose(resultado_proprio, resultado_referencia,\n"
            "             rel_tol=1e-9, abs_tol=1e-12)",
            language="python",
        )
        st.markdown(
            "A tolerância relativa acompanha a escala; a absoluta protege valores "
            "próximos de zero. Sequências são comparadas elemento a elemento. "
            "Textos e simulações controladas usam a propriedade documentada."
        )

    st.subheader("Catálogo completo do núcleo", anchor=False)
    st.caption("Todas as funções abaixo podem ser escolhidas no seletor desta página.")
    st.dataframe(
        inventario,
        hide_index=True,
        width="stretch",
        column_config={
            "Função": st.column_config.TextColumn(width="medium"),
            "Uso no projeto": st.column_config.TextColumn(width="large"),
            "Referência ou propriedade": st.column_config.TextColumn(width="large"),
            "Estado do exemplo": st.column_config.TextColumn(width="small"),
        },
    )
