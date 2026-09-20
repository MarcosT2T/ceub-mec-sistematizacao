"""Execução segura e validada do minieditor do núcleo estatístico."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import statistics
from time import perf_counter
from typing import Any

import numpy as np
from scipy import stats

from nucleo import minhastats


TOLERANCIA_RELATIVA = 1e-9
TOLERANCIA_ABSOLUTA = 1e-12
MAXIMO_ITENS = 10_000
MAXIMO_OPERACOES_SIMULACAO = 1_000_000
MAXIMO_CARACTERES_JSON = 100_000


@dataclass(frozen=True)
class ExecucaoEditor:
    """Saída auditável de uma execução solicitada pelo usuário."""

    proprio: Any
    referencia: Any
    aprovado: bool
    diferenca: float | None
    criterio: str
    duracao_ms: float
    avisos: tuple[str, ...] = ()


def _numero(valor: Any, nome: str) -> float:
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ValueError(f"'{nome}' deve ser um número real.")
    convertido = float(valor)
    if not math.isfinite(convertido):
        raise ValueError(f"'{nome}' deve ser finito; NaN e infinito não são aceitos.")
    return convertido


def _inteiro(
    valor: Any,
    nome: str,
    minimo: int | None = None,
    maximo: int | None = None,
) -> int:
    if isinstance(valor, bool) or not isinstance(valor, int):
        raise ValueError(f"'{nome}' deve ser um número inteiro.")
    if minimo is not None and valor < minimo:
        raise ValueError(f"'{nome}' deve ser maior ou igual a {minimo}.")
    if maximo is not None and valor > maximo:
        raise ValueError(f"'{nome}' deve ser menor ou igual a {maximo}.")
    return valor


def _booleano(valor: Any, nome: str) -> bool:
    if not isinstance(valor, bool):
        raise ValueError(f"'{nome}' deve ser true ou false.")
    return valor


def _lista_numerica(
    valor: Any,
    nome: str,
    minimo_itens: int = 1,
    somente_inteiros: bool = False,
) -> list[float] | list[int]:
    if not isinstance(valor, list):
        raise ValueError(f"'{nome}' deve ser uma lista JSON.")
    if len(valor) < minimo_itens:
        raise ValueError(
            f"'{nome}' deve conter pelo menos {minimo_itens} valor(es)."
        )
    if len(valor) > MAXIMO_ITENS:
        raise ValueError(
            f"'{nome}' excede o limite seguro de {MAXIMO_ITENS} valores."
        )
    if somente_inteiros:
        return [_inteiro(item, f"{nome}[{indice}]") for indice, item in enumerate(valor)]
    return [_numero(item, f"{nome}[{indice}]") for indice, item in enumerate(valor)]


def _lista_categorica(valor: Any, nome: str) -> list[str | int | float]:
    if not isinstance(valor, list) or not valor:
        raise ValueError(f"'{nome}' deve ser uma lista JSON não vazia.")
    if len(valor) > MAXIMO_ITENS:
        raise ValueError(
            f"'{nome}' excede o limite seguro de {MAXIMO_ITENS} valores."
        )
    resultado: list[str | int | float] = []
    for indice, item in enumerate(valor):
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            raise ValueError(
                f"'{nome}[{indice}]' deve ser texto ou número, sem listas ou objetos."
            )
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"'{nome}[{indice}]' deve ser finito.")
        resultado.append(item)
    return resultado


def _validar_chaves(
    entradas: dict[str, Any],
    obrigatorias: set[str],
    opcionais: set[str] | None = None,
) -> None:
    opcionais = opcionais or set()
    ausentes = obrigatorias - set(entradas)
    extras = set(entradas) - obrigatorias - opcionais
    if ausentes:
        raise ValueError(
            "Campos obrigatórios ausentes: " + ", ".join(sorted(ausentes)) + "."
        )
    if extras:
        raise ValueError(
            "Campos não reconhecidos: " + ", ".join(sorted(extras)) + "."
        )


def carregar_json_entradas(texto: str) -> dict[str, Any]:
    """Converte o conteúdo do editor sem avaliar código Python."""
    if len(texto) > MAXIMO_CARACTERES_JSON:
        raise ValueError(
            f"O JSON excede o limite de {MAXIMO_CARACTERES_JSON} caracteres."
        )
    try:
        entradas = json.loads(texto)
    except json.JSONDecodeError as erro:
        raise ValueError(
            f"JSON inválido na linha {erro.lineno}, coluna {erro.colno}: {erro.msg}."
        ) from erro
    if not isinstance(entradas, dict):
        raise ValueError("A raiz do JSON deve ser um objeto entre chaves: { ... }.")
    return entradas


def entradas_padrao(
    nome: str,
    x: list[float],
    y: list[float],
    percentil: float,
) -> dict[str, Any]:
    """Monta um exemplo editável coerente com a assinatura selecionada."""
    somente_x = {"media", "mediana", "moda", "amplitude"}
    if nome in somente_x:
        return {"valores": x}
    if nome in {"variancia", "desvio_padrao", "coeficiente_variacao"}:
        return {"valores": x, "amostral": True}
    if nome == "percentil":
        return {"valores": x, "p": percentil}
    if nome == "covariancia":
        return {"x": x, "y": y, "amostral": True}
    if nome in {"correlacao_pearson", "regressao_linear"}:
        return {"x": x, "y": y}
    inclinacao, intercepto = minhastats.regressao_linear(x, y)
    if nome == "predicao_linear":
        return {
            "x": minhastats.mediana(x),
            "inclinacao": inclinacao,
            "intercepto": intercepto,
        }
    if nome == "coeficiente_determinacao":
        estimados = [
            minhastats.predicao_linear(valor, inclinacao, intercepto)
            for valor in x
        ]
        return {"y": y, "y_estimado": estimados}
    if nome in {"limites_iqr", "detectar_outliers", "interpretar_assimetria"}:
        return {"valores": [10, 11, 12, 12, 13, 14, 100]}
    if nome == "simular_frequencia_relativa":
        return {
            "dados_categoricos": ["Ataque", "Normal", "Ataque", "Ataque"],
            "categoria_alvo": "Ataque",
            "n_sorteios": 1000,
            "semente": 42,
        }
    if nome == "gerar_medias_amostrais":
        return {
            "dados": [1, 2, 3, 4, 5, 6],
            "tamanho_amostra": 5,
            "n_repeticoes": 1000,
            "semente": 42,
        }
    if nome in {"pdf_normal", "cdf_normal"}:
        return {"x": 1.25, "mu": 0.5, "sigma": 2.0}
    if nome in {"pdf_exponencial", "cdf_exponencial"}:
        return {"x": 1.25, "lambd": 0.8}
    if nome == "pmf_poisson":
        return {"k": 4, "lambd": 0.8}
    if nome == "cdf_poisson":
        return {"x": 4, "lambd": 0.8}
    if nome == "distancia_kolmogorov_smirnov":
        return {
            "dados": [-1.1, -0.4, 0.0, 0.6, 1.2],
            "distribuicao": "normal",
            "mu": 0.0,
            "sigma": 1.0,
        }
    if nome == "distancia_kolmogorov_discreta":
        return {"dados": [0, 1, 1, 2, 3, 4], "lambd": 1.5}
    if nome == "_validar_parametros_distribuicao":
        return {"valores": [1, 2.5, "3.75"]}
    raise ValueError(f"A função '{nome}' não possui esquema de entrada registrado.")


def serializar_entradas(entradas: dict[str, Any]) -> str:
    return json.dumps(entradas, ensure_ascii=False, indent=2)


def validar_entradas(
    nome: str,
    entradas: dict[str, Any],
) -> dict[str, Any]:
    """Valida esquema, domínio, custo e condições matemáticas da função."""
    if nome in {"media", "mediana", "moda", "amplitude"}:
        _validar_chaves(entradas, {"valores"})
        return {"valores": _lista_numerica(entradas["valores"], "valores")}
    if nome in {"variancia", "desvio_padrao", "coeficiente_variacao"}:
        _validar_chaves(entradas, {"valores", "amostral"})
        amostral = _booleano(entradas["amostral"], "amostral")
        valores = _lista_numerica(
            entradas["valores"], "valores", minimo_itens=2 if amostral else 1
        )
        if nome == "coeficiente_variacao" and math.isclose(
            sum(valores) / len(valores), 0.0, abs_tol=1e-15
        ):
            raise ValueError("O coeficiente de variação exige média diferente de zero.")
        return {"valores": valores, "amostral": amostral}
    if nome == "percentil":
        _validar_chaves(entradas, {"valores", "p"})
        p = _numero(entradas["p"], "p")
        if not 0 <= p <= 100:
            raise ValueError("'p' deve estar entre 0 e 100.")
        return {"valores": _lista_numerica(entradas["valores"], "valores"), "p": p}
    if nome in {"covariancia", "correlacao_pearson", "regressao_linear"}:
        obrigatorias = {"x", "y"}
        if nome == "covariancia":
            obrigatorias.add("amostral")
        _validar_chaves(entradas, obrigatorias)
        amostral = _booleano(entradas["amostral"], "amostral") if nome == "covariancia" else True
        minimo = 2 if amostral or nome != "covariancia" else 1
        x = _lista_numerica(entradas["x"], "x")
        y = _lista_numerica(entradas["y"], "y")
        if len(x) != len(y):
            raise ValueError("'x' e 'y' devem ter o mesmo número de elementos.")
        if len(x) < minimo:
            raise ValueError(
                f"'x' e 'y' devem conter pelo menos {minimo} par(es)."
            )
        if nome in {"correlacao_pearson", "regressao_linear"} and len(set(x)) < 2:
            raise ValueError("'x' deve possuir pelo menos dois valores distintos.")
        if nome == "correlacao_pearson" and len(set(y)) < 2:
            raise ValueError("'y' deve possuir pelo menos dois valores distintos.")
        resultado = {"x": x, "y": y}
        if nome == "covariancia":
            resultado["amostral"] = amostral
        return resultado
    if nome == "predicao_linear":
        _validar_chaves(entradas, {"x", "inclinacao", "intercepto"})
        return {
            chave: _numero(entradas[chave], chave)
            for chave in ("x", "inclinacao", "intercepto")
        }
    if nome == "coeficiente_determinacao":
        _validar_chaves(entradas, {"y", "y_estimado"})
        y = _lista_numerica(entradas["y"], "y")
        estimados = _lista_numerica(entradas["y_estimado"], "y_estimado")
        if len(y) != len(estimados):
            raise ValueError("'y' e 'y_estimado' devem ter o mesmo tamanho.")
        if len(y) < 2:
            raise ValueError("'y' e 'y_estimado' devem conter pelo menos 2 pares.")
        if len(set(y)) < 2:
            raise ValueError("'y' deve possuir variação para que R² seja definido.")
        return {"y": y, "y_estimado": estimados}
    if nome in {"limites_iqr", "detectar_outliers", "interpretar_assimetria"}:
        _validar_chaves(entradas, {"valores"})
        return {"valores": _lista_numerica(entradas["valores"], "valores")}
    if nome == "simular_frequencia_relativa":
        _validar_chaves(
            entradas,
            {"dados_categoricos", "categoria_alvo", "n_sorteios", "semente"},
        )
        dados = _lista_categorica(entradas["dados_categoricos"], "dados_categoricos")
        alvo = entradas["categoria_alvo"]
        if isinstance(alvo, bool) or not isinstance(alvo, (str, int, float)):
            raise ValueError("'categoria_alvo' deve ser texto ou número.")
        if isinstance(alvo, float) and not math.isfinite(alvo):
            raise ValueError("'categoria_alvo' deve ser finita.")
        sorteios = _inteiro(entradas["n_sorteios"], "n_sorteios", 1, MAXIMO_OPERACOES_SIMULACAO)
        semente = entradas["semente"]
        if semente is not None:
            semente = _inteiro(semente, "semente")
        return {
            "dados_categoricos": dados,
            "categoria_alvo": alvo,
            "n_sorteios": sorteios,
            "semente": semente,
        }
    if nome == "gerar_medias_amostrais":
        _validar_chaves(
            entradas,
            {"dados", "tamanho_amostra", "n_repeticoes", "semente"},
        )
        dados = _lista_numerica(entradas["dados"], "dados")
        tamanho = _inteiro(entradas["tamanho_amostra"], "tamanho_amostra", 1, MAXIMO_ITENS)
        repeticoes = _inteiro(entradas["n_repeticoes"], "n_repeticoes", 1, MAXIMO_ITENS)
        if tamanho * repeticoes > MAXIMO_OPERACOES_SIMULACAO:
            raise ValueError(
                "tamanho_amostra × n_repeticoes excede o limite seguro de "
                f"{MAXIMO_OPERACOES_SIMULACAO} sorteios."
            )
        semente = entradas["semente"]
        if semente is not None:
            semente = _inteiro(semente, "semente")
        return {
            "dados": dados,
            "tamanho_amostra": tamanho,
            "n_repeticoes": repeticoes,
            "semente": semente,
        }
    if nome in {"pdf_normal", "cdf_normal"}:
        _validar_chaves(entradas, {"x", "mu", "sigma"})
        resultado = {chave: _numero(entradas[chave], chave) for chave in ("x", "mu", "sigma")}
        if resultado["sigma"] <= 0:
            raise ValueError("'sigma' deve ser positivo.")
        return resultado
    if nome in {"pdf_exponencial", "cdf_exponencial"}:
        _validar_chaves(entradas, {"x", "lambd"})
        resultado = {chave: _numero(entradas[chave], chave) for chave in ("x", "lambd")}
        if resultado["lambd"] <= 0:
            raise ValueError("'lambd' deve ser positivo.")
        return resultado
    if nome == "pmf_poisson":
        _validar_chaves(entradas, {"k", "lambd"})
        resultado = {chave: _numero(entradas[chave], chave) for chave in ("k", "lambd")}
        if resultado["lambd"] <= 0:
            raise ValueError("'lambd' deve ser positivo.")
        return resultado
    if nome == "cdf_poisson":
        _validar_chaves(entradas, {"x", "lambd"})
        resultado = {chave: _numero(entradas[chave], chave) for chave in ("x", "lambd")}
        if resultado["lambd"] <= 0:
            raise ValueError("'lambd' deve ser positivo.")
        return resultado
    if nome == "distancia_kolmogorov_smirnov":
        _validar_chaves(entradas, {"dados", "distribuicao", "mu", "sigma"})
        if entradas["distribuicao"] != "normal":
            raise ValueError("Nesta demonstração, 'distribuicao' deve ser 'normal'.")
        dados = _lista_numerica(entradas["dados"], "dados")
        mu, sigma = _numero(entradas["mu"], "mu"), _numero(entradas["sigma"], "sigma")
        if sigma <= 0:
            raise ValueError("'sigma' deve ser positivo.")
        return {"dados": dados, "distribuicao": "normal", "mu": mu, "sigma": sigma}
    if nome == "distancia_kolmogorov_discreta":
        _validar_chaves(entradas, {"dados", "lambd"})
        dados = _lista_numerica(entradas["dados"], "dados", somente_inteiros=True)
        lambd = _numero(entradas["lambd"], "lambd")
        if lambd <= 0:
            raise ValueError("'lambd' deve ser positivo.")
        return {"dados": dados, "lambd": lambd}
    if nome == "_validar_parametros_distribuicao":
        _validar_chaves(entradas, {"valores"})
        if not isinstance(entradas["valores"], list) or not entradas["valores"]:
            raise ValueError("'valores' deve ser uma lista JSON não vazia.")
        if len(entradas["valores"]) > MAXIMO_ITENS:
            raise ValueError(f"'valores' excede {MAXIMO_ITENS} itens.")
        convertidos = []
        for indice, item in enumerate(entradas["valores"]):
            if isinstance(item, bool):
                raise ValueError(f"'valores[{indice}]' não pode ser booleano.")
            try:
                numero = float(item)
            except (TypeError, ValueError, OverflowError) as erro:
                raise ValueError(f"'valores[{indice}]' não pode ser convertido para float.") from erro
            if not math.isfinite(numero):
                raise ValueError(f"'valores[{indice}]' deve ser finito.")
            convertidos.append(item)
        return {"valores": convertidos}
    raise ValueError(f"A função '{nome}' não possui validador de entradas.")


def _equivalentes(a: Any, b: Any) -> bool:
    if isinstance(a, (list, tuple, np.ndarray)) and isinstance(b, (list, tuple, np.ndarray)):
        return len(a) == len(b) and all(_equivalentes(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        return math.isclose(float(a), float(b), rel_tol=TOLERANCIA_RELATIVA, abs_tol=TOLERANCIA_ABSOLUTA)
    return a == b


def _diferenca(a: Any, b: Any) -> float | None:
    if isinstance(a, (list, tuple, np.ndarray)) and isinstance(b, (list, tuple, np.ndarray)):
        if len(a) != len(b):
            return None
        diferencas = [valor for x, y in zip(a, b) if (valor := _diferenca(x, y)) is not None]
        return max(diferencas, default=0.0)
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        return abs(float(a) - float(b))
    return None


def _classificar_assimetria(valores: list[float]) -> str:
    media, mediana = float(np.mean(valores)), float(np.median(valores))
    escala = max(abs(media), abs(mediana), 1e-12)
    if abs(media - mediana) / escala < 0.005:
        return "Equilíbrio compatível"
    return "Assimetria à direita" if media > mediana else "Assimetria à esquerda"


def _classificar_texto(texto: str) -> str:
    if "à direita" in texto:
        return "Assimetria à direita"
    if "à esquerda" in texto:
        return "Assimetria à esquerda"
    return "Equilíbrio compatível"


def _distancia_discreta_referencia(dados: list[int], lambd: float) -> float:
    valores, contagens = np.unique(dados, return_counts=True)
    acumuladas = np.cumsum(contagens)
    esquerda = (acumuladas - contagens) / len(dados)
    direita = acumuladas / len(dados)
    return float(max(
        np.max(np.abs(esquerda - stats.poisson.cdf(valores - 1, mu=lambd))),
        np.max(np.abs(direita - stats.poisson.cdf(valores, mu=lambd))),
    ))


def executar(nome: str, entradas: dict[str, Any]) -> ExecucaoEditor:
    """Executa somente funções registradas; nunca avalia o texto como Python."""
    dados = validar_entradas(nome, entradas)
    inicio = perf_counter()
    criterio = "equivalência numérica dentro das tolerâncias documentadas"
    avisos: tuple[str, ...] = ()

    if nome == "media":
        proprio, referencia = minhastats.media(dados["valores"]), np.mean(dados["valores"])
    elif nome == "mediana":
        proprio, referencia = minhastats.mediana(dados["valores"]), np.median(dados["valores"])
    elif nome == "moda":
        proprio, referencia = minhastats.moda(dados["valores"]), statistics.mode(dados["valores"])
        criterio = "mesmo valor e mesma política de primeiro empate"
    elif nome == "amplitude":
        proprio, referencia = minhastats.amplitude(dados["valores"]), np.ptp(dados["valores"])
    elif nome == "variancia":
        ddof = 1 if dados["amostral"] else 0
        proprio = minhastats.variancia(dados["valores"], dados["amostral"])
        referencia = np.var(dados["valores"], ddof=ddof)
        criterio += f"; ddof={ddof}"
    elif nome == "desvio_padrao":
        ddof = 1 if dados["amostral"] else 0
        proprio = minhastats.desvio_padrao(dados["valores"], dados["amostral"])
        referencia = np.std(dados["valores"], ddof=ddof)
        criterio += f"; ddof={ddof}"
    elif nome == "coeficiente_variacao":
        ddof = 1 if dados["amostral"] else 0
        proprio = minhastats.coeficiente_variacao(dados["valores"], dados["amostral"])
        referencia = 100 * stats.variation(dados["valores"], ddof=ddof)
        criterio += f"; ddof={ddof}; percentual"
    elif nome == "percentil":
        proprio = minhastats.percentil(dados["valores"], dados["p"])
        referencia = np.percentile(dados["valores"], dados["p"], method="linear")
        criterio += "; interpolação linear"
    elif nome == "covariancia":
        ddof = 1 if dados["amostral"] else 0
        proprio = minhastats.covariancia(dados["x"], dados["y"], dados["amostral"])
        referencia = np.cov(dados["x"], dados["y"], ddof=ddof)[0, 1]
        criterio += f"; ddof={ddof}; pareamento preservado"
    elif nome == "correlacao_pearson":
        proprio = minhastats.correlacao_pearson(dados["x"], dados["y"])
        referencia = stats.pearsonr(dados["x"], dados["y"]).statistic
    elif nome == "regressao_linear":
        proprio = minhastats.regressao_linear(dados["x"], dados["y"])
        ajuste = stats.linregress(dados["x"], dados["y"])
        referencia = (ajuste.slope, ajuste.intercept)
        criterio += "; inclinação e intercepto elemento a elemento"
    elif nome == "predicao_linear":
        proprio = minhastats.predicao_linear(dados["x"], dados["inclinacao"], dados["intercepto"])
        referencia = np.polynomial.polynomial.polyval(
            dados["x"], [dados["intercepto"], dados["inclinacao"]]
        )
    elif nome == "coeficiente_determinacao":
        proprio = minhastats.coeficiente_determinacao(dados["y"], dados["y_estimado"])
        y, estimados = np.asarray(dados["y"]), np.asarray(dados["y_estimado"])
        referencia = 1 - np.sum((y - estimados) ** 2) / np.sum((y - np.mean(y)) ** 2)
    elif nome == "limites_iqr":
        proprio = minhastats.limites_iqr(dados["valores"])
        q1, q3 = np.percentile(dados["valores"], [25, 75], method="linear")
        referencia = (q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
    elif nome == "detectar_outliers":
        proprio = minhastats.detectar_outliers(dados["valores"])
        q1, q3 = np.percentile(dados["valores"], [25, 75], method="linear")
        li, ls = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        referencia = [valor for valor in dados["valores"] if valor < li or valor > ls]
        criterio = "mesmos limites, valores e ordem"
    elif nome == "interpretar_assimetria":
        proprio = minhastats.interpretar_assimetria(dados["valores"])
        referencia = _classificar_assimetria(dados["valores"])
        aprovado = _classificar_texto(proprio) == referencia
        duracao = (perf_counter() - inicio) * 1000
        return ExecucaoEditor(
            proprio, referencia, aprovado, None,
            "decisão textual igual à regra recalculada com NumPy", duracao,
            ("A referência valida a decisão da heurística, não uma frase canônica.",),
        )
    elif nome == "simular_frequencia_relativa":
        proprio = minhastats.simular_frequencia_relativa(**dados)
        repeticao = minhastats.simular_frequencia_relativa(**dados)
        sucessos_anteriores = 0
        trajetoria_valida = len(proprio) == dados["n_sorteios"]
        for indice, proporcao in enumerate(proprio, start=1):
            sucessos = round(proporcao * indice)
            trajetoria_valida &= 0 <= proporcao <= 1
            trajetoria_valida &= sucessos - sucessos_anteriores in {0, 1}
            trajetoria_valida &= math.isclose(proporcao, sucessos / indice, abs_tol=1e-12)
            sucessos_anteriores = sucessos
        proprio_resumo = {
            "primeiras_proporcoes": proprio[:10],
            "proporcao_final": proprio[-1],
            "quantidade": len(proprio),
        }
        referencia = {
            "reprodutivel_com_semente": proprio == repeticao,
            "trajetoria_valida": trajetoria_valida,
            "proporcao_na_populacao": dados["dados_categoricos"].count(dados["categoria_alvo"]) / len(dados["dados_categoricos"]),
        }
        duracao = (perf_counter() - inicio) * 1000
        return ExecucaoEditor(
            proprio_resumo, referencia,
            proprio == repeticao and trajetoria_valida, None,
            "reprodutibilidade e invariantes da frequência acumulada", duracao,
            ("A proporção final simulada não precisa ser idêntica à proporção da população em um número finito de sorteios.",),
        )
    elif nome == "gerar_medias_amostrais":
        proprio = minhastats.gerar_medias_amostrais(**dados)
        repeticao = minhastats.gerar_medias_amostrais(**dados)
        dentro_da_amplitude = all(min(dados["dados"]) <= valor <= max(dados["dados"]) for valor in proprio)
        proprio_resumo = {
            "primeiras_medias": proprio[:10],
            "media_das_medias": float(np.mean(proprio)),
            "quantidade": len(proprio),
        }
        referencia = {
            "reprodutivel_com_semente": proprio == repeticao,
            "todas_na_amplitude": dentro_da_amplitude,
            "media_da_populacao": float(np.mean(dados["dados"])),
        }
        duracao = (perf_counter() - inicio) * 1000
        return ExecucaoEditor(
            proprio_resumo, referencia,
            proprio == repeticao and len(proprio) == dados["n_repeticoes"] and dentro_da_amplitude,
            None, "reprodutibilidade, cardinalidade e limites matemáticos", duracao,
            ("A média das médias aproxima a média populacional; igualdade exata não é exigida em simulação finita.",),
        )
    elif nome == "pdf_normal":
        proprio = minhastats.pdf_normal(**dados)
        referencia = stats.norm.pdf(dados["x"], loc=dados["mu"], scale=dados["sigma"])
    elif nome == "cdf_normal":
        proprio = minhastats.cdf_normal(**dados)
        referencia = stats.norm.cdf(dados["x"], loc=dados["mu"], scale=dados["sigma"])
    elif nome == "pdf_exponencial":
        proprio = minhastats.pdf_exponencial(**dados)
        referencia = stats.expon.pdf(dados["x"], scale=1 / dados["lambd"])
    elif nome == "cdf_exponencial":
        proprio = minhastats.cdf_exponencial(**dados)
        referencia = stats.expon.cdf(dados["x"], scale=1 / dados["lambd"])
    elif nome == "pmf_poisson":
        proprio = minhastats.pmf_poisson(**dados)
        referencia = stats.poisson.pmf(dados["k"], mu=dados["lambd"])
    elif nome == "cdf_poisson":
        proprio = minhastats.cdf_poisson(**dados)
        referencia = stats.poisson.cdf(dados["x"], mu=dados["lambd"])
    elif nome == "distancia_kolmogorov_smirnov":
        cdf_propria = lambda valor: minhastats.cdf_normal(valor, dados["mu"], dados["sigma"])
        cdf_referencia = lambda valor: stats.norm.cdf(valor, loc=dados["mu"], scale=dados["sigma"])
        proprio = minhastats.distancia_kolmogorov_smirnov(dados["dados"], cdf_propria)
        referencia = stats.kstest(dados["dados"], cdf_referencia).statistic
        criterio += "; somente a estatística descritiva D"
        avisos = ("O p-valor não é calculado porque parâmetros estimados mudam sua interpretação.",)
    elif nome == "distancia_kolmogorov_discreta":
        proprio = minhastats.distancia_kolmogorov_discreta(
            dados["dados"], lambda valor: minhastats.cdf_poisson(valor, dados["lambd"])
        )
        referencia = _distancia_discreta_referencia(dados["dados"], dados["lambd"])
        criterio += "; lados esquerdo e direito dos saltos"
        avisos = ("Compara-se a distância descritiva; não se apresenta um p-valor KS contínuo como se fosse válido para Poisson.",)
    elif nome == "_validar_parametros_distribuicao":
        proprio = minhastats._validar_parametros_distribuicao(*dados["valores"])
        referencia = tuple(float(valor) for valor in dados["valores"])
        criterio = "mesma tupla convertida e somente valores finitos"
    else:
        raise ValueError(f"Execução não registrada para '{nome}'.")

    duracao = (perf_counter() - inicio) * 1000
    return ExecucaoEditor(
        proprio,
        referencia,
        _equivalentes(proprio, referencia),
        _diferenca(proprio, referencia),
        criterio,
        duracao,
        avisos,
    )


def formatar_saida(valor: Any) -> str:
    """Serializa resultados NumPy e estruturas compostas para o terminal."""
    def normalizar(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return [normalizar(valor) for valor in item.tolist()]
        if isinstance(item, tuple):
            return [normalizar(valor) for valor in item]
        if isinstance(item, list):
            return [normalizar(valor) for valor in item]
        if isinstance(item, dict):
            return {str(chave): normalizar(valor) for chave, valor in item.items()}
        return item
    return json.dumps(normalizar(valor), ensure_ascii=False, indent=2)


def montar_terminal(
    nome: str,
    execucao: ExecucaoEditor | None = None,
    erro: str | None = None,
) -> str:
    """Cria um registro legível da validação e da execução."""
    linhas = [
        f"$ validar {nome}",
        "[SEGURANÇA] Execução restrita à lista autorizada; nenhum código arbitrário foi avaliado.",
    ]
    if erro is not None:
        linhas.extend([
            "[ENTRADAS] REPROVADAS",
            f"[ERRO] {erro}",
            "[RESULTADO] Execução cancelada antes de chamar o núcleo.",
        ])
        return "\n".join(linhas)
    if execucao is None:
        linhas.append("[AGUARDANDO] Revise as entradas JSON e pressione Executar comparação.")
        return "\n".join(linhas)
    linhas.extend([
        "[ENTRADAS] APROVADAS",
        "[NÚCLEO PRÓPRIO]",
        formatar_saida(execucao.proprio),
        "[REFERÊNCIA]",
        formatar_saida(execucao.referencia),
        f"[CRITÉRIO] {execucao.criterio}",
    ])
    if execucao.diferenca is not None:
        linhas.append(f"[MAIOR DIFERENÇA] {execucao.diferenca:.3e}")
    linhas.extend([
        f"[TEMPO] {execucao.duracao_ms:.3f} ms",
        f"[RESULTADO] {'APROVADO' if execucao.aprovado else 'REPROVADO'}",
    ])
    linhas.extend(f"[NOTA] {aviso}" for aviso in execucao.avisos)
    return "\n".join(linhas)
