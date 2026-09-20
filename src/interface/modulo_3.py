"""
Módulo 3 — Probabilidade e simulação de Monte Carlo.

O módulo demonstra a Lei dos Grandes Números e o Teorema Central do Limite
com amostragens da distribuição empírica do UNSW-NB15. Os sorteios são feitos
com reposição e podem ser reproduzidos por meio de uma semente controlada.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from nucleo import minhastats


VARIAVEIS_CATEGORICAS = {
    "attack_cat": "Categoria normal ou tipo de ataque",
    "proto": "Protocolo da transação",
    "service": "Serviço de rede identificado",
    "state": "Estado registrado para o fluxo",
}


VARIAVEIS_NUMERICAS = {
    "dur": "Duração do fluxo",
    "spkts": "Pacotes da origem para o destino",
    "dpkts": "Pacotes do destino para a origem",
    "sbytes": "Bytes da origem para o destino",
    "dbytes": "Bytes do destino para a origem",
    "rate": "Taxa calculada do fluxo",
    "sttl": "TTL da origem para o destino",
    "dttl": "TTL do destino para a origem",
}


COLUNAS_MODULO_3 = tuple(
    list(VARIAVEIS_CATEGORICAS) + list(VARIAVEIS_NUMERICAS)
)


def formatar_inteiro(valor: int) -> str:
    """Formata uma contagem com separador de milhar em português."""
    return f"{valor:,}".replace(",", ".")


def formatar_decimal(valor: float, casas: int = 4) -> str:
    """Formata um número decimal para a apresentação na interface."""
    return (
        f"{valor:,.{casas}f}"
        .replace(",", "TEMP")
        .replace(".", ",")
        .replace("TEMP", ".")
    )


def formatar_percentual(valor: float, casas: int = 3) -> str:
    """Formata uma proporção entre zero e um como percentual."""
    return f"{valor * 100:.{casas}f}%".replace(".", ",")


def colunas_ausentes(dados: pd.DataFrame) -> list[str]:
    """Retorna as colunas esperadas pelo Módulo 3 que não estão no dataset."""
    return [coluna for coluna in COLUNAS_MODULO_3 if coluna not in dados]


def categorias_por_frequencia(dados: list[object]) -> list[object]:
    """Ordena categorias da mais frequente para a menos frequente."""
    if not dados:
        raise ValueError("A variável categórica não possui valores válidos.")

    contagem: dict[object, int] = {}

    for valor in dados:
        contagem[valor] = contagem.get(valor, 0) + 1

    return [
        categoria
        for categoria, _ in sorted(
            contagem.items(),
            key=lambda item: (-item[1], str(item[0])),
        )
    ]


def calcular_frequencia_referencia(
    dados: list[object],
    categoria_alvo: object,
) -> tuple[int, float]:
    """Calcula a frequência empírica de uma categoria na partição."""
    if not dados:
        raise ValueError("A variável categórica não possui valores válidos.")

    frequencia = sum(valor == categoria_alvo for valor in dados)
    return frequencia, frequencia / len(dados)


def calcular_resumo_lgn(
    dados: list[object],
    categoria_alvo: object,
    numero_sorteios: int,
    semente: int,
) -> tuple[list[float], dict[str, float | int]]:
    """Executa e resume uma simulação da Lei dos Grandes Números."""
    frequencia, referencia = calcular_frequencia_referencia(
        dados,
        categoria_alvo,
    )
    proporcoes = minhastats.simular_frequencia_relativa(
        dados,
        categoria_alvo,
        numero_sorteios,
        semente=semente,
    )
    proporcao_final = proporcoes[-1]
    ocorrencias_observadas = round(proporcao_final * numero_sorteios)

    return proporcoes, {
        "frequencia_particao": frequencia,
        "referencia": referencia,
        "proporcao_final": proporcao_final,
        "erro_absoluto": abs(proporcao_final - referencia),
        "ocorrencias_esperadas": numero_sorteios * referencia,
        "ocorrencias_observadas": ocorrencias_observadas,
    }


def limites_verticais_lgn(
    proporcoes: list[float],
    referencia: float,
) -> tuple[float, float]:
    """Calcula limites que preservam a leitura de eventos comuns e raros."""
    if not proporcoes:
        raise ValueError("A trajetória da simulação não pode estar vazia.")

    inicio = min(9, len(proporcoes) - 1)
    valores = proporcoes[inicio:] + [referencia]
    minimo = min(valores)
    maximo = max(valores)
    extensao = max(maximo - minimo, referencia * 0.25, 0.005)
    margem = extensao * 0.15
    limite_inferior = max(0.0, minimo - margem)
    limite_superior = min(1.0, maximo + margem)

    if limite_superior <= limite_inferior:
        limite_superior = min(1.0, limite_inferior + 0.01)

    return limite_inferior, limite_superior


def calcular_resumo_tcl(
    dados: list[float | int],
    tamanho_amostra: int,
    numero_repeticoes: int,
    semente: int,
) -> tuple[list[float], dict[str, float | int | None]]:
    """Executa o TCL e compara a dispersão observada com a teórica."""
    if numero_repeticoes < 2:
        raise ValueError(
            "O resumo do TCL requer pelo menos duas amostras simuladas."
        )

    medias_amostrais = minhastats.gerar_medias_amostrais(
        dados,
        tamanho_amostra,
        numero_repeticoes,
        semente=semente,
    )
    media_particao = minhastats.media(dados)
    desvio_particao = minhastats.desvio_padrao(
        dados,
        amostral=False,
    )
    media_das_medias = minhastats.media(medias_amostrais)
    desvio_das_medias = minhastats.desvio_padrao(
        medias_amostrais,
        amostral=True,
    )
    erro_padrao_teorico = desvio_particao / math.sqrt(tamanho_amostra)
    erro_monte_carlo_centro = (
        erro_padrao_teorico / math.sqrt(numero_repeticoes)
    )

    if erro_padrao_teorico == 0:
        diferenca_dispersao_percentual = None
    else:
        diferenca_dispersao_percentual = (
            abs(desvio_das_medias - erro_padrao_teorico)
            / erro_padrao_teorico
            * 100
        )

    return medias_amostrais, {
        "tamanho_amostra": tamanho_amostra,
        "numero_repeticoes": numero_repeticoes,
        "media_particao": media_particao,
        "desvio_particao": desvio_particao,
        "media_das_medias": media_das_medias,
        "erro_centro": media_das_medias - media_particao,
        "erro_centro_absoluto": abs(media_das_medias - media_particao),
        "desvio_das_medias": desvio_das_medias,
        "erro_padrao_teorico": erro_padrao_teorico,
        "erro_monte_carlo_centro": erro_monte_carlo_centro,
        "diferenca_dispersao_percentual": diferenca_dispersao_percentual,
    }


def renderizar_lgn(dados: pd.DataFrame) -> None:
    """Renderiza a simulação da Lei dos Grandes Números."""
    st.subheader(
        "Lei dos Grandes Números",
        icon=":material/trending_up:",
    )
    st.markdown(
        "A aplicação sorteia registros de fluxo **com reposição** e acompanha "
        "a frequência acumulada de uma categoria. À medida que o número de "
        "sorteios cresce, essa frequência tende a se estabilizar em torno da "
        "proporção observada na partição. Uma execução finita ainda apresenta "
        "variação aleatória."
    )

    coluna_variavel, coluna_evento = st.columns(2)
    with coluna_variavel:
        variavel = st.selectbox(
            "Variável categórica",
            list(VARIAVEIS_CATEGORICAS),
            format_func=lambda nome: (
                f"{nome} — {VARIAVEIS_CATEGORICAS[nome]}"
            ),
            key="modulo3_lgn_variavel",
        )

    valores = dados[variavel].dropna().tolist()
    categorias = categorias_por_frequencia(valores)

    with coluna_evento:
        categoria_alvo = st.selectbox(
            "Categoria acompanhada",
            categorias,
            key="modulo3_lgn_categoria",
        )

    coluna_sorteios, coluna_semente = st.columns(2)
    with coluna_sorteios:
        numero_sorteios = st.slider(
            "Número de registros sorteados",
            min_value=100,
            max_value=50_000,
            value=5_000,
            step=100,
            key="modulo3_lgn_sorteios",
        )
    with coluna_semente:
        semente = st.number_input(
            "Semente da simulação",
            min_value=0,
            max_value=999_999,
            value=42,
            step=1,
            key="modulo3_lgn_semente",
        )

    with st.spinner("Sorteando registros da distribuição empírica..."):
        proporcoes, resumo = calcular_resumo_lgn(
            valores,
            categoria_alvo,
            numero_sorteios,
            int(semente),
        )

    metricas = st.columns(4)
    metricas[0].metric(
        "Referência na partição",
        formatar_percentual(float(resumo["referencia"])),
        border=True,
    )
    metricas[1].metric(
        "Frequência simulada final",
        formatar_percentual(float(resumo["proporcao_final"])),
        border=True,
    )
    metricas[2].metric(
        "Erro absoluto final",
        (
            f"{formatar_decimal(float(resumo['erro_absoluto']) * 100, 3)} "
            "p.p."
        ),
        border=True,
    )
    metricas[3].metric(
        "Ocorrências esperadas · observadas",
        (
            f"{formatar_decimal(float(resumo['ocorrencias_esperadas']), 2)} "
            f"· {formatar_inteiro(int(resumo['ocorrencias_observadas']))}"
        ),
        border=True,
    )

    limite_inferior, limite_superior = limites_verticais_lgn(
        proporcoes,
        float(resumo["referencia"]),
    )
    figura, eixo = plt.subplots(figsize=(11, 4.5))
    eixo.plot(
        range(1, numero_sorteios + 1),
        proporcoes,
        color="#6A1B9A",
        alpha=0.85,
        label="Frequência acumulada simulada",
    )
    eixo.axhline(
        float(resumo["referencia"]),
        color="#C62828",
        linestyle="dashed",
        linewidth=2,
        label="Frequência de referência na partição",
    )
    eixo.set_ylim(limite_inferior, limite_superior)
    eixo.set_title(
        f"Estabilização da frequência de {categoria_alvo!s}"
    )
    eixo.set_xlabel("Número acumulado de registros sorteados")
    eixo.set_ylabel("Frequência relativa")
    eixo.legend()
    figura.tight_layout()
    st.pyplot(figura)
    plt.close(figura)
    st.caption(
        "Para manter categorias raras visíveis, o eixo vertical é ajustado "
        "a partir do décimo sorteio. Oscilações dos nove primeiros sorteios "
        "podem ultrapassar a área exibida, mas continuam incluídas em todos "
        "os cálculos."
    )

    if float(resumo["ocorrencias_esperadas"]) < 5:
        st.warning(
            "A categoria é rara para o número de sorteios escolhido. É "
            "plausível observar poucas ocorrências ou nenhuma, e a trajetória "
            "pode permanecer instável. Aumente o número de sorteios para "
            "visualizar melhor a estabilização.",
            icon=":material/warning:",
        )

    with st.expander("Como interpretar esta simulação"):
        st.latex(r"\hat{p}_n = \frac{\text{ocorrências acumuladas}}{n}")
        st.markdown(
            "A linha vermelha é a frequência relativa calculada em toda a "
            "partição carregada; ela funciona como referência empírica, não "
            "como probabilidade universal de uma rede real. A semente permite "
            "reproduzir exatamente a mesma sequência de sorteios."
        )


def renderizar_tcl(dados: pd.DataFrame) -> None:
    """Renderiza a simulação do Teorema Central do Limite."""
    st.subheader(
        "Teorema Central do Limite",
        icon=":material/finance:",
    )
    st.markdown(
        "A aplicação sorteia várias amostras independentes **com reposição** "
        "da distribuição empírica e calcula a média de cada uma. Sob essa "
        "simulação e com média e variância finitas, a distribuição das médias "
        "tende a se aproximar de uma Normal quando o tamanho da amostra cresce."
    )

    variavel = st.selectbox(
        "Variável numérica",
        list(VARIAVEIS_NUMERICAS),
        format_func=lambda nome: f"{nome} — {VARIAVEIS_NUMERICAS[nome]}",
        key="modulo3_tcl_variavel",
    )
    coluna_tamanho, coluna_repeticoes, coluna_semente = st.columns(3)
    with coluna_tamanho:
        tamanho_amostra = st.slider(
            "Tamanho de cada amostra (n)",
            min_value=5,
            max_value=200,
            value=30,
            step=5,
            key="modulo3_tcl_tamanho",
        )
    with coluna_repeticoes:
        numero_repeticoes = st.slider(
            "Número de amostras",
            min_value=100,
            max_value=5_000,
            value=1_000,
            step=100,
            key="modulo3_tcl_repeticoes",
        )
    with coluna_semente:
        semente = st.number_input(
            "Semente da simulação",
            min_value=0,
            max_value=999_999,
            value=42,
            step=1,
            key="modulo3_tcl_semente",
        )

    valores = dados[variavel].dropna().tolist()

    with st.spinner("Sorteando amostras e calculando as médias..."):
        medias_amostrais, resumo = calcular_resumo_tcl(
            valores,
            tamanho_amostra,
            numero_repeticoes,
            int(semente),
        )

    linha_centro = st.columns(3)
    linha_centro[0].metric(
        "Média da partição",
        formatar_decimal(float(resumo["media_particao"]), 6),
        border=True,
    )
    linha_centro[1].metric(
        "Média das médias",
        formatar_decimal(float(resumo["media_das_medias"]), 6),
        border=True,
    )
    linha_centro[2].metric(
        "Diferença absoluta entre médias",
        formatar_decimal(float(resumo["erro_centro_absoluto"]), 6),
        border=True,
    )

    linha_dispersao = st.columns(3)
    linha_dispersao[0].metric(
        "DP observado das médias",
        formatar_decimal(float(resumo["desvio_das_medias"]), 6),
        border=True,
    )
    linha_dispersao[1].metric(
        "Erro padrão teórico",
        formatar_decimal(float(resumo["erro_padrao_teorico"]), 6),
        border=True,
    )
    diferenca_dispersao = resumo["diferenca_dispersao_percentual"]
    linha_dispersao[2].metric(
        "Diferença entre dispersões",
        (
            "Indefinida"
            if diferenca_dispersao is None
            else f"{formatar_decimal(float(diferenca_dispersao), 2)}%"
        ),
        border=True,
    )

    figura, eixos = plt.subplots(1, 2, figsize=(15, 5))
    eixos[0].hist(
        valores,
        bins=40,
        color="skyblue",
        edgecolor="black",
    )
    eixos[0].axvline(
        float(resumo["media_particao"]),
        color="#C62828",
        linestyle="dashed",
        linewidth=2,
        label="Média da partição",
    )
    eixos[0].set_title(f"Distribuição original de {variavel}")
    eixos[0].set_xlabel(variavel)
    eixos[0].set_ylabel("Frequência absoluta")
    eixos[0].legend()

    eixos[1].hist(
        medias_amostrais,
        bins=40,
        color="lightgreen",
        edgecolor="black",
    )
    eixos[1].axvline(
        float(resumo["media_particao"]),
        color="#C62828",
        linestyle="dashed",
        linewidth=2,
        label="Média da partição",
    )
    eixos[1].axvline(
        float(resumo["media_das_medias"]),
        color="#1565C0",
        linestyle="dotted",
        linewidth=2,
        label="Média das médias",
    )
    eixos[1].set_title(
        f"Médias de {numero_repeticoes} amostras com n={tamanho_amostra}"
    )
    eixos[1].set_xlabel("Média amostral")
    eixos[1].set_ylabel("Frequência absoluta")
    eixos[1].legend()
    figura.tight_layout()
    st.pyplot(figura)
    plt.close(figura)

    diferenca_percentual = resumo["diferenca_dispersao_percentual"]

    if diferenca_percentual is None:
        st.info(
            "A variável é constante: tanto o erro padrão teórico quanto a "
            "dispersão observada das médias são iguais a zero.",
            icon=":material/query_stats:",
        )
    elif float(diferenca_percentual) <= 10:
        st.info(
            "O desvio observado das médias ficou próximo de σ/√n nesta "
            "execução. Isso é compatível com o comportamento previsto para "
            "a distribuição das médias; o histograma ainda deve ser usado "
            "para avaliar a aproximação visual à forma Normal.",
            icon=":material/query_stats:",
        )
    else:
        st.warning(
            "A dispersão observada diferiu mais de 10% de σ/√n nesta "
            "execução. Variáveis com caudas longas e valores extremos podem "
            "produzir estimativas Monte Carlo instáveis. Aumente o número de "
            "amostras ou altere a semente para avaliar a variabilidade; uma "
            "execução isolada não refuta o TCL.",
            icon=":material/warning:",
        )

    st.caption(
        "A oscilação esperada para a média das médias nesta simulação é da "
        "ordem de "
        f"{formatar_decimal(float(resumo['erro_monte_carlo_centro']), 6)} "
        "unidades, calculada por (σ/√n)/√R, em que R é o número de amostras."
    )

    with st.expander("Hipóteses e limites da demonstração"):
        st.latex(r"E(\bar{X}) = \mu")
        st.latex(r"DP(\bar{X}) = \frac{\sigma}{\sqrt{n}}")
        st.markdown(
            "- Os sorteios são independentes porque a simulação os realiza "
            "com reposição. Isso não prova independência temporal entre os "
            "fluxos que originaram o dataset.\n"
            "- O TCL é assintótico. Amostras pequenas podem conservar parte "
            "da assimetria da variável original.\n"
            "- A forma aproximadamente Normal das médias não transforma a "
            "variável original em Normal.\n"
            "- Os resultados descrevem a distribuição empírica carregada e "
            "não estimam automaticamente o comportamento de outra rede."
        )


def renderizar_modulo_3(dados: pd.DataFrame) -> None:
    """Renderiza o Módulo 3 completo na aplicação Streamlit."""
    faltantes = colunas_ausentes(dados)

    st.header(
        "Probabilidade e simulação de Monte Carlo",
        icon=":material/casino:",
    )
    st.markdown(
        "Use sorteios reprodutíveis para observar a estabilização de "
        "frequências relativas e o comportamento das médias amostrais."
    )

    if faltantes:
        st.error(
            "O dataset não contém todas as colunas exigidas pelo Módulo 3: "
            + ", ".join(faltantes),
            icon=":material/error:",
        )
        return

    experimento = st.segmented_control(
        "Experimento",
        ["Lei dos Grandes Números", "Teorema Central do Limite"],
        default="Lei dos Grandes Números",
        key="modulo3_experimento",
    )

    if experimento == "Lei dos Grandes Números":
        renderizar_lgn(dados)
    else:
        renderizar_tcl(dados)
