"""
Módulo 4 — Distribuições teóricas.

O módulo compara a distribuição empírica de características do UNSW-NB15
com modelos teóricos cujos parâmetros são estimados pelo núcleo próprio. A
Normal é exibida em todas as análises. A segunda família respeita a natureza
da variável: Exponencial para medidas não negativas e Poisson para contagens.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from nucleo import minhastats


VARIAVEIS_MODULO_4 = {
    "dur": {
        "rotulo": "Duração do fluxo",
        "descricao": "Tempo total registrado para o fluxo de rede.",
        "unidade": "segundos",
        "familia": "continua",
        "segunda_distribuicao": "Exponencial",
        "cautela": (
            "A duração possui muitos valores muito pequenos, alguns zeros e "
            "uma cauda longa. Uma Exponencial simples pode representar o "
            "decaimento geral sem explicar todos esses componentes."
        ),
    },
    "rate": {
        "rotulo": "Taxa do fluxo",
        "descricao": "Taxa calculada de transmissão e recepção do fluxo.",
        "unidade": "taxa registrada no dataset",
        "familia": "continua",
        "segunda_distribuicao": "Exponencial",
        "cautela": (
            "A taxa apresenta limites, valores repetidos e concentração em "
            "faixas específicas. Esses padrões podem produzir um ajuste ruim "
            "tanto para a Normal quanto para a Exponencial."
        ),
    },
    "sbytes": {
        "rotulo": "Bytes da origem para o destino",
        "descricao": "Volume enviado da origem para o destino.",
        "unidade": "bytes",
        "familia": "continua",
        "segunda_distribuicao": "Exponencial",
        "cautela": (
            "Bytes são contagens inteiras, mas a alta cardinalidade permite "
            "uma aproximação contínua exploratória. A cauda é muito longa e "
            "poucos fluxos volumosos influenciam fortemente a média."
        ),
    },
    "dbytes": {
        "rotulo": "Bytes do destino para a origem",
        "descricao": "Volume enviado do destino para a origem.",
        "unidade": "bytes",
        "familia": "continua",
        "segunda_distribuicao": "Exponencial",
        "cautela": (
            "A variável combina muitos zeros com valores positivos de grande "
            "amplitude. Uma única distribuição contínua dificilmente descreve "
            "simultaneamente a massa em zero e a cauda positiva."
        ),
    },
    "spkts": {
        "rotulo": "Pacotes da origem para o destino",
        "descricao": "Quantidade de pacotes enviados da origem ao destino.",
        "unidade": "pacotes",
        "familia": "contagem",
        "segunda_distribuicao": "Poisson",
        "cautela": (
            "A Poisson pressupõe uma contagem produzida por uma taxa estável "
            "e possui média igual à variância. Misturar fluxos de protocolos, "
            "serviços e ataques diferentes costuma violar essa hipótese."
        ),
    },
    "dpkts": {
        "rotulo": "Pacotes do destino para a origem",
        "descricao": "Quantidade de pacotes enviados do destino à origem.",
        "unidade": "pacotes",
        "familia": "contagem",
        "segunda_distribuicao": "Poisson",
        "cautela": (
            "Há muitos fluxos sem pacote de resposta. Essa massa em zero e a "
            "mistura de comportamentos de rede tendem a produzir grande "
            "sobredispersão em relação à Poisson simples."
        ),
    },
}


GRUPOS_TRAFEGO = {
    "Todos os registros": None,
    "Tráfego normal": "Normal",
    "Todos os ataques": "Ataques",
}


COLUNAS_MODULO_4 = tuple(VARIAVEIS_MODULO_4) + ("attack_cat",)


def formatar_inteiro(valor: int) -> str:
    """Formata uma contagem com separador de milhar em português."""
    return f"{valor:,}".replace(",", ".")


def formatar_decimal(valor: float, casas: int = 4) -> str:
    """Formata um valor decimal para leitura em português."""
    numero = float(valor)
    absoluto = abs(numero)

    if numero == 0:
        return "0"
    if absoluto >= 1_000_000 or absoluto < 0.0001:
        texto = f"{numero:.4e}"
    else:
        texto = f"{numero:,.{casas}f}"

    return (
        texto.replace(",", "TEMP")
        .replace(".", ",")
        .replace("TEMP", ".")
    )


def formatar_percentual(valor: float, casas: int = 2) -> str:
    """Formata uma proporção entre zero e um como percentual."""
    return f"{valor * 100:.{casas}f}%".replace(".", ",")


def colunas_ausentes(dados: pd.DataFrame) -> list[str]:
    """Retorna as colunas necessárias ao Módulo 4 que não estão na base."""
    return [coluna for coluna in COLUNAS_MODULO_4 if coluna not in dados]


def validar_dados_numericos(dados: list[float | int]) -> None:
    """Valida o conjunto usado para estimar e comparar as distribuições."""
    if len(dados) < 2:
        raise ValueError("O ajuste requer pelo menos dois valores válidos.")

    try:
        invalidos = any(
            not math.isfinite(float(valor))
            for valor in dados
        )
    except (TypeError, ValueError, OverflowError) as erro:
        raise ValueError(
            "Todos os valores do ajuste devem ser números finitos."
        ) from erro

    if invalidos:
        raise ValueError(
            "Todos os valores do ajuste devem ser números finitos."
        )


def filtrar_grupo(
    dados: pd.DataFrame,
    grupo: str,
) -> pd.DataFrame:
    """Aplica apenas o recorte de tráfego solicitado pelo usuário."""
    if grupo not in GRUPOS_TRAFEGO:
        raise ValueError("O grupo de tráfego selecionado é inválido.")

    categoria = GRUPOS_TRAFEGO[grupo]

    if categoria is None:
        return dados
    if categoria == "Ataques":
        return dados[dados["attack_cat"] != "Normal"]

    return dados[dados["attack_cat"] == categoria]


def preparar_valores(
    dados: pd.DataFrame,
    variavel: str,
    grupo: str,
) -> list[float]:
    """Seleciona o grupo e converte a variável escolhida em lista numérica."""
    if variavel not in VARIAVEIS_MODULO_4:
        raise ValueError("A variável selecionada não pertence ao Módulo 4.")

    recorte = filtrar_grupo(dados, grupo)
    valores = [float(valor) for valor in recorte[variavel].dropna().tolist()]
    validar_dados_numericos(valores)

    return valores


def calcular_resumo_ajuste(
    dados: list[float | int],
    familia: str,
) -> dict[str, float | int | str | None]:
    """
    Estima parâmetros e calcula distâncias descritivas entre CDFs.

    Todos os parâmetros são estimados com a lista completa recebida. O recorte
    usado posteriormente no gráfico não altera o ajuste matemático.
    """
    validar_dados_numericos(dados)

    if familia not in {"continua", "contagem"}:
        raise ValueError("A família da variável deve ser contínua ou contagem.")

    valores = [float(valor) for valor in dados]
    media = minhastats.media(valores)
    mediana = minhastats.mediana(valores)
    variancia = minhastats.variancia(valores, amostral=False)
    desvio_padrao = minhastats.desvio_padrao(
        valores,
        amostral=False,
    )

    if desvio_padrao <= 0:
        raise ValueError(
            "Não é possível ajustar distribuições a uma variável constante."
        )

    distancia_normal = minhastats.distancia_kolmogorov_smirnov(
        valores,
        lambda x: minhastats.cdf_normal(x, media, desvio_padrao),
    )
    quantidade_zeros = sum(valor == 0 for valor in valores)
    resumo: dict[str, float | int | str | None] = {
        "n": len(valores),
        "media": media,
        "mediana": mediana,
        "variancia": variancia,
        "desvio_padrao": desvio_padrao,
        "minimo": min(valores),
        "maximo": max(valores),
        "quantidade_zeros": quantidade_zeros,
        "proporcao_zeros": quantidade_zeros / len(valores),
        "distancia_normal": distancia_normal,
        "segunda_distribuicao": None,
        "parametro_segunda": None,
        "distancia_segunda": None,
        "indice_dispersao": None,
        "razao_desvio_media": None,
    }

    if familia == "continua":
        if min(valores) < 0 or media <= 0:
            raise ValueError(
                "A Exponencial requer valores não negativos e média positiva."
            )

        taxa = 1.0 / media
        distancia_exponencial = minhastats.distancia_kolmogorov_smirnov(
            valores,
            lambda x: minhastats.cdf_exponencial(x, taxa),
        )
        resumo.update(
            {
                "segunda_distribuicao": "Exponencial",
                "parametro_segunda": taxa,
                "distancia_segunda": distancia_exponencial,
                "razao_desvio_media": desvio_padrao / media,
            }
        )
    else:
        if min(valores) < 0 or any(
            not valor.is_integer()
            for valor in valores
        ):
            raise ValueError(
                "A Poisson requer contagens inteiras e não negativas."
            )
        if media <= 0:
            raise ValueError("A taxa da Poisson deve ser positiva.")

        cache_poisson: dict[int, float] = {}

        def cdf_poisson_em_cache(x: float) -> float:
            chave = math.floor(x)

            if chave not in cache_poisson:
                cache_poisson[chave] = minhastats.cdf_poisson(x, media)

            return cache_poisson[chave]

        distancia_poisson = minhastats.distancia_kolmogorov_discreta(
            valores,
            cdf_poisson_em_cache,
        )
        resumo.update(
            {
                "segunda_distribuicao": "Poisson",
                "parametro_segunda": media,
                "distancia_segunda": distancia_poisson,
                "indice_dispersao": variancia / media,
            }
        )

    return resumo


def calcular_intervalo_visual(
    dados: list[float | int],
    modo: str,
) -> tuple[float, float]:
    """Define a janela do gráfico sem remover valores do ajuste."""
    validar_dados_numericos(dados)

    if modo == "Amplitude completa":
        minimo = float(min(dados))
        maximo = float(max(dados))
    elif modo == "Região central (IQR)":
        limite_inferior, limite_superior = minhastats.limites_iqr(dados)
        minimo = max(float(min(dados)), float(limite_inferior))
        maximo = min(float(max(dados)), float(limite_superior))
    else:
        raise ValueError("O modo de visualização selecionado é inválido.")

    if maximo <= minimo:
        minimo = float(min(dados))
        maximo = float(max(dados))

    if maximo <= minimo:
        raise ValueError("A variável não possui amplitude para o gráfico.")

    return minimo, maximo


def construir_histograma_densidade(
    dados: list[float | int],
    minimo: float,
    maximo: float,
    numero_classes: int,
) -> tuple[list[float], list[float], float, int]:
    """
    Constrói um histograma compatível com curvas de densidade teóricas.

    A densidade usa o total de registros do ajuste no denominador. Assim, um
    zoom central não renormaliza silenciosamente apenas os valores visíveis.
    """
    validar_dados_numericos(dados)

    if not isinstance(numero_classes, int) or isinstance(numero_classes, bool):
        raise ValueError("O número de classes deve ser inteiro.")
    if not 5 <= numero_classes <= 100:
        raise ValueError("O número de classes deve estar entre 5 e 100.")
    if not math.isfinite(minimo) or not math.isfinite(maximo):
        raise ValueError("Os limites do histograma devem ser finitos.")
    if maximo <= minimo:
        raise ValueError("O limite superior deve exceder o inferior.")

    largura = (maximo - minimo) / numero_classes
    contagens = [0] * numero_classes

    for valor_original in dados:
        valor = float(valor_original)

        if valor < minimo or valor > maximo:
            continue

        if valor == maximo:
            indice = numero_classes - 1
        else:
            indice = int((valor - minimo) / largura)
            indice = min(indice, numero_classes - 1)

        contagens[indice] += 1

    centros = [
        minimo + (indice + 0.5) * largura
        for indice in range(numero_classes)
    ]
    densidades = [
        contagem / (len(dados) * largura)
        for contagem in contagens
    ]

    return centros, densidades, largura, sum(contagens)


def descrever_comparacao(
    resumo: dict[str, float | int | str | None],
) -> str:
    """Produz uma comparação relativa sem transformar D em teste de hipótese."""
    distancia_normal = float(resumo["distancia_normal"])
    distancia_segunda = float(resumo["distancia_segunda"])
    segunda = str(resumo["segunda_distribuicao"])

    menor_distancia = min(distancia_normal, distancia_segunda)
    discrepancia = (
        f"A menor D ainda corresponde a uma separação máxima de "
        f"aproximadamente {menor_distancia * 100:.2f} pontos percentuais "
        "entre as proporções acumuladas observada e teórica."
    )

    if math.isclose(
        distancia_normal,
        distancia_segunda,
        rel_tol=0.01,
        abs_tol=0.001,
    ):
        return (
            "As duas distâncias são semelhantes neste recorte. O histograma "
            "e as hipóteses de cada família devem orientar a discussão. "
            + discrepancia
        )
    if distancia_normal < distancia_segunda:
        return (
            f"Entre as duas candidatas, a Normal apresenta menor distância "
            f"descritiva ({distancia_normal:.4f} contra "
            f"{distancia_segunda:.4f} da {segunda}). Isso indica maior "
            "proximidade relativa, mas não demonstra ajuste adequado. "
            + discrepancia
        )

    return (
        f"Entre as duas candidatas, a {segunda} apresenta menor distância "
        f"descritiva ({distancia_segunda:.4f} contra "
        f"{distancia_normal:.4f} da Normal). Isso indica maior proximidade "
        "relativa, mas não demonstra ajuste adequado. "
        + discrepancia
    )


def renderizar_grafico(
    dados: list[float],
    resumo: dict[str, float | int | str | None],
    configuracao: dict[str, str],
    minimo_visual: float,
    maximo_visual: float,
    numero_classes: int,
) -> tuple[plt.Figure, int]:
    """Monta o histograma e as duas distribuições teóricas comparadas."""
    centros, densidades, largura, quantidade_visivel = (
        construir_histograma_densidade(
            dados,
            minimo_visual,
            maximo_visual,
            numero_classes,
        )
    )
    media = float(resumo["media"])
    desvio_padrao = float(resumo["desvio_padrao"])
    parametro_segunda = float(resumo["parametro_segunda"])
    familia = configuracao["familia"]

    eixo_x = np.linspace(minimo_visual, maximo_visual, 800)
    curva_normal = [
        minhastats.pdf_normal(x, media, desvio_padrao)
        for x in eixo_x
    ]

    figura, eixo = plt.subplots(figsize=(12, 6.5))
    eixo.bar(
        centros,
        densidades,
        width=largura * 0.95,
        color="#72A9C9",
        edgecolor="#264653",
        alpha=0.62,
        label="Dados observados",
    )
    eixo.plot(
        eixo_x,
        curva_normal,
        color="#C83E4D",
        linewidth=2.4,
        linestyle="--",
        label="Normal estimada",
    )

    if familia == "continua":
        curva_segunda = [
            minhastats.pdf_exponencial(x, parametro_segunda)
            for x in eixo_x
        ]
        eixo.plot(
            eixo_x,
            curva_segunda,
            color="#6A4C93",
            linewidth=2.4,
            label="Exponencial estimada",
        )
    else:
        primeiro_k = max(0, math.ceil(minimo_visual))
        ultimo_k = math.floor(maximo_visual)
        limite_informativo = math.ceil(
            parametro_segunda
            + 8 * math.sqrt(parametro_segunda)
            + 10
        )
        ultimo_k_modelo = min(ultimo_k, limite_informativo)

        if ultimo_k_modelo >= primeiro_k:
            valores_k = list(range(primeiro_k, ultimo_k_modelo + 1))
            probabilidades = [
                minhastats.pmf_poisson(k, parametro_segunda)
                for k in valores_k
            ]
            eixo.plot(
                valores_k,
                probabilidades,
                color="#6A4C93",
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label="Poisson estimada",
            )

    eixo.set_xlim(minimo_visual, maximo_visual)
    eixo.set_title(
        f"Distribuição observada e modelos — {configuracao['rotulo']}"
    )
    eixo.set_xlabel(configuracao["unidade"].capitalize())
    eixo.set_ylabel("Densidade ou probabilidade por unidade")
    eixo.grid(axis="y", alpha=0.18)
    eixo.legend()
    figura.tight_layout()

    return figura, quantidade_visivel


def renderizar_modulo_4(dados: pd.DataFrame) -> None:
    """Renderiza o Módulo 4 na aplicação Streamlit."""
    st.header(
        "Distribuições teóricas e dados observados",
        icon=":material/function:",
    )
    st.markdown(
        "Compare o histograma de uma característica do tráfego com uma "
        "**Normal** e com outra distribuição coerente com a natureza da "
        "variável. Os parâmetros são estimados pelas funções próprias usando "
        "todos os registros do grupo selecionado."
    )
    st.warning(
        "Uma curva próxima ao histograma descreve apenas a forma dos dados "
        "neste recorte. Ela não identifica ataques, não calcula risco e não "
        "prova que o processo real gere dados exatamente por essa distribuição.",
        icon=":material/warning:",
    )

    faltantes = colunas_ausentes(dados)

    if faltantes:
        st.error(
            "O dataset não contém todas as colunas exigidas pelo Módulo 4: "
            + ", ".join(faltantes)
        )
        return

    with st.expander("Como este módulo compara os modelos"):
        st.markdown(
            "- **Normal:** estima μ pela média e σ pelo desvio padrão "
            "populacional. Ela é simétrica e tem suporte em toda a reta.\n"
            "- **Exponencial:** usa λ = 1/μ em medidas não negativas. Ela "
            "representa decaimento contínuo a partir de zero.\n"
            "- **Poisson:** usa λ = média nas contagens de pacotes. Nessa "
            "família, média e variância deveriam ser próximas.\n"
            "- **Distância D:** maior separação vertical entre a distribuição "
            "acumulada empírica e a teórica. Menor D significa maior "
            "proximidade relativa. Não exibimos p-valor porque os parâmetros "
            "foram estimados na própria amostra e algumas variáveis são "
            "discretas."
        )

    coluna_variavel, coluna_grupo, coluna_janela = st.columns(3)

    with coluna_variavel:
        variavel = st.selectbox(
            "Variável",
            list(VARIAVEIS_MODULO_4),
            format_func=lambda nome: (
                f"{nome} — {VARIAVEIS_MODULO_4[nome]['rotulo']}"
            ),
            key="modulo4_variavel",
        )

    with coluna_grupo:
        grupo = st.selectbox(
            "Grupo de tráfego",
            list(GRUPOS_TRAFEGO),
            key="modulo4_grupo",
            help=(
                "O filtro altera a população empírica analisada e, portanto, "
                "também altera os parâmetros estimados."
            ),
        )

    with coluna_janela:
        modo_visual = st.selectbox(
            "Janela do gráfico",
            ["Região central (IQR)", "Amplitude completa"],
            key="modulo4_janela",
            help=(
                "O zoom muda apenas o trecho exibido. Todos os registros "
                "continuam no ajuste e na distância D."
            ),
        )

    configuracao = VARIAVEIS_MODULO_4[variavel]

    try:
        valores = preparar_valores(dados, variavel, grupo)
        resumo = calcular_resumo_ajuste(
            valores,
            configuracao["familia"],
        )
        minimo_visual, maximo_visual = calcular_intervalo_visual(
            valores,
            modo_visual,
        )
    except ValueError as erro:
        st.error(str(erro))
        return

    numero_classes = st.slider(
        "Quantidade de classes do histograma",
        min_value=10,
        max_value=100,
        value=50,
        step=5,
        key="modulo4_classes",
        help=(
            "A quantidade de classes altera a aparência do histograma, mas "
            "não modifica os parâmetros nem a distância D."
        ),
    )

    figura, quantidade_visivel = renderizar_grafico(
        valores,
        resumo,
        configuracao,
        minimo_visual,
        maximo_visual,
        numero_classes,
    )
    proporcao_visivel = quantidade_visivel / int(resumo["n"])

    st.subheader(
        f"{configuracao['rotulo']} — {grupo}",
        icon=":material/analytics:",
    )
    st.caption(
        configuracao["descricao"]
        + " Os resultados abaixo pertencem somente ao grupo selecionado."
    )

    colunas_metricas = st.columns(5)
    colunas_metricas[0].metric(
        "Registros no ajuste",
        formatar_inteiro(int(resumo["n"])),
    )
    colunas_metricas[1].metric(
        "Visíveis no gráfico",
        formatar_percentual(proporcao_visivel),
    )
    colunas_metricas[2].metric(
        "Média (μ)",
        formatar_decimal(float(resumo["media"])),
    )
    colunas_metricas[3].metric(
        "Mediana",
        formatar_decimal(float(resumo["mediana"])),
    )
    colunas_metricas[4].metric(
        "Valores iguais a zero",
        formatar_percentual(float(resumo["proporcao_zeros"])),
    )

    st.pyplot(figura)
    plt.close(figura)

    segunda = str(resumo["segunda_distribuicao"])
    parametro_segunda = float(resumo["parametro_segunda"])
    parametros_segunda = (
        f"λ = {formatar_decimal(parametro_segunda, 6)}"
    )
    tabela_modelos = pd.DataFrame(
        [
            {
                "Modelo": "Normal",
                "Parâmetros estimados": (
                    f"μ = {formatar_decimal(float(resumo['media']))}; "
                    f"σ = {formatar_decimal(float(resumo['desvio_padrao']))}"
                ),
                "Distância D": float(resumo["distancia_normal"]),
            },
            {
                "Modelo": segunda,
                "Parâmetros estimados": parametros_segunda,
                "Distância D": float(resumo["distancia_segunda"]),
            },
        ]
    )
    st.dataframe(
        tabela_modelos,
        hide_index=True,
        width="stretch",
        column_config={
            "Distância D": st.column_config.NumberColumn(format="%.6f"),
        },
    )

    st.info(
        descrever_comparacao(resumo),
        icon=":material/compare_arrows:",
    )

    if configuracao["familia"] == "contagem":
        indice_dispersao = float(resumo["indice_dispersao"])

        if 0.8 <= indice_dispersao <= 1.2:
            st.markdown(
                f"**Dispersão da contagem:** variância/média = "
                f"**{formatar_decimal(indice_dispersao)}**. O valor está "
                "próximo de 1, uma condição compatível com a Poisson, embora "
                "isso sozinho não confirme o modelo."
            )
        elif indice_dispersao > 1.2:
            st.markdown(
                f"**Sobredispersão:** variância/média = "
                f"**{formatar_decimal(indice_dispersao)}**. A variabilidade "
                "é muito maior que a esperada por uma Poisson simples, o que "
                "é compatível com mistura de grupos, taxas diferentes e "
                "cauda longa."
            )
        else:
            st.markdown(
                f"**Subdispersão:** variância/média = "
                f"**{formatar_decimal(indice_dispersao)}**. A variabilidade "
                "é menor que a esperada pela Poisson simples."
            )
    else:
        razao = float(resumo["razao_desvio_media"])
        st.markdown(
            f"**Diagnóstico da Exponencial:** σ/μ = "
            f"**{formatar_decimal(razao)}**. Na Exponencial teórica essa "
            "razão é 1. A proximidade desse valor é apenas uma condição "
            "necessária de compatibilidade, não uma confirmação do modelo."
        )

    st.markdown(f"**Cautela específica:** {configuracao['cautela']}")

    if float(resumo["proporcao_zeros"]) >= 0.10:
        st.warning(
            "A variável possui uma concentração relevante exatamente em "
            "zero. Modelos simples de uma única família podem esconder a "
            "diferença entre não ocorrência e valores positivos.",
            icon=":material/filter_alt:",
        )

    if modo_visual == "Região central (IQR)":
        st.caption(
            "A região central foi ampliada apenas no gráfico. Os valores "
            "fora da janela permanecem na média, no desvio padrão, nos "
            "parâmetros e nas distâncias D. Por isso, a área visível do "
            "histograma pode ser menor que 100%."
        )

    with st.expander("Por que TTL não aparece nesta seleção?"):
        st.markdown(
            "`sttl` e `dttl` possuem poucos valores distintos e refletem "
            "configurações e comportamentos do protocolo. Elas formam grupos "
            "discretos, não uma medida contínua suave nem uma contagem de "
            "eventos homogêneos. Forçar Normal, Exponencial ou Poisson nesses "
            "campos produziria uma comparação visual de pouca utilidade. O "
            "Módulo 2 é mais adequado para examinar suas frequências."
        )

    st.caption(
        "Conclusões deste módulo descrevem a partição de treinamento do "
        "UNSW-NB15 carregada pela aplicação. A base é um conjunto de "
        "referência para pesquisa, não uma amostra aleatória de todo o "
        "tráfego real de redes."
    )
