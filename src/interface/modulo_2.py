"""
Módulo 2 — Estatística descritiva interativa.

Este arquivo mantém a preparação e a apresentação do Módulo 2 separadas do
arquivo principal da aplicação. As medidas estatísticas exibidas continuam
sendo calculadas pelo núcleo próprio, definido em ``minhastats.py``.
"""

from __future__ import annotations

import math

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from nucleo import minhastats


VARIAVEIS_NUMERICAS = {
    "dur": {
        "rotulo": "Duração do fluxo",
        "descricao": "Tempo total registrado para o fluxo de rede.",
        "natureza": "Quantitativa contínua e não negativa",
        "unidade": "segundos",
    },
    "spkts": {
        "rotulo": "Pacotes da origem para o destino",
        "descricao": "Quantidade de pacotes enviados da origem ao destino.",
        "natureza": "Quantitativa discreta (contagem)",
        "unidade": "pacotes",
    },
    "dpkts": {
        "rotulo": "Pacotes do destino para a origem",
        "descricao": "Quantidade de pacotes enviados do destino à origem.",
        "natureza": "Quantitativa discreta (contagem)",
        "unidade": "pacotes",
    },
    "sbytes": {
        "rotulo": "Bytes da origem para o destino",
        "descricao": "Volume de dados transmitido da origem ao destino.",
        "natureza": "Quantitativa discreta de alta cardinalidade",
        "unidade": "bytes",
    },
    "dbytes": {
        "rotulo": "Bytes do destino para a origem",
        "descricao": "Volume de dados transmitido do destino à origem.",
        "natureza": "Quantitativa discreta de alta cardinalidade",
        "unidade": "bytes",
    },
    "rate": {
        "rotulo": "Taxa do fluxo",
        "descricao": "Taxa calculada de transmissão e recepção do fluxo.",
        "natureza": "Quantitativa contínua e não negativa",
        "unidade": "taxa registrada no dataset",
    },
    "sttl": {
        "rotulo": "TTL da origem para o destino",
        "descricao": "Valor de Time To Live no sentido origem-destino.",
        "natureza": "Quantitativa discreta com poucos valores",
        "unidade": "valor de TTL",
    },
    "dttl": {
        "rotulo": "TTL do destino para a origem",
        "descricao": "Valor de Time To Live no sentido destino-origem.",
        "natureza": "Quantitativa discreta com poucos valores",
        "unidade": "valor de TTL",
    },
}


VARIAVEIS_CATEGORICAS = {
    "proto": {
        "rotulo": "Protocolo",
        "descricao": "Protocolo associado à transação de rede.",
        "natureza": "Qualitativa nominal",
    },
    "service": {
        "rotulo": "Serviço de rede",
        "descricao": "Serviço de rede identificado no fluxo.",
        "natureza": "Qualitativa nominal",
    },
    "state": {
        "rotulo": "Estado do fluxo",
        "descricao": "Estado registrado para o fluxo e seu protocolo.",
        "natureza": "Qualitativa nominal",
    },
    "attack_cat": {
        "rotulo": "Categoria do tráfego",
        "descricao": "Categoria normal ou tipo de ataque atribuído ao fluxo.",
        "natureza": "Qualitativa nominal",
    },
}


VARIAVEIS_MODULO_2 = {
    **VARIAVEIS_NUMERICAS,
    **VARIAVEIS_CATEGORICAS,
}


def formatar_numero(valor: float | int) -> str:
    """Formata números para leitura em português sem perder valores pequenos."""
    numero = float(valor)
    absoluto = abs(numero)

    if numero == 0:
        return "0"

    if numero.is_integer() and absoluto < 1_000_000_000:
        texto = f"{int(numero):,}"
    elif absoluto >= 1_000_000_000 or absoluto < 0.000001:
        texto = f"{numero:.4e}"
    elif absoluto >= 1_000_000:
        texto = f"{numero:,.2f}"
    elif absoluto >= 1:
        texto = f"{numero:,.4f}"
    else:
        texto = f"{numero:.6f}"

    return (
        texto.replace(",", "TEMP")
        .replace(".", ",")
        .replace("TEMP", ".")
    )


def formatar_inteiro(valor: int) -> str:
    """Formata uma contagem inteira com separador de milhar em português."""
    return f"{valor:,}".replace(",", ".")


def formatar_percentual(valor: float, casas: int = 2) -> str:
    """Formata um percentual já multiplicado por cem."""
    return f"{valor:.{casas}f}%".replace(".", ",")


def colunas_ausentes(dados: pd.DataFrame) -> list[str]:
    """Retorna as colunas necessárias ao Módulo 2 que não estão no dataset."""
    return [
        coluna
        for coluna in VARIAVEIS_MODULO_2
        if coluna not in dados.columns
    ]


def validar_dados_numericos(dados: list[float | int]) -> None:
    """Impede cálculos descritivos sobre listas vazias ou valores não finitos."""
    if not dados:
        raise ValueError("A variável selecionada não possui valores válidos.")

    if any(not math.isfinite(float(valor)) for valor in dados):
        raise ValueError(
            "A variável selecionada possui valores NaN ou infinitos."
        )


def calcular_resumo_numerico(
    dados: list[float | int],
) -> dict[str, float | int | None]:
    """
    Calcula as medidas apresentadas no Módulo 2 com o núcleo próprio.

    As medidas de variância, desvio padrão e coeficiente de variação usam a
    versão populacional porque descrevem todos os registros da partição
    carregada pela aplicação.
    """
    validar_dados_numericos(dados)

    media = minhastats.media(dados)
    mediana = minhastats.mediana(dados)
    moda = minhastats.moda(dados)
    variancia = minhastats.variancia(dados, amostral=False)
    desvio_padrao = minhastats.desvio_padrao(dados, amostral=False)
    amplitude = minhastats.amplitude(dados)
    q1 = minhastats.percentil(dados, 25)
    q3 = minhastats.percentil(dados, 75)
    iqr = q3 - q1
    limite_inferior, limite_superior = minhastats.limites_iqr(dados)
    quantidade_outliers = sum(
        valor < limite_inferior or valor > limite_superior
        for valor in dados
    )

    if media == 0:
        coeficiente_variacao = None
    else:
        coeficiente_variacao = minhastats.coeficiente_variacao(
            dados,
            amostral=False,
        )

    return {
        "n": len(dados),
        "media": media,
        "mediana": mediana,
        "moda": moda,
        "variancia": variancia,
        "desvio_padrao": desvio_padrao,
        "amplitude": amplitude,
        "coeficiente_variacao": coeficiente_variacao,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "limite_inferior": limite_inferior,
        "limite_superior": limite_superior,
        "quantidade_outliers": quantidade_outliers,
        "percentual_outliers": quantidade_outliers / len(dados) * 100,
    }


def adicionar_percentuais(
    linhas: list[dict[str, object]],
    total: int,
) -> pd.DataFrame:
    """Acrescenta frequências relativa e acumulada a uma tabela."""
    acumulada = 0

    for linha in linhas:
        frequencia = int(linha["Frequência absoluta"])
        acumulada += frequencia
        linha["Frequência relativa (%)"] = frequencia / total * 100
        linha["Frequência acumulada (%)"] = acumulada / total * 100

    return pd.DataFrame(linhas)


def tabela_frequencias_por_valor(
    dados: list[float | int],
) -> pd.DataFrame:
    """Cria uma tabela ordenada para variáveis numéricas discretas."""
    contagem: dict[float | int, int] = {}

    for valor in dados:
        contagem[valor] = contagem.get(valor, 0) + 1

    linhas = [
        {
            "Classe ou valor": formatar_numero(valor),
            "Tipo": "Valor observado",
            "Frequência absoluta": frequencia,
        }
        for valor, frequencia in sorted(contagem.items())
    ]

    return adicionar_percentuais(linhas, len(dados))


def tabela_frequencias_numerica(
    dados: list[float | int],
    numero_classes: int = 15,
) -> pd.DataFrame:
    """
    Agrupa uma variável numérica preservando também os valores extremos.

    Quando há poucos valores distintos, cada valor recebe uma linha. Nos
    demais casos, a faixa delimitada pelas cercas do IQR é dividida em classes
    de mesma largura. Valores externos permanecem contabilizados em linhas
    próprias, portanto a tabela sempre representa todos os registros.
    """
    validar_dados_numericos(dados)

    if not 5 <= numero_classes <= 30:
        raise ValueError("O número de classes deve estar entre 5 e 30.")

    valores_distintos = set(dados)

    if len(valores_distintos) <= 20:
        return tabela_frequencias_por_valor(dados)

    minimo = min(dados)
    maximo = max(dados)
    limite_inferior, limite_superior = minhastats.limites_iqr(dados)
    inicio_central = max(minimo, limite_inferior)
    fim_central = min(maximo, limite_superior)

    if inicio_central == fim_central:
        return tabela_frequencias_por_valor(dados)

    largura = (fim_central - inicio_central) / numero_classes
    contagens = [0] * numero_classes
    abaixo = 0
    acima = 0

    for valor in dados:
        if valor < inicio_central:
            abaixo += 1
        elif valor > fim_central:
            acima += 1
        else:
            indice = int((valor - inicio_central) / largura)

            if indice == numero_classes:
                indice -= 1

            contagens[indice] += 1

    linhas: list[dict[str, object]] = []

    if abaixo:
        linhas.append(
            {
                "Classe ou valor": (
                    f"Abaixo de {formatar_numero(inicio_central)}"
                ),
                "Tipo": "Além da cerca inferior do IQR",
                "Frequência absoluta": abaixo,
            }
        )

    for indice, frequencia in enumerate(contagens):
        inicio = inicio_central + indice * largura
        fim = inicio + largura
        fechamento = "]" if indice == numero_classes - 1 else ")"
        linhas.append(
            {
                "Classe ou valor": (
                    f"[{formatar_numero(inicio)}; "
                    f"{formatar_numero(fim)}{fechamento}"
                ),
                "Tipo": "Classe na faixa central",
                "Frequência absoluta": frequencia,
            }
        )

    if acima:
        linhas.append(
            {
                "Classe ou valor": (
                    f"Acima de {formatar_numero(fim_central)}"
                ),
                "Tipo": "Além da cerca superior do IQR",
                "Frequência absoluta": acima,
            }
        )

    return adicionar_percentuais(linhas, len(dados))


def tabela_frequencias_categorica(dados: list[object]) -> pd.DataFrame:
    """Conta categorias sem usar atalhos estatísticos de bibliotecas."""
    if not dados:
        raise ValueError("A variável selecionada não possui valores válidos.")

    contagem: dict[object, int] = {}

    for valor in dados:
        contagem[valor] = contagem.get(valor, 0) + 1

    itens_ordenados = sorted(
        contagem.items(),
        key=lambda item: (-item[1], str(item[0])),
    )
    linhas = [
        {
            "Categoria": str(categoria),
            "Frequência absoluta": frequencia,
        }
        for categoria, frequencia in itens_ordenados
    ]

    return adicionar_percentuais(linhas, len(dados))


def dados_grafico_categorias(
    tabela: pd.DataFrame,
    limite: int,
) -> pd.DataFrame:
    """Mantém as categorias mais frequentes e agrega a cauda como Outros."""
    if limite <= 0:
        raise ValueError("O limite de categorias deve ser maior que zero.")

    principais = tabela.head(limite)[
        ["Categoria", "Frequência absoluta"]
    ].copy()
    restantes = tabela.iloc[limite:]
    frequencia_restante = sum(
        int(valor)
        for valor in restantes["Frequência absoluta"].tolist()
    )

    if frequencia_restante:
        principais.loc[len(principais)] = {
            "Categoria": "Outras categorias",
            "Frequência absoluta": frequencia_restante,
        }

    return principais


def exibir_metricas_numericas(
    resumo: dict[str, float | int | None],
) -> None:
    """Organiza as medidas do núcleo próprio em três linhas temáticas."""
    st.subheader(
        "Medidas calculadas pelo núcleo próprio",
        icon=":material/calculate:",
    )

    linha_central = st.columns(4)
    linha_central[0].metric(
        "Registros válidos",
        formatar_inteiro(int(resumo["n"])),
        border=True,
    )
    linha_central[1].metric(
        "Média",
        formatar_numero(float(resumo["media"])),
        border=True,
    )
    linha_central[2].metric(
        "Mediana",
        formatar_numero(float(resumo["mediana"])),
        border=True,
    )
    linha_central[3].metric(
        "Moda",
        formatar_numero(float(resumo["moda"])),
        border=True,
    )

    linha_dispersao = st.columns(4)
    linha_dispersao[0].metric(
        "Variância populacional",
        formatar_numero(float(resumo["variancia"])),
        border=True,
    )
    linha_dispersao[1].metric(
        "Desvio padrão populacional",
        formatar_numero(float(resumo["desvio_padrao"])),
        border=True,
    )
    linha_dispersao[2].metric(
        "Amplitude",
        formatar_numero(float(resumo["amplitude"])),
        border=True,
    )
    cv = resumo["coeficiente_variacao"]
    linha_dispersao[3].metric(
        "Coeficiente de variação",
        (
            "Indefinido"
            if cv is None
            else formatar_percentual(float(cv))
        ),
        border=True,
    )

    linha_posicao = st.columns(4)
    linha_posicao[0].metric(
        "Primeiro quartil (Q1)",
        formatar_numero(float(resumo["q1"])),
        border=True,
    )
    linha_posicao[1].metric(
        "Terceiro quartil (Q3)",
        formatar_numero(float(resumo["q3"])),
        border=True,
    )
    linha_posicao[2].metric(
        "Intervalo interquartílico",
        formatar_numero(float(resumo["iqr"])),
        border=True,
    )
    linha_posicao[3].metric(
        "Valores além das cercas do IQR",
        f"{formatar_inteiro(int(resumo['quantidade_outliers']))} · "
        f"{formatar_percentual(float(resumo['percentual_outliers']))}",
        border=True,
    )

    st.caption(
        "Variância, desvio padrão e coeficiente de variação descrevem a "
        "partição carregada e, por isso, usam divisor n."
    )


def exibir_interpretacao_numerica(
    dados: list[float | int],
    resumo: dict[str, float | int | None],
) -> None:
    """Explica assimetria, dispersão e limites da regra do IQR."""
    st.info(
        minhastats.interpretar_assimetria(dados),
        icon=":material/query_stats:",
    )

    percentual = float(resumo["percentual_outliers"])

    with st.container(border=True):
        st.markdown("#### Como interpretar os valores além das cercas do IQR")
        st.markdown(
            f"A regra utilizou o intervalo de "
            f"**{formatar_numero(float(resumo['limite_inferior']))}** a "
            f"**{formatar_numero(float(resumo['limite_superior']))}**. "
            f"Foram identificados "
            f"**{formatar_inteiro(int(resumo['quantidade_outliers']))}** "
            f"registros (**{formatar_percentual(percentual)}**)."
        )

        if percentual > 5:
            st.warning(
                "A proporção é elevada segundo a regra do IQR. Em dados de "
                "rede, isso pode refletir caudas longas, mistura de tipos de "
                "tráfego ou fluxos legitimamente intensos. O resultado não "
                "classifica um registro como ataque, erro ou anomalia.",
                icon=":material/warning:",
            )
        else:
            st.caption(
                "A regra do IQR é uma descrição univariada. Mesmo uma baixa "
                "proporção não comprova ausência de ataques ou anomalias."
            )

        st.caption(
            "O coeficiente de variação mede dispersão relativa. Ele deve ser "
            "interpretado com cautela quando a média está próxima de zero ou "
            "quando se comparam variáveis de naturezas diferentes."
        )


def exibir_graficos_numericos(
    dados: list[float | int],
    resumo: dict[str, float | int | None],
    variavel: str,
    numero_classes: int,
) -> None:
    """Mostra gráficos adequados à cardinalidade da variável numérica."""
    metadados = VARIAVEIS_NUMERICAS[variavel]
    valores_distintos = len(set(dados))

    st.subheader("Visualização da distribuição", icon=":material/bar_chart:")

    if valores_distintos <= 20:
        tabela_valores = tabela_frequencias_por_valor(dados)
        grafico = alt.Chart(tabela_valores).mark_bar().encode(
            x=alt.X(
                "Classe ou valor:N",
                title=f"{variavel} — {metadados['unidade']}",
                sort=None,
            ),
            y=alt.Y(
                "Frequência absoluta:Q",
                title="Frequência absoluta",
            ),
            tooltip=[
                alt.Tooltip("Classe ou valor:N", title="Valor"),
                alt.Tooltip(
                    "Frequência absoluta:Q",
                    title="Registros",
                    format=",",
                ),
                alt.Tooltip(
                    "Frequência relativa (%):Q",
                    title="Percentual",
                    format=".3f",
                ),
            ],
        ).properties(height=380)
        st.altair_chart(grafico)
        st.caption(
            "Como a variável possui poucos valores distintos, o gráfico "
            "apresenta a frequência de cada valor, sem criar intervalos "
            "artificiais."
        )
        return

    recorte_visual = st.segmented_control(
        "Faixa exibida nos gráficos",
        ["Todos os registros", "Faixa central pelo IQR"],
        default="Todos os registros",
        key="modulo2_recorte_visual",
    )
    dados_plot = dados

    if recorte_visual == "Faixa central pelo IQR":
        limite_inferior = float(resumo["limite_inferior"])
        limite_superior = float(resumo["limite_superior"])
        dados_plot = [
            valor
            for valor in dados
            if limite_inferior <= valor <= limite_superior
        ]
        removidos = len(dados) - len(dados_plot)
        st.caption(
            f"A ampliação central omite visualmente "
            f"{formatar_inteiro(removidos)} registros "
            "além das cercas do IQR. As medidas e a tabela continuam usando "
            "todos os registros."
        )
    else:
        st.caption(
            "Os gráficos usam todos os registros. Caudas longas podem "
            "comprimir a região de maior concentração; use a faixa central "
            "apenas para ampliar essa região."
        )

    figura, eixos = plt.subplots(1, 2, figsize=(15, 5))
    eixos[0].hist(
        dados_plot,
        bins=numero_classes,
        color="skyblue",
        edgecolor="black",
    )
    eixos[0].axvline(
        float(resumo["media"]),
        color="red",
        linestyle="dashed",
        linewidth=2,
        label="Média completa",
    )
    eixos[0].axvline(
        float(resumo["mediana"]),
        color="green",
        linestyle="dashed",
        linewidth=2,
        label="Mediana completa",
    )
    eixos[0].set_title("Histograma de frequências")
    eixos[0].set_xlabel(f"{variavel} — {metadados['unidade']}")
    eixos[0].set_ylabel("Frequência absoluta")
    eixos[0].legend()

    eixos[1].boxplot(dados_plot, orientation="horizontal")
    eixos[1].set_title("Boxplot da faixa exibida")
    eixos[1].set_xlabel(f"{variavel} — {metadados['unidade']}")

    figura.tight_layout()
    st.pyplot(figura)
    plt.close(figura)


def renderizar_numerica(
    dados: pd.DataFrame,
    variavel: str,
    numero_classes: int,
) -> None:
    """Renderiza a análise de uma variável numérica."""
    serie = dados[variavel].dropna()
    valores = serie.tolist()

    try:
        resumo = calcular_resumo_numerico(valores)
        tabela = tabela_frequencias_numerica(
            valores,
            numero_classes=numero_classes,
        )
    except ValueError as erro:
        st.error(str(erro), icon=":material/error:")
        return

    exibir_metricas_numericas(resumo)
    exibir_interpretacao_numerica(valores, resumo)
    exibir_graficos_numericos(
        valores,
        resumo,
        variavel,
        numero_classes,
    )

    st.subheader(
        "Tabela de frequências",
        icon=":material/table_chart:",
    )
    st.caption(
        "Variáveis com muitos valores são agrupadas na faixa central "
        "delimitada pelo IQR. Os registros externos permanecem em classes "
        "próprias, portanto a soma da tabela representa a base completa."
    )
    st.dataframe(
        tabela,
        hide_index=True,
        width="stretch",
        column_config={
            "Frequência absoluta": st.column_config.NumberColumn(
                format="localized",
            ),
            "Frequência relativa (%)": st.column_config.NumberColumn(
                format="%.3f%%",
            ),
            "Frequência acumulada (%)": st.column_config.NumberColumn(
                format="%.3f%%",
            ),
        },
    )


def renderizar_categorica(
    dados: pd.DataFrame,
    variavel: str,
    limite_grafico: int,
) -> None:
    """Renderiza moda, frequências e gráfico de uma variável categórica."""
    valores = dados[variavel].dropna().tolist()

    try:
        tabela = tabela_frequencias_categorica(valores)
    except ValueError as erro:
        st.error(str(erro), icon=":material/error:")
        return

    moda = minhastats.moda(valores)
    frequencia_moda = int(tabela.iloc[0]["Frequência absoluta"])
    percentual_moda = float(tabela.iloc[0]["Frequência relativa (%)"])

    st.subheader("Resumo categórico", icon=":material/category:")
    metricas = st.columns(4)
    metricas[0].metric(
        "Registros válidos",
        formatar_inteiro(len(valores)),
        border=True,
    )
    metricas[1].metric(
        "Categorias distintas",
        formatar_inteiro(len(tabela)),
        border=True,
    )
    metricas[2].metric(
        "Moda",
        str(moda),
        border=True,
    )
    metricas[3].metric(
        "Frequência da moda",
        f"{formatar_inteiro(frequencia_moda)} · "
        f"{formatar_percentual(percentual_moda)}",
        border=True,
    )

    dados_grafico = dados_grafico_categorias(
        tabela,
        limite=limite_grafico,
    )
    grafico = alt.Chart(dados_grafico).mark_bar().encode(
        x=alt.X(
            "Frequência absoluta:Q",
            title="Frequência absoluta",
        ),
        y=alt.Y(
            "Categoria:N",
            title=None,
            sort="-x",
        ),
        tooltip=[
            alt.Tooltip("Categoria:N"),
            alt.Tooltip(
                "Frequência absoluta:Q",
                title="Registros",
                format=",",
            ),
        ],
    ).properties(height=max(320, len(dados_grafico) * 30))

    st.subheader("Distribuição das categorias", icon=":material/bar_chart:")
    st.altair_chart(grafico)

    if len(tabela) > limite_grafico:
        st.caption(
            f"O gráfico mostra as {limite_grafico} categorias mais "
            f"frequentes e agrega as outras {len(tabela) - limite_grafico} "
            "em “Outras categorias”. A tabela abaixo permanece completa."
        )

    if variavel == "attack_cat":
        st.info(
            "As proporções descrevem esta partição do UNSW-NB15. Elas não "
            "estimam a prevalência de ataques em uma rede real e não devem "
            "ser usadas como taxa esperada de incidentes.",
            icon=":material/info:",
        )

    st.subheader("Tabela de frequências", icon=":material/table_chart:")
    st.dataframe(
        tabela,
        hide_index=True,
        width="stretch",
        column_config={
            "Frequência absoluta": st.column_config.NumberColumn(
                format="localized",
            ),
            "Frequência relativa (%)": st.column_config.NumberColumn(
                format="%.3f%%",
            ),
            "Frequência acumulada (%)": st.column_config.NumberColumn(
                format="%.3f%%",
            ),
        },
    )


def renderizar_modulo_2(dados: pd.DataFrame) -> None:
    """Renderiza o Módulo 2 completo na aplicação Streamlit."""
    faltantes = colunas_ausentes(dados)

    st.sidebar.header("Configurações do Módulo 2")
    variavel = st.sidebar.selectbox(
        "Escolha uma variável para análise",
        list(VARIAVEIS_MODULO_2),
        format_func=lambda nome: (
            f"{nome} — {VARIAVEIS_MODULO_2[nome]['rotulo']}"
        ),
        key="modulo2_variavel",
    )

    if faltantes:
        st.error(
            "O dataset não contém todas as colunas exigidas pelo Módulo 2: "
            + ", ".join(faltantes),
            icon=":material/error:",
        )
        return

    metadados = VARIAVEIS_MODULO_2[variavel]
    st.header("Estatística descritiva", icon=":material/query_stats:")
    st.markdown(
        "Explore a distribuição de uma característica dos registros de "
        "fluxo. Os cálculos apresentados vêm da biblioteca própria da equipe."
    )

    with st.container(border=True):
        st.subheader(f"{metadados['rotulo']} (`{variavel}`)")
        st.markdown(metadados["descricao"])
        st.caption(f"Natureza estatística: {metadados['natureza']}.")

    if variavel in VARIAVEIS_NUMERICAS:
        numero_classes = st.sidebar.slider(
            "Número de classes para histograma e tabela",
            min_value=5,
            max_value=30,
            value=15,
            step=1,
            key="modulo2_numero_classes",
        )
        renderizar_numerica(
            dados,
            variavel,
            numero_classes,
        )
    else:
        quantidade_categorias = len(
            set(dados[variavel].dropna().tolist())
        )
        maximo_grafico = max(5, min(20, quantidade_categorias))
        padrao_grafico = min(12, maximo_grafico)
        limite_grafico = st.sidebar.slider(
            "Categorias exibidas individualmente no gráfico",
            min_value=5,
            max_value=maximo_grafico,
            value=padrao_grafico,
            step=1,
            key="modulo2_limite_categorias",
        )
        renderizar_categorica(
            dados,
            variavel,
            limite_grafico,
        )
