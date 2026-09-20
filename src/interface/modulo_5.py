"""
Módulo 5 — Correlação e regressão linear simples.

Este arquivo concentra a preparação dos recortes, os cálculos por meio
do núcleo estatístico próprio e toda a apresentação interativa do módulo.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from nucleo import minhastats


descricoes_m5 = {
    "dur": "Duração do fluxo (s)",
    "spkts": "Pacotes enviados pela origem",
    "dpkts": "Pacotes enviados pelo destino",
    "sbytes": "Bytes enviados pela origem",
    "dbytes": "Bytes enviados pelo destino",
    "rate": "Taxa de pacotes (pacotes/s)",
    "sttl": "TTL dos pacotes de origem",
    "dttl": "TTL dos pacotes de destino",
}
variaveis_m5 = list(descricoes_m5)
variaveis_discretas_m5 = {
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
}
variaveis_ttl_m5 = {"sttl", "dttl"}
colunas_modulo_5 = tuple(variaveis_m5) + (
    "attack_cat",
    "label",
    "proto",
)


def colunas_ausentes_m5(dados):
    """Retorna as colunas obrigatórias do Módulo 5 ausentes no dataset."""
    return [
        coluna
        for coluna in colunas_modulo_5
        if coluna not in dados.columns
    ]

def rotulo_variavel_m5(nome):
    return f"{nome} — {descricoes_m5[nome]}"

def formatar_inteiro_m5(valor):
    return f"{valor:,}".replace(",", ".")

def selecionar_grupo_m5(dados, grupo):
    """Seleciona um grupo rotulado sem alterar o DataFrame original."""
    if grupo == "Todos os registros":
        return dados
    if grupo == "Todos os ataques":
        return dados.loc[dados["label"] == 1]
    return dados.loc[dados["attack_cat"] == grupo]

def filtrar_protocolo_m5(dados, protocolo):
    """Mantém todos os protocolos ou seleciona somente um deles."""
    if protocolo == "Todos os protocolos":
        return dados
    return dados.loc[dados["proto"] == protocolo]

def preparar_pares_m5(dados, nome_x, nome_y):
    """Preserva o pareamento e remove somente ausentes e infinitos."""
    pares = dados[[nome_x, nome_y]].copy()
    return pares.replace(
        [float("inf"), float("-inf")],
        float("nan"),
    ).dropna(subset=[nome_x, nome_y])

def calcular_resumo_m5(pares, nome_x, nome_y):
    """Calcula o ajuste com a biblioteca própria e registra seus limites."""
    resumo = {
        "n": len(pares),
        "x": [],
        "y": [],
        "x_minimo": None,
        "x_maximo": None,
        "x_constante": False,
        "y_constante": False,
        "inclinacao": None,
        "intercepto": None,
        "r": None,
        "r2": None,
        "y_estimado": [],
        "x_reta": [],
        "y_reta": [],
        "erro": None,
    }

    if resumo["n"] < 3:
        resumo["erro"] = "São necessários pelo menos três pares válidos."
        return resumo

    dados_x = pares[nome_x].tolist()
    dados_y = pares[nome_y].tolist()

    resumo["x"] = dados_x
    resumo["y"] = dados_y
    resumo["x_minimo"] = min(dados_x)
    resumo["x_maximo"] = max(dados_x)
    resumo["x_constante"] = all(
        valor == dados_x[0]
        for valor in dados_x
    )
    resumo["y_constante"] = all(
        valor == dados_y[0]
        for valor in dados_y
    )

    if resumo["x_constante"]:
        resumo["erro"] = (
            "X é constante; não existe uma inclinação única para a reta."
        )
        return resumo

    try:
        inclinacao, intercepto = minhastats.regressao_linear(
            dados_x,
            dados_y,
        )

        y_estimado = [
            minhastats.predicao_linear(
                valor_x,
                inclinacao,
                intercepto,
            )
            for valor_x in dados_x
        ]

        r_pearson = None
        r_quadrado = None

        if not resumo["y_constante"]:
            r_pearson = minhastats.correlacao_pearson(
                dados_x,
                dados_y,
            )
            r_quadrado = minhastats.coeficiente_determinacao(
                dados_y,
                y_estimado,
            )

        x_reta = [
            resumo["x_minimo"],
            resumo["x_maximo"],
        ]

        y_reta = [
            minhastats.predicao_linear(
                valor_x,
                inclinacao,
                intercepto,
            )
            for valor_x in x_reta
        ]

    except (TypeError, ValueError, ArithmeticError) as erro:
        resumo["erro"] = str(erro)
        return resumo

    resumo.update(
        {
            "inclinacao": inclinacao,
            "intercepto": intercepto,
            "r": r_pearson,
            "r2": r_quadrado,
            "y_estimado": y_estimado,
            "x_reta": x_reta,
            "y_reta": y_reta,
        }
    )

    return resumo

def linha_comparacao_m5(nome_grupo, resumo):
    if resumo["n"] < 3:
        situacao = "Menos de 3 pares"
    elif resumo["x_constante"]:
        situacao = "X constante"
    elif resumo["y_constante"]:
        situacao = "Y constante"
    elif resumo["erro"]:
        situacao = "Ajuste indisponível"
    else:
        situacao = "Calculado"

    return {
        "Grupo": nome_grupo,
        "Pares válidos": resumo["n"],
        "Pearson (r)": resumo["r"],
        "R²": resumo["r2"],
        "Inclinação (b)": resumo["inclinacao"],
        "Intercepto (a)": resumo["intercepto"],
        "Situação": situacao,
    }

def renderizar_modulo_5(df: pd.DataFrame) -> None:
    st.header(
        "Correlação e regressão linear",
        icon=":material/query_stats:",
    )
    st.markdown(
        "Explore a associação linear entre duas características do tráfego "
        "e ajuste uma reta por mínimos quadrados. Todos os resultados se "
        "referem ao grupo, ao protocolo e às variáveis selecionadas."
    )
    st.warning(
        "Correlação e regressão descrevem padrões nos dados, mas não "
        "demonstram causalidade, não calculam a probabilidade de ataque e "
        "não classificam um fluxo como normal ou malicioso.",
        icon=":material/warning:",
    )

    faltantes_m5 = colunas_ausentes_m5(df)

    if faltantes_m5:
        st.error(
            "O dataset não contém todas as colunas exigidas pelo Módulo 5: "
            + ", ".join(faltantes_m5)
        )
        return

    st.sidebar.header(
        "Configurações do Módulo 5",
        icon=":material/tune:",
    )

    categorias_ataque_m5 = sorted(
        categoria
        for categoria in df["attack_cat"].dropna().unique()
        if categoria != "Normal"
    )

    protocolos_m5 = sorted(
        df["proto"].dropna().unique()
    )

    grupo_m5 = st.sidebar.selectbox(
        "Grupo de tráfego",
        options=[
            "Todos os registros",
            "Normal",
            "Todos os ataques",
        ]
        + categorias_ataque_m5,
        key="modulo5_grupo",
        help=(
            "As classes são rótulos já fornecidos pelo dataset e servem "
            "para comparações descritivas entre grupos conhecidos."
        ),
    )

    protocolo_m5 = st.sidebar.selectbox(
        "Protocolo",
        options=[
            "Todos os protocolos",
        ]
        + protocolos_m5,
        key="modulo5_protocolo",
        help=(
            "Restringir o protocolo reduz a mistura de comportamentos "
            "de comunicação diferentes."
        ),
    )

    variavel_x = st.sidebar.selectbox(
        "Variável X — eixo horizontal",
        options=variaveis_m5,
        index=variaveis_m5.index("spkts"),
        format_func=rotulo_variavel_m5,
        key="modulo5_x",
        help="Variável explicativa usada pela reta para estimar Y.",
    )

    variavel_y = st.sidebar.selectbox(
        "Variável Y — resposta estimada",
        options=variaveis_m5,
        index=variaveis_m5.index("dpkts"),
        format_func=rotulo_variavel_m5,
        key="modulo5_y",
        help="Variável apresentada no eixo vertical e estimada pela reta.",
    )

    if variavel_x == variavel_y:
        st.warning(
            "Escolha variáveis diferentes para X e Y.",
            icon=":material/info:",
        )
        st.stop()

    with st.expander(
        "Roteiros recomendados para explorar o UNSW-NB15",
        icon=":material/lightbulb:",
    ):
        st.markdown(
            "Use os roteiros como perguntas de investigação. Mantenha X e Y "
            "fixos e altere um filtro por vez para que a comparação continue "
            "interpretável."
        )

        st.markdown(
            "1. **Exemplo principal — Normal × Exploits × Fuzzers, TCP, "
            "`spkts → dpkts`:** compara três grupos com amostras grandes, "
            "o mesmo protocolo e as mesmas variáveis. Observe que inclinações "
            "parecidas ainda podem acompanhar valores diferentes de r e R².\n"
            "2. **Alternativa — Normal × Generic, UDP, "
            "`spkts → dpkts`:** avalia uma categoria concentrada em UDP e "
            "evita misturá-la com protocolos pouco representativos.\n"
            "3. **Correlação não resume a reta — Normal × Fuzzers, TCP, "
            "`spkts → dpkts`:** compare simultaneamente Pearson, inclinação "
            "e R²; grupos podem ter associação linear e inclinações muito "
            "diferentes.\n"
            "4. **Relação estrutural — `spkts → sbytes` ou "
            "`dpkts → dbytes`:** mostra a ligação entre contagem de pacotes "
            "e volume de bytes, com atenção aos valores extremos.\n"
            "5. **Exemplo de cautela — Reconnaissance, TCP, "
            "`spkts → dpkts`:** uma correlação muito alta pode ser dominada "
            "por poucos pontos extremos; confira sempre o gráfico."
        )

        st.caption(
            "`dur → rate` exige cautela porque rate é uma variável derivada. "
            "`sttl` e `dttl` têm poucos valores distintos e podem ser mais "
            "bem exploradas por frequências do que por uma reta."
        )

    dados_recorte_m5 = selecionar_grupo_m5(
        df,
        grupo_m5,
    )

    dados_recorte_m5 = filtrar_protocolo_m5(
        dados_recorte_m5,
        protocolo_m5,
    )

    dados_pares = preparar_pares_m5(
        dados_recorte_m5,
        variavel_x,
        variavel_y,
    )

    total_registros = len(df)
    total_recorte = len(dados_recorte_m5)
    total_pares = len(dados_pares)
    total_excluidos = total_recorte - total_pares

    with st.container(border=True):
        st.subheader(
            "Escopo da análise",
            icon=":material/filter_alt:",
        )

        with st.container(horizontal=True):
            st.metric(
                "Registros no arquivo",
                formatar_inteiro_m5(total_registros),
                border=True,
            )
            st.metric(
                "Registros no recorte",
                formatar_inteiro_m5(total_recorte),
                border=True,
            )
            st.metric(
                "Pares válidos",
                formatar_inteiro_m5(total_pares),
                border=True,
            )
            st.metric(
                "Pares excluídos",
                formatar_inteiro_m5(total_excluidos),
                border=True,
                help="Valores ausentes ou infinitos em X ou Y.",
            )

        st.markdown(
            f"**Recorte atual:** {grupo_m5} · {protocolo_m5}  \n"
            f"**Pergunta analisada:** como **{variavel_x}** e "
            f"**{variavel_y}** variam linearmente em conjunto?"
        )

        st.caption(
            "Os filtros alteram a população analisada. Compare grupos "
            "mantendo o mesmo par X–Y e o mesmo protocolo."
        )

        if grupo_m5 != "Todos os registros":
            st.info(
                "O grupo foi definido por um rótulo já conhecido no dataset. "
                "Este recorte permite comparação descritiva, mas não simula "
                "a classificação de um tráfego novo.",
                icon=":material/label:",
            )

        if protocolo_m5 == "Todos os protocolos":
            st.caption(
                "O recorte mistura protocolos. Parte do resultado pode "
                "refletir diferenças entre protocolos, além da relação "
                "entre X e Y."
            )

        if total_excluidos > 0:
            st.warning(
                f"{formatar_inteiro_m5(total_excluidos)} registros foram "
                "excluídos porque X ou Y continha valor ausente ou infinito.",
                icon=":material/data_alert:",
            )

    if total_pares < 3:
        st.warning(
            "Este recorte possui menos de três pares válidos. Dois pontos "
            "definem uma reta perfeita, mas não sustentam uma análise "
            "estatística útil. Escolha outro grupo, protocolo ou par.",
            icon=":material/info:",
        )
        st.stop()

    resumo_m5 = calcular_resumo_m5(
        dados_pares,
        variavel_x,
        variavel_y,
    )

    dados_x = resumo_m5["x"]
    dados_y = resumo_m5["y"]
    x_minimo = resumo_m5["x_minimo"]
    x_maximo = resumo_m5["x_maximo"]
    inclinacao = resumo_m5["inclinacao"]
    intercepto = resumo_m5["intercepto"]
    r_pearson = resumo_m5["r"]
    r_quadrado = resumo_m5["r2"]
    y_estimado = resumo_m5["y_estimado"]
    x_reta = resumo_m5["x_reta"]
    y_reta = resumo_m5["y_reta"]

    if resumo_m5["x_constante"]:
        st.warning(
            "X é constante neste recorte. Não existe uma inclinação única "
            "para a regressão, e Pearson também é indefinido. O diagrama "
            "de dispersão permanece disponível.",
            icon=":material/info:",
        )

    elif resumo_m5["erro"]:
        st.error(
            f"Não foi possível calcular o ajuste: {resumo_m5['erro']}",
            icon=":material/error:",
        )

    if inclinacao is not None:
        with st.container(border=True):
            st.subheader(
                "Resultados do ajuste",
                icon=":material/analytics:",
            )

            with st.container(horizontal=True):
                st.metric(
                    "Pearson (r)",
                    (
                        "Indefinido"
                        if r_pearson is None
                        else f"{r_pearson:.6f}"
                    ),
                    border=True,
                )

                st.metric(
                    "R² no recorte",
                    (
                        "Indefinido"
                        if r_quadrado is None
                        else f"{r_quadrado:.6f}"
                    ),
                    border=True,
                )

                st.metric(
                    "Inclinação (b)",
                    f"{inclinacao:.6g}",
                    border=True,
                )

                st.metric(
                    "Intercepto (a)",
                    f"{intercepto:.6g}",
                    border=True,
                )

            with st.container(border=True):
                sinal = "+" if inclinacao >= 0 else "−"

                st.markdown(
                    f"**Equação ajustada:** Ŷ = {intercepto:.8g} "
                    f"{sinal} {abs(inclinacao):.8g} × X"
                )

                st.caption(
                    f"X representa {variavel_x}; Y representa {variavel_y}. "
                    "O arredondamento ocorre somente na apresentação."
                )

                if resumo_m5["y_constante"]:
                    st.warning(
                        "Y é constante neste recorte. A reta horizontal pode "
                        "ser calculada, mas Pearson e R² são indefinidos "
                        "porque não existe variação em Y.",
                        icon=":material/info:",
                    )

                else:
                    if r_pearson > 0:
                        direcao = "positiva"
                    elif r_pearson < 0:
                        direcao = "negativa"
                    else:
                        direcao = "nula"

                    st.markdown(
                        f"**Pearson:** a associação linear é **{direcao}** "
                        f"e tem valor r = **{r_pearson:.6f}** neste recorte. "
                        "O sinal informa a direção; o módulo informa o grau "
                        "de alinhamento linear. Um valor próximo de zero não "
                        "descarta relações não lineares ou subgrupos."
                    )

                    st.markdown(
                        f"**R²:** a reta representa aproximadamente "
                        f"**{100 * r_quadrado:.2f}%** da variação observada "
                        f"de {variavel_y} em torno de sua média, nos mesmos "
                        "dados usados no ajuste. Isso não é uma porcentagem "
                        "de previsões corretas."
                    )

                st.markdown(
                    f"**Inclinação:** ao percorrer a reta, um aumento de uma "
                    f"unidade em {variavel_x} altera Ŷ em "
                    f"{inclinacao:.6g} unidades de {variavel_y}. "
                    "Esse coeficiente descreve o ajuste e não um efeito causal."
                )

                st.markdown(
                    f"**Intercepto:** quando X = 0, a reta retorna "
                    f"{intercepto:.6g} para {variavel_y}."
                )

                if not x_minimo <= 0 <= x_maximo:
                    st.caption(
                        "X = 0 está fora da faixa observada neste recorte. "
                        "Nesse caso, o intercepto é necessário para definir "
                        "a reta, mas sua interpretação prática envolve "
                        "extrapolação."
                    )

                estimativas_negativas = sum(
                    valor < 0
                    for valor in y_estimado)

                if min(dados_y) >= 0 and estimativas_negativas > 0:
                    percentual_negativas = (
                        100 * estimativas_negativas / total_pares
                    )

                    st.warning(
                        f"A reta gera "
                        f"{formatar_inteiro_m5(estimativas_negativas)} "
                        f"estimativas negativas ({percentual_negativas:.2f}% "
                        "dos pares), embora Y seja uma variável não negativa. "
                        "Esse resultado é matematicamente possível, mas "
                        "indica limitação prática do modelo linear.",
                        icon=":material/warning:",
                    )

    with st.container(border=True):
        st.subheader(
            "Diagrama de dispersão e reta ajustada",
            icon=":material/scatter_plot:",
        )

        fig_m5, ax_m5 = plt.subplots(
            figsize=(10, 5.5)
        )

        ax_m5.scatter(
            dados_x,
            dados_y,
            s=10,
            alpha=0.25,
            color="steelblue",
            linewidths=0,
            rasterized=True,
            label="Registros observados",
        )

        if inclinacao is not None:
            ax_m5.plot(
                x_reta,
                y_reta,
                color="darkorange",
                linewidth=2,
                label="Regressão por mínimos quadrados",
            )

        ax_m5.set_xlabel(
            rotulo_variavel_m5(variavel_x)
        )

        ax_m5.set_ylabel(
            rotulo_variavel_m5(variavel_y)
        )

        ax_m5.set_title(
            f"{variavel_y} em função de {variavel_x} "
            f"— {grupo_m5} · {protocolo_m5}"
        )

        ax_m5.grid(alpha=0.2)
        ax_m5.legend()
        fig_m5.tight_layout()

        st.pyplot(
            fig_m5,
            width="stretch",
        )

        plt.close(fig_m5)

        st.caption(
            "O gráfico utiliza todos os pares válidos do recorte, em "
            "escala linear, mantendo zeros e valores extremos finitos. "
            "A reta é exibida entre o menor e o maior X observado."
        )

    with st.container(border=True):
        st.subheader(
            "Como interpretar sem produzir falsos diagnósticos",
            icon=":material/fact_check:",
        )

        st.markdown(
            f"**O que está sendo avaliado:** a associação linear entre "
            f"{variavel_x} e {variavel_y} nos "
            f"{formatar_inteiro_m5(total_pares)} pares do recorte atual."
        )

        st.markdown(
            "**O que o resultado pode fornecer:** direção da associação "
            "linear, uma reta descritiva, comparação entre recortes "
            "equivalentes e uma estimativa numérica de Y."
        )

        st.markdown(
            "**O que o resultado não fornece:** causa do comportamento, "
            "significância estatística, probabilidade de ataque, classe "
            "de um novo fluxo ou garantia de desempenho fora destes dados."
        )

        st.markdown(
            "**Como comparar filtros:** mantenha X e Y fixos, use o mesmo "
            "protocolo e altere apenas o grupo. Diferenças podem decorrer "
            "da composição, do tamanho do recorte, dos zeros ou dos extremos."
        )

        st.info(
            "**Exemplo do próprio dataset:** uma associação positiva entre "
            "`spkts` e `dpkts` não prova que enviar mais pacotes pela origem "
            "cause mais pacotes no destino, nem que um valor elevado de r "
            "identifique tráfego normal. Protocolo, serviço, estado da "
            "conexão, duração e construção do fluxo podem afetar as duas "
            "contagens. Os grupos de ataque usados aqui já estavam rotulados.",
            icon=":material/science:",
        )

        if r_quadrado is not None:
            st.caption(
                "R² e Pearson resumem aspectos específicos da relação. "
                "Mesmo valores elevados precisam ser avaliados junto ao "
                "diagrama de dispersão e, em uma análise ampliada, aos resíduos."
            )

    with st.container(border=True):
        st.subheader(
            "Comparação controlada entre três grupos",
            icon=":material/compare_arrows:",
        )

        st.markdown(
            "Compare **tráfego Normal e dois tipos específicos de ataque** "
            "mantendo X, Y e protocolo iguais. As categorias são mutuamente "
            "exclusivas; por isso, o agregado `Todos os ataques` não aparece "
            "nesta comparação."
        )

        mostrar_comparacao_m5 = st.toggle(
            "Ativar comparação orientada",
            value=False,
            key="modulo5_mostrar_comparacao",
            help=(
                "Calcula três ajustes com a biblioteca própria somente "
                "quando esta opção estiver ativada."
            ),
        )

        if mostrar_comparacao_m5:
            ataque_a_padrao = (
                categorias_ataque_m5.index("Exploits")
                if "Exploits" in categorias_ataque_m5
                else 0
            )

            ataque_b_padrao = (
                categorias_ataque_m5.index("Fuzzers")
                if "Fuzzers" in categorias_ataque_m5
                else min(1, len(categorias_ataque_m5) - 1)
            )

            opcoes_protocolo_comparacao = [
                "Todos os protocolos",
            ] + protocolos_m5

            protocolo_padrao = (
                opcoes_protocolo_comparacao.index("tcp")
                if "tcp" in opcoes_protocolo_comparacao
                else 0
            )

            with st.container(horizontal=True):
                ataque_a_comparacao = st.selectbox(
                    "Primeiro tipo de ataque",
                    options=categorias_ataque_m5,
                    index=ataque_a_padrao,
                    key="modulo5_ataque_a_comparacao",
                )

                ataque_b_comparacao = st.selectbox(
                    "Segundo tipo de ataque",
                    options=categorias_ataque_m5,
                    index=ataque_b_padrao,
                    key="modulo5_ataque_b_comparacao",
                )

                protocolo_comparacao = st.selectbox(
                    "Protocolo mantido nos três grupos",
                    options=opcoes_protocolo_comparacao,
                    index=protocolo_padrao,
                    key="modulo5_protocolo_comparacao",
                )

            if ataque_a_comparacao == ataque_b_comparacao:
                st.warning(
                    "Escolha dois tipos de ataque diferentes para formar "
                    "três grupos distintos.",
                    icon=":material/warning:",
                )

            else:
                if protocolo_comparacao == "Todos os protocolos":
                    st.warning(
                        "A comparação mistura protocolos. Para uma leitura "
                        "mais controlada, escolha TCP ou UDP conforme as "
                        "categorias selecionadas.",
                        icon=":material/warning:",
                    )

                grupos_comparacao_m5 = [
                    "Normal",
                    ataque_a_comparacao,
                    ataque_b_comparacao,
                ]

                pares_por_grupo_m5 = {}
                resumos_por_grupo_m5 = {}

                for nome_grupo_m5 in grupos_comparacao_m5:
                    dados_grupo_m5 = selecionar_grupo_m5(
                        df,
                        nome_grupo_m5,
                    )
                    dados_grupo_m5 = filtrar_protocolo_m5(
                        dados_grupo_m5,
                        protocolo_comparacao,
                    )
                    pares_grupo_m5 = preparar_pares_m5(
                        dados_grupo_m5,
                        variavel_x,
                        variavel_y,
                    )
                    pares_por_grupo_m5[nome_grupo_m5] = pares_grupo_m5
                    resumos_por_grupo_m5[nome_grupo_m5] = (
                        calcular_resumo_m5(
                            pares_grupo_m5,
                            variavel_x,
                            variavel_y,
                        )
                    )

                tabela_comparacao_m5 = pd.DataFrame(
                    [
                        linha_comparacao_m5(
                            nome_grupo_m5,
                            resumos_por_grupo_m5[nome_grupo_m5],
                        )
                        for nome_grupo_m5 in grupos_comparacao_m5
                    ]
                )

                st.dataframe(
                    tabela_comparacao_m5,
                    hide_index=True,
                    width="stretch",
                )

                cores_comparacao_m5 = [
                    "#2E7D32",
                    "#C62828",
                    "#EF6C00",
                ]

                figura_comparacao_m5, eixos_comparacao_m5 = plt.subplots(
                    1,
                    3,
                    figsize=(16, 4.8),
                    sharex=True,
                    sharey=True,
                    constrained_layout=True,
                )

                for eixo_m5, nome_grupo_m5, cor_m5 in zip(
                    eixos_comparacao_m5,
                    grupos_comparacao_m5,
                    cores_comparacao_m5,
                ):
                    pares_grupo_m5 = pares_por_grupo_m5[nome_grupo_m5]
                    resumo_grupo_m5 = resumos_por_grupo_m5[nome_grupo_m5]

                    eixo_m5.scatter(
                        pares_grupo_m5[variavel_x],
                        pares_grupo_m5[variavel_y],
                        s=7,
                        alpha=0.20,
                        color=cor_m5,
                        edgecolors="none",
                        rasterized=True,
                    )

                    if resumo_grupo_m5["x_reta"]:
                        eixo_m5.plot(
                            resumo_grupo_m5["x_reta"],
                            resumo_grupo_m5["y_reta"],
                            color="#111827",
                            linewidth=2.2,
                        )

                    eixo_m5.set_title(
                        f"{nome_grupo_m5} "
                        f"(n={formatar_inteiro_m5(resumo_grupo_m5['n'])})"
                    )
                    eixo_m5.set_xlabel(variavel_x)
                    eixo_m5.grid(alpha=0.18)

                eixos_comparacao_m5[0].set_ylabel(variavel_y)
                figura_comparacao_m5.suptitle(
                    f"{variavel_x} → {variavel_y} | "
                    f"protocolo: {protocolo_comparacao}"
                )

                st.pyplot(figura_comparacao_m5)
                plt.close(figura_comparacao_m5)

                st.caption(
                    "Os três painéis usam os mesmos limites dos eixos e todos "
                    "os pares válidos de cada grupo. Se poucos extremos "
                    "comprimirem a nuvem central, isso é um sinal para não "
                    "interpretar r ou R² isoladamente."
                )

                comparacao_calculavel = all(
                    resumo["r"] is not None
                    and resumo["r2"] is not None
                    and resumo["inclinacao"] is not None
                    for resumo in resumos_por_grupo_m5.values()
                )

                if comparacao_calculavel:
                    resumo_normal_m5 = resumos_por_grupo_m5["Normal"]
                    resumo_ataque_a_m5 = resumos_por_grupo_m5[
                        ataque_a_comparacao
                    ]
                    resumo_ataque_b_m5 = resumos_por_grupo_m5[
                        ataque_b_comparacao
                    ]

                    grupo_maior_alinhamento = max(
                        grupos_comparacao_m5,
                        key=lambda nome: abs(
                            resumos_por_grupo_m5[nome]["r"]
                        ),
                    )

                    st.markdown(
                        f"**Leitura de Pearson:** |r| é "
                        f"{abs(resumo_normal_m5['r']):.6f} em Normal, "
                        f"{abs(resumo_ataque_a_m5['r']):.6f} em "
                        f"{ataque_a_comparacao} e "
                        f"{abs(resumo_ataque_b_m5['r']):.6f} em "
                        f"{ataque_b_comparacao}. O maior alinhamento linear "
                        f"aparece em **{grupo_maior_alinhamento}**, mas essa "
                        "ordenação não é uma regra de classificação."
                    )

                    st.markdown(
                        f"**Leitura das inclinações:** a variação de Ŷ para "
                        f"cada unidade adicional de {variavel_x} é "
                        f"{resumo_normal_m5['inclinacao']:.6g} em Normal, "
                        f"{resumo_ataque_a_m5['inclinacao']:.6g} em "
                        f"{ataque_a_comparacao} e "
                        f"{resumo_ataque_b_m5['inclinacao']:.6g} em "
                        f"{ataque_b_comparacao}. Inclinações parecidas não "
                        "obrigam os grupos a ter a mesma dispersão ou o mesmo "
                        "R²."
                    )

                    st.markdown(
                        f"**Leitura do R²:** a reta representa "
                        f"{100 * resumo_normal_m5['r2']:.2f}% da variação de "
                        f"{variavel_y} em Normal, "
                        f"{100 * resumo_ataque_a_m5['r2']:.2f}% em "
                        f"{ataque_a_comparacao} e "
                        f"{100 * resumo_ataque_b_m5['r2']:.2f}% em "
                        f"{ataque_b_comparacao}, dentro das próprias amostras "
                        "usadas em cada ajuste."
                    )

                    st.markdown(
                        f"**Tamanhos dos recortes:** "
                        f"{formatar_inteiro_m5(resumo_normal_m5['n'])} pares "
                        f"em Normal, "
                        f"{formatar_inteiro_m5(resumo_ataque_a_m5['n'])} em "
                        f"{ataque_a_comparacao} e "
                        f"{formatar_inteiro_m5(resumo_ataque_b_m5['n'])} em "
                        f"{ataque_b_comparacao}. Os tamanhos dos recortes "
                        "devem ser considerados na comparação, mas o número "
                        "de observações, isoladamente, não determina a "
                        "estabilidade dos coeficientes."
                    )

                    st.info(
                        "A comparação mostra que a estrutura linear pode "
                        "mudar entre rótulos conhecidos mesmo quando o "
                        "protocolo e as variáveis são mantidos. Ela não prova "
                        "que o tipo de ataque causou a diferença, não testa "
                        "formalmente se os coeficientes diferem e não valida "
                        "um detector de intrusão.",
                        icon=":material/fact_check:",
                    )

                else:
                    grupos_indisponiveis = [
                        nome_grupo_m5
                        for nome_grupo_m5 in grupos_comparacao_m5
                        if (
                            resumos_por_grupo_m5[nome_grupo_m5]["r"]
                            is None
                            or resumos_por_grupo_m5[nome_grupo_m5]["r2"]
                            is None
                        )
                    ]

                    st.info(
                        "Não foi possível completar Pearson e R² em: "
                        f"{', '.join(grupos_indisponiveis)}. Ao menos uma "
                        "variável ficou constante ou não houve pares "
                        "suficientes nesse recorte. Isso também é um resultado "
                        "do filtro, não uma falha do cálculo.",
                        icon=":material/info:",
                    )

                st.caption(
                    "Escopo inferencial: esta é uma comparação descritiva. "
                    "Afirmar que as inclinações ou correlações populacionais "
                    "são diferentes exigiria intervalos de confiança ou um "
                    "teste específico, fora do escopo obrigatório do módulo."
                )
    with st.container(border=True):
        st.subheader(
            "Predição interativa de Y",
            icon=":material/calculate:",
        )

        st.caption(
            "A predição consulta a reta do recorte principal. Ela estima "
            "somente Y; não calcula risco, probabilidade ou classe de ataque."
        )

        if inclinacao is None:
            st.info(
                "A predição requer uma reta ajustada. Escolha uma variável "
                "X que apresente variação no recorte.",
                icon=":material/info:",
            )

        else:
            st.caption(
                f"X: {descricoes_m5[variavel_x]}. "
                f"Faixa observada: {x_minimo:.10g} a {x_maximo:.10g}."
            )

            x_informado = st.number_input(
                f"Informe X — {variavel_x}",
                value=None,
                step=(
                    1.0
                    if variavel_x in variaveis_discretas_m5
                    else 0.1
                ),
                format="%.10g",
                placeholder="Digite um valor para calcular Ŷ",
                key=(
                    f"modulo5_predicao_{variavel_x}_{variavel_y}_"
                    f"{grupo_m5}_{protocolo_m5}"
                ),
                help=(
                    "Confirme com Enter ou saia do campo. Valores fora "
                    "da faixa observada serão identificados como extrapolação."
                ),
            )

            if x_informado is None:
                st.info(
                    "Informe um valor de X para consultar a reta.",
                    icon=":material/info:",
                )

            elif x_informado < 0:
                st.error(
                    f"{variavel_x} não admite valor negativo neste módulo.",
                    icon=":material/error:",
                )

            elif (
                variavel_x in variaveis_discretas_m5
                and not float(x_informado).is_integer()
            ):
                st.error(
                    f"{variavel_x} representa uma quantidade inteira. "
                    "Informe um valor sem parte fracionária.",
                    icon=":material/error:",
                )

            elif (
                variavel_x in variaveis_ttl_m5
                and x_informado > 255
            ):
                st.error(
                    "TTL deve estar entre 0 e 255.",
                    icon=":material/error:",
                )

            else:
                try:
                    y_previsto = minhastats.predicao_linear(
                        x_informado,
                        inclinacao,
                        intercepto,
                    )

                except (TypeError, ValueError, ArithmeticError) as erro:
                    st.error(
                        f"Não foi possível calcular a predição: {erro}",
                        icon=":material/error:",
                    )

                else:
                    st.metric(
                        f"Ŷ estimado — {variavel_y}",
                        f"{y_previsto:.10g}",
                        help=descricoes_m5[variavel_y],
                        border=True,
                    )

                    st.caption(
                        f"Para {variavel_x} = {x_informado:.10g}, "
                        f"a reta estima {variavel_y} = {y_previsto:.10g}."
                    )

                    if (
                        x_informado < x_minimo
                        or x_informado > x_maximo
                    ):
                        st.warning(
                            "Extrapolação: X está fora da faixa observada "
                            "neste recorte. A relação estimada pode não se "
                            "manter nessa região.",
                            icon=":material/warning:",
                        )

                    else:
                        st.caption(
                            "X está dentro da faixa observada, mas isso "
                            "não garante que a estimativa seja adequada "
                            "para um registro individual."
                        )

                    if y_previsto < 0:
                        st.warning(
                            "A reta retornou um valor negativo para uma "
                            "variável não negativa. Preserve o resultado "
                            "matemático, mas não o interprete como uma "
                            "quantidade fisicamente observável.",
                            icon=":material/warning:",
                        )

                    elif (
                        variavel_y in variaveis_ttl_m5
                        and y_previsto > 255
                    ):
                        st.warning(
                            "A estimativa ultrapassa 255, limite do campo "
                            "TTL. Isso indica inadequação prática da reta "
                            "para esse valor de X.",
                            icon=":material/warning:",
                        )

                    if variavel_y in variaveis_discretas_m5:
                        st.caption(
                            "A regressão é contínua e pode produzir Ŷ "
                            "fracionário mesmo quando Y é uma contagem "
                            "ou outro campo inteiro."
                        )

                    if grupo_m5 != "Todos os registros":
                        st.caption(
                            "A estimativa está condicionada a uma classe "
                            "já conhecida. Ela não descobre a classe do fluxo."
                        )

                    st.caption(
                        "Ŷ é uma estimativa pontual, sem intervalo de "
                        "predição. Não representa uma observação real nem "
                        "uma garantia de resultado futuro."
                    )

    with st.expander(
        "Conferência dos pares utilizados no recorte principal",
        icon=":material/table_view:",
    ):
        st.caption(
            "A tabela mostra somente os dez primeiros pares válidos. "
            "Todos os pares informados no escopo participam dos cálculos."
        )

        st.dataframe(
            dados_pares.head(10),
            hide_index=True,
            width="stretch",
        )
