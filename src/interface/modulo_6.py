from __future__ import annotations

from io import BytesIO
import math

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from nucleo import minhastats


COR_NORMAL = "#0072B2"
COR_ATAQUE = "#D55E00"
CORES_GRUPOS = {
    "Normal": COR_NORMAL,
    "Exploits": "#CC79A7",
    "Fuzzers": "#E69F00",
}
GRUPOS_COMPARACAO = ("Normal", "Exploits", "Fuzzers")
PROTOCOLO_COMPARACAO = "tcp"
VARIAVEL_X = "spkts"
VARIAVEL_Y = "dpkts"
COLUNAS_MODULO_6 = (
    "attack_cat",
    "proto",
    "dur",
    VARIAVEL_X,
    VARIAVEL_Y,
)


def formatar_inteiro(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def formatar_decimal(valor: float, casas: int = 2) -> str:
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def colunas_ausentes(dados: pd.DataFrame) -> list[str]:
    """Retorna as colunas necessárias ao relatório de descobertas."""
    return [
        coluna
        for coluna in COLUNAS_MODULO_6
        if coluna not in dados.columns
    ]


def figura_para_png(figura) -> bytes:
    arquivo = BytesIO()
    figura.savefig(
        arquivo,
        format="png",
        dpi=190,
        bbox_inches="tight",
        facecolor="white",
    )
    arquivo.seek(0)
    return arquivo.getvalue()


def adicionar_rodape(figura, texto: str) -> None:
    figura.text(
        0.01,
        0.01,
        texto,
        fontsize=8,
        color="#4B5563",
        ha="left",
    )


@st.cache_data(show_spinner=False)
def calcular_qualidade_dados(dados: pd.DataFrame) -> dict:
    colunas_sem_id = dados.drop(columns=["id"], errors="ignore")
    vetores_repetidos = int(
        colunas_sem_id.duplicated(keep="first").sum()
    )
    return {
        "registros": len(dados),
        "colunas": len(dados.columns),
        "ausentes": int(dados.isna().sum().sum()),
        "vetores_repetidos": vetores_repetidos,
        "percentual_vetores_repetidos": (
            100 * vetores_repetidos / len(dados)
            if len(dados)
            else 0.0
        ),
    }


@st.cache_data(show_spinner=False)
def calcular_composicao(dados: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    contagens: dict[str, int] = {}
    for categoria in dados["attack_cat"].dropna().tolist():
        contagens[categoria] = contagens.get(categoria, 0) + 1

    total = sum(contagens.values())
    linhas = []
    for categoria, quantidade in sorted(
        contagens.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        linhas.append(
            {
                "Categoria": categoria,
                "Classe": "Normal" if categoria == "Normal" else "Ataque",
                "Registros": quantidade,
                "Percentual": 100 * quantidade / total,
            }
        )

    tabela = pd.DataFrame(linhas)
    normal = contagens.get("Normal", 0)
    ataques = total - normal
    categorias_ataque = sum(
        1 for categoria in contagens if categoria != "Normal"
    )

    resumo = {
        "total": total,
        "normal": normal,
        "ataques": ataques,
        "percentual_normal": 100 * normal / total,
        "percentual_ataques": 100 * ataques / total,
        "razao_ataque_normal": ataques / normal,
        "categorias_ataque": categorias_ataque,
    }
    return tabela, resumo


@st.cache_data(show_spinner=False)
def calcular_resumo_duracao(dados: pd.DataFrame) -> dict:
    valores = dados["dur"].dropna().tolist()
    media = minhastats.media(valores)
    mediana = minhastats.mediana(valores)
    moda = minhastats.moda(valores)
    desvio = minhastats.desvio_padrao(valores, amostral=False)
    amplitude = minhastats.amplitude(valores)
    coeficiente_variacao = minhastats.coeficiente_variacao(
        valores,
        amostral=False,
    )
    q1 = minhastats.percentil(valores, 25)
    q3 = minhastats.percentil(valores, 75)
    limite_inferior, limite_superior = minhastats.limites_iqr(valores)
    outliers = minhastats.detectar_outliers(valores)
    outliers_por_classe = {}

    for nome, condicao in (
        ("Normal", dados["attack_cat"] == "Normal"),
        ("Ataques", dados["attack_cat"] != "Normal"),
    ):
        valores_grupo = dados.loc[condicao, "dur"].dropna().tolist()
        quantidade_grupo = len(valores_grupo)
        quantidade_outliers = sum(
            valor < limite_inferior or valor > limite_superior
            for valor in valores_grupo
        )
        outliers_por_classe[nome] = {
            "registros": quantidade_grupo,
            "outliers": quantidade_outliers,
            "percentual": (
                100 * quantidade_outliers / quantidade_grupo
                if quantidade_grupo
                else 0.0
            ),
        }

    return {
        "n": len(valores),
        "media": media,
        "mediana": mediana,
        "moda": moda,
        "desvio": desvio,
        "amplitude": amplitude,
        "coeficiente_variacao": coeficiente_variacao,
        "q1": q1,
        "q3": q3,
        "limite_inferior": limite_inferior,
        "limite_superior": limite_superior,
        "outliers": len(outliers),
        "percentual_outliers": 100 * len(outliers) / len(valores),
        "razao_media_mediana": media / mediana,
        "minimo": min(valores),
        "maximo": max(valores),
        "outliers_por_classe": outliers_por_classe,
    }


@st.cache_data(show_spinner=False)
def calcular_comparacao_tcp(
    dados: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    linhas = []
    pares_por_grupo: dict[str, pd.DataFrame] = {}

    for grupo in GRUPOS_COMPARACAO:
        pares = dados.loc[
            (dados["attack_cat"] == grupo)
            & (dados["proto"] == PROTOCOLO_COMPARACAO),
            [VARIAVEL_X, VARIAVEL_Y],
        ].copy()
        pares = pares.replace(
            [float("inf"), float("-inf")],
            float("nan"),
        ).dropna(subset=[VARIAVEL_X, VARIAVEL_Y])

        valores_x = pares[VARIAVEL_X].tolist()
        valores_y = pares[VARIAVEL_Y].tolist()

        inclinacao, intercepto = minhastats.regressao_linear(
            valores_x,
            valores_y,
        )
        estimados = [
            minhastats.predicao_linear(
                valor_x,
                inclinacao,
                intercepto,
            )
            for valor_x in valores_x
        ]
        correlacao = minhastats.correlacao_pearson(
            valores_x,
            valores_y,
        )
        r_quadrado = minhastats.coeficiente_determinacao(
            valores_y,
            estimados,
        )

        pares["estimado"] = estimados
        pares["residuo"] = [
            observado - estimado
            for observado, estimado in zip(valores_y, estimados)
        ]
        pares_por_grupo[grupo] = pares

        linhas.append(
            {
                "Grupo": grupo,
                "Pares": len(pares),
                "r": correlacao,
                "R²": r_quadrado,
                "Inclinação": inclinacao,
                "Intercepto": intercepto,
                "Equação": (
                    f"ŷ = {intercepto:.6f} "
                    f"{'+' if inclinacao >= 0 else '-'} "
                    f"{abs(inclinacao):.6f}x"
                ),
            }
        )

    return pd.DataFrame(linhas), pares_por_grupo


def calcular_sensibilidade_p99(
    tabela: pd.DataFrame,
    pares_por_grupo: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Recalcula os ajustes após retirar valores acima do P99 de X ou Y.

    A análise é diagnóstica. Ela não substitui o ajuste principal e não
    redefine valores extremos como erros; apenas mede a sensibilidade dos
    coeficientes à cauda superior de cada grupo.
    """
    linhas = []

    for grupo in GRUPOS_COMPARACAO:
        pares = pares_por_grupo[grupo]
        valores_x = pares[VARIAVEL_X].tolist()
        valores_y = pares[VARIAVEL_Y].tolist()
        limite_x = minhastats.percentil(valores_x, 99)
        limite_y = minhastats.percentil(valores_y, 99)
        centrais = pares.loc[
            (pares[VARIAVEL_X] <= limite_x)
            & (pares[VARIAVEL_Y] <= limite_y)
        ]
        x_central = centrais[VARIAVEL_X].tolist()
        y_central = centrais[VARIAVEL_Y].tolist()
        inclinacao_central, intercepto_central = (
            minhastats.regressao_linear(x_central, y_central)
        )
        estimados_centrais = [
            minhastats.predicao_linear(
                valor_x,
                inclinacao_central,
                intercepto_central,
            )
            for valor_x in x_central
        ]
        r_central = minhastats.correlacao_pearson(
            x_central,
            y_central,
        )
        r2_central = minhastats.coeficiente_determinacao(
            y_central,
            estimados_centrais,
        )
        linha_completa = tabela.loc[
            tabela["Grupo"] == grupo
        ].iloc[0]

        linhas.append(
            {
                "Grupo": grupo,
                "Pares completos": len(pares),
                "Pares até P99": len(centrais),
                "Retenção (%)": 100 * len(centrais) / len(pares),
                "P99 de spkts": limite_x,
                "P99 de dpkts": limite_y,
                "r completo": linha_completa["r"],
                "r até P99": r_central,
                "R² completo": linha_completa["R²"],
                "R² até P99": r2_central,
                "Inclinação completa": linha_completa["Inclinação"],
                "Inclinação até P99": inclinacao_central,
            }
        )

    return pd.DataFrame(linhas)


def criar_figura_composicao(
    tabela: pd.DataFrame,
    resumo: dict,
):
    figura, eixos = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        gridspec_kw={"width_ratios": [1.8, 1]},
    )

    categorias = tabela["Categoria"].tolist()
    percentuais = tabela["Percentual"].tolist()
    quantidades = tabela["Registros"].tolist()
    cores = [
        COR_NORMAL if classe == "Normal" else COR_ATAQUE
        for classe in tabela["Classe"].tolist()
    ]

    barras = eixos[0].barh(categorias, percentuais, color=cores)
    eixos[0].invert_yaxis()
    eixos[0].set_xlabel("Participação na partição (%)")
    eixos[0].set_title("Distribuição por categoria")
    eixos[0].grid(axis="x", alpha=0.22)
    eixos[0].set_axisbelow(True)
    eixos[0].set_xlim(0, max(percentuais) * 1.36)

    for barra, quantidade, percentual in zip(
        barras,
        quantidades,
        percentuais,
    ):
        eixos[0].text(
            barra.get_width() + 0.25,
            barra.get_y() + barra.get_height() / 2,
            f"{formatar_inteiro(quantidade)} ({percentual:.2f}%)",
            va="center",
            fontsize=9,
        )

    normal = resumo["percentual_normal"]
    ataques = resumo["percentual_ataques"]
    eixos[1].barh(["Registros"], [normal], color=COR_NORMAL, label="Normal")
    eixos[1].barh(
        ["Registros"],
        [ataques],
        left=[normal],
        color=COR_ATAQUE,
        label="Ataques",
    )
    eixos[1].text(
        normal / 2,
        0,
        f"Normal\n{normal:.2f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )
    eixos[1].text(
        normal + ataques / 2,
        0,
        f"Ataques\n{ataques:.2f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )
    eixos[1].set_xlim(0, 100)
    eixos[1].set_xlabel("Participação na partição (%)")
    eixos[1].set_title("Agregação binária")
    eixos[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=2)

    adicionar_rodape(
        figura,
        "Fonte: UNSW-NB15, partição analisada. Frequências calculadas pela aplicação.",
    )
    figura.tight_layout(rect=(0, 0.05, 1, 1))
    return figura


def criar_figura_duracao(valores: list[float], resumo: dict):
    valores_log = [math.log1p(valor) for valor in valores]
    figura, eixos = plt.subplots(1, 3, figsize=(17, 5.2))

    eixos[0].hist(
        valores,
        bins=60,
        color="#56B4E9",
        edgecolor="white",
    )
    eixos[0].axvline(
        resumo["media"],
        color="#D55E00",
        linestyle="--",
        linewidth=2,
        label="Média",
    )
    eixos[0].axvline(
        resumo["mediana"],
        color="#009E73",
        linestyle="-.",
        linewidth=2,
        label="Mediana",
    )
    eixos[0].set_yscale("log")
    eixos[0].set_title("Escala original")
    eixos[0].set_xlabel("dur")
    eixos[0].set_ylabel("Frequência (escala logarítmica)")
    eixos[0].legend()
    eixos[0].grid(alpha=0.18)

    eixos[1].hist(
        valores_log,
        bins=60,
        color="#009E73",
        edgecolor="white",
    )
    eixos[1].axvline(
        math.log1p(resumo["media"]),
        color="#D55E00",
        linestyle="--",
        linewidth=2,
        label="log1p da média bruta",
    )
    eixos[1].axvline(
        math.log1p(resumo["mediana"]),
        color="#0072B2",
        linestyle="-.",
        linewidth=2,
        label="log1p da mediana bruta",
    )
    eixos[1].set_title("Transformação visual log1p(dur)")
    eixos[1].set_xlabel("log(1 + dur)")
    eixos[1].set_ylabel("Frequência")
    eixos[1].legend(fontsize=8)
    eixos[1].grid(alpha=0.18)

    eixos[2].boxplot(
        valores,
        orientation="horizontal",
        showfliers=True,
        flierprops={
            "marker": ".",
            "markersize": 2,
            "alpha": 0.22,
            "markerfacecolor": COR_ATAQUE,
            "markeredgecolor": COR_ATAQUE,
        },
        boxprops={"color": COR_NORMAL},
        medianprops={"color": "#111827", "linewidth": 2},
    )
    eixos[2].axvline(
        resumo["limite_superior"],
        color=COR_ATAQUE,
        linestyle="--",
        linewidth=1.8,
        label="Limite superior do IQR",
    )
    eixos[2].set_title("Boxplot e limite do IQR")
    eixos[2].set_xlabel("dur")
    eixos[2].set_yticks([])
    eixos[2].legend(fontsize=8)
    eixos[2].grid(axis="x", alpha=0.18)

    adicionar_rodape(
        figura,
        "Medidas calculadas por minhastats.py. A transformação log1p é usada apenas para visualização.",
    )
    figura.tight_layout(rect=(0, 0.05, 1, 1))
    return figura


def criar_figura_coeficientes(tabela: pd.DataFrame):
    figura, eixos = plt.subplots(1, 3, figsize=(15.5, 4.8))
    grupos = tabela["Grupo"].tolist()
    cores = [CORES_GRUPOS[grupo] for grupo in grupos]

    configuracoes = (
        ("r", "Correlação de Pearson", "r"),
        ("R²", "Coeficiente de determinação", "R²"),
        ("Inclinação", "Inclinação da reta", "b"),
    )

    for eixo, (coluna, titulo, rotulo_y) in zip(eixos, configuracoes):
        valores = tabela[coluna].tolist()
        barras = eixo.bar(grupos, valores, color=cores)
        eixo.axhline(0, color="#111827", linewidth=0.8)
        eixo.set_title(titulo)
        eixo.set_ylabel(rotulo_y)
        eixo.grid(axis="y", alpha=0.20)
        eixo.set_axisbelow(True)

        menor = min(0.0, min(valores))
        maior = max(0.0, max(valores))
        margem = max((maior - menor) * 0.18, 0.08)
        eixo.set_ylim(menor - margem, maior + margem)

        for barra, valor in zip(barras, valores):
            deslocamento = margem * 0.10
            eixo.text(
                barra.get_x() + barra.get_width() / 2,
                valor + deslocamento if valor >= 0 else valor - deslocamento,
                f"{valor:.3f}",
                ha="center",
                va="bottom" if valor >= 0 else "top",
                fontsize=9,
                fontweight="bold",
            )

    adicionar_rodape(
        figura,
        "Recorte: protocolo TCP, X=spkts e Y=dpkts. Coeficientes calculados por minhastats.py.",
    )
    figura.tight_layout(rect=(0, 0.06, 1, 1))
    return figura


def criar_figura_regressoes(
    tabela: pd.DataFrame,
    pares_por_grupo: dict[str, pd.DataFrame],
    aplicar_zoom: bool,
):
    figura, eixos = plt.subplots(
        1,
        3,
        figsize=(17, 5.3),
        sharex=True,
        sharey=True,
    )

    limite_x = None
    limite_y = None
    if aplicar_zoom:
        valores_x = []
        valores_y = []
        for pares in pares_por_grupo.values():
            valores_x.extend(pares[VARIAVEL_X].tolist())
            valores_y.extend(pares[VARIAVEL_Y].tolist())
        limite_x = minhastats.percentil(valores_x, 99)
        limite_y = minhastats.percentil(valores_y, 99)

    for eixo, grupo in zip(eixos, GRUPOS_COMPARACAO):
        pares = pares_por_grupo[grupo]
        linha = tabela.loc[tabela["Grupo"] == grupo].iloc[0]
        eixo.scatter(
            pares[VARIAVEL_X],
            pares[VARIAVEL_Y],
            s=7,
            alpha=0.18,
            color=CORES_GRUPOS[grupo],
            edgecolors="none",
            rasterized=True,
        )

        x_minimo = min(pares[VARIAVEL_X].tolist())
        x_maximo = max(pares[VARIAVEL_X].tolist())
        if aplicar_zoom and limite_x is not None:
            x_maximo = min(x_maximo, limite_x)
        x_reta = [x_minimo, x_maximo]
        y_reta = [
            minhastats.predicao_linear(
                valor_x,
                linha["Inclinação"],
                linha["Intercepto"],
            )
            for valor_x in x_reta
        ]
        eixo.plot(x_reta, y_reta, color="#111827", linewidth=2.2)
        eixo.set_title(f"{grupo} (n={formatar_inteiro(len(pares))})")
        eixo.set_xlabel(VARIAVEL_X)
        eixo.grid(alpha=0.18)

        if aplicar_zoom and limite_x is not None and limite_y is not None:
            eixo.set_xlim(0, limite_x)
            eixo.set_ylim(0, limite_y)

    eixos[0].set_ylabel(VARIAVEL_Y)
    titulo = "Relação spkts → dpkts em tráfego TCP"
    if aplicar_zoom:
        titulo += " — zoom visual até P99"
    figura.suptitle(titulo, fontsize=14, fontweight="bold")
    adicionar_rodape(
        figura,
        "As regressões usam todos os pares. Quando ativo, o zoom altera somente os limites visuais dos eixos.",
    )
    figura.tight_layout(rect=(0, 0.06, 1, 0.94))
    return figura


def criar_figura_residuos(
    pares_por_grupo: dict[str, pd.DataFrame],
):
    figura, eixos = plt.subplots(
        1,
        3,
        figsize=(17, 4.8),
        sharey=False,
    )

    for eixo, grupo in zip(eixos, GRUPOS_COMPARACAO):
        pares = pares_por_grupo[grupo]
        passo = max(1, len(pares) // 6000)
        amostra_visual = pares.iloc[::passo]
        eixo.scatter(
            amostra_visual["estimado"],
            amostra_visual["residuo"],
            s=7,
            alpha=0.22,
            color=CORES_GRUPOS[grupo],
            edgecolors="none",
            rasterized=True,
        )
        eixo.axhline(0, color="#111827", linestyle="--", linewidth=1.4)
        eixo.set_title(grupo)
        eixo.set_xlabel("dpkts estimado")
        eixo.grid(alpha=0.18)

    eixos[0].set_ylabel("Resíduo: observado − estimado")
    figura.suptitle("Diagnóstico visual dos resíduos", fontsize=14, fontweight="bold")
    adicionar_rodape(
        figura,
        "Amostra visual determinística de até cerca de 6.000 pontos por grupo; os coeficientes usam todos os pares.",
    )
    figura.tight_layout(rect=(0, 0.06, 1, 0.94))
    return figura


def exibir_botao_download(
    figura,
    nome_arquivo: str,
    chave: str,
) -> None:
    st.download_button(
        "Baixar figura em PNG",
        data=figura_para_png(figura),
        file_name=nome_arquivo,
        mime="image/png",
        icon=":material/download:",
        key=chave,
    )


def renderizar_modulo_6(dados: pd.DataFrame) -> None:
    st.header(
        "Relatório de descobertas",
        icon=":material/analytics:",
    )
    st.markdown(
        "Esta página reúne três evidências reproduzíveis obtidas com o "
        "laboratório: composição das classes, distribuição de `dur` e "
        "comparação da relação `spkts → dpkts` entre grupos TCP."
    )
    st.warning(
        "O painel é descritivo. As associações observadas não demonstram "
        "causalidade, não estimam a prevalência de ataques em redes reais "
        "e não constituem um detector de intrusão.",
        icon=":material/warning:",
    )

    faltantes = colunas_ausentes(dados)

    if faltantes:
        st.error(
            "O dataset não contém todas as colunas exigidas pelo Módulo 6: "
            + ", ".join(faltantes)
        )
        return

    qualidade = calcular_qualidade_dados(dados)
    tabela_composicao, composicao = calcular_composicao(dados)
    resumo_duracao = calcular_resumo_duracao(dados)
    tabela_regressao, pares_por_grupo = calcular_comparacao_tcp(dados)
    tabela_sensibilidade = calcular_sensibilidade_p99(
        tabela_regressao,
        pares_por_grupo,
    )

    colunas_resumo = st.columns(4)
    with colunas_resumo[0]:
        st.metric(
            "Registros analisados",
            formatar_inteiro(qualidade["registros"]),
            border=True,
        )
    with colunas_resumo[1]:
        st.metric(
            "Ataques agregados",
            (
                f"{formatar_inteiro(composicao['ataques'])} · "
                f"{composicao['percentual_ataques']:.2f}%"
            ),
            border=True,
        )
    with colunas_resumo[2]:
        st.metric(
            "Tráfego Normal",
            (
                f"{formatar_inteiro(composicao['normal'])} · "
                f"{composicao['percentual_normal']:.2f}%"
            ),
            border=True,
        )
    with colunas_resumo[3]:
        st.metric(
            "Categorias de ataque",
            composicao["categorias_ataque"],
            border=True,
        )

    with st.expander(
        "Procedência, qualidade e unidade de análise",
        icon=":material/database:",
    ):
        st.markdown(
            "A fonte oficial descreve o UNSW-NB15 como um conjunto criado "
            "em laboratório com atividades normais e ataques sintéticos. "
            "A partição de 175.341 registros corresponde à partição oficial "
            "de treinamento, conforme sua cardinalidade e composição de "
            "classes. Cada linha é tratada como um registro de fluxo; não "
            "como um pacote individual."
        )
        st.markdown(
            "- [Fonte oficial do UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)\n"
            "- [Artigo original](https://doi.org/10.1109/MilCIS.2015.7348942)"
        )
        st.table(
            {
                "Registros": formatar_inteiro(qualidade["registros"]),
                "Colunas": formatar_inteiro(qualidade["colunas"]),
                "Valores ausentes": formatar_inteiro(qualidade["ausentes"]),
                "Vetores repetidos sem id": formatar_inteiro(
                    qualidade["vetores_repetidos"]
                ),
                "Percentual de vetores repetidos": (
                    f"{qualidade['percentual_vetores_repetidos']:.2f}%"
                ),
            },
            border="horizontal",
            width="content",
        )
        st.caption(
            "Vetores repetidos não foram removidos: sem os campos originais "
            "de endereço e tempo, não há base para afirmar que representem "
            "duplicações indevidas. Essa repetição limita a suposição de "
            "independência entre todos os registros."
        )

    st.subheader(
        "Mapa das evidências",
        icon=":material/fact_check:",
    )
    st.table(
        pd.DataFrame(
            [
                {
                    "Evidência": "Composição das classes",
                    "Sustenta": "Caracterização da partição analisada",
                    "Não sustenta": "Prevalência em redes reais",
                },
                {
                    "Evidência": "Distribuição de dur",
                    "Sustenta": "Assimetria e sensibilidade da média",
                    "Não sustenta": "Outlier igual a ataque",
                },
                {
                    "Evidência": "spkts → dpkts em TCP",
                    "Sustenta": "Diferenças descritivas entre grupos rotulados",
                    "Não sustenta": "Causalidade ou classificação automática",
                },
            ]
        ),
        border="horizontal",
    )

    with st.container(border=True):
        st.subheader(
            "Descoberta 1 — composição desigual das classes",
            icon=":material/bar_chart:",
        )
        st.markdown(
            "A agregação das nove categorias de ataque produz "
            f"**{formatar_inteiro(composicao['ataques'])} registros "
            f"({composicao['percentual_ataques']:.2f}%)**, contra "
            f"**{formatar_inteiro(composicao['normal'])} registros "
            f"Normais ({composicao['percentual_normal']:.2f}%)**. "
            f"A razão é de **{composicao['razao_ataque_normal']:.2f} para 1**."
        )

        figura_composicao = criar_figura_composicao(
            tabela_composicao,
            composicao,
        )
        st.pyplot(figura_composicao)
        exibir_botao_download(
            figura_composicao,
            "modulo6_composicao_classes.png",
            "m6_download_composicao",
        )
        plt.close(figura_composicao)

        tabela_composicao_exibicao = tabela_composicao.copy()
        tabela_composicao_exibicao["Registros"] = [
            formatar_inteiro(valor)
            for valor in tabela_composicao["Registros"].tolist()
        ]
        tabela_composicao_exibicao["Percentual"] = [
            f"{valor:.2f}%"
            for valor in tabela_composicao["Percentual"].tolist()
        ]
        st.dataframe(
            tabela_composicao_exibicao,
            hide_index=True,
            width="stretch",
        )

        st.markdown(
            "**Leitura técnica.** A divisão binária é desigual, mas a "
            "heterogeneidade mais severa ocorre entre as categorias de "
            "ataque. Generic e Exploits concentram grande parte dos casos, "
            "enquanto Worms possui somente 130 registros. Uma avaliação "
            "multiclasse futura deve apresentar métricas por categoria, pois "
            "uma medida global pode ocultar desempenho ruim nas classes raras."
        )
        st.warning(
            "68,06% é a frequência relativa dentro desta partição construída "
            "para pesquisa. Não é a probabilidade de ataque em uma rede real.",
            icon=":material/report:",
        )

    with st.container(border=True):
        st.subheader(
            "Descoberta 2 — duração fortemente assimétrica",
            icon=":material/stacked_line_chart:",
        )
        with st.container(horizontal=True):
            st.metric(
                "Média de dur",
                formatar_decimal(resumo_duracao["media"], 6),
                border=True,
            )
            st.metric(
                "Mediana de dur",
                formatar_decimal(resumo_duracao["mediana"], 6),
                border=True,
            )
            st.metric(
                "Coeficiente de variação",
                f"{formatar_decimal(resumo_duracao['coeficiente_variacao'], 2)}%",
                border=True,
            )
            st.metric(
                "Outliers pelo IQR",
                formatar_inteiro(resumo_duracao["outliers"]),
                f"{resumo_duracao['percentual_outliers']:.2f}% dos registros",
                border=True,
            )

        valores_duracao = dados["dur"].dropna().tolist()
        figura_duracao = criar_figura_duracao(
            valores_duracao,
            resumo_duracao,
        )
        st.pyplot(figura_duracao)
        exibir_botao_download(
            figura_duracao,
            "modulo6_distribuicao_dur.png",
            "m6_download_duracao",
        )
        plt.close(figura_duracao)

        st.table(
            {
                "Média": formatar_decimal(resumo_duracao["media"], 6),
                "Mediana": formatar_decimal(resumo_duracao["mediana"], 6),
                "Moda": formatar_decimal(resumo_duracao["moda"], 6),
                "Desvio-padrão populacional": formatar_decimal(
                    resumo_duracao["desvio"], 6
                ),
                "Amplitude": formatar_decimal(
                    resumo_duracao["amplitude"], 6
                ),
                "Q1": formatar_decimal(resumo_duracao["q1"], 6),
                "Q3": formatar_decimal(resumo_duracao["q3"], 6),
                "Limite superior do IQR": formatar_decimal(
                    resumo_duracao["limite_superior"], 6
                ),
            },
            border="horizontal",
        )

        st.markdown(
            "**Leitura técnica.** A média é aproximadamente "
            f"**{resumo_duracao['razao_media_mediana']:.2f} vezes** a "
            "mediana. Essa diferença, combinada com o coeficiente de "
            f"variação de {resumo_duracao['coeficiente_variacao']:.2f}% e "
            "com a cauda superior do histograma, "
            "mostra que a média isolada não representa o registro típico. "
            "A mediana e o intervalo interquartil são medidas mais robustas "
            "para descrever o centro dessa distribuição."
        )

        outliers_por_classe = resumo_duracao["outliers_por_classe"]
        tabela_outliers = pd.DataFrame(
            [
                {
                    "Grupo": grupo,
                    "Registros": valores["registros"],
                    "Além das cercas do IQR": valores["outliers"],
                    "Percentual no grupo": valores["percentual"],
                }
                for grupo, valores in outliers_por_classe.items()
            ]
        )
        st.dataframe(
            tabela_outliers,
            hide_index=True,
            width="stretch",
            column_config={
                "Registros": st.column_config.NumberColumn(format="%d"),
                "Além das cercas do IQR": (
                    st.column_config.NumberColumn(format="%d")
                ),
                "Percentual no grupo": (
                    st.column_config.NumberColumn(format="%.2f%%")
                ),
            },
        )
        st.markdown(
            "**Verificação por classe.** A regra global do IQR sinaliza "
            f"{outliers_por_classe['Normal']['percentual']:.2f}% dos "
            "registros Normais e "
            f"{outliers_por_classe['Ataques']['percentual']:.2f}% dos "
            "registros de ataque. A frequência é maior entre os ataques, "
            "mas há milhares de valores sinalizados nos dois grupos. Logo, "
            "a condição de outlier não separa as classes."
        )
        st.info(
            "O painel em log1p preserva a ordem dos valores e melhora a "
            "visibilidade da região próxima de zero. A transformação é apenas "
            "visual; as medidas exibidas foram calculadas na escala original.",
            icon=":material/info:",
        )
        st.warning(
            "A regra do IQR identifica valores distantes da região central, "
            "mas não determina a causa. Um valor extremo não é automaticamente "
            "erro, ataque ou anomalia de segurança.",
            icon=":material/report:",
        )

    with st.container(border=True):
        st.subheader(
            "Descoberta 3 — a relação entre pacotes muda entre grupos TCP",
            icon=":material/query_stats:",
        )
        st.markdown(
            "A comparação mantém **X = spkts**, **Y = dpkts** e "
            "**protocolo = TCP** nos grupos Normal, Exploits e Fuzzers. "
            "Os rótulos são conhecidos previamente; portanto, esta é uma "
            "análise condicional entre grupos e não um classificador."
        )

        tabela_regressao_exibicao = tabela_regressao.copy()
        tabela_regressao_exibicao["Pares"] = [
            formatar_inteiro(valor)
            for valor in tabela_regressao["Pares"].tolist()
        ]
        for coluna in ("r", "R²", "Inclinação", "Intercepto"):
            tabela_regressao_exibicao[coluna] = [
                formatar_decimal(valor, 6)
                for valor in tabela_regressao[coluna].tolist()
            ]
        st.dataframe(
            tabela_regressao_exibicao,
            hide_index=True,
            width="stretch",
        )

        figura_coeficientes = criar_figura_coeficientes(tabela_regressao)
        st.pyplot(figura_coeficientes)
        exibir_botao_download(
            figura_coeficientes,
            "modulo6_coeficientes_regressao.png",
            "m6_download_coeficientes",
        )
        plt.close(figura_coeficientes)

        modo_eixos = st.segmented_control(
            "Escala dos diagramas de dispersão",
            options=["Faixa completa", "Zoom central até P99"],
            default="Faixa completa",
            key="m6_escala_dispersao",
            help=(
                "O zoom altera somente os limites dos eixos. Os coeficientes "
                "continuam sendo calculados com todos os pares válidos."
            ),
        )
        figura_regressoes = criar_figura_regressoes(
            tabela_regressao,
            pares_por_grupo,
            aplicar_zoom=modo_eixos == "Zoom central até P99",
        )
        st.pyplot(figura_regressoes)
        exibir_botao_download(
            figura_regressoes,
            "modulo6_dispersao_regressoes_tcp.png",
            "m6_download_regressoes",
        )
        plt.close(figura_regressoes)

        linha_normal = tabela_regressao.loc[
            tabela_regressao["Grupo"] == "Normal"
        ].iloc[0]
        linha_exploits = tabela_regressao.loc[
            tabela_regressao["Grupo"] == "Exploits"
        ].iloc[0]
        linha_fuzzers = tabela_regressao.loc[
            tabela_regressao["Grupo"] == "Fuzzers"
        ].iloc[0]

        st.markdown(
            "**Leitura técnica.** Todos os coeficientes de Pearson são "
            "positivos, mas a intensidade do alinhamento linear varia. Em "
            f"Normal, r = **{linha_normal['r']:.6f}** e R² = "
            f"**{100 * linha_normal['R²']:.2f}%**. Em Exploits, r = "
            f"**{linha_exploits['r']:.6f}** e R² = "
            f"**{100 * linha_exploits['R²']:.2f}%**. Em Fuzzers, r = "
            f"**{linha_fuzzers['r']:.6f}** e R² = "
            f"**{100 * linha_fuzzers['R²']:.2f}%**."
        )
        st.markdown(
            "No ajuste completo, as inclinações de Exploits e Fuzzers são "
            "próximas, enquanto r e R² diferem. Essa observação descreve os "
            "dados completos, mas precisa ser examinada quanto à influência "
            "da cauda superior antes de ser tratada como um padrão estável."
        )

        st.markdown("**Análise de sensibilidade aos valores extremos**")
        tabela_sensibilidade_exibicao = tabela_sensibilidade.copy()
        tabela_sensibilidade_exibicao["Pares completos"] = [
            formatar_inteiro(valor)
            for valor in tabela_sensibilidade["Pares completos"].tolist()
        ]
        tabela_sensibilidade_exibicao["Pares até P99"] = [
            formatar_inteiro(valor)
            for valor in tabela_sensibilidade["Pares até P99"].tolist()
        ]
        tabela_sensibilidade_exibicao["Retenção (%)"] = [
            f"{formatar_decimal(valor, 2)}%"
            for valor in tabela_sensibilidade["Retenção (%)"].tolist()
        ]
        for coluna in ("P99 de spkts", "P99 de dpkts"):
            tabela_sensibilidade_exibicao[coluna] = [
                formatar_decimal(valor, 2)
                for valor in tabela_sensibilidade[coluna].tolist()
            ]
        for coluna in (
            "r completo",
            "r até P99",
            "R² completo",
            "R² até P99",
            "Inclinação completa",
            "Inclinação até P99",
        ):
            tabela_sensibilidade_exibicao[coluna] = [
                formatar_decimal(valor, 6)
                for valor in tabela_sensibilidade[coluna].tolist()
            ]
        st.dataframe(
            tabela_sensibilidade_exibicao,
            hide_index=True,
            width="stretch",
        )

        sensibilidade_normal = tabela_sensibilidade.loc[
            tabela_sensibilidade["Grupo"] == "Normal"
        ].iloc[0]
        sensibilidade_exploits = tabela_sensibilidade.loc[
            tabela_sensibilidade["Grupo"] == "Exploits"
        ].iloc[0]
        sensibilidade_fuzzers = tabela_sensibilidade.loc[
            tabela_sensibilidade["Grupo"] == "Fuzzers"
        ].iloc[0]

        st.markdown(
            "A retirada diagnóstica dos pares acima do P99 de `spkts` ou "
            "`dpkts`, calculado separadamente em cada grupo, mantém r elevado "
            "em Normal: "
            f"**{sensibilidade_normal['r completo']:.6f} → "
            f"{sensibilidade_normal['r até P99']:.6f}**. Em Exploits, r "
            f"cai de **{sensibilidade_exploits['r completo']:.6f}** para "
            f"**{sensibilidade_exploits['r até P99']:.6f}**; em Fuzzers, "
            f"de **{sensibilidade_fuzzers['r completo']:.6f}** para "
            f"**{sensibilidade_fuzzers['r até P99']:.6f}**."
        )
        st.warning(
            "A inclinação de Exploits muda de "
            f"{sensibilidade_exploits['Inclinação completa']:.6f} para "
            f"{sensibilidade_exploits['Inclinação até P99']:.6f} na análise "
            "de sensibilidade. Portanto, a semelhança observada entre as "
            "inclinações completas de Exploits e Fuzzers não é robusta à "
            "cauda superior. O recorte até P99 é diagnóstico e não autoriza "
            "descartar os valores extremos como erros.",
            icon=":material/warning:",
        )

        mostrar_residuos = st.toggle(
            "Mostrar diagnóstico visual dos resíduos",
            value=False,
            key="m6_mostrar_residuos",
        )
        if mostrar_residuos:
            figura_residuos = criar_figura_residuos(pares_por_grupo)
            st.pyplot(figura_residuos)
            exibir_botao_download(
                figura_residuos,
                "modulo6_residuos_regressao.png",
                "m6_download_residuos",
            )
            plt.close(figura_residuos)
            st.caption(
                "Resíduos sem distribuição aleatória em torno de zero, "
                "mudança de dispersão ou padrões curvos indicam limitações "
                "da reta. O gráfico é diagnóstico visual e não substitui "
                "testes formais das hipóteses do modelo."
            )

        st.warning(
            "Os R² são calculados nos mesmos dados usados no ajuste. Eles "
            "descrevem aderência dentro da amostra e não medem desempenho "
            "preditivo fora dela. A regressão não fornece probabilidade de "
            "ataque e correlação não implica causalidade.",
            icon=":material/report:",
        )

    with st.expander(
        "Fundamentação matemática e reprodutibilidade",
        icon=":material/function:",
    ):
        st.markdown("**Frequência relativa**")
        st.latex(r"f_r(c)=\frac{n_c}{N}")
        st.markdown("**Limites da regra do IQR**")
        st.latex(r"IQR=Q_3-Q_1")
        st.latex(
            r"L_{inf}=Q_1-1{,}5\,IQR\qquad "
            r"L_{sup}=Q_3+1{,}5\,IQR"
        )
        st.markdown("**Regressão linear simples**")
        st.latex(r"\hat{y}=a+bx")
        st.latex(
            r"b=\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}"
            r"{\sum_{i=1}^{n}(x_i-\bar{x})^2}"
        )
        st.latex(r"a=\bar{y}-b\bar{x}")
        st.markdown("**Coeficiente de determinação**")
        st.latex(
            r"R^2=1-\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}"
            r"{\sum_{i=1}^{n}(y_i-\bar{y})^2}"
        )
        st.caption(
            "Média, mediana, moda, desvio-padrão, percentis, IQR, Pearson, "
            "mínimos quadrados, predição e R² são calculados pelas funções "
            "próprias de minhastats.py. Pandas é usado para carregar, filtrar "
            "e organizar os dados; Matplotlib é usado para visualização."
        )

    st.subheader(
        "Síntese científica",
        icon=":material/science:",
    )
    st.markdown(
        "1. **A composição da partição é heterogênea.** O agregado de ataques "
        "é majoritário, e a distribuição entre suas categorias é muito "
        "desigual.\n"
        "2. **A duração não é bem resumida pela média isolada.** A forte "
        "assimetria e os valores extremos tornam mediana e IQR essenciais; "
        "os valores além das cercas aparecem tanto em tráfego Normal quanto "
        "em ataques.\n"
        "3. **A estrutura linear depende do grupo e da cauda.** Dentro do "
        "tráfego TCP, Normal mantém forte alinhamento após a análise até P99, "
        "enquanto os ajustes dos grupos de ataque estudados são mais "
        "sensíveis aos valores extremos."
    )
    st.info(
        "Essas conclusões caracterizam a partição analisada e geram hipóteses "
        "para investigações posteriores. Validar um detector exigiria uma "
        "tarefa supervisionada separada, divisão adequada entre treino e "
        "teste, métricas por classe e avaliação fora da amostra.",
        icon=":material/fact_check:",
    )
