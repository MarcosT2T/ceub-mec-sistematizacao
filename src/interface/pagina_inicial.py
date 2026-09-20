"""Página inicial e orientação de uso do laboratório estatístico."""

from __future__ import annotations

import pandas as pd
import streamlit as st


PARTICOES_OFICIAIS = {
    "Treinamento": {
        "registros": 175_341,
        "normal": 56_000,
        "ataques": 119_341,
    },
    "Teste": {
        "registros": 82_332,
        "normal": 37_000,
        "ataques": 45_332,
    },
}

INSTITUICAO = "Centro Universitário de Brasília (CEUB)"
CURSO = "Análise e Desenvolvimento de Sistemas"
DISCIPLINA = "Matemática e Estatística para Computação"
PROFESSOR_ORIENTADOR = "Prof. Romes Heriberto"
TIPO_ATIVIDADE = "Sistematização — Laboratório Estatístico Interativo"

INTEGRANTES = (
    {
        "Nome": "Iago Batista Gomes de Carvalho",
        "Matrícula": "72650448",
    },
    {
        "Nome": "Marcos Paulo dos Santos Júnior",
        "Matrícula": "72650390",
    },
    {
        "Nome": "Caio Sacramento Côrtes",
        "Matrícula": "72650430",
    },
)


def formatar_inteiro(valor: int) -> str:
    """Formata uma contagem com separador de milhar em português."""
    return f"{valor:,}".replace(",", ".")


def identificar_particao(
    registros: int,
    normal: int,
    ataques: int,
) -> str:
    """Compara cardinalidade e classes com as partições oficiais."""
    for nome, referencia in PARTICOES_OFICIAIS.items():
        if (
            registros == referencia["registros"]
            and normal == referencia["normal"]
            and ataques == referencia["ataques"]
        ):
            return nome

    return "Não identificada"


def calcular_resumo_inicial(dados: pd.DataFrame) -> dict[str, int | float | str]:
    """Resume procedência e integridade sem executar modelos estatísticos."""
    if "attack_cat" not in dados.columns:
        raise ValueError(
            "O dataset precisa conter a coluna attack_cat para a apresentação."
        )

    contagens: dict[object, int] = {}

    for categoria in dados["attack_cat"].dropna().tolist():
        contagens[categoria] = contagens.get(categoria, 0) + 1

    normal = contagens.get("Normal", 0)
    ataques = sum(
        quantidade
        for categoria, quantidade in contagens.items()
        if categoria != "Normal"
    )
    registros = len(dados)
    colunas_sem_id = dados.drop(columns=["id"], errors="ignore")
    vetores_repetidos = int(
        colunas_sem_id.duplicated(keep="first").sum()
    )

    return {
        "registros": registros,
        "colunas": len(dados.columns),
        "normal": normal,
        "ataques": ataques,
        "categorias_ataque": sum(
            categoria != "Normal"
            for categoria in contagens
        ),
        "ausentes": int(dados.isna().sum().sum()),
        "vetores_repetidos": vetores_repetidos,
        "percentual_repetidos": (
            100 * vetores_repetidos / registros
            if registros
            else 0.0
        ),
        "particao": identificar_particao(
            registros,
            normal,
            ataques,
        ),
    }


def renderizar_pagina_inicial(dados: pd.DataFrame) -> None:
    """Apresenta o projeto, a base carregada e o percurso dos módulos."""
    st.caption(f"{INSTITUICAO} · {CURSO}")
    st.title(
        "Laboratório Estatístico Interativo",
        anchor=False,
    )
    st.markdown(
        "### Estatística aplicada à análise responsável de tráfego de rede"
    )

    with st.container(border=True):
        coluna_academica, coluna_projeto = st.columns(2)
        with coluna_academica:
            st.markdown(
                f"**Instituição:** {INSTITUICAO}  \n"
                f"**Curso:** {CURSO}  \n"
                f"**Disciplina:** {DISCIPLINA}  \n"
                f"**Professor orientador:** {PROFESSOR_ORIENTADOR}"
            )
        with coluna_projeto:
            st.markdown(
                f"**Atividade:** {TIPO_ATIVIDADE}  \n"
                "**Tema:** análise estatística do UNSW-NB15  \n"
                "**Linguagem:** Python  \n"
                "**Interface:** Streamlit"
            )

    st.subheader("Apresentação do projeto", icon=":material/info:")
    st.markdown(
        "Este projeto acadêmico transforma os conceitos estudados em "
        "Matemática e Estatística para Computação em um laboratório "
        "interativo, reproduzível e auditável. A aplicação utiliza o conjunto "
        "de referência **UNSW-NB15** para explorar dados de fluxos de rede, "
        "descrever distribuições, simular resultados probabilísticos, comparar "
        "modelos teóricos e investigar relações lineares entre variáveis."
    )
    st.markdown(
        "O elemento central do trabalho é o núcleo `minhastats.py`. Média, "
        "mediana, moda, amplitude, variâncias, desvios-padrão, percentis, "
        "coeficiente de variação, covariância, correlação de Pearson, mínimos "
        "quadrados, predição, R² e funções de distribuições foram programados "
        "pela equipe a partir das fórmulas matemáticas. NumPy, SciPy e a "
        "biblioteca `statistics` são usadas como referências independentes "
        "para validar os resultados, sem substituir o núcleo próprio nos "
        "módulos apresentados ao usuário."
    )
    st.markdown(
        "A proposta não é construir um detector de intrusão. O laboratório "
        "ensina a formular perguntas, calcular medidas, visualizar padrões e "
        "reconhecer os limites das evidências. As conclusões descrevem a "
        "partição carregada e não devem ser convertidas automaticamente em "
        "diagnóstico de ataque, causalidade ou estimativa do risco de uma rede "
        "real."
    )

    st.subheader("Equipe responsável", icon=":material/groups:")
    st.dataframe(
        pd.DataFrame(INTEGRANTES),
        hide_index=True,
        width="stretch",
    )

    try:
        resumo = calcular_resumo_inicial(dados)
    except ValueError as erro:
        st.error(str(erro))
        return

    with st.container(border=True):
        st.subheader(
            "Base carregada nesta execução",
            icon=":material/database:",
        )

        colunas = st.columns(5)
        colunas[0].metric(
            "Registros de fluxo",
            formatar_inteiro(int(resumo["registros"])),
        )
        colunas[1].metric(
            "Variáveis",
            formatar_inteiro(int(resumo["colunas"])),
        )
        colunas[2].metric(
            "Tráfego normal",
            formatar_inteiro(int(resumo["normal"])),
        )
        colunas[3].metric(
            "Registros de ataque",
            formatar_inteiro(int(resumo["ataques"])),
        )
        colunas[4].metric(
            "Tipos de ataque",
            formatar_inteiro(int(resumo["categorias_ataque"])),
        )

        if resumo["particao"] == "Treinamento":
            st.success(
                "A cardinalidade e a composição das classes correspondem à "
                "partição oficial de treinamento do UNSW-NB15: 175.341 "
                "registros, sendo 56.000 normais e 119.341 ataques.",
                icon=":material/check_circle:",
            )
        elif resumo["particao"] == "Teste":
            st.success(
                "A cardinalidade e a composição das classes correspondem à "
                "partição oficial de teste do UNSW-NB15.",
                icon=":material/check_circle:",
            )
        else:
            st.warning(
                "A base carregada não coincide integralmente com as "
                "cardinalidades oficiais de treino ou teste. Trate-a como "
                "um recorte próprio e documente sua origem.",
                icon=":material/warning:",
            )

        st.caption(
            "Cada linha é tratada como um registro de fluxo de rede. Ela não "
            "representa um pacote individual nem uma ocorrência independente "
            "garantida."
        )

    coluna_dataset, coluna_tecnologias = st.columns(2)
    with coluna_dataset:
        with st.container(border=True):
            st.subheader("Dataset em contexto", icon=":material/dataset:")
            st.markdown(
                "O **UNSW-NB15** foi desenvolvido pela UNSW Canberra em um "
                "ambiente controlado de segurança de redes. A base combina "
                "atividades normais e comportamentos de ataque produzidos "
                "para pesquisa e disponibiliza atributos de protocolo, "
                "serviço, estado, duração, pacotes, bytes, taxas, TTL e "
                "categoria do tráfego."
            )
            st.markdown(
                "A aplicação usa a partição oficial de treinamento. Essa "
                "escolha é apropriada para exploração descritiva, mas não "
                "transforma as medidas exibidas em avaliação preditiva fora "
                "da amostra."
            )

    with coluna_tecnologias:
        with st.container(border=True):
            st.subheader(
                "Tecnologias e reprodutibilidade",
                icon=":material/code:",
            )
            st.markdown(
                "- **Python 3.12+** para a implementação.\n"
                "- **Streamlit** para a interface interativa.\n"
                "- **Pandas** para carregar e organizar a base.\n"
                "- **Matplotlib e Altair** para visualização.\n"
                "- **NumPy, SciPy e statistics** como referências.\n"
                "- **pytest** para a validação automatizada."
            )
            st.markdown(
                "As dependências são registradas em `requirements.txt`; as "
                "instruções, decisões e estrutura do projeto estão no "
                "`README.md`."
            )

    coluna_objetivo, coluna_regra = st.columns(2)

    with coluna_objetivo:
        with st.container(border=True):
            st.subheader(
                "Objetivo acadêmico",
                icon=":material/school:",
            )
            st.markdown(
                "O laboratório conecta fórmulas, código e interpretação. O "
                "usuário pode observar distribuições, simular teoremas, "
                "comparar grupos e estudar relações lineares sem transformar "
                "uma associação estatística em diagnóstico de segurança."
            )
            st.markdown(
                "**Pergunta central:** o que os números sustentam sobre esta "
                "partição e o que permanece fora do alcance da análise?"
            )

    with coluna_regra:
        with st.container(border=True):
            st.subheader(
                "Regra de ouro",
                icon=":material/function:",
            )
            st.markdown(
                "Média, mediana, moda, variância, desvio-padrão, percentis, "
                "covariância, Pearson, mínimos quadrados, predição, R² e "
                "funções de distribuição são calculados pelo núcleo próprio."
            )
            st.markdown(
                "Pandas organiza os dados; Matplotlib e Altair constroem os "
                "gráficos; NumPy e SciPy servem como referências de validação "
                "nos testes automatizados."
            )

    st.subheader("Organização da solução", icon=":material/account_tree:")
    arquitetura = pd.DataFrame(
        [
            {
                "Camada": "Dados",
                "Responsabilidade": (
                    "Carregar a partição de treinamento do UNSW-NB15 e "
                    "preservar a unidade de análise como registro de fluxo."
                ),
            },
            {
                "Camada": "Núcleo matemático",
                "Responsabilidade": (
                    "Implementar as fórmulas próprias em minhastats.py, sem "
                    "delegar as medidas exibidas às bibliotecas científicas."
                ),
            },
            {
                "Camada": "Validação",
                "Responsabilidade": (
                    "Comparar os resultados próprios com NumPy, SciPy e "
                    "statistics dentro de tolerâncias numéricas documentadas."
                ),
            },
            {
                "Camada": "Interface",
                "Responsabilidade": (
                    "Permitir escolhas, filtros, simulações, gráficos e "
                    "predições por meio do Streamlit."
                ),
            },
            {
                "Camada": "Interpretação",
                "Responsabilidade": (
                    "Distinguir evidência descritiva, hipótese, limitação e "
                    "conclusão não sustentada."
                ),
            },
        ]
    )
    st.dataframe(
        arquitetura,
        hide_index=True,
        width="stretch",
    )

    st.subheader(
        "Percurso recomendado",
        icon=":material/route:",
    )
    st.markdown(
        "Os módulos formam uma sequência de investigação. É possível navegar "
        "diretamente para qualquer etapa, mas a ordem abaixo facilita a "
        "compreensão dos resultados."
    )

    percurso = pd.DataFrame(
        [
            {
                "Etapa": "Módulo 0",
                "Foco": "Dados reais e procedência",
                "Pergunta orientadora": (
                    "Qual é a origem, a estrutura e a limitação da base?"
                ),
            },
            {
                "Etapa": "Módulo 1",
                "Foco": "Núcleo estatístico próprio",
                "Pergunta orientadora": (
                    "Como cada medida é construída a partir de sua fórmula?"
                ),
            },
            {
                "Etapa": "Validação",
                "Foco": "Fórmulas próprias × bibliotecas de referência",
                "Pergunta orientadora": (
                    "Os resultados implementados pela equipe coincidem com "
                    "referências independentes?"
                ),
            },
            {
                "Etapa": "Módulo 2",
                "Foco": "Estatística descritiva",
                "Pergunta orientadora": (
                    "Como a variável se distribui e quais valores merecem "
                    "investigação?"
                ),
            },
            {
                "Etapa": "Módulo 3",
                "Foco": "Probabilidade e simulação",
                "Pergunta orientadora": (
                    "Como frequências e médias amostrais se estabilizam?"
                ),
            },
            {
                "Etapa": "Módulo 4",
                "Foco": "Distribuições teóricas",
                "Pergunta orientadora": (
                    "Quais modelos se aproximam dos dados e onde falham?"
                ),
            },
            {
                "Etapa": "Módulo 5",
                "Foco": "Correlação e regressão",
                "Pergunta orientadora": (
                    "Como duas variáveis mudam juntas dentro de um recorte?"
                ),
            },
            {
                "Etapa": "Módulo 6",
                "Foco": "Relatório de descobertas",
                "Pergunta orientadora": (
                    "Quais conclusões são sustentadas pelas evidências?"
                ),
            },
        ]
    )
    st.dataframe(
        percurso,
        hide_index=True,
        width="stretch",
    )

    with st.expander(
        "Integridade, procedência e limitações da base",
        icon=":material/fact_check:",
    ):
        st.markdown(
            f"- **Valores ausentes:** "
            f"{formatar_inteiro(int(resumo['ausentes']))}.\n"
            f"- **Vetores de atributos repetidos, desconsiderando `id`:** "
            f"{formatar_inteiro(int(resumo['vetores_repetidos']))} "
            f"({float(resumo['percentual_repetidos']):.2f}%).\n"
            "- **Origem:** ambiente de laboratório com tráfego normal e "
            "comportamentos de ataque produzidos para pesquisa.\n"
            "- **Escopo:** os percentuais descrevem a partição carregada, "
            "sem estimar a prevalência de ataques em redes reais."
        )
        st.caption(
            "Vetores iguais não foram removidos. Sem os campos brutos de "
            "endereço e tempo, não é possível afirmar que sejam duplicações "
            "indevidas do mesmo evento. A repetição também limita a hipótese "
            "de independência entre todas as linhas."
        )
        st.markdown(
            "[Fonte oficial do UNSW-NB15]"
            "(https://research.unsw.edu.au/projects/unsw-nb15-dataset) · "
            "[Artigo original]"
            "(https://doi.org/10.1109/MilCIS.2015.7348942) · "
            "[Cópia pública usada como referência pelo projeto]"
            "(https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)"
        )

    st.subheader(
        "Regras para interpretar com responsabilidade",
        icon=":material/gpp_good:",
    )
    st.markdown(
        "1. **Frequência na base não é risco real.** A proporção de ataques "
        "reflete a construção desta partição.\n"
        "2. **Outlier não é sinônimo de ataque.** A regra do IQR sinaliza "
        "distância estatística, sem determinar a causa.\n"
        "3. **Correlação não implica causalidade.** Um r elevado não prova "
        "mecanismo causal nem identifica a classe do fluxo.\n"
        "4. **R² dentro da amostra não mede generalização.** Avaliação "
        "preditiva exigiria dados separados e métricas próprias.\n"
        "5. **Filtros mudam a população analisada.** Compare grupos mantendo "
        "variáveis e protocolo constantes."
    )

    st.info(
        "Comece pela página de Validação para conferir o núcleo matemático. "
        "Depois, use o menu lateral para percorrer os Módulos 2 a 6.",
        icon=":material/arrow_back:",
    )
