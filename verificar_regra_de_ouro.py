"""
Verifica se o núcleo e a interface respeitam a regra de ouro do projeto.

Regras verificadas:

1. src/nucleo/minhastats.py não pode importar NumPy, Pandas, SciPy ou
   statistics.
2. src/app.py não deve calcular as medidas estatísticas exigidas por meio
   de atalhos de Pandas, NumPy, SciPy ou statistics.
3. NumPy e Pandas ainda podem ser usados na interface para carregamento,
   preparação dos dados, simulação, criação de eixos e visualização.
4. Os arquivos de validação podem usar bibliotecas consolidadas somente para
   comparar, de forma explícita, o núcleo próprio com referências externas.
5. O minieditor não pode chamar eval(), exec() ou compile().

Execute a partir da raiz:

    python verificar_regra_de_ouro.py
"""

from __future__ import annotations

import ast
import sys
import warnings
from pathlib import Path


RAIZ_PROJETO = Path(__file__).resolve().parent
ARQUIVO_NUCLEO = RAIZ_PROJETO / "src" / "nucleo" / "minhastats.py"
ARQUIVOS_INTERFACE = (
    RAIZ_PROJETO / "src" / "app.py",
    RAIZ_PROJETO / "src" / "interface" / "pagina_inicial.py",
    RAIZ_PROJETO / "src" / "interface" / "modulo_2.py",
    RAIZ_PROJETO / "src" / "interface" / "modulo_3.py",
    RAIZ_PROJETO / "src" / "interface" / "modulo_4.py",
    RAIZ_PROJETO / "src" / "interface" / "modulo_5.py",
    RAIZ_PROJETO / "src" / "interface" / "modulo_6.py",
)
ARQUIVO_VALIDACAO = (
    RAIZ_PROJETO / "src" / "interface" / "pagina_validacao.py"
)
ARQUIVO_EDITOR = (
    RAIZ_PROJETO / "src" / "interface" / "editor_validacao.py"
)

BIBLIOTECAS_PROIBIDAS_NO_NUCLEO = {
    "numpy",
    "pandas",
    "scipy",
    "statistics",
}

METODOS_ESTATISTICOS_SUSPEITOS = {
    "mean",
    "median",
    "std",
    "var",
    "corr",
    "quantile",
    "describe",
    "mode",
    "cov",
    "skew",
}

FUNCOES_NUMPY_SUSPEITAS = {
    "average",
    "corrcoef",
    "cov",
    "mean",
    "median",
    "percentile",
    "polyfit",
    "quantile",
    "std",
    "var",
}

FUNCOES_SCIPY_STATS_SUSPEITAS = {
    "linregress",
    "pearsonr",
    "scoreatpercentile",
    "variation",
    "zscore",
}


def carregar_arvore(caminho: Path) -> ast.AST:
    """
    Lê um arquivo Python e retorna sua árvore sintática.

    SyntaxWarning é silenciado porque a responsabilidade deste verificador
    é detectar violações da regra de ouro. Erros de sintaxe continuam sendo
    capturados normalmente.
    """
    codigo = caminho.read_text(encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)

        return ast.parse(
            codigo,
            filename=str(caminho),
        )


def nome_completo_no(candidato: ast.AST) -> str | None:
    """
    Reconstrói o nome de uma chamada representada pela AST.

    Exemplos:

        np.mean
        scipy.stats.linregress
        df.describe
    """
    if isinstance(candidato, ast.Name):
        return candidato.id

    if isinstance(candidato, ast.Attribute):
        prefixo = nome_completo_no(candidato.value)

        if prefixo:
            return f"{prefixo}.{candidato.attr}"

    return None


def verificar_importacoes_nucleo(caminho: Path) -> list[str]:
    """
    Localiza importações proibidas no núcleo estatístico.

    O núcleo deve implementar os cálculos matemáticos sem NumPy,
    Pandas, SciPy ou statistics.
    """
    arvore = carregar_arvore(caminho)
    problemas = []

    for no in ast.walk(arvore):
        if isinstance(no, ast.Import):
            for nome in no.names:
                biblioteca = nome.name.split(".")[0]

                if biblioteca in BIBLIOTECAS_PROIBIDAS_NO_NUCLEO:
                    problemas.append(
                        f"linha {no.lineno}: "
                        f"importação proibida de {nome.name}"
                    )

        elif isinstance(no, ast.ImportFrom) and no.module:
            biblioteca = no.module.split(".")[0]

            if biblioteca in BIBLIOTECAS_PROIBIDAS_NO_NUCLEO:
                problemas.append(
                    f"linha {no.lineno}: "
                    f"importação proibida de {no.module}"
                )

    return problemas


def mapear_importacoes_app(
    arvore: ast.AST,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Relaciona nomes locais aos módulos e funções importadas.

    Isso permite detectar também casos com apelidos, por exemplo:

        import numpy as np
        from scipy.stats import linregress as regressao_pronta
    """
    modulos = {}
    funcoes = {}

    for no in ast.walk(arvore):
        if isinstance(no, ast.Import):
            for nome in no.names:
                apelido = (
                    nome.asname
                    or nome.name.split(".")[0]
                )

                modulos[apelido] = nome.name

        elif isinstance(no, ast.ImportFrom) and no.module:
            for nome in no.names:
                apelido = nome.asname or nome.name

                funcoes[apelido] = (
                    f"{no.module}.{nome.name}"
                )

    return modulos, funcoes


def verificar_calculos_app(caminho: Path) -> list[str]:
    """
    Localiza atalhos estatísticos suspeitos na interface.

    A verificação não impede Pandas e NumPy de serem utilizados para
    carregamento, filtragem, preparação, simulação e visualização.
    """
    arvore = carregar_arvore(caminho)
    modulos, funcoes = mapear_importacoes_app(arvore)
    problemas = []

    for no in ast.walk(arvore):
        if not isinstance(no, ast.Call):
            continue

        nome_chamada = nome_completo_no(no.func)

        if not nome_chamada:
            continue

        if isinstance(no.func, ast.Attribute):
            metodo = no.func.attr

            if metodo in METODOS_ESTATISTICOS_SUSPEITOS:
                problemas.append(
                    f"linha {no.lineno}: "
                    f"chamada suspeita {nome_chamada}()"
                )
                continue

            raiz = nome_chamada.split(".")[0]
            modulo_real = modulos.get(raiz, raiz)

            if (
                modulo_real == "numpy"
                and metodo in FUNCOES_NUMPY_SUSPEITAS
            ):
                problemas.append(
                    f"linha {no.lineno}: "
                    f"cálculo estatístico com {nome_chamada}()"
                )
                continue

            if (
                modulo_real.startswith("scipy.stats")
                and metodo in FUNCOES_SCIPY_STATS_SUSPEITAS
            ):
                problemas.append(
                    f"linha {no.lineno}: "
                    f"cálculo estatístico com {nome_chamada}()"
                )

        elif isinstance(no.func, ast.Name):
            origem = funcoes.get(no.func.id)

            if not origem:
                continue

            modulo, _, funcao = origem.rpartition(".")

            usa_numpy = (
                modulo == "numpy"
                and funcao in FUNCOES_NUMPY_SUSPEITAS
            )

            usa_scipy = (
                modulo.startswith("scipy.stats")
                and funcao in FUNCOES_SCIPY_STATS_SUSPEITAS
            )

            usa_statistics = modulo == "statistics"

            if usa_numpy or usa_scipy or usa_statistics:
                problemas.append(
                    f"linha {no.lineno}: "
                    f"cálculo estatístico com {no.func.id}()"
                )

    return problemas


def verificar_pagina_validacao(caminho: Path) -> list[str]:
    """Confirma a separação explícita entre cálculo próprio e referência."""
    arvore = carregar_arvore(caminho)
    importacoes = set()
    funcoes = set()

    for no in ast.walk(arvore):
        if isinstance(no, ast.Import):
            importacoes.update(nome.name.split(".")[0] for nome in no.names)
        elif isinstance(no, ast.ImportFrom) and no.module:
            importacoes.add(no.module.split(".")[0])
            importacoes.update(nome.name.split(".")[0] for nome in no.names)
        elif isinstance(no, ast.FunctionDef):
            funcoes.add(no.name)

    problemas = []
    for biblioteca in ("minhastats", "numpy", "scipy", "statistics"):
        if biblioteca not in importacoes:
            problemas.append(
                f"a página de validação não importa {biblioteca}"
            )

    funcoes_esperadas = {
        "_codigo_funcao",
        "executar_validacao_funcao",
        "resultados_equivalentes",
        "validar_catalogo_completo",
    }
    for funcao in sorted(funcoes_esperadas - funcoes):
        problemas.append(f"função de comparação ausente: {funcao}()")

    return problemas


def verificar_editor_validacao(caminho: Path) -> list[str]:
    """Confirma referências explícitas e impede execução arbitrária de código."""
    arvore = carregar_arvore(caminho)
    importacoes = set()
    funcoes = set()
    problemas = []

    for no in ast.walk(arvore):
        if isinstance(no, ast.Import):
            importacoes.update(nome.name.split(".")[0] for nome in no.names)
        elif isinstance(no, ast.ImportFrom) and no.module:
            importacoes.add(no.module.split(".")[0])
            importacoes.update(nome.name.split(".")[0] for nome in no.names)
        elif isinstance(no, ast.FunctionDef):
            funcoes.add(no.name)
        elif isinstance(no, ast.Call) and isinstance(no.func, ast.Name):
            if no.func.id in {"eval", "exec", "compile"}:
                problemas.append(
                    f"linha {no.lineno}: execução arbitrária com {no.func.id}()"
                )

    for biblioteca in ("minhastats", "numpy", "scipy", "statistics"):
        if biblioteca not in importacoes:
            problemas.append(f"o minieditor não importa {biblioteca}")

    funcoes_esperadas = {
        "carregar_json_entradas",
        "entradas_padrao",
        "executar",
        "montar_terminal",
        "validar_entradas",
    }
    for funcao in sorted(funcoes_esperadas - funcoes):
        problemas.append(f"função do minieditor ausente: {funcao}()")

    return problemas


def exibir_resultado(
    caminho: Path,
    problemas: list[str],
) -> bool:
    """
    Exibe o diagnóstico de um arquivo.

    Retorna True quando nenhuma violação foi encontrada.
    """
    caminho_relativo = caminho.relative_to(
        RAIZ_PROJETO
    )

    if not problemas:
        print(f"[OK] {caminho_relativo}")
        return True

    print(f"[ERRO] {caminho_relativo}")

    for problema in problemas:
        print(f"  - {problema}")

    return False


def main() -> int:
    """
    Executa as verificações do núcleo e da interface.

    Retorna:

        0: regra de ouro respeitada;
        1: violação, arquivo ausente ou erro de sintaxe.
    """
    try:
        problemas_nucleo = verificar_importacoes_nucleo(
            ARQUIVO_NUCLEO
        )

        problemas_interfaces = [
            (
                caminho,
                verificar_calculos_app(caminho),
            )
            for caminho in ARQUIVOS_INTERFACE
        ]
        problemas_validacao = verificar_pagina_validacao(
            ARQUIVO_VALIDACAO
        )
        problemas_editor = verificar_editor_validacao(
            ARQUIVO_EDITOR
        )

    except FileNotFoundError as erro:
        print(
            f"[ERRO] Arquivo não encontrado: "
            f"{erro.filename}"
        )

        return 1

    except SyntaxError as erro:
        print(
            f"[ERRO] Sintaxe inválida em {erro.filename}, "
            f"linha {erro.lineno}: {erro.msg}"
        )

        return 1

    nucleo_ok = exibir_resultado(
        ARQUIVO_NUCLEO,
        problemas_nucleo,
    )

    resultados_interfaces = [
        exibir_resultado(caminho, problemas)
        for caminho, problemas in problemas_interfaces
    ]
    interfaces_ok = all(resultados_interfaces)
    validacao_ok = exibir_resultado(
        ARQUIVO_VALIDACAO,
        problemas_validacao,
    )
    editor_ok = exibir_resultado(
        ARQUIVO_EDITOR,
        problemas_editor,
    )
    if nucleo_ok and interfaces_ok and validacao_ok and editor_ok:
        print()
        print("Regra de ouro respeitada.")

        return 0

    print()
    print(
        "Corrija os pontos indicados antes da entrega."
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())
