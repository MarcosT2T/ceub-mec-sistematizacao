# Relatório da Sistematização de Matemática e Estatística para Computação

## Laboratório Estatístico Interativo aplicado ao UNSW-NB15

### Identificação acadêmica

| Campo | Informação |
| --- | --- |
| Instituição | Centro Universitário de Brasília — CEUB |
| Curso | Análise e Desenvolvimento de Sistemas |
| Disciplina | Matemática e Estatística para Computação |
| Professor orientador | Prof. Romes Heriberto |
| Atividade | Sistematização — Laboratório Estatístico Interativo |

### Equipe

| Integrante | Matrícula |
| --- | ---: |
| Iago Batista Gomes de Carvalho | 72650448 |
| Marcos Paulo dos Santos Júnior | 72650390 |
| Caio Sacramento Côrtes | 72650430 |

---

## 1. Objetivo e escopo

O projeto transforma fórmulas de Matemática e Estatística em uma aplicação
interativa desenvolvida em Python e Streamlit. O núcleo matemático foi
implementado pela equipe em `src/nucleo/minhastats.py`; Pandas organiza os
dados, Matplotlib e Altair geram visualizações, e NumPy, SciPy e `statistics`
atuam como referências independentes de validação.

A unidade de análise é um **registro de fluxo de rede** da partição de
treinamento do UNSW-NB15. A aplicação não é um sistema de detecção de intrusão,
não classifica novos fluxos e não estabelece causalidade. Seu objetivo é
demonstrar cálculos estatísticos, explorar padrões presentes na partição e
ensinar como distinguir evidência descritiva de conclusões que os dados não
sustentam.

---

## 2. Dataset utilizado

O UNSW-NB15 foi desenvolvido pela UNSW Canberra em um ambiente controlado de
pesquisa em segurança de redes. O conjunto reúne atividades normais e
comportamentos de ataque gerados para experimentação acadêmica.

- Fonte oficial: <https://research.unsw.edu.au/projects/unsw-nb15-dataset>
- Cópia pública utilizada como referência: <https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15>
- Arquivo analisado: `data/UNSW_NB15_training-set.csv`

### 2.1 Integridade da partição carregada

| Verificação | Resultado |
| --- | ---: |
| Registros | 175.341 |
| Variáveis | 45 |
| Valores ausentes | 0 |
| Registros normais | 56.000 |
| Registros de ataque | 119.341 |
| Categorias de ataque | 9 |
| Vetores repetidos, desconsiderando `id` | 67.601 (38,55%) |

Os vetores repetidos não foram removidos automaticamente. Em tráfego de rede,
linhas com os mesmos atributos podem representar repetições reais do processo
de captura, duplicações de preparação ou registros cuja distinção dependia de
informações não presentes nesta tabela. Excluí-los sem investigar a procedência
alteraria frequências, médias e proporções.

### 2.2 Variáveis centrais utilizadas no relatório

| Variável | Natureza | Significado no laboratório |
| --- | --- | --- |
| `attack_cat` | Categórica | Categoria atribuída ao fluxo: Normal ou tipo de ataque |
| `proto` | Categórica | Protocolo de rede associado ao fluxo |
| `dur` | Numérica contínua | Duração registrada para o fluxo |
| `spkts` | Numérica discreta | Número de pacotes da origem para o destino |
| `dpkts` | Numérica discreta | Número de pacotes do destino para a origem |

---

## 3. Núcleo estatístico próprio

As medidas apresentadas ao usuário são calculadas por funções próprias. As
bibliotecas científicas são usadas para carregar e manipular dados, construir
gráficos e conferir os resultados, conforme permitido no enunciado.

### 3.1 Média aritmética

Para valores \(x_1, x_2, \ldots, x_n\):

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i
$$

### 3.2 Variância

Na forma populacional:

$$
\sigma^2=\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2
$$

Na forma amostral, o divisor é \(n-1\):

$$
s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2
$$

O desvio padrão é a raiz quadrada da variância. A interface informa qual divisor
foi utilizado para evitar a mistura entre as duas definições.

### 3.3 Percentis, quartis e regra do IQR

Os percentis são obtidos após ordenar os dados e interpolar linearmente a
posição. Com \(Q_1=P_{25}\), \(Q_3=P_{75}\) e
\(IQR=Q_3-Q_1\), as cercas exploratórias são:

$$
L_i=Q_1-1{,}5\,IQR
\qquad\text{e}\qquad
L_s=Q_3+1{,}5\,IQR
$$

Um valor além dessas cercas é distante em relação à distribuição. Isso não o
transforma automaticamente em erro, ataque ou anomalia operacional.

### 3.4 Correlação de Pearson

Para pares \((x_i,y_i)\):

$$
r=\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}
\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}}
$$

O coeficiente mede associação **linear**. Valores próximos de \(+1\) ou \(-1\)
indicam associação linear forte; valores próximos de zero não excluem relações
não lineares. Correlação não implica causalidade e não constitui, isoladamente,
um classificador de ataque.

### 3.5 Regressão linear simples por mínimos quadrados

A reta ajustada tem a forma:

$$
\hat{y}=b_0+b_1x
$$

com:

$$
b_1=\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
{\sum_{i=1}^{n}(x_i-\bar{x})^2}
\qquad\text{e}\qquad
b_0=\bar{y}-b_1\bar{x}
$$

Os coeficientes minimizam a soma dos quadrados dos resíduos
\(e_i=y_i-\hat{y}_i\). O coeficiente de determinação exibido é:

$$
R^2=1-\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
{\sum_{i=1}^{n}(y_i-\bar{y})^2}
$$

O \(R^2\) descreve o ajuste linear nos dados analisados. Ele não mede
generalização fora da amostra e não deve ser interpretado como acurácia de um
detector de intrusão.

---

## 4. Organização dos módulos

### Módulo 0 — Dados reais

A aplicação carrega a partição oficial de treinamento, identifica sua
cardinalidade, descreve a unidade de análise e mantém explícitas as limitações
de procedência e representatividade.

### Módulo 1 — Núcleo estatístico próprio

O arquivo `src/nucleo/minhastats.py` contém média, mediana, moda, amplitude,
variâncias, desvios padrão, percentis, coeficiente de variação, covariância,
Pearson, regressão, predição, R², simulações e funções de distribuição.

A página de validação permite escolher cada uma das 27 funções catalogadas,
visualizar o código próprio e a referência lado a lado, editar entradas JSON e
executar uma comparação controlada. A execução é restrita a uma lista fechada
de funções; a aplicação pública não usa `eval`, `exec` ou compilação de código
fornecido pelo usuário.

### Módulo 2 — Estatística descritiva interativa

O usuário escolhe uma variável e recebe medidas de posição, tendência central
e dispersão, tabela de frequências, gráfico adequado e análise de valores além
das cercas do IQR. Nas variáveis de cauda longa, a ampliação visual não remove
observações do cálculo.

### Módulo 3 — Probabilidade e simulação

A Lei dos Grandes Números é demonstrada por frequências relativas acumuladas.
O Teorema Central do Limite é explorado por amostras repetidas de uma variável
do dataset, com tamanho de amostra, número de repetições e semente controláveis.
As simulações usam amostragem com reposição e exibem o erro padrão teórico
\(\sigma/\sqrt{n}\).

### Módulo 4 — Distribuições teóricas

A interface compara o histograma observado com a Normal e com uma segunda
distribuição coerente com a natureza da variável: Exponencial para medidas
contínuas não negativas ou Poisson para contagens. Os parâmetros vêm do núcleo
próprio e a distância entre distribuições acumuladas é apresentada como medida
descritiva da qualidade de ajuste.

### Módulo 5 — Correlação e regressão linear

O usuário seleciona duas variáveis numéricas e pode filtrar os registros por
grupo e protocolo. A aplicação apresenta o diagrama de dispersão, Pearson, reta
de mínimos quadrados, equação, R² e predição interativa. Uma comparação guiada
mantém o mesmo protocolo e o mesmo par de variáveis entre tráfego Normal,
Exploits e Fuzzers.

### Módulo 6 — Relatório de descobertas

O relatório interativo reúne as três descobertas descritas a seguir. Os
gráficos de composição, duração, coeficientes, regressões e resíduos são
renderizados pela própria aplicação e podem ser baixados em PNG para o relatório
acadêmico final.

---

## 5. Descoberta 1 — A composição da partição é desigual

### Evidência observada

| Grupo | Registros | Percentual |
| --- | ---: | ---: |
| Normal | 56.000 | 31,94% |
| Ataques | 119.341 | 68,06% |

Há aproximadamente **2,13 registros de ataque para cada registro normal**.
Entre os ataques, as categorias com maior número de registros são Generic
(40.000), Exploits (33.393) e Fuzzers (18.184).

### Interpretação

Se um registro for sorteado uniformemente desta partição, a probabilidade
empírica de ele pertencer à classe de ataque é 68,06%. A demonstração da Lei
dos Grandes Números pode convergir para esse valor porque amostra a própria
partição.

### Limite da conclusão

Essa proporção descreve a construção do arquivo de treinamento. Ela **não é uma
estimativa da prevalência de ataques em redes reais** e não pode ser convertida
em risco operacional sem uma amostra representativa do ambiente de interesse.
Treinar ou avaliar classificadores sem considerar esse desenho também pode
produzir métricas influenciadas pelo desbalanceamento.

---

## 6. Descoberta 2 — A duração possui forte assimetria à direita

### Evidência observada

| Medida de `dur` | Resultado |
| --- | ---: |
| Média | 1,359389 |
| Mediana | 0,001582 |
| Moda | 0,000009 |
| Desvio padrão populacional | 6,480230 |
| Coeficiente de variação | 476,70% |
| Q1 | 0,000008 |
| Q3 | 0,668069 |
| Cerca superior do IQR | 1,670161 |
| Mínimo | 0,000000 |
| Máximo | 59,999989 |

A média é aproximadamente **859 vezes a mediana**. Foram identificados 15.741
valores além das cercas globais do IQR, equivalentes a 8,98% dos registros. Ao
aplicar essas mesmas cercas, 6,71% dos registros normais e 10,04% dos registros
de ataque ficam além dos limites.

### Interpretação

A maior parte das durações está concentrada perto de zero, enquanto uma cauda
superior relativamente pequena eleva a média e o desvio padrão. Por isso, a
mediana e os quartis descrevem o centro da distribuição com mais robustez do que
a média isolada. O histograma com ampliação da região central facilita a leitura,
mas todos os registros continuam nos cálculos.

### Limite da conclusão

As cercas usadas são calculadas para a distribuição conjunta. Comparar os
percentuais por classe com as mesmas cercas é uma descrição do recorte, não uma
prova de que valores longos causem ou identifiquem ataques. Um fluxo distante
da distribuição pode ser legítimo, e um ataque pode ocorrer dentro da região
central.

---

## 7. Descoberta 3 — A relação `spkts → dpkts` depende do grupo TCP

Para evitar comparar populações de protocolo diferentes, o laboratório fixa:

- protocolo: TCP;
- variável explicativa \(X\): `spkts`;
- variável resposta \(Y\): `dpkts`;
- grupos: Normal, Exploits e Fuzzers.

### 7.1 Ajuste com todos os pares válidos

| Grupo | Pares | Pearson \(r\) | \(R^2\) | Inclinação | Intercepto |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal | 39.121 | 0,961722 | 0,924909 | 1,827924 | -24,859683 |
| Exploits | 19.689 | 0,426988 | 0,182319 | 0,167671 | 28,697390 |
| Fuzzers | 11.761 | 0,753142 | 0,567223 | 0,161934 | 6,595091 |

No grupo Normal, a relação linear é forte e a reta explica aproximadamente
92,49% da variação observada de `dpkts` dentro desse recorte. Em Exploits, a
associação linear é mais fraca e o R² cai para 18,23%. Fuzzers ocupa uma posição
intermediária, com R² de 56,72%.

As inclinações têm unidade de pacotes de destino por pacote de origem. O
intercepto negativo do grupo Normal não deve ser lido como previsão física para
\(x=0\) sem verificar se esse ponto pertence ao domínio observado; ele é o termo
matemático da reta ajustada.

### 7.2 Análise de sensibilidade até o P99

A aplicação recalcula cada regressão após retirar, apenas para diagnóstico,
pares acima do P99 de `spkts` ou de `dpkts` dentro do respectivo grupo.

| Grupo | Retenção | \(r\) completo | \(r\) até P99 | Inclinação completa | Inclinação até P99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal | 99,03% | 0,961722 | 0,949488 | 1,827924 | 1,793333 |
| Exploits | 98,38% | 0,426988 | 0,299874 | 0,167671 | 0,368849 |
| Fuzzers | 98,80% | 0,753142 | 0,706571 | 0,161934 | 0,156192 |

Normal e Fuzzers mantêm coeficientes relativamente próximos após o recorte. Em
Exploits, a inclinação muda de 0,167671 para 0,368849 e o R² passa de 0,182319
para 0,089924. Esse resultado mostra que a conclusão sobre a inclinação de
Exploits é sensível à cauda superior, mesmo com mais de 98% dos pares retidos.

### Limite da conclusão

As diferenças entre retas descrevem associações condicionadas ao protocolo e às
categorias atribuídas no dataset. Elas não provam que a categoria causou o
padrão, não medem capacidade de classificação e não validam predições em outra
rede. A retirada até o P99 é uma análise de sensibilidade; os valores excluídos
não foram declarados erros.

---

## 8. Validação dos cálculos

A suíte automatizada contém **237 testes**, todos aprovados. A cobertura está
organizada por responsabilidade:

| Arquivo | Casos |
| --- | ---: |
| `test_minhastats.py` | 97 |
| `test_editor_validacao.py` | 31 |
| `test_modulo_2.py` | 9 |
| `test_modulo_3.py` | 25 |
| `test_modulo_4.py` | 44 |
| `test_modulo_5.py` | 13 |
| `test_modulo_6.py` | 7 |
| `test_pagina_inicial.py` | 5 |
| `test_pagina_validacao.py` | 6 |
| **Total** | **237** |

As funções próprias foram comparadas com NumPy, SciPy e `statistics` usando as
mesmas entradas e parâmetros equivalentes. Para resultados de ponto flutuante,
a tolerância principal é relativa de \(10^{-9}\) e absoluta de \(10^{-12}\).
Também foram testados resultados conhecidos, listas vazias, tamanhos
incompatíveis, NaN, infinitos, variáveis constantes, limites de domínio e
preservação das entradas.

Simulações aleatórias são validadas por propriedades matemáticas e
reprodutibilidade com semente, pois exigir que geradores diferentes produzam a
mesma sequência não seria uma comparação estatisticamente válida.

A aprovação dos testes aumenta a confiança na implementação para os casos
cobertos. Ela não constitui prova formal para todo número real possível nem
elimina limitações do desenho do dataset.

---

## 9. Limitações gerais

1. **Ambiente controlado:** a base não representa automaticamente o tráfego de
   qualquer organização, período ou topologia de rede.
2. **Partição de treinamento:** as análises são descritivas dentro da amostra;
   não foi realizada avaliação preditiva fora da amostra.
3. **Dependência entre registros:** fluxos podem compartilhar contexto temporal
   ou operacional, portanto independência estatística não é garantida.
4. **Vetores repetidos:** 38,55% das linhas repetem todos os atributos exceto
   `id`; elas foram preservadas por falta de justificativa para exclusão.
5. **Correlação e regressão:** os métodos estudam associação linear e são
   sensíveis a escala, caudas e valores influentes.
6. **Rótulos disponíveis:** as comparações dependem das categorias fornecidas
   pelo dataset e não auditam o processo original de rotulagem.
7. **Causalidade:** nenhum resultado identifica mecanismo causal.
8. **Uso em segurança:** uma análise estatística isolada não substitui regras de
   rede, contexto temporal, engenharia de atributos e avaliação de um modelo de
   detecção.

---

## 10. Reprodutibilidade

O repositório público desta versão é:

<https://github.com/MarcosT2T/ceub-mec-sistematizacao>

Com Python 3.12, a aplicação pode ser reproduzida a partir da raiz:

```bash
python -m venv .venv
```

No Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -v
python verificar_regra_de_ouro.py
python -m streamlit run src/app.py
```

No Linux ou macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -v
python verificar_regra_de_ouro.py
python -m streamlit run src/app.py
```

Para futura publicação no Streamlit Community Cloud, utilize o branch `main`,
o arquivo principal `src/app.py` e Python 3.12. O aplicativo não requer
segredos.

---

## 11. Conclusão

O laboratório cumpre o objetivo de conectar fórmulas, implementação, validação
e interpretação. As três descobertas são sustentadas por medidas calculadas
pelo núcleo próprio e por gráficos reproduzíveis na interface:

1. a partição contém 68,06% de registros de ataque, proporção que descreve o
   arquivo de treinamento e não a prevalência em redes reais;
2. `dur` tem forte assimetria à direita, tornando mediana, quartis e análise da
   cauda indispensáveis para interpretar o centro da distribuição;
3. a relação linear entre `spkts` e `dpkts` sob TCP muda entre Normal, Exploits
   e Fuzzers, e o ajuste de Exploits é especialmente sensível à cauda superior.

O resultado acadêmico mais relevante não é apenas obter números, mas explicar
o que eles medem, em qual população foram calculados e quais conclusões
permanecem fora do alcance dos dados.
