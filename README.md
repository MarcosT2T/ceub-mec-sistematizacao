# Sistematização de Matemática e Estatística para a Computação

## 📊 Sobre o Projeto

Sistema acadêmico desenvolvido para aplicar conceitos fundamentais de Matemática e Estatística à análise de um conjunto de referência de tráfego de rede utilizado em cibersegurança.

O projeto utiliza o dataset público **UNSW-NB15**, amplamente empregado em pesquisas de detecção de intrusões, para demonstrar visualmente e matematicamente conceitos estatísticos estudados durante a disciplina de Matemática e Estatística para Computação do curso de Análise e Desenvolvimento de Sistemas do CEUB.

A aplicação foi construída em Python e disponibiliza recursos para análise estatística, visualização de distribuições, demonstrações probabilísticas e validação matemática através de uma biblioteca própria desenvolvida pela equipe.

---

## 🏛️ Identificação Acadêmica

* **Instituição:** Centro Universitário de Brasília (CEUB)
* **Curso:** Análise e Desenvolvimento de Sistemas
* **Disciplina:** Matemática e Estatística para Computação
* **Professor orientador:** Prof. Romes Heriberto
* **Atividade:** Sistematização — Laboratório Estatístico Interativo

---

## 🎯 Objetivo

Proporcionar um ambiente interativo para estudo e aplicação prática de conceitos estatísticos através da análise de tráfego de rede real.

O sistema busca conectar teoria e prática por meio de:

* Estatística Descritiva
* Probabilidade
* Lei dos Grandes Números
* Teorema Central do Limite
* Correlação de Pearson
* Regressão Linear
* Coeficiente de Determinação (R²)
* Detecção de Outliers
* Análise de Distribuições

---

## 👥 Participantes do Projeto

* Iago Batista Gomes de Carvalho - 72650448 
* Marcos Paulo Dos Santos Júnior - 72650390
* Caio Sacramento Côrtes - 72650430

---

## 📂 Dataset Utilizado

### UNSW-NB15

Dataset público desenvolvido pela UNSW Canberra, em ambiente de laboratório,
com atividades normais e comportamentos de ataque produzidos para pesquisa.

Contém milhões de registros de tráfego de rede simulando atividades legítimas e ataques cibernéticos.

O arquivo carregado pela aplicação corresponde à partição oficial de
**treinamento**, com 175.341 registros: 56.000 Normais e 119.341 ataques. Cada
linha é tratada como registro de fluxo, não como pacote individual.

Links:

* Fonte oficial: https://research.unsw.edu.au/projects/unsw-nb15-dataset
* Cópia pública utilizada como referência: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

### Principais Variáveis

| Variável   | Descrição                      |
| ---------- | ------------------------------ |
| dur        | Duração da conexão             |
| spkts      | Pacotes enviados pela origem   |
| dpkts      | Pacotes enviados pelo destino  |
| sbytes     | Bytes enviados pela origem     |
| dbytes     | Bytes enviados pelo destino    |
| rate       | Taxa de transferência          |
| sttl       | Time To Live da origem         |
| dttl       | Time To Live do destino        |
| attack_cat | Categoria do ataque            |
| label      | Classificação normal ou ataque |

---

## 🚀 Funcionalidades

### Validação Visual do Núcleo

* escolha entre exemplo didático reproduzível e recorte do dataset
* fórmulas matemáticas renderizadas na interface
* comparação lado a lado com NumPy, SciPy e `statistics`
* diferença absoluta e check de equivalência numérica
* tolerância relativa de `1e-9` e absoluta de `1e-12`
* validação de medidas descritivas, relações, regressão e distribuições

As bibliotecas consolidadas são usadas somente como referências independentes
nessa página e nos testes. Os resultados analíticos dos módulos continuam sendo
produzidos pelo núcleo `minhastats.py`.

### Estatística Descritiva

* Média
* Mediana
* Moda
* Amplitude
* Variância Populacional
* Variância Amostral
* Desvio Padrão Populacional
* Desvio Padrão Amostral
* Coeficiente de Variação
* Quartis e intervalo interquartílico
* Tabelas de frequência adequadas à natureza da variável
* Gráficos específicos para variáveis contínuas, discretas e categóricas
* Detecção de valores além das cercas do IQR com interpretação cautelosa

O Módulo 2 calcula as medidas sobre todos os registros válidos da partição.
Para variáveis numéricas com caudas longas, a interface permite ampliar apenas
a região central do gráfico, sem excluir esses registros dos cálculos ou da
tabela de frequências. Valores além das cercas do IQR não são classificados
automaticamente como ataques, erros ou anomalias.

### Probabilidade e Distribuições

* Percentis
* Quartis
* IQR (Intervalo Interquartílico)
* Detecção automática de outliers
* Interpretação de assimetria

### Estatística Inferencial

* Covariância
* Correlação de Pearson
* Regressão Linear Simples
* Predição Linear
* Coeficiente de Determinação (R²)

### Demonstrações Acadêmicas

#### Lei dos Grandes Números (LGN)

Permite visualizar a convergência da frequência relativa de eventos conforme o número de observações aumenta.

Exemplos:

* Frequência de ataques DoS
* Frequência de tráfego normal
* Frequência de categorias específicas de ataques

#### Teorema Central do Limite (TCL)

Demonstra como distribuições amostrais das médias convergem para uma distribuição normal conforme o tamanho das amostras aumenta.

Aplicado sobre:

* duração das conexões
* quantidade de pacotes
* taxas de transferência
* tamanho dos pacotes

---

## 🛠 Tecnologias Utilizadas

### Linguagem

* Python 3.12+

### Interface

* Streamlit

### Computação Científica

* NumPy
* SciPy
* Pandas

### Visualização

* Matplotlib
* Altair

### Testes

* PyTest

---

## 📁 Estrutura do Projeto

```text
ceub-mec-sistematizacao/
│
├── src/
│   ├── app.py
│   ├── interface/
│   │   ├── pagina_inicial.py
│   │   ├── pagina_validacao.py
│   │   ├── editor_validacao.py
│   │   ├── modulo_2.py
│   │   ├── modulo_3.py
│   │   ├── modulo_4.py
│   │   ├── modulo_5.py
│   │   └── modulo_6.py
│   └── nucleo/
│       └── minhastats.py
│
├── tests/
│   ├── conftest.py
│   ├── test_minhastats.py
│   ├── test_editor_validacao.py
│   ├── test_pagina_inicial.py
│   ├── test_pagina_validacao.py
│   ├── test_modulo_2.py
│   ├── test_modulo_3.py
│   ├── test_modulo_4.py
│   ├── test_modulo_5.py
│   └── test_modulo_6.py
│
├── data/
│   └── UNSW_NB15_training-set.csv
│
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── verificar_regra_de_ouro.py
├── README.md
└── RELATORIO.md
```

### Arquivos Principais

#### app.py

Interface principal da aplicação utilizando Streamlit.

Responsável por:

* carregamento dos dados
* renderização dos gráficos
* interação com o usuário
* execução dos módulos estatísticos

#### pagina_inicial.py

Apresentação do projeto e orientação de navegação.

Responsável por:

* identificar a partição oficial pela cardinalidade e composição das classes
* apresentar instituição, curso, disciplina, professor, equipe e objetivo
* explicar a arquitetura, as tecnologias e a regra de ouro
* orientar o percurso entre o núcleo, a validação e os Módulos 2 a 6
* explicitar procedência, repetição de vetores e limites de interpretação

#### test_pagina_inicial.py

Valida os metadados acadêmicos, a identificação das partições oficiais, o CSV
real, o resumo de integridade e a abertura na página de apresentação.

#### pagina_validacao.py

Demonstração visual da validação do núcleo estatístico próprio.

Responsável por:

* permitir a escolha entre um exemplo didático e um recorte do dataset
* catalogar as 27 funções existentes em `minhastats.py`, incluindo o auxiliar
  interno das distribuições
* selecionar e inspecionar uma função por vez, organizada por grupo e módulo
* exibir a fórmula, a assinatura e o código real carregado do núcleo próprio
* mostrar lado a lado o código de referência, o resultado próprio e o resultado
  independente usado na comparação
* comparar com NumPy, SciPy e `statistics` quando existe equivalente adequado
* validar heurísticas e simulações por propriedades matemáticas controladas
* calcular a diferença absoluta e aplicar tolerâncias numéricas documentadas
* distinguir a evidência visual de um exemplo da cobertura ampliada pelos testes

#### editor_validacao.py

Executor seguro do minieditor exibido na página de validação.

Responsável por:

* gerar exemplos JSON coerentes com cada uma das 27 funções
* validar campos, tipos, finitude, tamanhos, domínios e custo computacional
* impedir a execução de código Python arbitrário no servidor
* chamar somente funções previamente registradas do núcleo e das referências
* produzir o histórico textual do terminal com entradas, resultados e critério
* limitar listas a 10.000 itens e simulações a 1.000.000 de sorteios elementares

#### test_pagina_validacao.py

Confirma que o catálogo corresponde exatamente às funções do núcleo, executa os
27 exemplos controlados, verifica diferentes tipos de resultado, a tolerância,
a leitura do código real, o terminal e a comparação visual lado a lado.

#### test_editor_validacao.py

Executa as 27 funções pelo minieditor e cobre JSON inválido, campos ausentes ou
extras, booleanos usados como números, NaN, limites de tamanho, pares
incompatíveis, variáveis constantes, parâmetros fora do domínio e simulações
acima do custo permitido.

#### conftest.py

Fornece uma base sintética pequena e determinística para os testes de interface.
Essa otimização evita recalcular gráficos sobre 175.341 registros em cada teste
visual. Um teste separado continua lendo o CSV real e confirma suas 45 colunas,
175.341 linhas, composição binária e nove categorias de ataque.

#### modulo_2.py

Implementação isolada do Módulo 2 — Estatística Descritiva Interativa.

Responsável por:

* descrever as variáveis utilizadas na análise
* calcular os resumos exibidos por meio do núcleo `minhastats.py`
* construir tabelas de frequências sem perder os valores extremos
* selecionar gráficos coerentes com variáveis contínuas, discretas e categóricas
* apresentar as limitações da regra do IQR e da interpretação da assimetria

#### test_modulo_2.py

Valida os resumos numéricos, as tabelas de frequências, a agregação visual de
categorias e a execução da interface do Módulo 2 com o framework de testes do
Streamlit.

#### modulo_3.py

Implementação isolada do Módulo 3 — Probabilidade e Simulação.

Responsável por:

* simular a Lei dos Grandes Números por amostragem com reposição
* demonstrar o Teorema Central do Limite com médias amostrais
* permitir a reprodução dos experimentos por meio de sementes controladas
* comparar o desvio observado das médias com o erro padrão `σ/√n`
* apresentar limites de interpretação para eventos raros e fluxos de rede

#### test_modulo_3.py

Valida a reprodutibilidade das simulações, as entradas inválidas, a preservação
dos dados originais, a convergência em experimentos controlados, o erro padrão
teórico e a execução das duas experiências na interface Streamlit.

#### modulo_4.py

Implementação isolada do Módulo 4 — Distribuições Teóricas.

Responsável por:

* estimar os parâmetros da Normal, Exponencial e Poisson com o núcleo próprio
* escolher a segunda distribuição de acordo com a natureza da variável
* comparar as curvas teóricas com a distribuição observada
* medir a distância descritiva entre as distribuições acumuladas
* preservar todos os registros no ajuste mesmo quando o gráfico amplia a região central
* explicar limites, massa em zero, caudas longas e sobredispersão

#### test_modulo_4.py

Compara as funções de densidade, massa e distribuição acumulada com o SciPy;
valida a distância entre distribuições, os parâmetros estimados, o histograma,
os filtros de tráfego e a execução das famílias contínua e discreta na
interface Streamlit.

#### modulo_5.py

Implementação isolada do Módulo 5 — Correlação e Regressão Linear.

Responsável por:

* selecionar grupos, protocolos e pares de variáveis numéricas
* preservar o pareamento de X e Y durante a preparação dos dados
* calcular Pearson, mínimos quadrados, predições e R² com `minhastats.py`
* apresentar a equação, o diagrama de dispersão e a reta ajustada
* comparar tráfego Normal com dois tipos de ataque sob o mesmo protocolo
* informar os limites de causalidade, extrapolação e classificação

#### test_modulo_5.py

Valida filtros, pareamento, casos constantes, recortes insuficientes, uma reta
de resultado conhecido, predição interativa e a comparação controlada entre
Normal, Exploits e Fuzzers na interface Streamlit. Os cálculos matemáticos do
núcleo permanecem validados contra o SciPy em `test_minhastats.py`.

#### modulo_6.py

Relatório interativo das três descobertas sustentadas pela aplicação.

Responsável por:

* caracterizar a composição da partição de treinamento
* demonstrar a assimetria da duração e comparar outliers entre classes
* comparar `spkts → dpkts` em Normal, Exploits e Fuzzers sob TCP
* avaliar a sensibilidade dos coeficientes à cauda superior até P99
* disponibilizar tabelas, gráficos e figuras para o relatório acadêmico
* separar evidências observadas de inferências que os dados não sustentam

#### test_modulo_6.py

Valida composição, integridade, medidas populacionais de duração, comparação
TCP, sensibilidade a valores extremos, colunas obrigatórias e renderização do
relatório na interface Streamlit.

#### minhastats.py

Biblioteca estatística desenvolvida pela equipe.

Implementa algoritmos próprios para:

* estatística descritiva
* probabilidade
* regressão linear
* correlação
* análise de distribuições

#### test_minhastats.py

Suíte completa de validação matemática comparando os resultados obtidos pela biblioteca própria com:

* NumPy
* SciPy
* statistics (biblioteca padrão do Python)

---

## 🔄 Fluxo da Aplicação

1. Inicialização da aplicação Streamlit
2. Carregamento do dataset UNSW-NB15
3. Seleção do módulo estatístico
4. Processamento dos dados
5. Geração dos gráficos
6. Exibição dos resultados
7. Interpretação estatística dos resultados

---

## 📊 Módulos Acadêmicos

### Módulo 1:

Fundamentos de Estatística Descritiva.

### Módulo 2:

Análise exploratória dos dados.

### Módulo 3:

#### Lei dos Grandes Números

Demonstra a convergência das frequências relativas.

#### Teorema Central do Limite

Demonstra a convergência das distribuições amostrais para a distribuição normal.

### Módulo 4:

Sobrepõe modelos teóricos aos dados observados. A Normal é comparada com a
Exponencial nas medidas não negativas e com a Poisson nas contagens de
pacotes. A interface discute a qualidade relativa do ajuste sem transformar
proximidade visual em diagnóstico de ataque ou em prova de um modelo gerador.

### Módulo 5

Permite escolher X e Y, calcular a correlação de Pearson, ajustar uma reta por
mínimos quadrados, apresentar a equação e o R² e realizar uma predição
interativa. Os filtros por grupo e protocolo tornam o escopo explícito, e a
comparação orientada mantém as mesmas variáveis e protocolo entre tráfego
Normal e dois tipos de ataque. A interface ressalta que associação linear não
demonstra causalidade nem funciona como classificador de ataques.

### Módulo 6

Consolida três descobertas: composição desigual das classes; duração fortemente
assimétrica, com valores além das cercas do IQR nos dois grupos; e diferenças
na relação `spkts → dpkts` entre Normal, Exploits e Fuzzers sob TCP. Uma análise
de sensibilidade até P99 verifica quais coeficientes permanecem estáveis quando
a influência da cauda superior é examinada.

---

## 🧪 Testes Executados

Durante a validação deste projeto foram executados todos os testes automatizados disponíveis no repositório.

### Resultado

```text
237 passed
```

### Taxa de Sucesso

* Testes executados: 237
* Testes aprovados: 237
* Falhas: 0
* Erros: 0

### Organização e Otimização da Suíte

| Arquivo | Casos coletados |
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

Os testes matemáticos usam exemplos conhecidos, bibliotecas de referência,
entradas inválidas e casos limite; esses grupos têm finalidades diferentes e
foram preservados. Os testes visuais usam uma base sintética determinística de
300 linhas, suficiente para percorrer todos os componentes sem reconstruir
gráficos sobre o CSV inteiro em cada caso. O arquivo oficial continua sendo
validado separadamente quanto a colunas, cardinalidade e composição das classes.

As duas inicializações repetidas da interface do Módulo 5 foram consolidadas em
um único fluxo: o mesmo teste confere o ajuste principal e, em seguida, ativa a
comparação controlada. Essa mudança preserva as asserções e reduz trabalho
redundante.

### Funcionalidades Validadas

#### Estatística Descritiva

* Média
* Mediana
* Moda
* Variância
* Desvio padrão
* Coeficiente de variação
* Percentis

#### Associação entre Variáveis

* Covariância
* Correlação de Pearson

#### Modelagem Estatística

* Regressão Linear
* Predição Linear
* Coeficiente de Determinação (R²)

#### Distribuições Teóricas

* Densidade e distribuição acumulada da Normal
* Densidade e distribuição acumulada da Exponencial
* Massa de probabilidade e distribuição acumulada da Poisson
* Distância entre a distribuição empírica e o modelo teórico
* Separação entre recorte visual e conjunto utilizado no ajuste

#### Casos de Borda

Validação de:

* listas vazias
* valores infinitos
* valores NaN
* variáveis constantes
* tamanhos incompatíveis
* resultados numericamente inválidos

#### Comparação Científica

Os resultados da biblioteca própria foram comparados com:

* NumPy
* SciPy
* statistics

Todos os resultados apresentaram concordância dentro das tolerâncias definidas nos testes.

---

## 📈 Qualidade Técnica Observada

### Pontos Fortes

* Implementação própria dos algoritmos estatísticos.
* Documentação interna das funções e decisões matemáticas principais.
* Tratamento robusto de erros.
* Cobertura abrangente de testes.
* Comparação dos cálculos com bibliotecas científicas reconhecidas.
* Código modularizado.
* Boa separação entre interface e lógica estatística.
* Uso de validações matemáticas rigorosas.

### Destaques

O módulo de regressão linear apresenta:

* validação de entradas
* proteção contra NaN e infinito
* prevenção de divisão por zero
* preservação da precisão numérica
* testes comparativos com SciPy

O módulo de correlação de Pearson valida corretamente cenários onde a correlação é matematicamente indefinida.

O cálculo de R² contempla inclusive casos onde o modelo é pior que utilizar simplesmente a média dos dados.

---

## 🎮 Como Executar

### Execução via Docker

A aplicação foi empacotada em um contêiner Linux leve configurado com *multi-stage build* e permissões *rootless* para garantir segurança e isolamento total de dependências.

#### Pré-requisitos
* **Linux / macOS:** Docker Engine instalado.
* **Windows:** Instale o [Docker Desktop](https://www.docker.com/products/docker-desktop/). Durante a instalação, mantenha a opção **"Use WSL 2 instead of Hyper-V"** ativada. O WSL 2 utiliza um kernel Linux real, garantindo máxima performance e compatibilidade com o contêiner.

#### Opção 1: Construir a imagem desta versão

Esta é a opção adequada para executar exatamente o código presente no fork:

```bash
docker build -t ceub-mec-sistematizacao:local .
docker run -d --name mec-stats -p 8501:8501 ceub-mec-sistematizacao:local
```

#### Opção 2: Imagem publicada pelo projeto de origem

A imagem abaixo pertence à versão `0.1.0` publicada pelo projeto de origem e
pode não conter as alterações deste fork:

```bash
docker run -d --name mec-stats -p 8501:8501 iagobgc/ceub-mec-sistematizacao:0.1.0
```

#### Acessando a Aplicação
Independente da opção escolhida, após iniciar o contêiner, abra o seu navegador e acesse:
👉 **http://localhost:8501**

#### Comandos Úteis do Docker
* Para visualizar os logs de execução da análise estatística em tempo real:
  `docker logs -f mec-stats`
* Para parar o laboratório:
  `docker stop mec-stats`
* Para reiniciar o laboratório:
  `docker start mec-stats`
* Para remover o contêiner do seu sistema:
  `docker rm -f mec-stats`

### Execução via instalação local

#### Pré-requisitos

* Python 3.12 ou superior

#### Clonar Repositório

```bash
git clone https://github.com/MarcosT2T/ceub-mec-sistematizacao.git
```

#### Entrar no Diretório

```bash
cd ceub-mec-sistematizacao
```

#### Criar Ambiente Virtual

Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### Instalar Dependências

```bash
pip install -r requirements.txt
```

Para executar também a suíte automatizada, instale as dependências de
desenvolvimento:

```bash
pip install -r requirements-dev.txt
```

#### Executar Aplicação

```bash
streamlit run src/app.py
```

ou

```bash
python -m streamlit run src/app.py
```

#### Executar os testes

```bash
python -m pytest -v
```

### Publicar no Streamlit Community Cloud

O repositório já está preparado para uma futura publicação no Streamlit
Community Cloud. A plataforma executa o aplicativo a partir da raiz do
repositório, instala as dependências declaradas e aceita o arquivo principal em
uma subpasta.

Use estas coordenadas na tela **Create app**:

| Campo | Valor |
| --- | --- |
| Repositório | `MarcosT2T/ceub-mec-sistematizacao` |
| Branch | `main` |
| Main file path | `src/app.py` |
| Python | `3.12` |

O aplicativo não exige chaves, senhas ou variáveis secretas. O arquivo de dados
necessário está versionado em `data/UNSW_NB15_training-set.csv`, e o
`requirements.txt` contém somente as seis dependências importadas diretamente
pela aplicação. As dependências exclusivas dos testes ficam separadas em
`requirements-dev.txt`, evitando instalá-las no servidor público.

Referências oficiais:

* [Organização dos arquivos no Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
* [Publicação de uma aplicação](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

---

## 🔍 Casos de Uso

* Ensino de Estatística Aplicada
* Ensino de Probabilidade
* Demonstração da Lei dos Grandes Números
* Demonstração do Teorema Central do Limite
* Introdução à Ciência de Dados
* Introdução à Cibersegurança
* Aprendizado de Regressão Linear
* Aprendizado de Correlação Estatística
* Estudos sobre tráfego de rede
* Validação de algoritmos estatísticos

---

## 📄 Licença

Consultar o repositório para informações de licenciamento.

---

## 👨‍💻 Desenvolvimento

Projeto desenvolvido para a disciplina de Matemática e Estatística para Computação do Centro Universitário de Brasília (CEUB).

Participantes:

* Iago Batista Gomes de Carvalho - 72650448 
* Marcos Paulo dos Santos Júnior - 72650390 
* Caio Sacramento Côrtes - 72650430
