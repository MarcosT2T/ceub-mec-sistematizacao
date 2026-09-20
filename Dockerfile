# ==============================================================================
# Dockerfile — ceub-mec-sistematizacao. (Linux container)
# ==============================================================================
FROM python:3.12-slim AS builder

# build-essential = compilador C/C++ (gcc, make, etc). se algum pacote não tiver wheel pronta, o pip vai conseguir compilar do zero.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cria um ambiente virtual dentro da imagem 
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copia apenas o requirements.txt antes do resto do código. No cache
# de camadas do Docker, cada instrução do Dockerfile vira uma "camada", e o
# Docker só reexecuta uma camada (e as que vêm depois dela) se o conteúdo que
# ela usa tiver mudado desde o último build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (OPCIONAL) — "test": portão de qualidade antes da imagem final
#
# docker build --target test -t ceub-mec-sistematizacao:test .
#
# Se qualquer um dos 237 testes falhar, OU se a regra de ouro do projeto for
# violada, o RUN abaixo retorna código de saída diferente de zero 
# e o build para exatamente aqui — antes de qualquer imagem de produção ser gerada.
# ---------------------------------------------------------------------------
FROM builder AS test
 
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt
 
COPY src/ ./src/
COPY tests/ ./tests/
COPY data/ ./data/
COPY pytest.ini verificar_regra_de_ouro.py ./
 
RUN python -m pytest -q \
    && python verificar_regra_de_ouro.py

# Importante: usa a MESMA tag de imagem base (python:3.12-slim) nas duas
# etapas. O ambiente virtual criado na etapa anterior guarda links simbólicos
# apontando para o Python "de sistema" (ex: /usr/local/bin/python3.12). Se a
# etapa final usasse uma imagem base diferente, esses links quebrariam.
FROM python:3.12-slim

# Cria um usuário e grupo dedicados, sem privilégios de root, para rodar a
# aplicação para diminuir a superficie de ataque caso o contêiner seja atacado.
RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app --create-home app

# Copia o ambiente virtual já pronto (com todas as dependências instaladas)
# da etapa builder — sem compilador, sem cache do pip, sem listas do apt.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copia o código-fonte e o dataset preservando a MESMA estrutura de pastas do
# repositório (src/ e data/ lado a lado)
COPY src/ ./src/
COPY data/ ./data/

# Variáveis de ambiente:
# - PYTHONDONTWRITEBYTECODE: não gera arquivos .pyc — desnecessários num
#   container que é recriado do zero a cada deploy.
# - PYTHONUNBUFFERED: manda os prints/logs do Python direto para o stdout
#   sem buffer interno, essencial para `docker logs` mostrar tudo em tempo
#   real.
# - MPLCONFIGDIR: o matplotlib tenta escrever um cache de fontes na pasta
#   HOME do usuário na primeira execução. Como "app" é um usuário sem
#   privilégios, é necessário aponta esse cache para /tmp (sempre gravável por
#   qualquer usuário) para evitar erro de permissão na primeira chamada de
#   plt.subplots().
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

# Garante que o usuário "app" seja dono de tudo dentro de /app antes de
# trocar de usuário.
RUN chown -R app:app /app

# A partir daqui, todo comando roda como "app", não como root.
USER app

# Documenta que a aplicação escuta na porta 8501, a porta padrão do Streamlit.
EXPOSE 8501

# Verifica periodicamente se o Streamlit está respondendo. Com isso,
# `docker ps` passa a mostrar "healthy"/"unhealthy" no lugar de só "Up", e o
# Docker Compose consegue reagir a um container travado (ex: reiniciar).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
