FROM python:3.11-slim

# Instala ffmpeg + dependências de compilação para o Whisper
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    gcc \
    g++ \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Instala dependências pesadas primeiro (cache layer)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir openai-whisper

# Instala o resto
RUN pip install --no-cache-dir fastapi uvicorn yt-dlp anthropic notion-client python-dotenv httpx

# Pré-baixa o modelo Whisper base (evita download no primeiro request)
RUN python -c "import whisper; whisper.load_model('base')"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
