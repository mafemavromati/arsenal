import os
import tempfile
import hashlib
import hmac
import asyncio
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from pydantic import BaseModel
import yt_dlp
import whisper
import anthropic
from notion_client import Client as NotionClient
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Arsenal AI — TikTok Processor")

# Carrega modelo Whisper uma vez na inicialização
print("Carregando modelo Whisper...")
whisper_model = whisper.load_model("base")
print("Whisper pronto!")

# Clientes
anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
notion = NotionClient(auth=os.environ["NOTION_TOKEN"])
NOTION_DATABASE_ID = os.environ["NOTION_DATABASE_ID"]
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")

# ─── Modelos ──────────────────────────────────────────────────────────────────

class VideoRequest(BaseModel):
    url: str
    fonte: Optional[str] = "TikTok"

class BatchRequest(BaseModel):
    urls: list[str]
    fonte: Optional[str] = "TikTok"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def verificar_assinatura(assinatura: str, corpo: bytes) -> bool:
    """Verifica o HMAC do webhook para segurança."""
    if not WEBHOOK_SECRET:
        return True
    esperado = hmac.new(WEBHOOK_SECRET.encode(), corpo, hashlib.sha256).hexdigest()
    return hmac.compare_digest(assinatura, esperado)


def baixar_audio(url: str, tmpdir: str) -> str:
    """Baixa apenas o áudio do vídeo usando yt-dlp."""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{tmpdir}/audio.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        titulo = info.get("title", "")
        descricao = info.get("description", "")

    return titulo, descricao, f"{tmpdir}/audio.mp3"


def transcrever_audio(caminho_audio: str) -> str:
    """Transcreve o áudio com Whisper."""
    resultado = whisper_model.transcribe(caminho_audio, language="pt")
    return resultado["text"].strip()


def classificar_com_claude(transcricao: str, titulo: str, descricao: str) -> dict:
    """Envia a transcrição ao Claude para classificar e estruturar."""
    prompt = f"""Você é um assistente especializado em catalogar ferramentas e tendências de IA.

Analise este vídeo e extraia as informações de forma UNIVERSAL.
Os casos de uso devem ser exemplos práticos que qualquer pessoa ou profissional poderia aplicar,
independente de área ou background. Não direcione para nichos específicos.

O vídeo pode ser em português ou inglês.

TÍTULO DO VÍDEO: {titulo}
DESCRIÇÃO: {descricao}
TRANSCRIÇÃO: {transcricao}

REGRAS PARA CASOS DE USO:
- Seja concreto e variado: cubra perfis diferentes (criador de conteúdo, pequeno empresário, profissional liberal, gestor, freelancer)
- Seja específico: "automatizar respostas de atendimento no WhatsApp" é melhor que "melhorar atendimento"
- Evite casos óbvios ou genéricos demais
- Pense em como alguém usaria isso HOJE no trabalho ou negócio

Retorne APENAS um JSON válido, sem markdown, sem backticks, com esta estrutura exata:
{{
  "nome_ferramenta": "nome da ferramenta principal mencionada, ou 'Tendência/Dica' se não for sobre ferramenta específica",
  "categoria": "exatamente uma de: IA Generativa, Automação, Produtividade, Design, Dev, Marketing, Dados, Agentes, Outro",
  "o_que_faz": "descrição objetiva em 1-2 frases sem jargão técnico",
  "casos_de_uso": "3 casos de uso práticos e específicos separados por vírgula — exemplos reais de como perfis diferentes usariam isso",
  "perfil_de_cliente": "2-3 perfis diferentes que mais se beneficiam — ex: criadores de conteúdo, donos de e-commerce, gestores de RH",
  "relevancia": número inteiro de 1 a 5 baseado em impacto potencial e acessibilidade para o público geral,
  "novidade": "exatamente uma de: 🔥 Alta, 🟡 Média, 🧊 Baixa",
  "tags": ["tag1", "tag2"],
  "para_cliente": true se pode ser recomendada para clientes de consultoria de IA, false caso contrário,
  "observacoes": "contexto adicional: limitações, preço estimado, alternativas ou por que se destaca"
}}"""

    resposta = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    import json
    texto = resposta.content[0].text.strip()
    # Remove backticks se o modelo os incluir mesmo assim
    texto = texto.replace("```json", "").replace("```", "").strip()
    return json.loads(texto)


def salvar_no_notion(dados: dict, url_tiktok: str, fonte: str, transcricao: str = "") -> str:
    """Cria uma página no database do Notion com os dados classificados."""
    print(f"[NOTION] Tentando salvar: {dados.get('nome_ferramenta')} no database {NOTION_DATABASE_ID}")
    print(f"[NOTION] Token presente: {'sim' if os.environ.get('NOTION_TOKEN') else 'NAO!'}")
    page = notion.pages.create(
        parent={"database_id": NOTION_DATABASE_ID},
        properties={
            "Nome da Ferramenta": {
                "title": [{"text": {"content": dados["nome_ferramenta"]}}]
            },
            "Categoria": {
                "select": {"name": dados["categoria"]}
            },
            "O que faz": {
                "rich_text": [{"text": {"content": dados["o_que_faz"]}}]
            },
            "Casos de Uso": {
                "rich_text": [{"text": {"content": dados["casos_de_uso"]}}]
            },
            "Perfil de Cliente": {
                "rich_text": [{"text": {"content": dados["perfil_de_cliente"]}}]
            },
            "Relevância": {
                "number": dados["relevancia"]
            },
            "Novidade": {
                "select": {"name": dados["novidade"]}
            },
            "Tags": {
                "multi_select": [{"name": t} for t in dados.get("tags", [])]
            },
            "Status": {
                "select": {"name": "📥 Na fila"}
            },
            "Para Cliente": {
                "checkbox": dados.get("para_cliente", False)
            },
            "URL TikTok": {
                "url": url_tiktok
            },
            "Fonte": {
                "select": {"name": fonte}
            },
            "Data de Descoberta": {
                "date": {"start": datetime.now().strftime("%Y-%m-%d")}
            },
            "Observações": {
                "rich_text": [{"text": {"content": dados.get("observacoes", "")}}]
            },
            "Transcrição": {
                "rich_text": [{"text": {"content": transcricao[:2000]}}]
            },
        }
    )
    print(f"[NOTION] Salvo com sucesso! URL: {page['url']}")
    return page["url"]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "whisper": "loaded", "timestamp": datetime.now().isoformat()}


def processar_em_background(url: str, fonte: str):
    """Executa o processamento completo em background — sem bloquear o Shortcut."""
    try:
        print(f"[BG] Iniciando processamento: {url}")
        with tempfile.TemporaryDirectory() as tmpdir:
            titulo, descricao, caminho_audio = baixar_audio(url, tmpdir)
            transcricao = transcrever_audio(caminho_audio)
            dados = classificar_com_claude(transcricao, titulo, descricao)
            url_notion = salvar_no_notion(dados, url, fonte, transcricao)
            print(f"[BG] Concluído! Notion: {url_notion}")
    except Exception as e:
        print(f"[BG ERRO] {type(e).__name__}: {str(e)}")


@app.post("/processar")
async def processar_video(request: VideoRequest, background_tasks: BackgroundTasks):
    """
    Endpoint assíncrono: responde imediatamente ao Shortcut e processa em background.
    Resolve o problema de timeout do iOS Shortcuts (30s).
    """
    background_tasks.add_task(processar_em_background, request.url, request.fonte)
    return {"status": "processando", "mensagem": "Vídeo recebido! O card vai aparecer no Notion em ~60 segundos."}


@app.post("/processar-sync")
async def processar_video_sync(request: VideoRequest):
    """Versão síncrona para testes via curl — aguarda o processamento completo."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Baixa o áudio
            titulo, descricao, caminho_audio = baixar_audio(request.url, tmpdir)

            # 2. Transcreve com Whisper
            transcricao = transcrever_audio(caminho_audio)

            # 3. Classifica com Claude
            dados = classificar_com_claude(transcricao, titulo, descricao)

            # 4. Salva no Notion
            url_notion = salvar_no_notion(dados, request.url, request.fonte, transcricao)

        return {
            "sucesso": True,
            "ferramenta": dados["nome_ferramenta"],
            "categoria": dados["categoria"],
            "relevancia": dados["relevancia"],
            "notion_url": url_notion,
        }

    except Exception as e:
        print(f"[ERRO SYNC] {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch")
async def processar_batch(request: BatchRequest):
    """
    Processa múltiplos vídeos salvos (para o batch do último mês).
    Roda sequencialmente para não sobrecarregar o Railway.
    """
    resultados = []
    erros = []

    for url in request.urls:
        try:
            resultado = await processar_video(VideoRequest(url=url, fonte=request.fonte))
            resultados.append(resultado)
            # Pequena pausa para não sobrecarregar APIs
            await asyncio.sleep(2)
        except Exception as e:
            erros.append({"url": url, "erro": str(e)})

    return {
        "processados": len(resultados),
        "erros": len(erros),
        "resultados": resultados,
        "erros_detalhes": erros,
    }
