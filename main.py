import os
import tempfile
import hashlib
import hmac
import asyncio
import subprocess
import base64
import glob as glob_module
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import yt_dlp
import anthropic
from openai import OpenAI
from notion_client import Client as NotionClient
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Arsenal AI — Video Processor")

# Clientes
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
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

class EnrichRequest(BaseModel):
    page_id: str

class EnrichBatchRequest(BaseModel):
    limite: Optional[int] = 50

# ─── Helpers ──────────────────────────────────────────────────────────────────

def verificar_assinatura(assinatura: str, corpo: bytes) -> bool:
    """Verifica o HMAC do webhook para segurança."""
    if not WEBHOOK_SECRET:
        return True
    esperado = hmac.new(WEBHOOK_SECRET.encode(), corpo, hashlib.sha256).hexdigest()
    return hmac.compare_digest(assinatura, esperado)


def baixar_video(url: str, tmpdir: str) -> tuple:
    """Baixa o vídeo (até 720p), extrai o áudio e captura a URL do thumbnail."""
    ydl_opts = {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "outtmpl": f"{tmpdir}/video.%(ext)s",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        titulo = info.get("title", "")
        descricao = info.get("description", "")
        thumbnail_url = info.get("thumbnail", "")

    arquivos = glob_module.glob(f"{tmpdir}/video.*")
    caminho_video = arquivos[0] if arquivos else None

    caminho_audio = f"{tmpdir}/audio.mp3"
    subprocess.run(
        ["ffmpeg", "-i", caminho_video, "-vn", "-ar", "16000", "-ac", "1",
         "-b:a", "128k", caminho_audio, "-y", "-loglevel", "quiet"],
        check=True,
    )

    return titulo, descricao, caminho_video, caminho_audio, thumbnail_url


def extrair_frames(caminho_video: str, tmpdir: str, n: int = 5) -> list[str]:
    """Extrai N frames distribuídos ao longo do vídeo."""
    import json as _json

    resultado = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", caminho_video],
        capture_output=True, text=True, check=True,
    )
    duracao = float(_json.loads(resultado.stdout)["format"]["duration"])

    frames = []
    for i in range(n):
        t = duracao * (i + 1) / (n + 1)
        caminho_frame = f"{tmpdir}/frame_{i:02d}.jpg"
        subprocess.run(
            ["ffmpeg", "-ss", str(t), "-i", caminho_video,
             "-vframes", "1", "-q:v", "2", caminho_frame, "-y", "-loglevel", "quiet"],
            check=True,
        )
        if os.path.exists(caminho_frame):
            frames.append(caminho_frame)

    return frames


def frame_para_base64(caminho: str) -> str:
    with open(caminho, "rb") as f:
        return base64.b64encode(f.read()).decode()


def transcrever_audio(caminho_audio: str) -> str:
    """Transcreve o áudio via OpenAI Whisper API."""
    with open(caminho_audio, "rb") as f:
        resultado = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="pt",
        )
    return resultado.text.strip()


def classificar_com_claude(
    transcricao: str, titulo: str, descricao: str, frames: list[str] = None
) -> dict:
    """Classifica o conteúdo com Claude usando transcrição + frames visuais."""
    tem_transcricao = len(transcricao.strip()) > 50

    prompt = f"""Você é um assistente especializado em catalogar conteúdo de vídeo salvo para referência futura.

O vídeo pode ser sobre qualquer tema relevante: ferramentas de IA, dicas de marketing, inspirações de conteúdo, estratégias de negócio, produtividade, design, desenvolvimento, dados, etc.
{"Use os frames do vídeo como fonte principal — o áudio tem pouco ou nenhum conteúdo textual." if not tem_transcricao else "Use tanto a transcrição quanto os frames para uma análise completa."}

O vídeo pode ser em português ou inglês.

TÍTULO: {titulo}
DESCRIÇÃO: {descricao}
TRANSCRIÇÃO: {transcricao if transcricao else "(sem conteúdo de áudio)"}

Retorne APENAS um JSON válido, sem markdown, sem backticks, com esta estrutura exata:
{{
  "nome_ferramenta": "título curto e descritivo para este conteúdo (nome de ferramenta, tema da dica, assunto da inspiração, etc.)",
  "categoria": "exatamente uma de: Ferramenta IA, Automação, Marketing, Conteúdo, Produtividade, Design, Dev, Dados, Negócios, Outro",
  "o_que_faz": "resumo objetivo em 1-2 frases do que este vídeo ensina ou mostra",
  "casos_de_uso": "3 aplicações práticas e específicas separadas por vírgula",
  "perfil_de_cliente": "2-3 perfis que mais se beneficiam deste conteúdo",
  "relevancia": número inteiro de 1 a 10 baseado em impacto e aplicabilidade,
  "novidade": "exatamente uma de: 🔥 Alta, 🟡 Média, 🧊 Baixa",
  "tags": ["tag1", "tag2"],
  "observacoes": "contexto adicional, limitações, alternativas ou por que se destaca"
}}"""

    content = []
    if frames:
        content.append({"type": "text", "text": "Frames do vídeo para análise visual:"})
        for caminho in frames:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame_para_base64(caminho),
                },
            })
    content.append({"type": "text", "text": prompt})

    resposta = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": content}],
    )

    import json
    texto = resposta.content[0].text.strip()
    texto = texto.replace("```json", "").replace("```", "").strip()
    return json.loads(texto)


def enriquecer_com_claude(titulo: str, observacoes: str, categoria: str, url_fonte: str) -> dict:
    """Gera campos faltantes a partir do título e observações já existentes."""
    prompt = f"""Você é um assistente especializado em catalogar conteúdo de referência.

A entrada abaixo foi criada com dados parciais. Com base nas informações disponíveis, preencha os campos faltantes.

TÍTULO: {titulo}
CATEGORIA: {categoria}
URL: {url_fonte}
OBSERVAÇÕES (análise editorial existente):
{observacoes}

Retorne APENAS um JSON válido, sem markdown, sem backticks:
{{
  "o_que_faz": "resumo objetivo em 1-2 frases do que este conteúdo ou ferramenta faz",
  "casos_de_uso": "3 aplicações práticas e específicas separadas por vírgula",
  "perfil_de_cliente": "2-3 perfis que mais se beneficiam",
  "novidade": "exatamente uma de: 🔥 Alta, 🟡 Média, 🧊 Baixa"
}}"""

    resposta = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    import json
    texto = resposta.content[0].text.strip()
    texto = texto.replace("```json", "").replace("```", "").strip()
    return json.loads(texto)


def salvar_no_notion(
    dados: dict, url_video: str, fonte: str, transcricao: str = "", thumbnail_url: str = ""
) -> str:
    """Cria uma página no Notion com os dados classificados e thumbnail como cover."""
    print(f"[NOTION] Salvando: {dados.get('nome_ferramenta')} | {dados.get('categoria')}")

    create_params = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Título": {
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
            "URL Fonte": {
                "url": url_video
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
        },
    }

    if thumbnail_url:
        create_params["cover"] = {"type": "external", "external": {"url": thumbnail_url}}

    page = notion.pages.create(**create_params)
    print(f"[NOTION] Salvo! URL: {page['url']}")
    return page["url"]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "whisper": "api",
        "vision": "enabled",
        "enrich": "enabled",
        "timestamp": datetime.now().isoformat(),
    }


def processar_em_background(url: str, fonte: str):
    """Executa o processamento completo em background — sem bloquear o Shortcut."""
    try:
        print(f"[BG] Iniciando: {url}")
        with tempfile.TemporaryDirectory() as tmpdir:
            titulo, descricao, caminho_video, caminho_audio, thumbnail_url = baixar_video(url, tmpdir)
            transcricao = transcrever_audio(caminho_audio)
            frames = extrair_frames(caminho_video, tmpdir)
            dados = classificar_com_claude(transcricao, titulo, descricao, frames)
            url_notion = salvar_no_notion(dados, url, fonte, transcricao, thumbnail_url)
            print(f"[BG] Concluído! {dados['nome_ferramenta']} → {url_notion}")
    except Exception as e:
        print(f"[BG ERRO] {type(e).__name__}: {str(e)}")


@app.post("/processar")
async def processar_video(request: VideoRequest, background_tasks: BackgroundTasks):
    """Endpoint assíncrono — responde imediatamente, processa em background."""
    background_tasks.add_task(processar_em_background, request.url, request.fonte)
    return {"status": "processando", "mensagem": "Vídeo recebido! O card vai aparecer no Notion em ~60 segundos."}


@app.post("/processar-sync")
async def processar_video_sync(request: VideoRequest):
    """Versão síncrona para testes via curl."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            titulo, descricao, caminho_video, caminho_audio, thumbnail_url = baixar_video(request.url, tmpdir)
            transcricao = transcrever_audio(caminho_audio)
            frames = extrair_frames(caminho_video, tmpdir)
            dados = classificar_com_claude(transcricao, titulo, descricao, frames)
            url_notion = salvar_no_notion(dados, request.url, request.fonte, transcricao, thumbnail_url)

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


@app.post("/enriquecer")
async def enriquecer_entrada(request: EnrichRequest):
    """Preenche campos vazios de uma entrada existente usando Claude."""
    try:
        page = notion.pages.retrieve(request.page_id)
        props = page["properties"]

        def get_text(prop_name):
            items = props.get(prop_name, {}).get("rich_text", [])
            return items[0]["text"]["content"] if items else ""

        def get_select(prop_name):
            sel = props.get(prop_name, {}).get("select")
            return sel["name"] if sel else ""

        def get_title(prop_name):
            items = props.get(prop_name, {}).get("title", [])
            return items[0]["text"]["content"] if items else ""

        def get_url(prop_name):
            return props.get(prop_name, {}).get("url", "") or ""

        titulo = get_title("Título")
        observacoes = get_text("Observações")
        o_que_faz = get_text("O que faz")
        categoria = get_select("Categoria")
        url_fonte = get_url("URL Fonte")

        if o_que_faz:
            return {"status": "skip", "mensagem": "Entrada já está completa.", "page_id": request.page_id}

        if not observacoes and not titulo:
            return {"status": "skip", "mensagem": "Dados insuficientes para enriquecer.", "page_id": request.page_id}

        dados = enriquecer_com_claude(titulo, observacoes, categoria, url_fonte)

        update_props = {}
        if dados.get("o_que_faz"):
            update_props["O que faz"] = {"rich_text": [{"text": {"content": dados["o_que_faz"]}}]}
        if dados.get("casos_de_uso"):
            update_props["Casos de Uso"] = {"rich_text": [{"text": {"content": dados["casos_de_uso"]}}]}
        if dados.get("perfil_de_cliente"):
            update_props["Perfil de Cliente"] = {"rich_text": [{"text": {"content": dados["perfil_de_cliente"]}}]}
        if dados.get("novidade") and not get_select("Novidade"):
            update_props["Novidade"] = {"select": {"name": dados["novidade"]}}

        notion.pages.update(request.page_id, properties=update_props)
        print(f"[ENRICH] ✅ {titulo} — campos: {list(update_props.keys())}")

        return {
            "status": "ok",
            "page_id": request.page_id,
            "titulo": titulo,
            "campos_preenchidos": list(update_props.keys()),
        }

    except Exception as e:
        print(f"[ENRICH ERRO] {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def enriquecer_batch_em_background(limite: int):
    """Busca entradas com 'O que faz' vazio e enriquece em batch."""
    import time

    try:
        print(f"[ENRICH-BATCH] Buscando até {limite} entradas vazias...")
        response = notion.databases.query(
            database_id=NOTION_DATABASE_ID,
            filter={"property": "O que faz", "rich_text": {"is_empty": True}},
            page_size=min(limite, 100),
        )
        pages = response.get("results", [])
        print(f"[ENRICH-BATCH] {len(pages)} entradas encontradas")

        ok, skip, erros = 0, 0, 0
        for page in pages:
            try:
                props = page["properties"]

                def get_text(prop_name):
                    items = props.get(prop_name, {}).get("rich_text", [])
                    return items[0]["text"]["content"] if items else ""

                def get_select(prop_name):
                    sel = props.get(prop_name, {}).get("select")
                    return sel["name"] if sel else ""

                def get_title(prop_name):
                    items = props.get(prop_name, {}).get("title", [])
                    return items[0]["text"]["content"] if items else ""

                def get_url(prop_name):
                    return props.get(prop_name, {}).get("url", "") or ""

                titulo = get_title("Título")
                observacoes = get_text("Observações")
                categoria = get_select("Categoria")
                url_fonte = get_url("URL Fonte")

                if not observacoes and not titulo:
                    skip += 1
                    continue

                dados = enriquecer_com_claude(titulo, observacoes, categoria, url_fonte)

                update_props = {}
                if dados.get("o_que_faz"):
                    update_props["O que faz"] = {"rich_text": [{"text": {"content": dados["o_que_faz"]}}]}
                if dados.get("casos_de_uso"):
                    update_props["Casos de Uso"] = {"rich_text": [{"text": {"content": dados["casos_de_uso"]}}]}
                if dados.get("perfil_de_cliente"):
                    update_props["Perfil de Cliente"] = {"rich_text": [{"text": {"content": dados["perfil_de_cliente"]}}]}
                if dados.get("novidade") and not get_select("Novidade"):
                    update_props["Novidade"] = {"select": {"name": dados["novidade"]}}

                if update_props:
                    notion.pages.update(page["id"], properties=update_props)
                    ok += 1
                    print(f"[ENRICH-BATCH] ✅ {titulo}")
                else:
                    skip += 1

                time.sleep(0.5)  # respeita rate limit da Notion API

            except Exception as e:
                erros += 1
                print(f"[ENRICH-BATCH ERRO] {page.get('id')}: {str(e)}")

        print(f"[ENRICH-BATCH] Concluído — ok:{ok} skip:{skip} erros:{erros}")

    except Exception as e:
        print(f"[ENRICH-BATCH FATAL] {type(e).__name__}: {str(e)}")


@app.post("/enriquecer-batch")
async def enriquecer_batch(request: EnrichBatchRequest, background_tasks: BackgroundTasks):
    """Enriquece em batch todas as entradas com 'O que faz' vazio."""
    background_tasks.add_task(enriquecer_batch_em_background, request.limite)
    return {
        "status": "processando",
        "mensagem": f"Enriquecimento iniciado para até {request.limite} entradas. Acompanhe nos logs do Railway.",
    }


def transcrever_audio_longo(caminho_audio: str) -> str:
    """Transcreve áudios longos dividindo em chunks de 24 MB."""
    tamanho = os.path.getsize(caminho_audio)
    limite = 24 * 1024 * 1024  # 24 MB

    if tamanho <= limite:
        with open(caminho_audio, "rb") as f:
            return openai_client.audio.transcriptions.create(
                model="whisper-1", file=f, language="pt"
            ).text.strip()

    # Descobre duração total
    import json as _json
    resultado = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", caminho_audio],
        capture_output=True, text=True, check=True,
    )
    duracao = float(_json.loads(resultado.stdout)["format"]["duration"])

    # Calcula quantos chunks são necessários
    n_chunks = -(-tamanho // limite)  # divisão com teto
    dur_chunk = duracao / n_chunks

    transcricoes = []
    with tempfile.TemporaryDirectory() as tmpchunks:
        for i in range(n_chunks):
            inicio = i * dur_chunk
            chunk_path = os.path.join(tmpchunks, f"chunk_{i:02d}.mp3")
            subprocess.run(
                ["ffmpeg", "-ss", str(inicio), "-t", str(dur_chunk),
                 "-i", caminho_audio, "-ar", "16000", "-ac", "1",
                 chunk_path, "-y", "-loglevel", "quiet"],
                check=True,
            )
            print(f"[TRANSCRICAO] Chunk {i+1}/{n_chunks}...")
            with open(chunk_path, "rb") as f:
                texto = openai_client.audio.transcriptions.create(
                    model="whisper-1", file=f, language="pt"
                ).text.strip()
            transcricoes.append(texto)

    return " ".join(transcricoes)


@app.post("/transcrever")
async def transcrever_video(request: VideoRequest):
    """Transcreve o áudio de qualquer vídeo, incluindo vídeos longos (chunking automático)."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": f"{tmpdir}/audio.%(ext)s",
                "postprocessors": [{"key": "FFmpegExtractAudio",
                                    "preferredcodec": "mp3", "preferredquality": "64"}],
                "quiet": True, "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(request.url, download=True)
                titulo = info.get("title", "")

            arquivos = glob_module.glob(f"{tmpdir}/audio.*")
            caminho_audio = arquivos[0] if arquivos else f"{tmpdir}/audio.mp3"

            print(f"[TRANSCRICAO] Iniciando: {titulo} ({os.path.getsize(caminho_audio) / 1024 / 1024:.1f} MB)")
            transcricao = transcrever_audio_longo(caminho_audio)

        return {"titulo": titulo, "transcricao": transcricao, "caracteres": len(transcricao)}

    except Exception as e:
        print(f"[TRANSCRICAO ERRO] {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch")
async def processar_batch(request: BatchRequest):
    """Processa múltiplos vídeos sequencialmente."""
    resultados = []
    erros = []

    for url in request.urls:
        try:
            resultado = await processar_video(VideoRequest(url=url, fonte=request.fonte))
            resultados.append(resultado)
            await asyncio.sleep(2)
        except Exception as e:
            erros.append({"url": url, "erro": str(e)})

    return {
        "processados": len(resultados),
        "erros": len(erros),
        "resultados": resultados,
        "erros_detalhes": erros,
    }
