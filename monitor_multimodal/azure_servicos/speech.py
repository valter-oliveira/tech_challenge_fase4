"""
speech.py — Transcrição de áudio (Azure Speech to Text) com fallback local.

Cadeia de provedores (controlada por MODO_SERVICOS no .env):
  1. azure      : Azure Cognitive Services — Speech SDK, reconhecimento contínuo pt-BR,
                  com offsets/durações por trecho (usados para taxa de fala).
  2. whisper    : faster-whisper local (opcional: pip install faster-whisper).
  3. referencia : arquivo <audio>.transcricao_referencia.txt ao lado do áudio
                  (gerado junto com os áudios sintéticos) — permite demonstrar o restante
                  do pipeline sem credenciais nem modelos pesados.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import config
from ..utils import log


@dataclass
class Transcricao:
    texto: str
    provedor: str                       # azure_speech | faster_whisper | referencia_local | nenhum
    idioma: str = config.IDIOMA_FALA
    confianca: float | None = None
    segmentos: list[dict[str, Any]] = field(default_factory=list)  # {texto, inicio_s, duracao_s}
    duracao_processamento_s: float = 0.0
    erro: str | None = None

    @property
    def n_palavras(self) -> int:
        return len(self.texto.split())

    def to_dict(self) -> dict[str, Any]:
        return {
            "texto": self.texto, "provedor": self.provedor, "idioma": self.idioma,
            "confianca": self.confianca, "n_palavras": self.n_palavras,
            "segmentos": self.segmentos, "duracao_processamento_s": round(self.duracao_processamento_s, 2),
            "erro": self.erro,
        }


def azure_speech_disponivel() -> bool:
    return bool(config.AZURE_SPEECH_KEY and config.AZURE_SPEECH_REGION)


def transcrever(caminho: Path | str, idioma: str = config.IDIOMA_FALA, modo: str | None = None) -> Transcricao:
    caminho = Path(caminho)
    modo = (modo or config.MODO_SERVICOS).lower()
    if modo in ("auto", "azure") and azure_speech_disponivel():
        try:
            return _transcrever_azure(caminho, idioma)
        except Exception as exc:  # falha de rede/credencial → fallback (exceto modo estrito)
            log.warning(f"Azure Speech falhou: {exc}")
            if modo == "azure":
                return Transcricao(texto="", provedor="azure_speech", erro=str(exc))
    elif modo == "azure":
        return Transcricao(texto="", provedor="azure_speech", erro="Credenciais AZURE_SPEECH_* ausentes")
    try:
        return _transcrever_whisper(caminho, idioma)
    except ImportError:
        pass
    except Exception as exc:
        log.warning(f"faster-whisper falhou: {exc}")
    return _transcrever_referencia(caminho, idioma)


# ─── Azure ───────────────────────────────────────────────────────────────────
def _transcrever_azure(caminho: Path, idioma: str) -> Transcricao:
    import azure.cognitiveservices.speech as speechsdk

    t0 = time.time()
    cfg = speechsdk.SpeechConfig(subscription=config.AZURE_SPEECH_KEY, region=config.AZURE_SPEECH_REGION)
    cfg.speech_recognition_language = idioma
    cfg.output_format = speechsdk.OutputFormat.Detailed
    audio = speechsdk.audio.AudioConfig(filename=str(caminho))
    rec = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio)

    segmentos: list[dict] = []
    confiancas: list[float] = []
    erro: list[str] = []
    terminou = [False]

    def on_recognized(evt):
        r = evt.result
        if r.reason == speechsdk.ResultReason.RecognizedSpeech and r.text:
            seg = {"texto": r.text, "inicio_s": r.offset / 1e7, "duracao_s": r.duration.total_seconds()}
            try:
                import json
                det = json.loads(r.json)
                nbest = det.get("NBest", [])
                if nbest:
                    confiancas.append(float(nbest[0].get("Confidence", 0)))
            except Exception:
                pass
            segmentos.append(seg)

    def on_canceled(evt):
        if evt.reason == speechsdk.CancellationReason.Error:
            erro.append(f"{evt.error_code}: {evt.error_details}")
        terminou[0] = True

    def on_stopped(evt):
        terminou[0] = True

    rec.recognized.connect(on_recognized)
    rec.canceled.connect(on_canceled)
    rec.session_stopped.connect(on_stopped)
    rec.start_continuous_recognition()
    while not terminou[0]:
        time.sleep(0.2)
    rec.stop_continuous_recognition()
    if erro:
        raise RuntimeError(erro[0])
    texto = " ".join(s["texto"] for s in segmentos).strip()
    log.info(f"[Azure Speech] {caminho.name}: {len(texto.split())} palavras em {time.time()-t0:.1f}s")
    return Transcricao(
        texto=texto, provedor="azure_speech", idioma=idioma,
        confianca=(sum(confiancas) / len(confiancas)) if confiancas else None,
        segmentos=segmentos, duracao_processamento_s=time.time() - t0,
    )


# ─── faster-whisper (opcional) ───────────────────────────────────────────────
_WHISPER = {}


def _transcrever_whisper(caminho: Path, idioma: str) -> Transcricao:
    from faster_whisper import WhisperModel  # ImportError → próximo fallback

    t0 = time.time()
    if "modelo" not in _WHISPER:
        _WHISPER["modelo"] = WhisperModel("small", device="cpu", compute_type="int8")
    segs, info = _WHISPER["modelo"].transcribe(str(caminho), language=idioma.split("-")[0], vad_filter=True)
    segmentos = [{"texto": s.text.strip(), "inicio_s": s.start, "duracao_s": s.end - s.start} for s in segs]
    texto = " ".join(s["texto"] for s in segmentos).strip()
    log.info(f"[faster-whisper] {caminho.name}: {len(texto.split())} palavras em {time.time()-t0:.1f}s")
    return Transcricao(texto=texto, provedor="faster_whisper", idioma=idioma,
                       confianca=getattr(info, "language_probability", None),
                       segmentos=segmentos, duracao_processamento_s=time.time() - t0)


# ─── Referência local ────────────────────────────────────────────────────────
def _transcrever_referencia(caminho: Path, idioma: str) -> Transcricao:
    ref = caminho.with_name(caminho.stem + ".transcricao_referencia.txt")
    if ref.exists():
        texto = ref.read_text(encoding="utf-8").strip()
        log.info(f"[referência local] {caminho.name}: transcrição de referência ({len(texto.split())} palavras)")
        return Transcricao(texto=texto, provedor="referencia_local", idioma=idioma, confianca=None)
    log.warning(f"Sem provedor de transcrição disponível para {caminho.name} "
                "(configure AZURE_SPEECH_KEY ou instale faster-whisper)")
    return Transcricao(texto="", provedor="nenhum", idioma=idioma, erro="sem provedor")
