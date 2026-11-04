"""
movimento.py — Energia de movimento e fluxo óptico (OpenCV), sem modelos pesados.

Para cada frame processado:
  energia_movimento : média do |frame_t − frame_{t−1}| em escala de cinza (0-255)
  fluxo_medio       : magnitude média do fluxo óptico de Farnebäck (px/frame, em 320 px de largura)
  fluxo_vertical    : componente vertical média (positivo = para baixo) — útil para quedas

Serve como camada independente dos modelos YOLO: funciona em qualquer vídeo e
alimenta os detectores de anomalia (z-score deslizante, Isolation Forest, regras).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd

from .. import config


def reencodar_h264(caminho: Path | str) -> Path:
    """
    OpenCV grava MP4 com codec mp4v (MPEG-4 parte 2), que navegadores não reproduzem.
    Se o ffmpeg estiver instalado, reencoda em H.264 (yuv420p) no mesmo caminho; senão mantém o arquivo.
    """
    import shutil
    import subprocess

    caminho = Path(caminho)
    if shutil.which("ffmpeg") is None or not caminho.exists():
        return caminho
    tmp = caminho.with_suffix(".h264.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(caminho), "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)]
    try:
        subprocess.run(cmd, check=True, timeout=600)
        tmp.replace(caminho)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        tmp.unlink(missing_ok=True)
    return caminho


def info_video(caminho: Path | str) -> dict:
    cap = cv2.VideoCapture(str(caminho))
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {caminho}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "largura": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "altura": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duracao_s"] = info["n_frames"] / info["fps"] if info["fps"] else 0.0
    cap.release()
    return info


def iterar_frames(caminho: Path | str, passo: int = config.VIDEO_PASSO_FRAMES,
                  max_frames: int | None = None) -> Iterator[tuple[int, float, np.ndarray]]:
    """Gera (indice_frame, tempo_s, frame BGR) a cada `passo` frames."""
    cap = cv2.VideoCapture(str(caminho))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    i, n = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % passo == 0:
            yield i, i / fps, frame
            n += 1
            if max_frames and n >= max_frames:
                break
        i += 1
    cap.release()


def calcular_movimento(caminho: Path | str, passo: int = config.VIDEO_PASSO_FRAMES,
                       largura_proc: int = 320, max_frames: int | None = None) -> pd.DataFrame:
    linhas = []
    anterior = None
    for idx, t, frame in iterar_frames(caminho, passo, max_frames):
        h, w = frame.shape[:2]
        escala = largura_proc / w
        pequeno = cv2.resize(frame, (largura_proc, int(h * escala)))
        cinza = cv2.GaussianBlur(cv2.cvtColor(pequeno, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if anterior is None:
            energia, fluxo_med, fluxo_v = 0.0, 0.0, 0.0
        else:
            energia = float(np.mean(cv2.absdiff(cinza, anterior)))
            fluxo = cv2.calcOpticalFlowFarneback(anterior, cinza, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(fluxo[..., 0] ** 2 + fluxo[..., 1] ** 2)
            ativo = mag > 0.5                    # só pixels em movimento
            fluxo_med = float(mag[ativo].mean()) if ativo.any() else 0.0
            fluxo_v = float(fluxo[..., 1][ativo].mean()) if ativo.any() else 0.0
        linhas.append({"frame": idx, "tempo_s": round(t, 3), "energia_movimento": round(energia, 3),
                       "fluxo_medio": round(fluxo_med, 3), "fluxo_vertical": round(fluxo_v, 3)})
        anterior = cinza
    return pd.DataFrame(linhas)
