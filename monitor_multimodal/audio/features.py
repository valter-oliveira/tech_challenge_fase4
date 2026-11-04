"""
features.py — Biomarcadores vocais extraídos do sinal de áudio (librosa).

Características (todas interpretáveis clinicamente):
  Prosódia / fluência
    taxa_silabica        : sílabas/s (picos de energia na banda 300–2500 Hz)  ↓ fadiga, disartria
    razao_pausa          : tempo em pausa / tempo total                         ↑ fadiga, dispneia
    pausas_por_min       : nº de pausas (≥150 ms) por minuto                    ↑ dispneia (frases curtas)
    duracao_media_frase  : duração média dos trechos de fala contínua (s)       ↓ dispneia
    pausa_maxima         : maior pausa (s)
  Fonação (f0 via pYIN)
    f0_mediana_hz        : altura da voz
    f0_desvio_semitons   : variabilidade de pitch                               ↓ monotonia (fadiga, disartria)
    jitter_proxy         : variação ciclo-a-ciclo de f0 (aprox. por quadro)     ↑ disartria, instabilidade
    shimmer_proxy        : variação de amplitude entre quadros vozeados         ↑ rouquidão
    hnr_db               : relação harmônico/ruído (via HPSS)                   ↓ soprosidade / esforço respiratório
  Energia / timbre
    rms_db_medio, rms_db_desvio, centroide_espectral_hz, mfcc_1..13 (médias)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
from scipy.signal import find_peaks

from .. import config


def carregar_audio(caminho: Path | str, sr: int = config.AUDIO_SR) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(str(caminho), sr=sr, mono=True)
    y = y - np.mean(y)
    pico = float(np.max(np.abs(y)))
    if pico < 1e-4:                      # silêncio digital: não normaliza (evitaria amplificar ruído numérico)
        return y.astype(np.float32), sr
    return (y / pico * 0.9).astype(np.float32), sr


def _segmentos_fala(y: np.ndarray, sr: int, top_db: float = 35.0) -> np.ndarray:
    """Intervalos [inicio, fim] (amostras) com fala, via limiar de energia (VAD simples)."""
    iv = librosa.effects.split(y, top_db=top_db, frame_length=1024, hop_length=256)
    # une intervalos separados por < 120 ms (micro-pausas intra-frase não contam como pausa)
    if len(iv) == 0:
        return iv
    unidos = [list(iv[0])]
    for a, b in iv[1:]:
        if a - unidos[-1][1] < 0.12 * sr:
            unidos[-1][1] = b
        else:
            unidos.append([a, b])
    return np.array(unidos)


def _taxa_silabica(y: np.ndarray, sr: int, segmentos: np.ndarray) -> tuple[float, int]:
    """Conta núcleos silábicos como picos do envelope de energia na banda 300–2500 Hz."""
    hop = 160  # 10 ms
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    banda = (freqs >= 300) & (freqs <= 2500)
    env = S[banda].sum(axis=0)
    env = np.convolve(env, np.hanning(7) / np.hanning(7).sum(), mode="same")
    env_db = librosa.power_to_db(env + 1e-10, ref=np.max)
    # picos distantes ≥ 110 ms e com proeminência ≥ 3 dB, acima de -30 dB
    picos, _ = find_peaks(env_db, distance=11, prominence=3.0, height=-30)
    # mantém apenas picos dentro de segmentos de fala
    t_picos = picos * hop
    dentro = np.zeros(len(t_picos), dtype=bool)
    for a, b in segmentos:
        dentro |= (t_picos >= a) & (t_picos <= b)
    n = int(dentro.sum())
    dur_fala = float(sum(b - a for a, b in segmentos) / sr) if len(segmentos) else 0.0
    return (n / dur_fala if dur_fala > 0 else 0.0), n


def extrair_features_vocais(caminho: Path | str) -> dict[str, Any]:
    y, sr = carregar_audio(caminho)
    duracao = len(y) / sr
    seg = _segmentos_fala(y, sr) if np.max(np.abs(y)) > 1e-4 else np.empty((0, 2), dtype=int)
    feats: dict[str, Any] = {"duracao_s": round(duracao, 2), "n_segmentos_fala": int(len(seg))}

    if len(seg) == 0:
        feats.update(tempo_fala_s=0.0, razao_pausa=1.0, pausas_por_min=0.0, n_pausas=0,
                     duracao_media_frase_s=0.0, pausa_maxima_s=duracao, taxa_silabica=0.0, n_silabas=0)
        return feats

    inicio, fim = seg[0][0], seg[-1][1]
    tempo_util = (fim - inicio) / sr
    tempo_fala = sum(b - a for a, b in seg) / sr
    pausas = [(seg[i + 1][0] - seg[i][1]) / sr for i in range(len(seg) - 1)]
    pausas = [p for p in pausas if p >= 0.15]
    feats.update(
        tempo_fala_s=round(float(tempo_fala), 2),
        razao_pausa=round(float(1 - tempo_fala / tempo_util), 3) if tempo_util > 0 else 0.0,
        n_pausas=len(pausas),
        pausas_por_min=round(60 * len(pausas) / tempo_util, 1) if tempo_util > 0 else 0.0,
        duracao_media_pausa_s=round(float(np.mean(pausas)), 2) if pausas else 0.0,
        pausa_maxima_s=round(float(max(pausas)), 2) if pausas else 0.0,
        duracao_media_frase_s=round(float(np.mean([(b - a) / sr for a, b in seg])), 2),
    )
    taxa, n_sil = _taxa_silabica(y, sr, seg)
    feats.update(taxa_silabica=round(float(taxa), 2), n_silabas=n_sil)

    # ── f0 (pYIN) apenas nos trechos de fala ────────────────────────────────
    y_fala = np.concatenate([y[a:b] for a, b in seg])
    f0, vflag, vprob = librosa.pyin(y_fala, fmin=65, fmax=400, sr=sr, frame_length=2048, hop_length=256)
    f0v = f0[vflag & ~np.isnan(f0)]
    if len(f0v) >= 5:
        st = 12 * np.log2(f0v / np.median(f0v))
        jitter = float(np.mean(np.abs(np.diff(f0v))) / np.mean(f0v))
        feats.update(f0_mediana_hz=round(float(np.median(f0v)), 1), f0_desvio_semitons=round(float(np.std(st)), 2),
                     f0_faixa_semitons=round(float(np.percentile(st, 95) - np.percentile(st, 5)), 2),
                     jitter_proxy=round(jitter, 4), fracao_vozeada=round(float(np.mean(vflag)), 3))
    else:
        feats.update(f0_mediana_hz=0.0, f0_desvio_semitons=0.0, f0_faixa_semitons=0.0, jitter_proxy=0.0,
                     fracao_vozeada=0.0)

    # ── energia / shimmer ───────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y_fala, frame_length=2048, hop_length=256)[0]
    rms_v = rms[: len(vflag)][vflag[: len(rms)]] if len(rms) >= len(vflag) else rms
    rms_v = rms_v[rms_v > 1e-4]
    feats.update(
        rms_db_medio=round(float(np.mean(librosa.amplitude_to_db(rms + 1e-9))), 1),
        rms_db_desvio=round(float(np.std(librosa.amplitude_to_db(rms + 1e-9))), 1),
        shimmer_proxy=round(float(np.mean(np.abs(np.diff(rms_v))) / np.mean(rms_v)), 4) if len(rms_v) > 2 else 0.0,
    )

    # ── HNR (harmônico/ruído) via separação harmônico-percussivo ────────────
    h, p = librosa.effects.hpss(y_fala)
    feats["hnr_db"] = round(float(10 * np.log10((np.sum(h**2) + 1e-9) / (np.sum(p**2) + 1e-9))), 1)

    # ── timbre ──────────────────────────────────────────────────────────────
    cent = librosa.feature.spectral_centroid(y=y_fala, sr=sr)[0]
    feats.update(centroide_espectral_hz=round(float(np.mean(cent)), 0),
                 centroide_espectral_desvio=round(float(np.std(cent)), 0))
    mfcc = librosa.feature.mfcc(y=y_fala, sr=sr, n_mfcc=13).mean(axis=1)
    feats.update({f"mfcc_{i+1}": round(float(v), 2) for i, v in enumerate(mfcc)})
    return feats


# ─── Indicadores clínicos derivados ──────────────────────────────────────────
def _sig(x: float, centro: float, largura: float, invertido: bool = False) -> float:
    """Sigmoide suave em [0,1]: 0.5 no centro; `invertido` → cresce quando x diminui."""
    z = (centro - x) / largura if invertido else (x - centro) / largura
    return float(1 / (1 + np.exp(-z)))


def indicadores_clinicos(f: dict[str, Any]) -> dict[str, float]:
    """
    Converte as características em três indicadores em [0,1], com valores de referência
    de fala conversacional adulta (taxa ~4–6 síl/s, pausas <25%, variação de pitch 2–4 st,
    HNR > 10 dB, frases > 1.5 s). Pesos definidos a partir da literatura de fala patológica.
    """
    if f.get("tempo_fala_s", 0) == 0:
        return {"fadiga": 0.0, "dispneia": 0.0, "disartria": 0.0, "qualidade_sinal": 0.0}
    lento = _sig(f["taxa_silabica"], 3.6, 0.5, invertido=True)
    muitas_pausas = _sig(f["razao_pausa"], 0.30, 0.07)
    pausas_longas = _sig(f.get("duracao_media_pausa_s", 0), 0.6, 0.15)
    monotono = _sig(f["f0_desvio_semitons"], 2.0, 0.6, invertido=True)
    frases_curtas = _sig(f["duracao_media_frase_s"], 1.4, 0.35, invertido=True)
    pausas_frequentes = _sig(f["pausas_por_min"], 18, 5)
    soproso = _sig(f["hnr_db"], 9.0, 2.5, invertido=True)
    instavel = _sig(f["jitter_proxy"], 0.030, 0.008)
    tremulo = _sig(f["shimmer_proxy"], 0.18, 0.05)

    fadiga = 0.35 * lento + 0.25 * muitas_pausas + 0.20 * pausas_longas + 0.20 * monotono
    dispneia = 0.35 * frases_curtas + 0.30 * pausas_frequentes + 0.20 * soproso + 0.15 * muitas_pausas
    disartria = 0.35 * lento + 0.25 * monotono + 0.20 * instavel + 0.20 * tremulo
    return {
        "fadiga": round(fadiga, 3), "dispneia": round(dispneia, 3), "disartria": round(disartria, 3),
        "qualidade_sinal": round(min(1.0, f.get("fracao_vozeada", 0) * 1.5), 3),
        "componentes": {
            "fala_lenta": round(lento, 2), "muitas_pausas": round(muitas_pausas, 2),
            "pausas_longas": round(pausas_longas, 2), "monotonia": round(monotono, 2),
            "frases_curtas": round(frases_curtas, 2), "pausas_frequentes": round(pausas_frequentes, 2),
            "soprosidade": round(soproso, 2), "instabilidade_f0": round(instavel, 2), "tremor_amplitude": round(tremulo, 2),
        },
    }
