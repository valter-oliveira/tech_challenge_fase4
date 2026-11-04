"""
dados_sinteticos.py — Geração de dados sintéticos com anomalias ROTULADAS.

Por que sintético?
  Dados reais de pacientes (PhysioNet/MIMIC) exigem credenciamento e não podem ser
  redistribuídos. Os geradores abaixo reproduzem padrões fisiológicos plausíveis
  (ritmo circadiano, ruído de sensor, deterioração progressiva) e injetam anomalias
  com rótulo (`anomalia_injetada`), o que permite medir precisão/recall dos detectores.

Pacientes simulados:
  P001 — estável (controle)
  P002 — deterioração progressiva tipo sepse (FC↑, PA↓, T↑, SpO₂↓, FR↑) + prescrições com erros
  P003 — episódios agudos (dessaturação, taquiarritmia) + artefatos de sensor + agitação/queda

Também gera:
  • áudios de consulta via TTS do macOS (`say`, voz pt-BR) — normal, fadiga, dispneia
  • vídeo sintético de "sessão de fisioterapia" com evento de queda (para testar o pipeline)
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .utils import log

RNG = np.random.default_rng(config.RANDOM_STATE)
OUT = config.DADOS_SINTETICOS_DIR


# ─── Sinais vitais ───────────────────────────────────────────────────────────
def gerar_sinais_vitais(paciente_id: str, perfil: str, horas: int = 48, passo_min: int = 5,
                        inicio: str = "2026-11-01 08:00") -> pd.DataFrame:
    n = horas * 60 // passo_min
    t = pd.date_range(inicio, periods=n, freq=f"{passo_min}min")
    h = np.arange(n) * passo_min / 60.0                      # horas decorridas
    circ = np.sin(2 * np.pi * (h - 6) / 24)                   # ritmo circadiano

    fc = 72 + 6 * circ + RNG.normal(0, 2.5, n)
    pas = 118 + 6 * circ + RNG.normal(0, 4, n)
    pad = 76 + 3 * circ + RNG.normal(0, 3, n)
    spo2 = 97 + 0.5 * circ + RNG.normal(0, 0.6, n)
    fr = 15 + 1.5 * circ + RNG.normal(0, 0.9, n)
    temp = 36.6 + 0.3 * circ + RNG.normal(0, 0.1, n)
    rotulo = np.zeros(n, dtype=bool)
    tipo = np.array([""] * n, dtype=object)

    def janela(h_ini, h_fim):
        return (h >= h_ini) & (h < h_fim)

    if perfil == "deterioracao":
        # a partir da hora 24, deterioração progressiva (sepse): rampa de 0→1 em 12h
        ramp = np.clip((h - 24) / 12, 0, 1)
        fc += 45 * ramp
        pas -= 35 * ramp
        pad -= 20 * ramp
        spo2 -= 8 * ramp
        fr += 12 * ramp
        temp += 2.6 * ramp
        rotulo |= ramp > 0.25
        tipo[ramp > 0.25] = "deterioracao_progressiva"
    elif perfil == "episodios":
        # dessaturação aguda 10h-10h40
        m = janela(10, 10.7)
        spo2[m] -= 9
        fr[m] += 7
        rotulo |= m
        tipo[m] = "dessaturacao_aguda"
        # taquiarritmia 30h-30h30
        m = janela(30, 30.5)
        fc[m] += 55
        pas[m] -= 15
        rotulo |= m
        tipo[m] = "taquiarritmia"
        # febre noturna 40h-44h
        m = janela(40, 44)
        temp[m] += 2.0
        fc[m] += 15
        rotulo |= m
        tipo[m] = "febre"
        # artefatos de sensor (pontuais): leitura absurda isolada
        for idx in RNG.choice(np.flatnonzero(~rotulo), 3, replace=False):
            spo2[idx] = 60 + RNG.uniform(0, 10)
            rotulo[idx] = True
            tipo[idx] = "artefato_sensor"
    # perfil "estavel": só ruído

    df = pd.DataFrame({
        "timestamp": t,
        "paciente_id": paciente_id,
        "frequencia_cardiaca": np.round(fc, 0),
        "pressao_sistolica": np.round(pas, 0),
        "pressao_diastolica": np.round(pad, 0),
        "saturacao_o2": np.round(np.clip(spo2, 50, 100), 0),
        "frequencia_respiratoria": np.round(np.clip(fr, 4, 45), 0),
        "temperatura": np.round(temp, 1),
        "anomalia_injetada": rotulo,
        "tipo_injetado": tipo,
    })
    return df


# ─── Prescrições ─────────────────────────────────────────────────────────────
_ESQUEMAS = {  # medicamento: (dose_mg típica, frequência/dia, via)
    "paracetamol": (500, 4, "oral"), "dipirona": (500, 4, "oral"), "metformina": (850, 2, "oral"),
    "enalapril": (10, 2, "oral"), "omeprazol": (20, 1, "oral"), "amoxicilina": (500, 3, "oral"),
    "ceftriaxona": (1000, 2, "iv"), "insulina": (10, 3, "sc"), "morfina": (5, 4, "iv"),
    "varfarina": (5, 1, "oral"), "heparina": (5000, 3, "sc"), "furosemida": (40, 2, "iv"),
    "losartana": (50, 1, "oral"), "atenolol": (50, 1, "oral"),
}
_PRESCRITORES = ["Dr. Almeida", "Dra. Souza", "Dr. Lima", "Dra. Castro"]


def gerar_prescricoes(paciente_id: str, perfil: str, dias: int = 5, inicio: str = "2026-11-01") -> pd.DataFrame:
    meds = {
        "estavel": ["paracetamol", "metformina", "enalapril", "omeprazol", "atenolol"],
        "deterioracao": ["ceftriaxona", "dipirona", "omeprazol", "heparina", "furosemida"],
        "episodios": ["dipirona", "insulina", "amoxicilina", "losartana", "furosemida"],
    }[perfil]
    linhas = []
    base = pd.Timestamp(inicio)
    for d in range(dias):
        for med in meds:
            dose, freq, via = _ESQUEMAS[med]
            hora = int(RNG.integers(7, 11))
            linhas.append({
                "timestamp": base + pd.Timedelta(days=d, hours=hora, minutes=int(RNG.integers(0, 59))),
                "paciente_id": paciente_id, "medicamento": med,
                "dose_mg": float(dose), "frequencia_dia": freq, "via": via,
                "prescritor": _PRESCRITORES[d % 2], "anomalia_injetada": False, "tipo_injetado": "",
            })
    df = pd.DataFrame(linhas)

    def adicionar(ts, med, dose, freq, via, prescritor, tipo):
        nonlocal df
        df = pd.concat([df, pd.DataFrame([{
            "timestamp": pd.Timestamp(ts), "paciente_id": paciente_id, "medicamento": med, "dose_mg": dose,
            "frequencia_dia": freq, "via": via, "prescritor": prescritor,
            "anomalia_injetada": True, "tipo_injetado": tipo}])], ignore_index=True)

    if perfil == "deterioracao":
        # erro de digitação: dipirona 5000 mg em vez de 500 mg (10x) → dose acima do máximo
        adicionar(base + pd.Timedelta(days=2, hours=14), "dipirona", 5000, 4, "oral", "Dr. Lima", "dose_10x")
        # varfarina iniciada com heparina ativa → interação crítica
        adicionar(base + pd.Timedelta(days=3, hours=9), "varfarina", 5, 1, "oral", "Dr. Almeida", "interacao")
        # ceftriaxona dobrada abruptamente (escalonamento) → aumento abrupto
        adicionar(base + pd.Timedelta(days=3, hours=16), "ceftriaxona", 2000, 2, "iv", "Dra. Castro", "aumento_abrupto")
    elif perfil == "episodios":
        # duplicidade: paracetamol por dois prescritores no mesmo dia
        med0 = meds[0]
        adicionar(base + pd.Timedelta(days=1, hours=20), med0, _ESQUEMAS[med0][0], _ESQUEMAS[med0][1],
                  _ESQUEMAS[med0][2], "Dra. Castro", "duplicidade")
        # troca de via inesperada
        med1 = meds[1]
        adicionar(base + pd.Timedelta(days=2, hours=22), med1, _ESQUEMAS[med1][0], _ESQUEMAS[med1][1],
                  "iv" if _ESQUEMAS[med1][2] != "iv" else "oral", "Dr. Lima", "troca_via")
    return df.sort_values("timestamp").reset_index(drop=True)


# ─── Movimentação ────────────────────────────────────────────────────────────
def gerar_movimentacao(paciente_id: str, perfil: str, horas: int = 72, inicio: str = "2026-11-01 00:00") -> pd.DataFrame:
    t = pd.date_range(inicio, periods=horas, freq="h")
    hora = t.hour.to_numpy()
    diurno = (hora >= 7) & (hora < 22)
    atividade = np.where(diurno, RNG.normal(0.45, 0.10, horas), RNG.normal(0.12, 0.05, horas))
    mudancas = np.where(diurno, RNG.poisson(3, horas), RNG.poisson(1.2, horas))
    fora = (diurno & (RNG.random(horas) < 0.25)).astype(int)
    rotulo = np.zeros(horas, dtype=bool)
    tipo = np.array([""] * horas, dtype=object)

    if perfil == "deterioracao":
        # imobilidade de 7h (horas 30-37) — paciente prostrado
        m = slice(30, 37)
        atividade[m] = RNG.normal(0.03, 0.01, 7)
        mudancas[m] = 0
        fora[m] = 0
        rotulo[m] = True
        tipo[m] = "imobilidade_prolongada"
    elif perfil == "episodios":
        # agitação noturna (hora 50 = dia 3, 02h), seguida de queda na hora 51
        atividade[49:51] = [0.88, 0.92]
        mudancas[49:51] = [7, 9]
        rotulo[49:51] = True
        tipo[49:51] = "agitacao_noturna"
        atividade[51] = 0.02
        mudancas[51] = 0
        fora[51] = 1
        rotulo[51] = True
        tipo[51] = "queda"
    return pd.DataFrame({
        "timestamp": t, "paciente_id": paciente_id,
        "indice_atividade": np.round(np.clip(atividade, 0, 1), 3),
        "mudancas_decubito": mudancas.astype(int), "fora_do_leito": fora,
        "anomalia_injetada": rotulo, "tipo_injetado": tipo,
    })


# ─── Áudio (TTS do macOS) ────────────────────────────────────────────────────
ROTEIROS_AUDIO = {
    "consulta_normal": dict(
        voz="Luciana", rate=175,
        texto="Bom dia, doutora. Estou me sentindo bem. Dormi normalmente, não tive dor nem febre. "
              "Tomei os remédios nos horários certos e consegui caminhar um pouco no corredor hoje de manhã. "
              "O apetite está bom e não tenho nenhuma queixa.",
    ),
    "consulta_fadiga": dict(
        voz="Luciana", rate=105,
        texto="Doutora [[slnc 600]] eu estou muito cansada. [[slnc 900]] Não consigo nem levantar da cama. "
              "[[slnc 800]] Qualquer esforço [[slnc 700]] me deixa exausta. [[slnc 1000]] Sinto uma fraqueza "
              "nas pernas [[slnc 700]] e uma tontura quando fico de pé. [[slnc 900]] Esse cansaço extremo "
              "não melhora com o descanso.",
    ),
    "consulta_dispneia": dict(
        voz="Eddy", rate=120,
        texto="Doutor [[slnc 500]] estou com falta de ar. [[slnc 1200]] Não consigo respirar direito "
              "[[slnc 1000]] desde ontem à noite. [[slnc 1100]] Sinto uma dor no peito [[slnc 800]] quando "
              "respiro fundo [[slnc 1200]] e meus lábios ficaram roxos. [[slnc 900]] Estou com muito medo.",
    ),
}


def gerar_audios(forcar: bool = False) -> list[Path]:
    """Gera WAVs de consulta com o TTS do macOS (`say`). Em outros sistemas, gera tom sintético."""
    pasta = OUT / "audio"
    pasta.mkdir(parents=True, exist_ok=True)
    saidas = []
    tem_say = shutil.which("say") is not None
    for nome, r in ROTEIROS_AUDIO.items():
        wav = pasta / f"{nome}.wav"
        txt = pasta / f"{nome}.transcricao_referencia.txt"
        if wav.exists() and not forcar:
            saidas.append(wav)
            continue
        texto_limpo = _remover_marcas(r["texto"])
        txt.write_text(texto_limpo, encoding="utf-8")
        if tem_say:
            cmd = ["say", "-v", r["voz"], "-r", str(r["rate"]), "-o", str(wav),
                   "--data-format=LEI16@16000", r["texto"]]
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=120)
                log.info(f"Áudio gerado: {wav.name}")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                log.warning(f"`say` falhou ({exc}); gerando tom sintético para {nome}")
                _tom_sintetico(wav, nome)
        else:
            _tom_sintetico(wav, nome)
        saidas.append(wav)
    return saidas


def _remover_marcas(texto: str) -> str:
    import re
    return re.sub(r"\[\[.*?\]\]\s*", "", texto).strip()


def _tom_sintetico(wav: Path, nome: str) -> None:
    """Fallback sem TTS: 'fala' sintética com envelope de sílabas e pausas crescentes."""
    import soundfile as sf
    sr = config.AUDIO_SR
    dur = 10.0
    t = np.arange(int(sr * dur)) / sr
    f0 = {"consulta_normal": 180, "consulta_fadiga": 150, "consulta_dispneia": 200}.get(nome, 170)
    taxa_silabas = {"consulta_normal": 5.0, "consulta_fadiga": 2.5, "consulta_dispneia": 3.0}.get(nome, 4)
    env = (np.sin(2 * np.pi * taxa_silabas * t) > 0).astype(float)
    if nome != "consulta_normal":
        env *= (np.sin(2 * np.pi * 0.4 * t) > -0.3)  # pausas longas
    sinal = env * (np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t))
    sinal += RNG.normal(0, 0.02, len(t))
    sf.write(wav, (0.5 * sinal / np.abs(sinal).max()).astype(np.float32), sr)


# ─── Vídeo sintético ─────────────────────────────────────────────────────────
def gerar_video_sintetico(forcar: bool = False, segundos: int = 24, fps: int = 20) -> Path:
    """
    Simula uma sessão de fisioterapia filmada: um "paciente" (figura humanoide simplificada)
    faz agachamentos rítmicos; aos 14 s sofre uma queda e fica imóvel no chão; aos 20 s um
    objeto entra em cena. Serve para validar o pipeline de energia de movimento/eventos.
    """
    import cv2

    pasta = OUT / "video"
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / "fisioterapia_sintetica.mp4"
    if caminho.exists() and not forcar:
        return caminho
    W, H = 640, 480
    writer = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(segundos * fps):
        t = i / fps
        frame = np.full((H, W, 3), (235, 235, 235), dtype=np.uint8)
        cv2.rectangle(frame, (0, 400), (W, H), (120, 120, 120), -1)          # chão
        cv2.rectangle(frame, (40, 300), (160, 400), (150, 110, 70), -1)      # maca
        if t < 14:                                                           # agachamentos rítmicos
            y_offset = int(40 * (0.5 - 0.5 * np.cos(2 * np.pi * t / 2.0)))
            _desenhar_pessoa(frame, 330, 190 + y_offset, agachado=y_offset > 20)
        elif t < 15:                                                         # queda (rotação rápida)
            frac = (t - 14)
            _desenhar_pessoa(frame, 330 + int(90 * frac), 190 + int(170 * frac), caido=frac > 0.5)
        else:                                                                # imóvel no chão
            _desenhar_pessoa(frame, 420, 360, caido=True)
        if t >= 20:                                                          # objeto entra em cena
            cv2.rectangle(frame, (500 - int(60 * (t - 20)), 150), (540 - int(60 * (t - 20)), 230), (40, 40, 200), -1)
        cv2.putText(frame, f"t={t:5.1f}s  SESSAO FISIOTERAPIA (SINTETICO)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)
        writer.write(frame)
    writer.release()
    from .video.movimento import reencodar_h264
    reencodar_h264(caminho)
    log.info(f"Vídeo sintético gerado: {caminho}")
    return caminho


def _desenhar_pessoa(frame, x, y, agachado=False, caido=False):
    import cv2
    cor_pele, cor_roupa = (170, 200, 240), (60, 90, 180)
    if caido:
        cv2.ellipse(frame, (x, y), (80, 18), 0, 0, 360, cor_roupa, -1)
        cv2.circle(frame, (x - 95, y - 5), 18, cor_pele, -1)
        return
    cv2.circle(frame, (x, y), 20, cor_pele, -1)                       # cabeça
    tronco_h = 70 if not agachado else 55
    cv2.rectangle(frame, (x - 25, y + 20), (x + 25, y + 20 + tronco_h), cor_roupa, -1)
    perna_h = 90 if not agachado else 60
    cv2.rectangle(frame, (x - 22, y + 20 + tronco_h), (x - 6, y + 20 + tronco_h + perna_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x + 6, y + 20 + tronco_h), (x + 22, y + 20 + tronco_h + perna_h), (50, 50, 50), -1)
    cv2.line(frame, (x - 25, y + 30), (x - 55, y + 30 + (40 if agachado else 60)), cor_pele, 8)
    cv2.line(frame, (x + 25, y + 30), (x + 55, y + 30 + (40 if agachado else 60)), cor_pele, 8)


# ─── Orquestração ────────────────────────────────────────────────────────────
PACIENTES = {"P001": "estavel", "P002": "deterioracao", "P003": "episodios"}


def gerar_tudo(forcar: bool = False) -> dict[str, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    saidas: dict[str, Path] = {}
    sv, pr, mv = [], [], []
    for pid, perfil in PACIENTES.items():
        sv.append(gerar_sinais_vitais(pid, perfil))
        pr.append(gerar_prescricoes(pid, perfil))
        mv.append(gerar_movimentacao(pid, perfil))
    for nome, frames in (("sinais_vitais", sv), ("prescricoes", pr), ("movimentacao", mv)):
        caminho = OUT / f"{nome}.csv"
        pd.concat(frames, ignore_index=True).to_csv(caminho, index=False)
        saidas[nome] = caminho
        log.info(f"Gerado: {caminho.relative_to(config.BASE_DIR)}")
    for wav in gerar_audios(forcar):
        saidas[wav.stem] = wav
    saidas["video"] = gerar_video_sintetico(forcar)
    return saidas


if __name__ == "__main__":
    gerar_tudo(forcar=True)
