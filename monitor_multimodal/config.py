"""
config.py — Configurações centrais do projeto.

Todas as constantes, caminhos e limiares clínicos ficam aqui.
As credenciais sensíveis vêm de variáveis de ambiente (arquivo .env).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ─── Caminhos ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DADOS_DIR = BASE_DIR / "dados"
DADOS_SINTETICOS_DIR = DADOS_DIR / "sinteticos" / "gerados"
DADOS_AMOSTRAS_DIR = DADOS_DIR / "amostras"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELOS_DIR = BASE_DIR / "modelos"

for _d in (DADOS_SINTETICOS_DIR, OUTPUTS_DIR, MODELOS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

# ─── Reprodutibilidade ──────────────────────────────────────────────────────
RANDOM_STATE = 42

# ─── Azure ──────────────────────────────────────────────────────────────────
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "brazilsouth")
AZURE_LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY", "")
AZURE_LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT", "")
MODO_SERVICOS = os.getenv("MODO_SERVICOS", "auto").lower()  # auto | azure | local
IDIOMA_FALA = "pt-BR"

# ─── Canais de alerta ───────────────────────────────────────────────────────
ALERTA_WEBHOOK_URL = os.getenv("ALERTA_WEBHOOK_URL", "")
ALERTA_SMTP = {
    "host": os.getenv("ALERTA_SMTP_HOST", ""),
    "port": int(os.getenv("ALERTA_SMTP_PORT", "587") or 587),
    "user": os.getenv("ALERTA_SMTP_USER", ""),
    "password": os.getenv("ALERTA_SMTP_PASSWORD", ""),
    "destino": os.getenv("ALERTA_EMAIL_DESTINO", ""),
}

# ─── Sinais vitais: faixas de referência (adulto) ───────────────────────────
# Cada entrada: (mínimo normal, máximo normal, mínimo crítico, máximo crítico)
# Fontes: limiares usuais de escores de alerta precoce (NEWS2) adaptados.
SINAIS_VITAIS = {
    "frequencia_cardiaca": {"unidade": "bpm", "normal": (50, 100), "critico": (40, 130)},
    "pressao_sistolica": {"unidade": "mmHg", "normal": (90, 140), "critico": (80, 180)},
    "pressao_diastolica": {"unidade": "mmHg", "normal": (60, 90), "critico": (50, 110)},
    "saturacao_o2": {"unidade": "%", "normal": (94, 100), "critico": (90, 100)},
    "frequencia_respiratoria": {"unidade": "irpm", "normal": (12, 20), "critico": (8, 25)},
    "temperatura": {"unidade": "°C", "normal": (36.0, 37.8), "critico": (35.0, 39.5)},
}

# ─── Detecção de anomalias ──────────────────────────────────────────────────
JANELA_ROLLING = 30          # amostras usadas pelo z-score deslizante
ZSCORE_LIMIAR = 3.0          # |z| acima disso = anomalia estatística
IFOREST_CONTAMINACAO = 0.01  # fração esperada de anomalias (Isolation Forest)
PERSISTENCIA_MINIMA = 3      # nº de repetições do mesmo alerta para escalar a severidade
JANELA_DEDUP_MIN = 30        # minutos: alertas iguais do mesmo paciente são agrupados nesta janela

# ─── Prescrições ────────────────────────────────────────────────────────────
# Dose diária máxima usual (mg) por medicamento — usada para checar doses fora da faixa.
DOSE_MAXIMA_DIARIA_MG = {
    "paracetamol": 4000,
    "dipirona": 4000,
    "metformina": 2550,
    "enalapril": 40,
    "losartana": 100,
    "atenolol": 100,
    "omeprazol": 40,
    "amoxicilina": 3000,
    "ceftriaxona": 4000,
    "insulina": 100,       # UI (tratado como mg para simplificação do modelo)
    "morfina": 60,
    "varfarina": 10,
    "heparina": 40000,     # UI
    "furosemida": 600,
}
VARIACAO_DOSE_SUSPEITA = 0.50   # +50% de dose em relação à última prescrição → suspeito
# Pares com interação medicamentosa relevante (simplificado para fins didáticos)
INTERACOES_CRITICAS = {
    frozenset({"varfarina", "heparina"}),
    frozenset({"varfarina", "dipirona"}),
    frozenset({"enalapril", "losartana"}),
    frozenset({"morfina", "atenolol"}),
}

# ─── Movimentação do paciente ───────────────────────────────────────────────
IMOBILIDADE_HORAS_LIMITE = 4      # horas sem mudança de decúbito → risco de lesão por pressão
AGITACAO_NOTURNA_LIMIAR = 0.80    # índice de atividade acima disso entre 22h-6h → agitação

# ─── Vídeo ──────────────────────────────────────────────────────────────────
YOLO_MODELO_DETECCAO = "yolov8n.pt"
YOLO_MODELO_POSE = "yolov8n-pose.pt"
YOLO_CONFIANCA = 0.35
VIDEO_PASSO_FRAMES = 2            # processa 1 a cada N frames (desempenho)
# Classes COCO consideradas "áreas/objetos críticos" num ambiente clínico
CLASSES_CRITICAS_VIDEO = {"knife", "scissors", "bottle", "cell phone", "chair", "bed"}
QUEDA_VELOCIDADE_QUADRIL = 0.25   # queda do quadril (fração da altura do frame/seg) → possível queda
ASSIMETRIA_LIMIAR = 0.15          # diferença relativa entre lados do corpo → assimetria postural

# ─── Áudio ──────────────────────────────────────────────────────────────────
AUDIO_SR = 16000
TERMOS_CRITICOS_AUDIO = [
    "dor no peito", "falta de ar", "não consigo respirar", "tontura", "desmaio",
    "sangramento", "febre alta", "convulsão", "confusão", "dormência", "fraqueza",
    "cansaço extremo", "palpitação", "visão turva", "dificuldade para falar",
    "dor forte", "vômito", "sangue", "inconsciente", "alergia",
]

# ─── Fusão multimodal: peso de cada fonte no score de risco ─────────────────
PESOS_FUSAO = {
    "sinais_vitais": 0.35,
    "prescricoes": 0.15,
    "movimentacao": 0.15,
    "audio": 0.20,
    "video": 0.15,
}
