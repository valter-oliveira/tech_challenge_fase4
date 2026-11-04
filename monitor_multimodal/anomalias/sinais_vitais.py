"""
sinais_vitais.py — Detecção de anomalias em séries temporais de sinais vitais.

Entrada: DataFrame com coluna `timestamp` e colunas de sinais (ver config.SINAIS_VITAIS):
  frequencia_cardiaca, pressao_sistolica, pressao_diastolica, saturacao_o2,
  frequencia_respiratoria, temperatura

Pipeline:
  1. Regras clínicas (limiares NEWS2-like)          → alertas explicáveis por variável
  2. Z-score robusto deslizante (janela do paciente) → mudanças abruptas
  3. Isolation Forest multivariado                   → combinações atípicas
  4. Tendência (regressão linear em janela curta)    → deterioração progressiva (ex.: SpO2 caindo)
  5. Escore de alerta precoce agregado (simplificado do NEWS2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .. import config
from ..alertas import Alerta, Severidade
from .detectores import DetectorIsolationForest, DetectorRegras, DetectorZScoreDeslizante

NOMES_LEGIVEIS = {
    "frequencia_cardiaca": "Frequência cardíaca",
    "pressao_sistolica": "Pressão sistólica",
    "pressao_diastolica": "Pressão diastólica",
    "saturacao_o2": "Saturação de O₂",
    "frequencia_respiratoria": "Frequência respiratória",
    "temperatura": "Temperatura",
}


def escore_alerta_precoce(linha: pd.Series) -> int:
    """
    Versão simplificada do NEWS2 (sem nível de consciência / O₂ suplementar).
    0-4 baixo · 5-6 médio · ≥7 alto risco.
    """
    s = 0
    fr = linha.get("frequencia_respiratoria", np.nan)
    if not np.isnan(fr):
        s += 3 if fr <= 8 or fr >= 25 else 2 if fr >= 21 else 1 if fr <= 11 else 0
    spo2 = linha.get("saturacao_o2", np.nan)
    if not np.isnan(spo2):
        s += 3 if spo2 <= 91 else 2 if spo2 <= 93 else 1 if spo2 <= 95 else 0
    pas = linha.get("pressao_sistolica", np.nan)
    if not np.isnan(pas):
        s += 3 if pas <= 90 or pas >= 220 else 2 if pas <= 100 else 1 if pas <= 110 else 0
    fc = linha.get("frequencia_cardiaca", np.nan)
    if not np.isnan(fc):
        s += 3 if fc <= 40 or fc >= 131 else 2 if fc >= 111 else 1 if fc <= 50 or fc >= 91 else 0
    t = linha.get("temperatura", np.nan)
    if not np.isnan(t):
        s += 3 if t <= 35.0 else 2 if t >= 39.1 else 1 if t <= 36.0 or t >= 38.1 else 0
    return int(s)


def _tendencia(serie: pd.Series, janela: int = 10) -> pd.Series:
    """Inclinação (unidades por amostra) de uma regressão linear na janela passada."""
    x = np.arange(janela)
    x = x - x.mean()
    denom = (x**2).sum()

    def _slope(v):
        if np.isnan(v).any():
            return np.nan
        return float(((v - v.mean()) * x).sum() / denom)

    return serie.rolling(janela, min_periods=janela).apply(_slope, raw=True)


@dataclass
class ResultadoSinaisVitais:
    paciente_id: str
    dados: pd.DataFrame                 # dados + colunas de anomalia/score/ews
    alertas: list[Alerta] = field(default_factory=list)
    resumo: dict[str, Any] = field(default_factory=dict)


def analisar_sinais_vitais(df: pd.DataFrame, paciente_id: str) -> ResultadoSinaisVitais:
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    colunas = [c for c in config.SINAIS_VITAIS if c in df.columns]
    X = df[colunas]

    regras = DetectorRegras().detectar(X)
    zscore = DetectorZScoreDeslizante().detectar(X)
    iforest = DetectorIsolationForest().detectar(X)

    df["score_regras"] = regras.score
    df["score_zscore"] = zscore.score
    df["score_iforest"] = iforest.score
    df["score_anomalia"] = np.maximum.reduce([regras.score, zscore.score, iforest.score])
    df["anomalia"] = regras.anomalia | zscore.anomalia | iforest.anomalia
    df["ews"] = df.apply(escore_alerta_precoce, axis=1)
    if "saturacao_o2" in df:
        df["tendencia_spo2"] = _tendencia(df["saturacao_o2"])

    alertas: list[Alerta] = []
    cat = regras.detalhes["categoria_por_coluna"]
    z = zscore.detalhes["z"]

    for i, linha in df.iterrows():
        ts = pd.Timestamp(linha["timestamp"]).isoformat()
        ews_agrupa = linha["ews"] >= 5   # com escore agregado alto, as variáveis vão como evidência do EWS
        variaveis_alteradas = {c: {"valor": float(linha[c]), "categoria": cat[c][i]}
                               for c in colunas if cat[c][i] != "normal"}
        # 1. Regras clínicas — um alerta por variável fora da faixa (quando não agrupadas no EWS)
        for c in colunas:
            k = cat[c][i]
            if k == "normal" or ews_agrupa:
                continue
            unidade = config.SINAIS_VITAIS[c]["unidade"]
            sc = float(regras.detalhes["score_por_coluna"][c][i])
            sev = Severidade.CRITICA if k == "critico" else Severidade.ALTA if sc >= 0.75 else Severidade.ATENCAO
            alertas.append(Alerta(
                paciente_id=paciente_id, fonte="sinais_vitais", tipo=f"{c}_{k}",
                severidade=sev, timestamp=ts,
                mensagem=f"{NOMES_LEGIVEIS[c]} {k}: {linha[c]:.1f} {unidade} "
                         f"(normal {config.SINAIS_VITAIS[c]['normal'][0]}–{config.SINAIS_VITAIS[c]['normal'][1]})",
                evidencias={"variavel": c, "valor": float(linha[c]), "ews": int(linha["ews"])},
                recomendacao="Avaliar paciente à beira-leito; repetir medida e comunicar médico plantonista."
                if sev == Severidade.CRITICA else "Reavaliar em 15 min e observar tendência.",
                score=float(regras.detalhes["score_por_coluna"][c][i]),
            ))
        # 2. Mudança abrupta (z-score) sem violar limiar absoluto
        if zscore.anomalia[i] and not regras.anomalia[i]:
            c = zscore.detalhes["coluna_responsavel"][i]
            alertas.append(Alerta(
                paciente_id=paciente_id, fonte="sinais_vitais", tipo=f"{c}_mudanca_abrupta",
                severidade=Severidade.ATENCAO, timestamp=ts,
                mensagem=f"Mudança abrupta em {NOMES_LEGIVEIS.get(c, c)}: {linha[c]:.1f} "
                         f"(z robusto = {z[c].iloc[i]:.1f} vs. histórico recente do paciente)",
                evidencias={"variavel": c, "valor": float(linha[c]), "z": float(z[c].iloc[i])},
                recomendacao="Confirmar leitura do sensor e observar próximas medidas.",
                score=float(zscore.score[i]),
            ))
        # 3. Padrão multivariado atípico
        if iforest.anomalia[i] and not regras.anomalia[i]:
            alertas.append(Alerta(
                paciente_id=paciente_id, fonte="sinais_vitais", tipo="padrao_multivariado_atipico",
                severidade=Severidade.ATENCAO, timestamp=ts,
                mensagem="Combinação atípica de sinais vitais (Isolation Forest) — "
                         + ", ".join(f"{NOMES_LEGIVEIS[c].split()[0]} {linha[c]:.0f}" for c in colunas[:4]),
                evidencias={c: float(linha[c]) for c in colunas},
                recomendacao="Correlacionar com quadro clínico; pode indicar deterioração inicial.",
                score=float(iforest.score[i]),
            ))
        # 4. Tendência de queda de SpO2 (deterioração progressiva)
        if "tendencia_spo2" in df and not np.isnan(linha.get("tendencia_spo2", np.nan)):
            if linha["tendencia_spo2"] <= -0.25 and linha["saturacao_o2"] < 96:
                alertas.append(Alerta(
                    paciente_id=paciente_id, fonte="sinais_vitais", tipo="spo2_tendencia_queda",
                    severidade=Severidade.ATENCAO, timestamp=ts,
                    mensagem=f"SpO₂ em queda progressiva ({linha['tendencia_spo2']:.2f} pts/amostra), "
                             f"atual {linha['saturacao_o2']:.0f}%",
                    evidencias={"inclinacao": float(linha["tendencia_spo2"]), "spo2": float(linha["saturacao_o2"])},
                    recomendacao="Avaliar necessidade de O₂ suplementar antes de atingir limiar crítico.",
                    score=min(1.0, abs(float(linha["tendencia_spo2"])) / 1.0),
                ))
        # 5. Escore agregado alto
        if linha["ews"] >= 7:
            alertas.append(Alerta(
                paciente_id=paciente_id, fonte="sinais_vitais", tipo="ews_alto",
                severidade=Severidade.CRITICA, timestamp=ts,
                mensagem=f"Escore de alerta precoce = {int(linha['ews'])} (≥7: risco alto de deterioração) — "
                         + ", ".join(f"{NOMES_LEGIVEIS[c]} {v['valor']:.0f} ({v['categoria']})" for c, v in variaveis_alteradas.items()),
                evidencias={c: float(linha[c]) for c in colunas} | {"variaveis_alteradas": variaveis_alteradas},
                recomendacao="Acionar time de resposta rápida.",
                score=min(1.0, linha["ews"] / 12),
            ))
        elif linha["ews"] >= 5:
            alertas.append(Alerta(
                paciente_id=paciente_id, fonte="sinais_vitais", tipo="ews_medio",
                severidade=Severidade.ALTA, timestamp=ts,
                mensagem=f"Escore de alerta precoce = {int(linha['ews'])} (5–6: risco médio) — "
                         + ", ".join(f"{NOMES_LEGIVEIS[c]} {v['valor']:.0f} ({v['categoria']})" for c, v in variaveis_alteradas.items()),
                evidencias={c: float(linha[c]) for c in colunas} | {"variaveis_alteradas": variaveis_alteradas},
                recomendacao="Aumentar frequência de monitoramento e avaliação médica em até 1h.",
                score=min(1.0, linha["ews"] / 12),
            ))

    resumo = {
        "n_amostras": int(len(df)),
        "n_anomalias": int(df["anomalia"].sum()),
        "pct_anomalias": round(100 * df["anomalia"].mean(), 2),
        "por_metodo": {
            "regras_clinicas": int(regras.n_anomalias),
            "zscore_deslizante": int(zscore.n_anomalias),
            "isolation_forest": int(iforest.n_anomalias),
        },
        "ews_maximo": int(df["ews"].max()),
        "score_risco": float(np.clip(df["score_anomalia"].quantile(0.95), 0, 1)),
        "periodo": [str(df["timestamp"].min()), str(df["timestamp"].max())],
    }
    return ResultadoSinaisVitais(paciente_id=paciente_id, dados=df, alertas=alertas, resumo=resumo)
