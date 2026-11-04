"""
prescricoes.py — Detecção de alterações inesperadas na evolução de prescrições.

Entrada: DataFrame com colunas
  timestamp, paciente_id, medicamento, dose_mg, frequencia_dia, via, prescritor

Verificações (regras de domínio + modelo não supervisionado):
  • dose diária acima do máximo usual do medicamento              → CRÍTICA
  • aumento abrupto (> VARIACAO_DOSE_SUSPEITA) vs. última prescrição → ALTA
  • interação medicamentosa crítica com fármaco ativo             → CRÍTICA
  • duplicidade terapêutica (mesmo fármaco, prescritores distintos, mesmo dia) → ATENÇÃO
  • troca de via de administração sem mudança de fármaco          → ATENÇÃO
  • Isolation Forest sobre (dose relativa, frequência, hora, Δdose) → ATENÇÃO
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .. import config
from ..alertas import Alerta, Severidade
from .detectores import DetectorIsolationForest


@dataclass
class ResultadoPrescricoes:
    paciente_id: str
    dados: pd.DataFrame
    alertas: list[Alerta] = field(default_factory=list)
    resumo: dict[str, Any] = field(default_factory=dict)


def analisar_prescricoes(df: pd.DataFrame, paciente_id: str) -> ResultadoPrescricoes:
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    df["medicamento"] = df["medicamento"].str.lower().str.strip()
    df["dose_diaria_mg"] = df["dose_mg"] * df["frequencia_dia"]
    df["dose_max_mg"] = df["medicamento"].map(config.DOSE_MAXIMA_DIARIA_MG)
    df["dose_relativa"] = df["dose_diaria_mg"] / df["dose_max_mg"]
    df["dose_anterior_mg"] = df.groupby("medicamento")["dose_diaria_mg"].shift(1)
    df["variacao_dose"] = (df["dose_diaria_mg"] - df["dose_anterior_mg"]) / df["dose_anterior_mg"]
    df["via_anterior"] = df.groupby("medicamento")["via"].shift(1)
    df["hora"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["anomalia"] = False
    df["motivos"] = [[] for _ in range(len(df))]

    alertas: list[Alerta] = []
    ativos: dict[str, pd.Timestamp] = {}  # medicamento → última prescrição (considerado ativo por 48h)
    pares_alertados: set[frozenset] = set()  # interações já sinalizadas (alerta único por par)

    def marcar(i: int, motivo: str):
        df.at[i, "anomalia"] = True
        df.at[i, "motivos"] = list(df.at[i, "motivos"]) + [motivo]

    for i, r in df.iterrows():
        ts = pd.Timestamp(r["timestamp"])
        med = r["medicamento"]
        # Remove da lista de ativos fármacos com última prescrição há mais de 48h
        ativos = {m: t for m, t in ativos.items() if (ts - t) <= pd.Timedelta(hours=48)}

        if not np.isnan(r["dose_relativa"]) and r["dose_relativa"] > 1.0:
            marcar(i, "dose_acima_maximo")
            alertas.append(Alerta(
                paciente_id=paciente_id, fonte="prescricoes", tipo="dose_acima_maximo",
                severidade=Severidade.CRITICA, timestamp=ts.isoformat(),
                mensagem=f"{med.capitalize()}: dose diária {r['dose_diaria_mg']:.0f} mg excede o máximo usual "
                         f"({r['dose_max_mg']:.0f} mg) em {100*(r['dose_relativa']-1):.0f}%",
                evidencias={"medicamento": med, "dose_diaria_mg": float(r["dose_diaria_mg"]),
                            "dose_max_mg": float(r["dose_max_mg"]), "prescritor": r.get("prescritor")},
                recomendacao="Bloquear dispensação e confirmar dose com o prescritor / farmácia clínica.",
                score=min(1.0, float(r["dose_relativa"]) - 0.5),
            ))
        if not np.isnan(r["variacao_dose"]) and r["variacao_dose"] > config.VARIACAO_DOSE_SUSPEITA:
            marcar(i, "aumento_abrupto_dose")
            alertas.append(Alerta(
                paciente_id=paciente_id, fonte="prescricoes", tipo="aumento_abrupto_dose",
                severidade=Severidade.ALTA, timestamp=ts.isoformat(),
                mensagem=f"{med.capitalize()}: dose diária subiu {100*r['variacao_dose']:.0f}% "
                         f"({r['dose_anterior_mg']:.0f} → {r['dose_diaria_mg']:.0f} mg)",
                evidencias={"medicamento": med, "variacao": float(r["variacao_dose"])},
                recomendacao="Verificar se o escalonamento foi intencional (possível erro de digitação).",
                score=min(1.0, float(r["variacao_dose"]) / 2),
            ))
        for ativo in ativos:
            par = frozenset({ativo, med})
            if ativo != med and par in config.INTERACOES_CRITICAS and par not in pares_alertados:
                pares_alertados.add(par)
                marcar(i, f"interacao_{ativo}")
                alertas.append(Alerta(
                    paciente_id=paciente_id, fonte="prescricoes", tipo="interacao_medicamentosa",
                    severidade=Severidade.CRITICA, timestamp=ts.isoformat(),
                    mensagem=f"Interação crítica: {med} prescrito com {ativo} ativo",
                    evidencias={"novo": med, "ativo": ativo},
                    recomendacao="Revisar com farmácia clínica antes da administração.",
                    score=0.9,
                ))
        mesmo_dia = df[(df["medicamento"] == med) & (pd.to_datetime(df["timestamp"]).dt.date == ts.date())
                       & (df.index < i)]
        if len(mesmo_dia) and (mesmo_dia["prescritor"] != r["prescritor"]).any():
            marcar(i, "duplicidade_terapeutica")
            alertas.append(Alerta(
                paciente_id=paciente_id, fonte="prescricoes", tipo="duplicidade_terapeutica",
                severidade=Severidade.ATENCAO, timestamp=ts.isoformat(),
                mensagem=f"{med.capitalize()} prescrito por mais de um profissional no mesmo dia",
                evidencias={"medicamento": med, "prescritores": sorted(set(mesmo_dia["prescritor"]) | {r["prescritor"]})},
                recomendacao="Conciliar prescrições para evitar dose duplicada.",
                score=0.5,
            ))
        if isinstance(r["via_anterior"], str) and r["via_anterior"] != r["via"]:
            marcar(i, "troca_de_via")
            alertas.append(Alerta(
                paciente_id=paciente_id, fonte="prescricoes", tipo="troca_de_via",
                severidade=Severidade.ATENCAO, timestamp=ts.isoformat(),
                mensagem=f"{med.capitalize()}: via alterada de {r['via_anterior']} para {r['via']}",
                evidencias={"medicamento": med, "de": r["via_anterior"], "para": r["via"]},
                recomendacao="Confirmar se a mudança de via é compatível com a dose prescrita.",
                score=0.4,
            ))
        ativos[med] = ts

    # Modelo não supervisionado: combinações atípicas (dose relativa, frequência, hora, Δdose)
    feats = df[["dose_relativa", "frequencia_dia", "hora", "variacao_dose"]].fillna(0.0)
    if len(feats) >= 20:
        res = DetectorIsolationForest(contaminacao=0.05).detectar(feats)
        df["score_iforest"] = res.score
        for i in res.indices:
            if not df.at[i, "anomalia"]:
                marcar(i, "padrao_atipico_iforest")
                r = df.loc[i]
                alertas.append(Alerta(
                    paciente_id=paciente_id, fonte="prescricoes", tipo="padrao_atipico",
                    severidade=Severidade.ATENCAO, timestamp=pd.Timestamp(r["timestamp"]).isoformat(),
                    mensagem=f"{r['medicamento'].capitalize()}: prescrição com padrão atípico "
                             f"(dose {r['dose_mg']:.0f} mg {int(r['frequencia_dia'])}x/dia às {int(r['hora']):02d}h)",
                    evidencias={"medicamento": r["medicamento"], "score": float(res.score[i])},
                    recomendacao="Revisão de rotina pela farmácia clínica.",
                    score=float(res.score[i]),
                ))
    else:
        df["score_iforest"] = 0.0

    resumo = {
        "n_prescricoes": int(len(df)),
        "n_anomalias": int(df["anomalia"].sum()),
        "medicamentos": sorted(df["medicamento"].unique().tolist()),
        "motivos": pd.Series([m for ms in df["motivos"] for m in ms]).value_counts().to_dict(),
        "score_risco": float(min(1.0, 0.25 * sum(a.severidade >= Severidade.ALTA for a in alertas)
                                 + 0.05 * len(alertas))),
    }
    return ResultadoPrescricoes(paciente_id=paciente_id, dados=df, alertas=alertas, resumo=resumo)
