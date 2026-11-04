"""
movimentacao.py — Padrões de movimentação do paciente durante a internação.

Entrada: DataFrame horário com colunas
  timestamp, indice_atividade (0-1, de sensor de leito/acelerômetro ou da análise de vídeo),
  mudancas_decubito (nº de mudanças de posição na hora), fora_do_leito (0/1)

Anomalias:
  • imobilidade prolongada (≥ IMOBILIDADE_HORAS_LIMITE h sem mudança de decúbito) → ALTA
      ↳ risco de lesão por pressão / TEV
  • agitação noturna (atividade alta entre 22h e 6h)                               → ATENÇÃO
      ↳ possível delirium / dor
  • pico abrupto de atividade seguido de imobilidade total                          → CRÍTICA
      ↳ padrão compatível com queda
  • saída do leito em paciente com restrição                                        → ALTA
  • z-score deslizante e Isolation Forest sobre o índice de atividade               → ATENÇÃO
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .. import config
from ..alertas import Alerta, Severidade
from .detectores import DetectorIsolationForest, DetectorZScoreDeslizante


@dataclass
class ResultadoMovimentacao:
    paciente_id: str
    dados: pd.DataFrame
    alertas: list[Alerta] = field(default_factory=list)
    resumo: dict[str, Any] = field(default_factory=dict)


def analisar_movimentacao(df: pd.DataFrame, paciente_id: str, restricao_leito: bool = False) -> ResultadoMovimentacao:
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hora"] = df["timestamp"].dt.hour
    df["noturno"] = (df["hora"] >= 22) | (df["hora"] < 6)

    # horas consecutivas sem mudança de decúbito
    sem_mudanca = (df["mudancas_decubito"] == 0).astype(int)
    grupos = (sem_mudanca != sem_mudanca.shift()).cumsum()
    df["horas_imovel"] = sem_mudanca.groupby(grupos).cumsum()

    zres = DetectorZScoreDeslizante(janela=24, limiar=3.0).detectar(df[["indice_atividade"]])
    ires = DetectorIsolationForest(contaminacao=0.02).detectar(df[["indice_atividade", "mudancas_decubito", "hora"]])
    df["score_zscore"] = zres.score
    df["score_iforest"] = ires.score
    df["anomalia"] = False
    df["motivo"] = ""

    alertas: list[Alerta] = []

    def marcar(i, motivo):
        df.at[i, "anomalia"] = True
        df.at[i, "motivo"] = (df.at[i, "motivo"] + ";" + motivo).strip(";")

    for i, r in df.iterrows():
        ts = r["timestamp"].isoformat()
        if r["horas_imovel"] >= config.IMOBILIDADE_HORAS_LIMITE:
            marcar(i, "imobilidade_prolongada")
        if r["horas_imovel"] >= config.IMOBILIDADE_HORAS_LIMITE and r["horas_imovel"] % config.IMOBILIDADE_HORAS_LIMITE == 0:
            alertas.append(Alerta(
                paciente_id=paciente_id, fonte="movimentacao", tipo="imobilidade_prolongada",
                severidade=Severidade.ALTA, timestamp=ts,
                mensagem=f"{int(r['horas_imovel'])}h sem mudança de decúbito",
                evidencias={"horas_imovel": int(r["horas_imovel"]), "indice_atividade": float(r["indice_atividade"])},
                recomendacao="Realizar mudança de decúbito e avaliar pele (risco de lesão por pressão).",
                score=min(1.0, r["horas_imovel"] / 8),
            ))
        if r["noturno"] and r["indice_atividade"] >= config.AGITACAO_NOTURNA_LIMIAR:
            marcar(i, "agitacao_noturna")
            alertas.append(Alerta(
                paciente_id=paciente_id, fonte="movimentacao", tipo="agitacao_noturna",
                severidade=Severidade.ATENCAO, timestamp=ts,
                mensagem=f"Agitação noturna às {int(r['hora']):02d}h (índice de atividade {r['indice_atividade']:.2f})",
                evidencias={"indice_atividade": float(r["indice_atividade"]), "hora": int(r["hora"])},
                recomendacao="Avaliar dor, delirium (CAM-ICU) e segurança do leito.",
                score=float(r["indice_atividade"]),
            ))
        if i >= 1:
            ant = df.loc[i - 1]
            if ant["indice_atividade"] >= 0.85 and r["indice_atividade"] <= 0.05 and r["fora_do_leito"] == 1:
                marcar(i, "padrao_queda")
                alertas.append(Alerta(
                    paciente_id=paciente_id, fonte="movimentacao", tipo="padrao_queda",
                    severidade=Severidade.CRITICA, timestamp=ts,
                    mensagem="Pico de atividade seguido de imobilidade fora do leito — padrão compatível com queda",
                    evidencias={"atividade_anterior": float(ant["indice_atividade"]),
                                "atividade_atual": float(r["indice_atividade"])},
                    recomendacao="Verificar o paciente imediatamente.",
                    score=0.95,
                ))
        if restricao_leito and r["fora_do_leito"] == 1 and "padrao_queda" not in df.at[i, "motivo"]:
            marcar(i, "saida_do_leito")
            alertas.append(Alerta(
                paciente_id=paciente_id, fonte="movimentacao", tipo="saida_do_leito",
                severidade=Severidade.ALTA, timestamp=ts,
                mensagem="Saída do leito em paciente com restrição",
                evidencias={"hora": int(r["hora"])},
                recomendacao="Acionar enfermagem do setor.",
                score=0.7,
            ))
        if (zres.anomalia[i] or ires.anomalia[i]) and not df.at[i, "anomalia"]:
            marcar(i, "atividade_atipica")
            alertas.append(Alerta(
                paciente_id=paciente_id, fonte="movimentacao", tipo="atividade_atipica",
                severidade=Severidade.ATENCAO, timestamp=ts,
                mensagem=f"Nível de atividade atípico para o paciente ({r['indice_atividade']:.2f}) às {int(r['hora']):02d}h",
                evidencias={"indice_atividade": float(r["indice_atividade"]),
                            "score_zscore": float(zres.score[i]), "score_iforest": float(ires.score[i])},
                recomendacao="Observar e correlacionar com sinais vitais.",
                score=float(max(zres.score[i], ires.score[i])),
            ))

    resumo = {
        "n_horas": int(len(df)),
        "n_anomalias": int(df["anomalia"].sum()),
        "max_horas_imovel": int(df["horas_imovel"].max()),
        "motivos": pd.Series([m for ms in df["motivo"] if ms for m in ms.split(";")]).value_counts().to_dict(),
        "score_risco": float(min(1.0, max([a.score for a in alertas], default=0.0))),
    }
    return ResultadoMovimentacao(paciente_id=paciente_id, dados=df, alertas=alertas, resumo=resumo)
