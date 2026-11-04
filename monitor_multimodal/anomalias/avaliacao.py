"""
avaliacao.py — Métricas dos detectores contra as anomalias injetadas nos dados sintéticos.

Além de precisão/recall/F1 por amostra, calcula o tempo de antecipação (lead time):
quanto antes do início de um episódio rotulado o sistema emitiu o primeiro alerta.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def metricas_binarias(y_true, y_pred, score=None) -> dict[str, Any]:
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    out = {
        "n": int(len(y_true)), "positivos_reais": int(y_true.sum()), "positivos_previstos": int(y_pred.sum()),
        "precisao": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
        "taxa_falsos_positivos": round(float(((y_pred) & (~y_true)).sum() / max((~y_true).sum(), 1)), 4),
    }
    if score is not None and 0 < y_true.sum() < len(y_true):
        out["auc_roc"] = round(float(roc_auc_score(y_true, np.asarray(score, float))), 3)
    return out


def tempo_antecipacao(df: pd.DataFrame, col_rotulo: str = "anomalia_injetada", col_pred: str = "anomalia",
                      col_tipo: str = "tipo_injetado") -> list[dict[str, Any]]:
    """Para cada episódio rotulado contíguo, mede o atraso/antecipação do 1º alerta (em minutos)."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    rot = df[col_rotulo].to_numpy(bool)
    grupos = (pd.Series(rot) != pd.Series(rot).shift()).cumsum()
    episodios = []
    for g, idx in df.groupby(grupos).groups.items():
        idx = list(idx)
        if not rot[idx[0]]:
            continue
        ini, fim = idx[0], idx[-1]
        t_ini = pd.Timestamp(df.at[ini, "timestamp"])
        # procura o primeiro alerta desde 2 h antes do início do episódio até o fim dele
        janela = df[(df["timestamp"] >= t_ini - pd.Timedelta(hours=2)) & (df.index <= fim)]
        det = janela[janela[col_pred].astype(bool)]
        primeiro = pd.Timestamp(det.iloc[0]["timestamp"]) if len(det) else None
        episodios.append({
            "tipo": df.at[ini, col_tipo], "inicio": str(t_ini), "fim": str(df.at[fim, "timestamp"]),
            "n_amostras": len(idx), "detectado": primeiro is not None,
            "antecipacao_min": round((t_ini - primeiro).total_seconds() / 60, 1) if primeiro is not None else None,
            "cobertura": round(float(df.loc[ini:fim, col_pred].astype(bool).mean()), 3),
        })
    return episodios
