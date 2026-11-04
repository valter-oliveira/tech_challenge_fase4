"""
stream.py — Monitoramento em tempo real (streaming) de sinais vitais.

O MonitorTempoReal recebe uma amostra por vez (como viria de um monitor
multiparamétrico), mantém uma janela deslizante por paciente e aplica:
  • regras clínicas            (imediato)
  • z-score robusto deslizante (histórico recente do paciente)
  • Isolation Forest           (treinado nas primeiras `aquecimento` amostras normais e
                                re-treinado periodicamente)
emitindo alertas pelo GerenciadorAlertas com latência de uma amostra.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Iterator

import numpy as np
import pandas as pd

from .. import config
from ..alertas import Alerta, GerenciadorAlertas, Severidade
from .detectores import DetectorIsolationForest, DetectorRegras, resolucao
from .sinais_vitais import NOMES_LEGIVEIS, escore_alerta_precoce


class MonitorTempoReal:
    def __init__(self, gerenciador: GerenciadorAlertas, janela: int = config.JANELA_ROLLING,
                 aquecimento: int = 60, retreino: int = 120, limiar_z: float = config.ZSCORE_LIMIAR):
        self.g = gerenciador
        self.janela = janela
        self.aquecimento = aquecimento
        self.retreino = retreino
        self.limiar_z = limiar_z
        self.regras = DetectorRegras()
        self._hist: dict[str, deque] = {}
        self.historico_max = 288  # 24 h a cada 5 min
        self._iforest: dict[str, DetectorIsolationForest] = {}
        self._n: dict[str, int] = {}
        self.colunas = list(config.SINAIS_VITAIS)

    def processar(self, paciente_id: str, amostra: dict) -> list[Alerta]:
        """Processa uma amostra {timestamp, sinal1, sinal2, ...}; devolve os alertas emitidos."""
        hist = self._hist.setdefault(paciente_id, deque(maxlen=self.historico_max))
        self._n[paciente_id] = self._n.get(paciente_id, 0) + 1
        cols = [c for c in self.colunas if c in amostra]
        ts = pd.Timestamp(amostra["timestamp"]).isoformat()
        emitidos: list[Alerta] = []

        # 0. Escore agregado (calculado antes: com EWS ≥ 5 as variáveis viram evidência do EWS)
        ews = escore_alerta_precoce(pd.Series(amostra))
        alteradas = {c: {"valor": float(amostra[c]), "categoria": self.regras.pontuar_valor(c, amostra[c])[1]}
                     for c in cols if self.regras.pontuar_valor(c, amostra[c])[1] != "normal"}

        # 1. Regras clínicas (uma por variável, quando o EWS não agrupa)
        for c in cols:
            score, cat = self.regras.pontuar_valor(c, amostra[c])
            if cat != "normal" and ews < 5:
                sev = Severidade.CRITICA if cat == "critico" else Severidade.ALTA if score >= 0.75 else Severidade.ATENCAO
                a = self.g.emitir(Alerta(
                    paciente_id=paciente_id, fonte="sinais_vitais", tipo=f"{c}_{cat}", severidade=sev,
                    timestamp=ts, score=score,
                    mensagem=f"{NOMES_LEGIVEIS[c]} {cat}: {amostra[c]:.1f} {config.SINAIS_VITAIS[c]['unidade']}",
                    evidencias={"variavel": c, "valor": float(amostra[c]), "modo": "tempo_real"},
                    recomendacao="Avaliar paciente à beira-leito." if sev == Severidade.CRITICA else "Reavaliar em 15 min.",
                ))
                if a:
                    emitidos.append(a)

        # 2. Z-score robusto contra o histórico do paciente
        if len(hist) >= 8:
            H = pd.DataFrame(list(hist))[cols].tail(self.janela)
            med = H.median()
            mad = (H - med).abs().median()
            mad = pd.Series({c: max(mad[c], resolucao(H[c])) for c in cols})
            for c in cols:
                z = 0.6745 * (amostra[c] - med[c]) / mad[c]
                if abs(z) > self.limiar_z and self.regras.pontuar_valor(c, amostra[c])[1] == "normal":
                    a = self.g.emitir(Alerta(
                        paciente_id=paciente_id, fonte="sinais_vitais", tipo=f"{c}_mudanca_abrupta",
                        severidade=Severidade.ATENCAO, timestamp=ts, score=float(min(1, abs(z) / 6)),
                        mensagem=f"Mudança abrupta em {NOMES_LEGIVEIS[c]}: {amostra[c]:.1f} (z={z:.1f})",
                        evidencias={"variavel": c, "z": float(z), "modo": "tempo_real"},
                        recomendacao="Confirmar leitura do sensor.",
                    ))
                    if a:
                        emitidos.append(a)

        # 3. Isolation Forest (treina no aquecimento, re-treina a cada `retreino` amostras)
        n = self._n[paciente_id]
        if n == self.aquecimento or (n > self.aquecimento and n % self.retreino == 0):
            self._iforest[paciente_id] = DetectorIsolationForest().ajustar(pd.DataFrame(list(hist))[cols])
        if paciente_id in self._iforest and n > self.aquecimento:
            res = self._iforest[paciente_id].detectar(pd.DataFrame([amostra])[cols])
            if res.anomalia[0] and res.score[0] >= 0.6 and not alteradas and ews < 5:
                a = self.g.emitir(Alerta(
                    paciente_id=paciente_id, fonte="sinais_vitais", tipo="padrao_multivariado_atipico",
                    severidade=Severidade.ATENCAO, timestamp=ts, score=float(res.score[0]),
                    mensagem="Combinação atípica de sinais vitais (Isolation Forest, tempo real)",
                    evidencias={c: float(amostra[c]) for c in cols} | {"modo": "tempo_real"},
                    recomendacao="Correlacionar com quadro clínico.",
                ))
                if a:
                    emitidos.append(a)

        # 4. Escore de alerta precoce (agrupa as variáveis alteradas)
        if ews >= 5:
            detalhe = ", ".join(f"{NOMES_LEGIVEIS[c]} {v['valor']:.0f} ({v['categoria']})" for c, v in alteradas.items())
            a = self.g.emitir(Alerta(
                paciente_id=paciente_id, fonte="sinais_vitais", tipo="ews_alto" if ews >= 7 else "ews_medio",
                severidade=Severidade.CRITICA if ews >= 7 else Severidade.ALTA,
                timestamp=ts, score=min(1.0, ews / 12),
                mensagem=f"Escore de alerta precoce = {ews} ({'risco alto' if ews >= 7 else 'risco médio'}) — {detalhe}",
                evidencias={c: float(amostra[c]) for c in cols} | {"modo": "tempo_real", "variaveis_alteradas": alteradas},
                recomendacao="Acionar time de resposta rápida." if ews >= 7 else "Avaliação médica em até 1h.",
            ))
            if a:
                emitidos.append(a)

        hist.append({c: amostra[c] for c in cols})
        return emitidos


def simular_stream(df: pd.DataFrame, paciente_id: str, monitor: MonitorTempoReal,
                   intervalo_s: float = 0.0) -> Iterator[tuple[dict, list[Alerta]]]:
    """Reproduz um DataFrame como se fosse um fluxo em tempo real (intervalo_s entre amostras)."""
    for _, linha in df.sort_values("timestamp").iterrows():
        amostra = {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in linha.items()}
        alertas = monitor.processar(paciente_id, amostra)
        yield amostra, alertas
        if intervalo_s > 0:
            time.sleep(intervalo_s)
