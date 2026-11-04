"""
fusao.py — Fusão multimodal: consolida as evidências de todas as fontes em um
perfil de risco por paciente.

Estratégia (late fusion, no nível de decisão):
  1. Cada modalidade produz um score de risco em [0,1] (resumo["score_risco"]) e alertas.
  2. Score global = Σ peso_fonte · score_fonte  (pesos em config.PESOS_FUSAO, renormalizados
     para as fontes disponíveis) + bônus de corroboração quando ≥2 modalidades distintas
     emitem alertas ALTA/CRÍTICA na mesma janela de tempo.
  3. Nível de risco: baixo (<0.35) · moderado (<0.55) · alto (<0.75) · crítico.
  4. Linha do tempo unificada de alertas + alerta de "corroboração multimodal" quando aplicável.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from .. import config
from ..alertas import Alerta, Severidade

NIVEIS = [(0.35, "baixo"), (0.55, "moderado"), (0.75, "alto"), (1.01, "crítico")]


def nivel_risco(score: float) -> str:
    for limite, nome in NIVEIS:
        if score < limite:
            return nome
    return "crítico"


@dataclass
class PerfilRisco:
    paciente_id: str
    score_global: float
    nivel: str
    scores_por_fonte: dict[str, float]
    pesos_usados: dict[str, float]
    corroboracoes: list[dict[str, Any]]
    alertas: list[Alerta]
    linha_do_tempo: pd.DataFrame
    resumo_alertas: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paciente_id": self.paciente_id, "score_global": round(self.score_global, 3), "nivel": self.nivel,
            "scores_por_fonte": {k: round(v, 3) for k, v in self.scores_por_fonte.items()},
            "pesos_usados": self.pesos_usados, "corroboracoes": self.corroboracoes,
            "resumo_alertas": self.resumo_alertas, "n_alertas": len(self.alertas),
        }


# Faixa de score permitida para uma fonte conforme a severidade máxima dos seus alertas:
# evita que anomalias meramente estatísticas (ATENÇÃO) inflem o risco de um paciente estável.
FAIXA_POR_SEVERIDADE = {0: (0.0, 0.10), 1: (0.10, 0.20), 2: (0.20, 0.45), 3: (0.50, 0.75), 4: (0.80, 1.00)}


def _score_fonte(resumo: dict, alertas: list[Alerta]) -> float:
    """Score da fonte: score do módulo, limitado à faixa da severidade máxima dos alertas."""
    base = float(resumo.get("score_risco", 0.0) or 0.0)
    sev_max = max((int(a.severidade) for a in alertas), default=0)
    lo, hi = FAIXA_POR_SEVERIDADE[sev_max]
    return float(min(hi, max(lo, base)))


def fundir(paciente_id: str, resultados: dict[str, Any], alertas_emitidos: list[Alerta] | None = None,
           janela_corroboracao_min: int = 60) -> PerfilRisco:
    """
    resultados: {"sinais_vitais": ResultadoSinaisVitais, "prescricoes": ..., "movimentacao": ...,
                 "audio": [ResultadoAudio, ...] ou ResultadoAudio, "video": [ResultadoVideo, ...] ou ResultadoVideo}
    alertas_emitidos: alertas que passaram pelo GerenciadorAlertas (deduplicados). Se None, usa os brutos.
    """
    scores: dict[str, float] = {}
    alertas_brutos: list[Alerta] = []
    for fonte, res in resultados.items():
        itens = res if isinstance(res, list) else [res]
        itens = [r for r in itens if r is not None]
        if not itens:
            continue
        scores[fonte] = max(_score_fonte(r.resumo, r.alertas) for r in itens)
        for r in itens:
            alertas_brutos.extend(r.alertas)

    alertas = alertas_emitidos if alertas_emitidos is not None else alertas_brutos
    alertas = [a for a in alertas if a.paciente_id == paciente_id]

    pesos = {f: config.PESOS_FUSAO.get(f, 0.1) for f in scores}
    soma = sum(pesos.values()) or 1.0
    pesos = {f: p / soma for f, p in pesos.items()}
    score = sum(pesos[f] * scores[f] for f in scores)

    # Corroboração entre modalidades: alertas ≥ ALTA de fontes distintas próximos no tempo
    corroboracoes: list[dict] = []
    graves = sorted([a for a in alertas if a.severidade >= Severidade.ALTA], key=lambda a: a.timestamp)
    usados: set[int] = set()
    for i, a in enumerate(graves):
        if i in usados:
            continue
        ta = _ts(a.timestamp)
        grupo = [a]
        for j in range(i + 1, len(graves)):
            b = graves[j]
            if j in usados or b.fonte == a.fonte:
                continue
            tb = _ts(b.timestamp)
            if ta is None or tb is None or abs((tb - ta).total_seconds()) <= janela_corroboracao_min * 60:
                if b.fonte not in {g.fonte for g in grupo}:
                    grupo.append(b)
                    usados.add(j)
        if len(grupo) >= 2:
            usados.add(i)
            corroboracoes.append({"timestamp": a.timestamp, "fontes": [g.fonte for g in grupo],
                                  "alertas": [g.resumo() for g in grupo]})
    if corroboracoes:
        n_fontes = max(len(c["fontes"]) for c in corroboracoes)
        score = min(1.0, score + 0.10 * (n_fontes - 1))

    nivel = nivel_risco(score)
    alertas_fusao: list[Alerta] = []
    for c in corroboracoes[:5]:
        alertas_fusao.append(Alerta(
            paciente_id=paciente_id, fonte="fusao", tipo="corroboracao_multimodal", severidade=Severidade.CRITICA,
            timestamp=c["timestamp"], score=score,
            mensagem=f"Evidências convergentes de {len(c['fontes'])} modalidades ({', '.join(c['fontes'])}) — "
                     f"risco global {nivel} ({score:.2f})",
            evidencias={"alertas": c["alertas"], "scores_por_fonte": scores},
            recomendacao="Priorizar avaliação médica presencial; revisar painel do paciente.",
        ))
    if nivel in ("alto", "crítico") and not corroboracoes:
        alertas_fusao.append(Alerta(
            paciente_id=paciente_id, fonte="fusao", tipo="risco_global", severidade=Severidade.ALTA if nivel == "alto" else Severidade.CRITICA,
            score=score, mensagem=f"Risco global {nivel} ({score:.2f}) — fonte dominante: {max(scores, key=scores.get)}",
            evidencias={"scores_por_fonte": scores, "pesos": pesos},
            recomendacao="Revisar painel do paciente e definir plano de monitoramento intensificado.",
        ))

    todos = sorted(alertas + alertas_fusao, key=lambda a: a.timestamp)
    linha = pd.DataFrame([a.to_dict() for a in todos]) if todos else pd.DataFrame(
        columns=["timestamp", "fonte", "tipo", "severidade", "mensagem"])
    resumo_alertas = {
        "total": len(todos),
        "por_fonte": linha["fonte"].value_counts().to_dict() if len(linha) else {},
        "por_severidade": linha["severidade"].value_counts().to_dict() if len(linha) else {},
        "primeiro_critico": next((a.timestamp for a in todos if a.severidade == Severidade.CRITICA), None),
    }
    return PerfilRisco(paciente_id, float(score), nivel, scores, pesos, corroboracoes, todos, linha, resumo_alertas)


def _ts(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s).replace(tzinfo=None)
    except Exception:
        return None
