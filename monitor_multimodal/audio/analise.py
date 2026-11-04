"""
analise.py — Pipeline de análise de áudio de consulta.

  áudio ──► biomarcadores vocais (librosa) ──► indicadores (fadiga, dispneia, disartria)
     └────► Azure Speech to Text ──► transcrição ──► Azure AI Language
                                                     (sentimento, frases-chave, entidades clínicas,
                                                      termos críticos) ──► alertas
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import config
from ..alertas import Alerta, Severidade
from ..azure_servicos import AnaliseTexto, Transcricao, analisar_texto, transcrever
from ..utils import agora_iso, log
from .features import extrair_features_vocais, indicadores_clinicos

TERMOS_EMERGENCIA = {"não consigo respirar", "dor no peito", "inconsciente", "convulsão", "desmaio", "sangramento"}
TERMOS_FADIGA = {"cansaço extremo", "fraqueza", "exausta", "exausto", "cansada", "cansado"}
TERMOS_RESPIRATORIOS = {"falta de ar", "não consigo respirar", "dificuldade para respirar", "lábios roxos"}


@dataclass
class ResultadoAudio:
    paciente_id: str
    arquivo: str
    features: dict[str, Any]
    indicadores: dict[str, Any]
    transcricao: Transcricao
    texto: AnaliseTexto
    alertas: list[Alerta] = field(default_factory=list)
    resumo: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paciente_id": self.paciente_id, "arquivo": self.arquivo, "features": self.features,
            "indicadores": self.indicadores, "transcricao": self.transcricao.to_dict(),
            "analise_texto": self.texto.to_dict(), "alertas": [a.to_dict() for a in self.alertas],
            "resumo": self.resumo,
        }


def analisar_audio(caminho: Path | str, paciente_id: str, timestamp: str | None = None,
                   modo: str | None = None) -> ResultadoAudio:
    caminho = Path(caminho)
    ts = timestamp or agora_iso()
    log.info(f"[áudio] {caminho.name} — extraindo biomarcadores vocais")
    feats = extrair_features_vocais(caminho)
    ind = indicadores_clinicos(feats)

    transcricao = transcrever(caminho, modo=modo)
    texto = analisar_texto(transcricao.texto, modo=modo) if transcricao.texto else \
        AnaliseTexto("neutro", {}, [], [], [], provedor="nenhum", erro="sem transcrição")

    # Fusão intra-modalidade: o conteúdo do que o paciente diz reforça o indicador acústico
    termos = set(texto.termos_criticos)
    if termos & TERMOS_RESPIRATORIOS:
        ind["dispneia"] = round(min(1.0, ind["dispneia"] + 0.25), 3)
    if termos & TERMOS_FADIGA:
        ind["fadiga"] = round(min(1.0, ind["fadiga"] + 0.20), 3)

    alertas: list[Alerta] = []
    base = dict(paciente_id=paciente_id, fonte="audio", timestamp=ts)
    emergencia = termos & TERMOS_EMERGENCIA
    if emergencia:
        alertas.append(Alerta(**base, tipo="termo_emergencia", severidade=Severidade.CRITICA,
                              mensagem=f"Paciente relata: {', '.join(sorted(emergencia))}",
                              evidencias={"termos": sorted(emergencia), "transcricao": transcricao.texto[:300],
                                          "provedor": transcricao.provedor},
                              recomendacao="Avaliação médica imediata.", score=1.0))
    outros = termos - emergencia
    if outros:
        alertas.append(Alerta(**base, tipo="termo_critico", severidade=Severidade.ALTA,
                              mensagem=f"Termos críticos na fala: {', '.join(sorted(outros))}",
                              evidencias={"termos": sorted(outros), "entidades": texto.entidades[:10],
                                          "provedor": texto.provedor},
                              recomendacao="Registrar sintomas no prontuário e avaliar em consulta.",
                              score=min(1.0, 0.5 + 0.1 * len(outros))))
    if ind["dispneia"] >= 0.6:
        alertas.append(Alerta(**base, tipo="dispneia_vocal", severidade=Severidade.ALTA,
                              mensagem=f"Padrão de fala compatível com dificuldade respiratória "
                                       f"(indicador {ind['dispneia']:.2f}): frases de {feats['duracao_media_frase_s']:.1f}s, "
                                       f"{feats['pausas_por_min']:.0f} pausas/min, HNR {feats['hnr_db']:.0f} dB",
                              evidencias={"indicador": ind["dispneia"], "componentes": ind["componentes"]},
                              recomendacao="Verificar SpO₂ e frequência respiratória.", score=ind["dispneia"]))
    if ind["fadiga"] >= 0.6:
        alertas.append(Alerta(**base, tipo="fadiga_vocal", severidade=Severidade.ATENCAO,
                              mensagem=f"Padrão de fala compatível com fadiga (indicador {ind['fadiga']:.2f}): "
                                       f"{feats['taxa_silabica']:.1f} síl/s, {100*feats['razao_pausa']:.0f}% de pausas, "
                                       f"variação de pitch {feats['f0_desvio_semitons']:.1f} st",
                              evidencias={"indicador": ind["fadiga"], "componentes": ind["componentes"]},
                              recomendacao="Investigar causas de fadiga (anemia, infecção, sono, medicação).",
                              score=ind["fadiga"]))
    if ind["disartria"] >= 0.65:
        alertas.append(Alerta(**base, tipo="disartria_suspeita", severidade=Severidade.ALTA,
                              mensagem=f"Alterações de articulação/fonação sugestivas de disartria (indicador {ind['disartria']:.2f})",
                              evidencias={"indicador": ind["disartria"], "componentes": ind["componentes"]},
                              recomendacao="Avaliação neurológica (descartar AVC) e fonoaudiológica.",
                              score=ind["disartria"]))
    if texto.sentimento == "negativo" and texto.scores_sentimento.get("negativo", 0) >= 0.7 and not alertas:
        alertas.append(Alerta(**base, tipo="sentimento_negativo", severidade=Severidade.INFO,
                              mensagem="Relato com sentimento predominantemente negativo",
                              evidencias={"scores": texto.scores_sentimento, "frases_chave": texto.frases_chave},
                              recomendacao="Considerar avaliação de bem-estar/psicologia.", score=0.3))

    resumo = {
        "provedor_transcricao": transcricao.provedor, "provedor_texto": texto.provedor,
        "n_palavras": transcricao.n_palavras, "sentimento": texto.sentimento,
        "termos_criticos": sorted(termos), "n_entidades": len(texto.entidades),
        "indicadores": {k: ind[k] for k in ("fadiga", "dispneia", "disartria")},
        "score_risco": float(max([a.score for a in alertas], default=0.0)),
        "n_alertas": len(alertas),
    }
    return ResultadoAudio(paciente_id, str(caminho), feats, ind, transcricao, texto, alertas, resumo)
