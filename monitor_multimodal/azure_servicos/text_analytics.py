"""
text_analytics.py — Análise de texto clínico (Azure AI Language / Text Analytics) com fallback local.

Azure:
  • analyze_sentiment               → sentimento (positivo/neutro/negativo) com scores
  • extract_key_phrases             → frases-chave
  • begin_analyze_healthcare_entities (Text Analytics for health) → entidades clínicas
    (sintomas, partes do corpo, medicamentos…) com categorias e confiança

Fallback local (sem credenciais): léxico de sentimento em português, dicionário de
sintomas/termos clínicos e busca de termos críticos (config.TERMOS_CRITICOS_AUDIO).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from .. import config
from ..utils import log


@dataclass
class AnaliseTexto:
    sentimento: str                               # positivo | neutro | negativo | misto
    scores_sentimento: dict[str, float]
    frases_chave: list[str]
    entidades: list[dict[str, Any]]               # {texto, categoria, confianca, [negada]}
    termos_criticos: list[str]
    provedor: str                                 # azure_language | lexico_local
    erro: str | None = None
    detalhes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentimento": self.sentimento, "scores_sentimento": self.scores_sentimento,
            "frases_chave": self.frases_chave, "entidades": self.entidades,
            "termos_criticos": self.termos_criticos, "provedor": self.provedor, "erro": self.erro,
        }


def azure_language_disponivel() -> bool:
    return bool(config.AZURE_LANGUAGE_KEY and config.AZURE_LANGUAGE_ENDPOINT)


def normalizar(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def termos_criticos(texto: str, termos: list[str] | None = None) -> list[str]:
    t = normalizar(texto)
    return [termo for termo in (termos or config.TERMOS_CRITICOS_AUDIO) if normalizar(termo) in t]


def analisar_texto(texto: str, modo: str | None = None, idioma: str = "pt") -> AnaliseTexto:
    modo = (modo or config.MODO_SERVICOS).lower()
    if not texto.strip():
        return AnaliseTexto("neutro", {}, [], [], [], provedor="nenhum", erro="texto vazio")
    if modo in ("auto", "azure") and azure_language_disponivel():
        try:
            return _analisar_azure(texto, idioma)
        except Exception as exc:
            log.warning(f"Azure Language falhou: {exc}")
            if modo == "azure":
                return AnaliseTexto("neutro", {}, [], [], termos_criticos(texto), "azure_language", erro=str(exc))
    elif modo == "azure":
        return AnaliseTexto("neutro", {}, [], [], termos_criticos(texto), "azure_language",
                            erro="Credenciais AZURE_LANGUAGE_* ausentes")
    return _analisar_local(texto)


# ─── Azure AI Language ───────────────────────────────────────────────────────
def _analisar_azure(texto: str, idioma: str) -> AnaliseTexto:
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    cli = TextAnalyticsClient(endpoint=config.AZURE_LANGUAGE_ENDPOINT,
                              credential=AzureKeyCredential(config.AZURE_LANGUAGE_KEY))
    docs = [texto]
    sent = cli.analyze_sentiment(docs, language=idioma, show_opinion_mining=True)[0]
    if sent.is_error:
        raise RuntimeError(sent.error)
    kp = cli.extract_key_phrases(docs, language=idioma)[0]
    frases = [] if kp.is_error else list(kp.key_phrases)

    entidades: list[dict] = []
    detalhes: dict[str, Any] = {}
    try:
        poller = cli.begin_analyze_healthcare_entities(docs, language=idioma)
        for doc in poller.result():
            if doc.is_error:
                continue
            for e in doc.entities:
                entidades.append({
                    "texto": e.text, "categoria": e.category, "confianca": round(e.confidence_score, 3),
                    "negada": bool(getattr(e, "assertion", None) and e.assertion.certainty == "negative"),
                })
            detalhes["relacoes"] = [
                {"tipo": r.relation_type, "papeis": [{"nome": p.name, "texto": p.entity.text} for p in r.roles]}
                for r in doc.entity_relations
            ]
    except Exception as exc:  # TA4H pode não estar habilitado/região
        log.warning(f"Text Analytics for health indisponível ({exc}); usando reconhecimento de entidades genérico")
        ner = cli.recognize_entities(docs, language=idioma)[0]
        entidades = [] if ner.is_error else [
            {"texto": e.text, "categoria": e.category, "confianca": round(e.confidence_score, 3)} for e in ner.entities
        ]
    sentimento = {"positive": "positivo", "neutral": "neutro", "negative": "negativo", "mixed": "misto"}[sent.sentiment]
    scores = {"positivo": sent.confidence_scores.positive, "neutro": sent.confidence_scores.neutral,
              "negativo": sent.confidence_scores.negative}
    detalhes["sentencas"] = [{"texto": s.text, "sentimento": s.sentiment} for s in sent.sentences]
    log.info(f"[Azure Language] sentimento={sentimento} entidades={len(entidades)} frases-chave={len(frases)}")
    return AnaliseTexto(sentimento, scores, frases, entidades, termos_criticos(texto), "azure_language", detalhes=detalhes)


# ─── Fallback local ──────────────────────────────────────────────────────────
_NEGATIVAS = {
    "dor", "cansado", "cansada", "cansaço", "exausta", "exausto", "fraqueza", "fraco", "fraca", "tontura",
    "medo", "pior", "piorou", "falta", "dificil", "dificuldade", "nao consigo", "sangue", "febre", "vomito",
    "roxos", "roxo", "sofrendo", "ruim", "mal", "angustia", "ansioso", "ansiosa", "desespero", "enjoo",
    "insuportavel", "preocupado", "preocupada", "triste", "assustado", "assustada",
}
_POSITIVAS = {
    "bem", "melhor", "melhorei", "melhorou", "otimo", "otima", "bom", "boa", "tranquilo", "tranquila",
    "normalmente", "consegui", "sem dor", "nenhuma queixa", "feliz", "disposto", "disposta", "animado",
}
_SINTOMAS = {  # termo normalizado → categoria (imitando as categorias do Text Analytics for health)
    "falta de ar": "SymptomOrSign", "dispneia": "SymptomOrSign", "dor no peito": "SymptomOrSign",
    "dor": "SymptomOrSign", "febre": "SymptomOrSign", "tontura": "SymptomOrSign", "fraqueza": "SymptomOrSign",
    "cansaco": "SymptomOrSign", "cansaco extremo": "SymptomOrSign", "exausta": "SymptomOrSign",
    "tosse": "SymptomOrSign", "vomito": "SymptomOrSign", "nausea": "SymptomOrSign", "desmaio": "SymptomOrSign",
    "sangramento": "SymptomOrSign", "labios roxos": "SymptomOrSign", "cianose": "SymptomOrSign",
    "palpitacao": "SymptomOrSign", "confusao": "SymptomOrSign", "dormencia": "SymptomOrSign",
    "peito": "BodyStructure", "pernas": "BodyStructure", "labios": "BodyStructure", "cabeca": "BodyStructure",
    "coracao": "BodyStructure", "pulmao": "BodyStructure", "braco": "BodyStructure", "estomago": "BodyStructure",
    "remedios": "MedicationClass", "medicamento": "MedicationClass", "antibiotico": "MedicationClass",
    "insulina": "MedicationName", "dipirona": "MedicationName", "paracetamol": "MedicationName",
    "ontem a noite": "Time", "hoje de manha": "Time", "desde ontem": "Time",
    "cama": "Direction", "corredor": "Direction",
}


def _analisar_local(texto: str) -> AnaliseTexto:
    t = normalizar(texto)
    palavras = re.findall(r"[a-z]+", t)
    def _negado(w: str) -> bool:
        return any(re.search(r"\b(nao|nem|sem)\s+(\w+\s+){0,2}" + re.escape(w), t) for _ in [0])
    neg = sum(1 for w in _NEGATIVAS if w in t and not _negado(w))
    pos = sum(1 for w in _POSITIVAS if w in t)
    total = max(neg + pos, 1)
    s_neg, s_pos = neg / total, pos / total
    s_neu = max(0.0, 1 - abs(s_pos - s_neg))
    if neg and pos and abs(s_pos - s_neg) < 0.3:
        sentimento = "misto"
    elif s_neg > 0.6:
        sentimento = "negativo"
    elif s_pos > 0.6:
        sentimento = "positivo"
    else:
        sentimento = "neutro"
    scores = {"positivo": round(s_pos, 3), "neutro": round(s_neu, 3), "negativo": round(s_neg, 3)}

    entidades = []
    ocupados: list[tuple[int, int]] = []
    for termo in sorted(_SINTOMAS, key=len, reverse=True):     # termos mais longos primeiro
        for m in re.finditer(r"\b" + re.escape(termo) + r"\b", t):
            if any(a <= m.start() < b for a, b in ocupados):
                continue
            ocupados.append((m.start(), m.end()))
            entidades.append({"texto": texto[m.start():m.end()], "categoria": _SINTOMAS[termo], "confianca": 0.75,
                              "negada": bool(re.search(r"\b(nao|sem)\s+(\w+\s+){0,2}$", t[max(0, m.start()-25):m.start()]))})
    entidades.sort(key=lambda e: t.find(normalizar(e["texto"])))

    stop = {"de", "a", "o", "e", "que", "nao", "um", "uma", "me", "nem", "os", "as", "do", "da", "no", "na",
            "com", "em", "para", "por", "eu", "esse", "essa", "quando", "fico", "estou", "esta", "meus", "minha"}
    frases = []
    for n in (3, 2):
        for i in range(len(palavras) - n + 1):
            ng = palavras[i:i + n]
            if ng[0] not in stop and ng[-1] not in stop and any(normalizar(s) in " ".join(ng) for s in _SINTOMAS):
                f = " ".join(ng)
                if f not in frases:
                    frases.append(f)
    return AnaliseTexto(sentimento, scores, frases[:8], entidades, termos_criticos(texto), "lexico_local",
                        detalhes={"n_palavras": len(palavras), "neg": neg, "pos": pos})
