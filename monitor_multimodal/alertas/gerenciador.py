"""
gerenciador.py — Gerenciador de alertas para a equipe médica.

Responsabilidades:
  • Padronizar o alerta (paciente, fonte, severidade, mensagem, evidências)
  • Deduplicar alertas repetidos em janela curta (evita "fadiga de alarme")
  • Escalar severidade quando a mesma anomalia persiste
  • Despachar para canais: console, arquivo JSONL, webhook (Teams/Slack), e-mail SMTP
"""

from __future__ import annotations

import json
import smtplib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from email.message import EmailMessage
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable

import requests

from .. import config
from ..utils import _EncoderJSON, agora_iso, log


class Severidade(IntEnum):
    INFO = 1
    ATENCAO = 2
    ALTA = 3
    CRITICA = 4

    @property
    def rotulo(self) -> str:
        return {1: "INFO", 2: "ATENÇÃO", 3: "ALTA", 4: "CRÍTICA"}[int(self)]

    @property
    def cor(self) -> str:
        return {1: "cyan", 2: "yellow", 3: "orange3", 4: "bold red"}[int(self)]


@dataclass
class Alerta:
    paciente_id: str
    fonte: str                     # sinais_vitais | prescricoes | movimentacao | audio | video | fusao
    tipo: str                      # ex.: "spo2_critico", "queda_detectada"
    severidade: Severidade
    mensagem: str
    timestamp: str = field(default_factory=agora_iso)
    evidencias: dict[str, Any] = field(default_factory=dict)
    recomendacao: str = ""
    score: float = 0.0             # 0-1, confiança/intensidade da anomalia

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severidade"] = self.severidade.rotulo
        d["severidade_nivel"] = int(self.severidade)
        return d

    def resumo(self) -> str:
        return f"[{self.severidade.rotulo}] {self.paciente_id} · {self.fonte} · {self.mensagem}"


class GerenciadorAlertas:
    """Centraliza a emissão de alertas e o despacho para os canais configurados."""

    def __init__(
        self,
        caminho_log: Path | str | None = None,
        janela_dedup: timedelta = timedelta(minutes=config.JANELA_DEDUP_MIN),
        console: bool = True,
        webhook_url: str | None = None,
        smtp: dict | None = None,
    ):
        self.caminho_log = Path(caminho_log) if caminho_log else config.OUTPUTS_DIR / "alertas.jsonl"
        self.janela_dedup = janela_dedup
        self.console = console
        self.webhook_url = webhook_url if webhook_url is not None else config.ALERTA_WEBHOOK_URL
        self.smtp = smtp if smtp is not None else config.ALERTA_SMTP
        self.alertas: list[Alerta] = []
        self.suprimidos = 0
        self._ultimo_por_chave: dict[tuple, datetime] = {}
        self._persistencia: dict[tuple, int] = {}
        self._assinantes: list[Callable[[Alerta], None]] = []

    # ── API pública ─────────────────────────────────────────────────────────
    def assinar(self, callback: Callable[[Alerta], None]) -> None:
        """Permite que outros componentes (dashboard, fusão) recebam alertas em tempo real."""
        self._assinantes.append(callback)

    def emitir(self, alerta: Alerta) -> Alerta | None:
        """Registra e despacha o alerta. Devolve None se foi deduplicado."""
        chave = (alerta.paciente_id, alerta.fonte, alerta.tipo)
        ts = datetime.fromisoformat(alerta.timestamp)

        ultimo = self._ultimo_por_chave.get(chave)
        if ultimo is not None and (ultimo.tzinfo is None) != (ts.tzinfo is None):
            ultimo = None  # timestamps com e sem fuso não são comparáveis → não deduplica
        if ultimo is not None and 0 <= (ts - ultimo).total_seconds() < self.janela_dedup.total_seconds():
            # Mesma anomalia dentro da janela → conta persistência. Na N-ésima repetição
            # (PERSISTENCIA_MINIMA) escala a severidade UMA vez e reemite; o restante é suprimido.
            self._persistencia[chave] = self._persistencia.get(chave, 1) + 1
            if (
                self._persistencia[chave] == config.PERSISTENCIA_MINIMA
                and alerta.severidade < Severidade.CRITICA
            ):
                alerta.severidade = Severidade(int(alerta.severidade) + 1)
                alerta.mensagem += f" (persistente: {self._persistencia[chave]}ª ocorrência em {int(self.janela_dedup.total_seconds()//60)} min)"
            else:
                self.suprimidos += 1
                return None
        else:
            self._persistencia[chave] = 1

        self._ultimo_por_chave[chave] = ts
        self.alertas.append(alerta)
        self._despachar(alerta)
        return alerta

    def emitir_lote(self, alertas: list[Alerta]) -> list[Alerta]:
        return [a for a in (self.emitir(x) for x in alertas) if a is not None]

    def por_paciente(self, paciente_id: str) -> list[Alerta]:
        return [a for a in self.alertas if a.paciente_id == paciente_id]

    def resumo(self) -> dict[str, Any]:
        contagem = {s.rotulo: 0 for s in Severidade}
        for a in self.alertas:
            contagem[a.severidade.rotulo] += 1
        return {
            "total": len(self.alertas),
            "suprimidos_por_deduplicacao": self.suprimidos,
            "por_severidade": contagem,
            "por_fonte": self._contar("fonte"),
            "por_paciente": self._contar("paciente_id"),
        }

    def _contar(self, campo: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self.alertas:
            out[getattr(a, campo)] = out.get(getattr(a, campo), 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    # ── Canais ──────────────────────────────────────────────────────────────
    def _despachar(self, alerta: Alerta) -> None:
        if self.console:
            log.info(f"[{alerta.severidade.cor}]{alerta.resumo()}[/]")
        self._gravar_jsonl(alerta)
        if self.webhook_url and alerta.severidade >= Severidade.ALTA:
            self._enviar_webhook(alerta)
        if self.smtp.get("host") and alerta.severidade >= Severidade.CRITICA:
            self._enviar_email(alerta)
        for cb in self._assinantes:
            try:
                cb(alerta)
            except Exception as exc:  # pragma: no cover
                log.warning(f"Assinante de alerta falhou: {exc}")

    def _gravar_jsonl(self, alerta: Alerta) -> None:
        self.caminho_log.parent.mkdir(parents=True, exist_ok=True)
        with self.caminho_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(alerta.to_dict(), cls=_EncoderJSON, ensure_ascii=False) + "\n")

    def _enviar_webhook(self, alerta: Alerta) -> None:
        payload = {
            "text": f"🚨 {alerta.resumo()}\nRecomendação: {alerta.recomendacao or '-'}",
            **alerta.to_dict(),
        }
        try:
            requests.post(self.webhook_url, json=payload, timeout=5)
        except requests.RequestException as exc:
            log.warning(f"Webhook falhou: {exc}")

    def _enviar_email(self, alerta: Alerta) -> None:
        msg = EmailMessage()
        msg["Subject"] = f"[{alerta.severidade.rotulo}] Alerta clínico — paciente {alerta.paciente_id}"
        msg["From"] = self.smtp["user"]
        msg["To"] = self.smtp["destino"]
        msg.set_content(
            f"{alerta.mensagem}\n\nFonte: {alerta.fonte}\nQuando: {alerta.timestamp}\n"
            f"Recomendação: {alerta.recomendacao}\n\nEvidências:\n"
            f"{json.dumps(alerta.evidencias, cls=_EncoderJSON, ensure_ascii=False, indent=2)}"
        )
        try:
            with smtplib.SMTP(self.smtp["host"], self.smtp["port"], timeout=10) as s:
                s.starttls()
                s.login(self.smtp["user"], self.smtp["password"])
                s.send_message(msg)
        except Exception as exc:  # pragma: no cover
            log.warning(f"E-mail falhou: {exc}")
