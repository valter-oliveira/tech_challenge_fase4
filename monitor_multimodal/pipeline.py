"""
pipeline.py — Orquestração ponta a ponta (demonstração completa).

  1. gera/carrega dados (sinais vitais, prescrições, movimentação, áudios, vídeos)
  2. roda cada módulo de análise por paciente
  3. despacha alertas pelo GerenciadorAlertas (dedup + canais)
  4. fusão multimodal → perfil de risco
  5. relatórios (Markdown/JSON/gráficos) + avaliação contra rótulos injetados
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from . import config, dados_sinteticos
from .alertas import GerenciadorAlertas
from .anomalias.avaliacao import metricas_binarias, tempo_antecipacao
from .anomalias.movimentacao import analisar_movimentacao
from .anomalias.prescricoes import analisar_prescricoes
from .anomalias.sinais_vitais import analisar_sinais_vitais
from .audio import analisar_audio
from .azure_servicos.speech import azure_speech_disponivel
from .azure_servicos.text_analytics import azure_language_disponivel
from .fusao import fundir
from .relatorios import gerar_relatorio_geral, gerar_relatorio_paciente
from .utils import log, salvar_json
from .video import analisar_video

# Mapeamento da demonstração: qual áudio/vídeo pertence a qual paciente e quando foi gravado.
# Os horários coincidem com episódios dos sinais vitais/movimentação para exercitar a fusão.
MIDIA_DEMO = {
    "P001": {"audio": [("consulta_normal.wav", "2026-11-01T10:00:00")],
             "video": [("vtest.avi", "2026-11-02T10:00:00", "enfermaria")]},
    "P002": {"audio": [("consulta_fadiga.wav", "2026-11-02T14:00:00")], "video": []},
    "P003": {"audio": [("consulta_dispneia.wav", "2026-11-01T18:30:00")],
             "video": [("fisioterapia_sintetica.mp4", "2026-11-03T03:00:00", "fisioterapia")]},
}


def _caminho_midia(nome: str) -> Path | None:
    for pasta in (config.DADOS_SINTETICOS_DIR / "audio", config.DADOS_SINTETICOS_DIR / "video", config.DADOS_AMOSTRAS_DIR):
        if (pasta / nome).exists():
            return pasta / nome
    return None


def executar_demo(pacientes: list[str] | None = None, com_video: bool = True, com_audio: bool = True,
                  gerar_dados: bool = True, max_frames_video: int | None = None, modo_servicos: str | None = None) -> dict[str, Any]:
    t0 = time.time()
    if gerar_dados or not (config.DADOS_SINTETICOS_DIR / "sinais_vitais.csv").exists():
        dados_sinteticos.gerar_tudo()
    sv = pd.read_csv(config.DADOS_SINTETICOS_DIR / "sinais_vitais.csv", parse_dates=["timestamp"])
    pr = pd.read_csv(config.DADOS_SINTETICOS_DIR / "prescricoes.csv", parse_dates=["timestamp"])
    mv = pd.read_csv(config.DADOS_SINTETICOS_DIR / "movimentacao.csv", parse_dates=["timestamp"])
    pacientes = pacientes or sorted(sv["paciente_id"].unique())

    log_alertas = config.OUTPUTS_DIR / "alertas.jsonl"
    if log_alertas.exists():
        log_alertas.unlink()
    gerenciador = GerenciadorAlertas(caminho_log=log_alertas, console=False)

    perfis, avaliacoes, todos_resultados = [], {}, {}
    for pid in pacientes:
        log.info(f"[bold]═══ Paciente {pid} ═══[/]")
        resultados: dict[str, Any] = {}
        resultados["sinais_vitais"] = analisar_sinais_vitais(sv[sv.paciente_id == pid], pid)
        resultados["prescricoes"] = analisar_prescricoes(pr[pr.paciente_id == pid], pid)
        resultados["movimentacao"] = analisar_movimentacao(mv[mv.paciente_id == pid], pid)
        midia = MIDIA_DEMO.get(pid, {"audio": [], "video": []})
        if com_audio:
            resultados["audio"] = [analisar_audio(c, pid, timestamp=ts, modo=modo_servicos)
                                   for nome, ts in midia["audio"] if (c := _caminho_midia(nome))]
        if com_video:
            resultados["video"] = [analisar_video(c, pid, tipo_sessao=tipo, timestamp=ts, max_frames=max_frames_video)
                                   for nome, ts, tipo in midia["video"] if (c := _caminho_midia(nome))]
        # despacho dos alertas (dedup/escalada/canais)
        brutos = []
        for r in resultados.values():
            for x in (r if isinstance(r, list) else [r]):
                brutos.extend(x.alertas)
        emitidos = gerenciador.emitir_lote(sorted(brutos, key=lambda a: a.timestamp))
        perfil = fundir(pid, resultados, alertas_emitidos=emitidos)
        gerenciador.emitir_lote([a for a in perfil.alertas if a.fonte == "fusao"])
        log.info(f"{pid}: risco [bold]{perfil.nivel}[/] ({perfil.score_global:.2f}) · {len(brutos)} detecções → {len(emitidos)} alertas emitidos")

        # avaliação contra rótulos
        av: dict[str, Any] = {}
        d = resultados["sinais_vitais"].dados
        if "anomalia_injetada" in d and d["anomalia_injetada"].any():
            av["sinais_vitais"] = {"metricas": metricas_binarias(d["anomalia_injetada"], d["anomalia"], d["score_anomalia"]),
                                   "episodios": tempo_antecipacao(d)}
        elif "anomalia_injetada" in d:
            av["sinais_vitais"] = {"metricas": metricas_binarias(d["anomalia_injetada"], d["anomalia"]), "episodios": []}
        d = resultados["movimentacao"].dados
        if "anomalia_injetada" in d:
            av["movimentacao"] = {"metricas": metricas_binarias(d["anomalia_injetada"], d["anomalia"])}
        d = resultados["prescricoes"].dados
        if "anomalia_injetada" in d:
            av["prescricoes"] = {"metricas": metricas_binarias(d["anomalia_injetada"], d["anomalia"])}
        avaliacoes[pid] = av
        gerar_relatorio_paciente(perfil, resultados, avaliacao=av)
        perfis.append(perfil)
        todos_resultados[pid] = resultados

    info = {
        "pacientes": ", ".join(pacientes), "modo_servicos": modo_servicos or config.MODO_SERVICOS,
        "azure_speech": "configurado" if azure_speech_disponivel() else "não configurado (fallback local)",
        "azure_language": "configurado" if azure_language_disponivel() else "não configurado (fallback local)",
        "video": com_video, "audio": com_audio, "duracao_execucao_s": round(time.time() - t0, 1),
    }
    rel = gerar_relatorio_geral(perfis, gerenciador, info, avaliacoes)
    salvar_json({"info": info, "perfis": [p.to_dict() for p in perfis], "avaliacoes": avaliacoes,
                 "alertas": gerenciador.resumo()}, config.OUTPUTS_DIR / "resultado_demo.json")
    log.info(f"Relatório geral: {rel}")
    return {"perfis": perfis, "gerenciador": gerenciador, "resultados": todos_resultados, "avaliacoes": avaliacoes,
            "info": info, "relatorio": rel}
