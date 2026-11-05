"""
gerador.py — Relatórios automáticos em Markdown (+ JSON) por paciente, por vídeo e geral.

Os relatórios são o entregável "Gerar relatórios automáticos indicando desvios ou falhas"
e alimentam o relatório técnico final (docs/relatorio_tecnico.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .. import config
from ..alertas import Severidade
from ..utils import agora_iso, salvar_json
from . import graficos as G


def _tabela(df: pd.DataFrame, max_linhas: int = 40) -> str:
    if df is None or len(df) == 0:
        return "_(nenhum registro)_\n"
    df = df.head(max_linhas).copy()
    cols = [str(c) for c in df.columns]
    linhas = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        linhas.append("| " + " | ".join(("-" if (isinstance(v, float) and pd.isna(v)) or v is None else str(v))
                                        .replace("\n", " ").replace("|", "/")[:120] for v in r.values) + " |")
    return "\n".join(linhas) + "\n"


def _rel(p: Path | str, base: Path) -> str:
    try:
        return str(Path(p).relative_to(base))
    except ValueError:
        return str(p)


def _tabela_alertas(alertas, base_dir: Path | None = None, n: int = 30) -> str:
    if not alertas:
        return "_(nenhum alerta)_\n"
    ordem = sorted(alertas, key=lambda a: (-int(a.severidade), a.timestamp))[:n]
    df = pd.DataFrame([{"quando": a.timestamp[:16].replace("T", " "), "severidade": a.severidade.rotulo, "fonte": a.fonte,
                        "tipo": a.tipo, "mensagem": a.mensagem[:110], "recomendação": a.recomendacao[:70]} for a in ordem])
    return _tabela(df, n)


# ─── Relatório de vídeo ──────────────────────────────────────────────────────
def gerar_relatorio_video(res, pasta: Path | None = None) -> Path:
    pasta = Path(pasta) if pasta else config.OUTPUTS_DIR / "video" / Path(res.arquivo).stem
    pasta.mkdir(parents=True, exist_ok=True)
    nome = Path(res.arquivo).stem
    graf = G.grafico_video(res.serie, res.eventos, pasta / f"{nome}_grafico.png", f"Vídeo: {Path(res.arquivo).name}")
    r = res.resumo
    md = [f"# Relatório automático de vídeo — `{Path(res.arquivo).name}`", "",
          f"Paciente **{res.paciente_id}** · gerado em {agora_iso()}", "",
          "## Resumo", "",
          f"- Duração: {r['duracao_s']} s · frames processados: {r['frames_processados']} (1 a cada {config.VIDEO_PASSO_FRAMES})",
          f"- Modelos: energia de movimento + fluxo óptico (OpenCV){' + YOLOv8 detecção + YOLOv8-pose' if r['yolo'] else ' (YOLOv8 indisponível)'}",
          f"- Pessoas detectadas (máx. simultâneas): {r['pessoas_max']} · classes vistas: {', '.join(r['objetos_detectados']) or '-'}",
          f"- **Eventos anômalos: {r['n_eventos']}** → {r['eventos_por_tipo']}",
          f"- Score de risco do vídeo: **{r['score_risco']:.2f}**", ""]
    if res.video_anotado:
        md += [f"- Vídeo anotado (caixas + esqueleto): `{_rel(res.video_anotado, config.BASE_DIR)}`", ""]
    md += ["## Linha do tempo dos eventos", ""]
    if res.eventos:
        df = pd.DataFrame([{"tempo (s)": f"{e['tempo_s']:.1f}", "severidade": e["severidade"], "tipo": e["tipo"],
                            "descrição": e["descricao"], "score": e["score"]} for e in res.eventos])
        md.append(_tabela(df, 60))
    else:
        md.append("_Nenhum desvio detectado._\n")
    md += ["", f"![gráfico]({graf.name})", ""]
    if res.frames_chave:
        md += ["## Frames-chave dos eventos", ""]
        for e in res.eventos:
            if e.get("frame_chave"):
                md += [f"**{e['tempo_s']:.1f}s — {e['tipo']} ({e['severidade']})**  ", f"![{e['tipo']}]({Path(e['frame_chave']).name})", ""]
    md += ["## Interpretação", "",
           "- `pico_seguido_de_imobilidade` / `queda_pose`: movimento brusco seguido de ausência de movimento ou queda rápida do quadril — padrão de queda/colapso; exige verificação imediata.",
           "- `imobilidade_prolongada`: paciente parado durante sessão ativa — possível mal-estar, dor ou interrupção do procedimento.",
           "- `quebra_de_ritmo` / `perda_amplitude`: a cadência ou a amplitude do exercício mudou — fadiga, dor ou execução incorreta.",
           "- `assimetria_postural` / `corpo_horizontal`: compensação unilateral ou paciente deitado/caído.",
           "- `objeto_critico` / `pessoa_saiu_de_cena`: objetos de atenção em área crítica ou paciente fora do campo de visão.",
           "- `movimento_atipico`: desvio estatístico (z-score robusto / Isolation Forest) do padrão de movimento do próprio vídeo.", ""]
    destino = pasta / f"{nome}_relatorio.md"
    destino.write_text("\n".join(md), encoding="utf-8")
    salvar_json(res.to_dict(), pasta / f"{nome}_resultado.json")
    return destino


# ─── Relatório por paciente ──────────────────────────────────────────────────
def gerar_relatorio_paciente(perfil, resultados: dict[str, Any], pasta: Path | None = None,
                             avaliacao: dict[str, Any] | None = None) -> Path:
    pid = perfil.paciente_id
    pasta = Path(pasta) if pasta else config.OUTPUTS_DIR / "pacientes" / pid
    pasta.mkdir(parents=True, exist_ok=True)
    md = [f"# Painel do paciente {pid}", "", f"Gerado em {agora_iso()}", "",
          f"## Risco global: **{perfil.nivel.upper()}** ({perfil.score_global:.2f})", "",
          _tabela(pd.DataFrame([{"fonte": f, "score": f"{s:.2f}", "peso": f"{perfil.pesos_usados[f]:.2f}"} for f, s in perfil.scores_por_fonte.items()])),
          f"Alertas: {perfil.resumo_alertas['total']} · por severidade: {perfil.resumo_alertas['por_severidade']} · por fonte: {perfil.resumo_alertas['por_fonte']}", ""]
    if perfil.corroboracoes:
        md += ["### Corroboração multimodal", ""]
        for c in perfil.corroboracoes[:5]:
            md += [f"- **{c['timestamp'][:16].replace('T', ' ')}** — {', '.join(c['fontes'])}:"] + [f"    - {a}" for a in c["alertas"]]
        md.append("")
    md += ["## Alertas prioritários", "", _tabela_alertas(perfil.alertas), ""]

    sv = resultados.get("sinais_vitais")
    if sv is not None:
        g = G.grafico_sinais_vitais(sv.dados, pasta / "sinais_vitais.png", f"{pid} — sinais vitais")
        r = sv.resumo
        md += ["## Sinais vitais", "",
               f"- Amostras: {r['n_amostras']} · anomalias: {r['n_anomalias']} ({r['pct_anomalias']}%) · por método: {r['por_metodo']}",
               f"- Escore de alerta precoce máximo: {r['ews_maximo']} · score de risco: {r['score_risco']:.2f}", "",
               f"![sinais vitais]({g.name})", ""]
        if avaliacao and "sinais_vitais" in avaliacao:
            a = avaliacao["sinais_vitais"]
            md += ["**Avaliação contra anomalias injetadas**", "", _tabela(pd.DataFrame([a["metricas"]]))]
            if a.get("episodios"):
                md += [_tabela(pd.DataFrame(a["episodios"]))]
            md.append("")
    pr = resultados.get("prescricoes")
    if pr is not None:
        g = G.grafico_prescricoes(pr.dados, pasta / "prescricoes.png", f"{pid} — prescrições")
        r = pr.resumo
        md += ["## Prescrições", "", f"- Prescrições: {r['n_prescricoes']} · anômalas: {r['n_anomalias']} · motivos: {r['motivos']}", "",
               f"![prescrições]({g.name})", ""]
        anom = pr.dados[pr.dados["anomalia"]][["timestamp", "medicamento", "dose_mg", "frequencia_dia", "via", "prescritor", "motivos"]]
        md += [_tabela(anom), ""]
    mv = resultados.get("movimentacao")
    if mv is not None:
        g = G.grafico_movimentacao(mv.dados, pasta / "movimentacao.png", f"{pid} — movimentação")
        r = mv.resumo
        md += ["## Movimentação", "", f"- Horas monitoradas: {r['n_horas']} · anômalas: {r['n_anomalias']} · motivos: {r['motivos']} · máx. horas imóvel: {r['max_horas_imovel']}", "",
               f"![movimentação]({g.name})", ""]
        if avaliacao and "movimentacao" in avaliacao:
            md += [_tabela(pd.DataFrame([avaliacao["movimentacao"]["metricas"]])), ""]
    auds = resultados.get("audio") or []
    auds = auds if isinstance(auds, list) else [auds]
    if auds:
        md += ["## Áudio de consulta", ""]
        G.grafico_audio({Path(a.arquivo).stem: a.features for a in auds}, {Path(a.arquivo).stem: a.indicadores for a in auds}, pasta / "audio.png")
        for a in auds:
            f, ind, t = a.features, a.indicadores, a.texto
            entidades = ", ".join(f"{e['texto']} ({e['categoria']})" for e in t.entidades[:10]) or "-"
            md += [f"### `{Path(a.arquivo).name}`", "",
                   f"- Transcrição ({a.transcricao.provedor}, {a.transcricao.n_palavras} palavras): _\"{a.transcricao.texto[:400]}\"_",
                   f"- Sentimento ({t.provedor}): **{t.sentimento}** {t.scores_sentimento}",
                   f"- Termos críticos: {', '.join(t.termos_criticos) or '-'}",
                   f"- Entidades clínicas: {entidades}",
                   f"- Frases-chave: {', '.join(t.frases_chave[:8]) or '-'}",
                   f"- Biomarcadores: {f.get('taxa_silabica')} síl/s · pausas {100*f.get('razao_pausa', 0):.0f}% ({f.get('pausas_por_min')}/min, média {f.get('duracao_media_pausa_s')}s) · "
                   f"f0 {f.get('f0_mediana_hz')} Hz ± {f.get('f0_desvio_semitons')} st · jitter {f.get('jitter_proxy')} · shimmer {f.get('shimmer_proxy')} · HNR {f.get('hnr_db')} dB",
                   f"- **Indicadores**: fadiga {ind['fadiga']:.2f} · dispneia {ind['dispneia']:.2f} · disartria {ind['disartria']:.2f}", ""]
        md += ["![áudio](audio.png)", ""]
    vids = resultados.get("video") or []
    vids = vids if isinstance(vids, list) else [vids]
    if vids:
        md += ["## Vídeo", ""]
        for v in vids:
            rel = gerar_relatorio_video(v)
            md += [f"- `{Path(v.arquivo).name}`: {v.resumo['n_eventos']} eventos {v.resumo['eventos_por_tipo']} · "
                   f"pessoas máx. {v.resumo['pessoas_max']} · score {v.resumo['score_risco']:.2f} → relatório: `{_rel(rel, config.BASE_DIR)}`"]
            for e in v.eventos:
                if e["severidade_nivel"] >= 3:
                    md.append(f"    - **{e['tempo_s']:.1f}s {e['severidade']}** {e['tipo']}: {e['descricao'][:120]}")
        md.append("")
    destino = pasta / "relatorio.md"
    destino.write_text("\n".join(md), encoding="utf-8")
    perfil.linha_do_tempo.to_csv(pasta / "alertas.csv", index=False)
    for nome in ("sinais_vitais", "prescricoes", "movimentacao"):
        if resultados.get(nome) is not None:
            resultados[nome].dados.to_csv(pasta / f"{nome}.csv", index=False)
    if auds:
        salvar_json([a.to_dict() for a in auds], pasta / "audio.json")
    if vids:
        salvar_json([v.to_dict() for v in vids], pasta / "video.json")
    salvar_json(perfil.to_dict() | {"resumos": {k: (v.resumo if not isinstance(v, list) else [x.resumo for x in v])
                                                 for k, v in resultados.items() if v is not None}}, pasta / "perfil.json")
    return destino


# ─── Relatório geral ─────────────────────────────────────────────────────────
def gerar_relatorio_geral(perfis: list, gerenciador, info: dict[str, Any], avaliacoes: dict[str, Any],
                          pasta: Path | None = None) -> Path:
    pasta = Path(pasta) if pasta else config.OUTPUTS_DIR
    pasta.mkdir(parents=True, exist_ok=True)
    g = G.grafico_risco(perfis, pasta / "risco_pacientes.png")
    res = gerenciador.resumo()
    md = ["# Relatório geral da execução — monitoramento multimodal", "", f"Gerado em {agora_iso()}", "",
          "## Configuração", "", _tabela(pd.DataFrame([{"parâmetro": k, "valor": str(v)} for k, v in info.items()])), "",
          "## Risco por paciente", "",
          _tabela(pd.DataFrame([{"paciente": p.paciente_id, "nível": p.nivel, "score": f"{p.score_global:.2f}",
                                 **{f"score_{f}": f"{s:.2f}" for f, s in p.scores_por_fonte.items()},
                                 "corroborações": len(p.corroboracoes), "alertas": p.resumo_alertas["total"]} for p in perfis])),
          f"![risco]({g.name})", "",
          "## Alertas despachados pelo gerenciador", "",
          f"- Total emitidos: **{res['total']}** · suprimidos por deduplicação/persistência: {res['suprimidos_por_deduplicacao']}",
          f"- Por severidade: {res['por_severidade']}", f"- Por fonte: {res['por_fonte']}", f"- Por paciente: {res['por_paciente']}",
          f"- Log completo: `outputs/alertas.jsonl`", "",
          "## Avaliação dos detectores (anomalias injetadas nos dados sintéticos)", ""]
    linhas = []
    for pid, av in avaliacoes.items():
        for fonte, a in av.items():
            linhas.append({"paciente": pid, "fonte": fonte, **a["metricas"]})
    md += [_tabela(pd.DataFrame(linhas)), ""]
    epis = [{"paciente": pid, **e} for pid, av in avaliacoes.items() for e in av.get("sinais_vitais", {}).get("episodios", [])]
    if epis:
        md += ["### Episódios rotulados × detecção (tempo de antecipação)", "", _tabela(pd.DataFrame(epis)), ""]
    md += ["## Relatórios por paciente", ""] + [f"- `outputs/pacientes/{p.paciente_id}/relatorio.md`" for p in perfis] + [""]
    destino = pasta / "relatorio_geral.md"
    destino.write_text("\n".join(md), encoding="utf-8")
    return destino
