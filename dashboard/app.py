"""
app.py — Painel da equipe médica (Streamlit).

Lê os artefatos gerados pelo pipeline (outputs/) e permite:
  • visão geral do risco por paciente e feed de alertas
  • detalhe por paciente: sinais vitais, prescrições, movimentação, áudio (transcrição,
    entidades, indicadores vocais), vídeo (eventos, frames-chave, vídeo anotado)
  • simulação em tempo real do monitor de sinais vitais com alertas ao vivo
  • análise de um arquivo de áudio/vídeo enviado pelo usuário

Executar:  python main.py dashboard   (ou  streamlit run dashboard/app.py)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from monitor_multimodal import config  # noqa: E402
from monitor_multimodal.alertas import GerenciadorAlertas  # noqa: E402

st.set_page_config(page_title="Monitoramento Multimodal — Equipe Médica", page_icon="🏥", layout="wide")
OUT = config.OUTPUTS_DIR
CORES_SEV = {"CRÍTICA": "#d62728", "ALTA": "#ff7f0e", "ATENÇÃO": "#ffbf00", "INFO": "#1f77b4"}
CORES_NIVEL = {"baixo": "🟢", "moderado": "🟡", "alto": "🟠", "crítico": "🔴"}


# ─── Carregamento ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def carregar_resultado(mtime: float):
    p = OUT / "resultado_demo.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def carregar_alertas() -> pd.DataFrame:
    p = OUT / "alertas.jsonl"
    if not p.exists():
        return pd.DataFrame()
    df = pd.DataFrame([json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()])
    if len(df):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
        df = df.sort_values("timestamp", ascending=False)
    return df


def pasta_paciente(pid: str) -> Path:
    return OUT / "pacientes" / pid


def _mtime() -> float:
    p = OUT / "resultado_demo.json"
    return p.stat().st_mtime if p.exists() else 0.0


# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("🏥 Monitoramento Multimodal")
st.sidebar.caption("Tech Challenge Fase 4 — PosTech FIAP")
resultado = carregar_resultado(_mtime())
if resultado is None:
    st.sidebar.warning("Nenhuma execução encontrada em `outputs/`.")
    if st.sidebar.button("▶️ Executar demonstração completa", type="primary"):
        from monitor_multimodal.pipeline import executar_demo
        with st.spinner("Executando pipeline multimodal (vídeo + áudio + anomalias)… pode levar alguns minutos"):
            executar_demo(gerar_dados=False)
        st.cache_data.clear()
        st.rerun()
    st.stop()

pacientes = [p["paciente_id"] for p in resultado["perfis"]]
info = resultado["info"]
st.sidebar.markdown(f"**Última execução:** {info['duracao_execucao_s']}s  \n"
                    f"**Azure Speech:** {info['azure_speech']}  \n**Azure Language:** {info['azure_language']}")
if st.sidebar.button("🔄 Reexecutar demonstração"):
    from monitor_multimodal.pipeline import executar_demo
    with st.spinner("Reexecutando pipeline…"):
        executar_demo(gerar_dados=False)
    st.cache_data.clear()
    st.rerun()

aba_geral, aba_pac, aba_rt, aba_arq = st.tabs(["📊 Visão geral", "🧑‍⚕️ Paciente", "⏱️ Tempo real", "📁 Analisar arquivo"])

# ─── Visão geral ─────────────────────────────────────────────────────────────
with aba_geral:
    st.subheader("Risco por paciente (fusão multimodal)")
    cols = st.columns(len(pacientes))
    for col, p in zip(cols, resultado["perfis"]):
        with col:
            st.metric(f"{CORES_NIVEL.get(p['nivel'], '')} {p['paciente_id']}", f"{p['score_global']:.2f}", p["nivel"].upper())
            st.caption(" · ".join(f"{k}: {v:.2f}" for k, v in p["scores_por_fonte"].items()))
            if p["corroboracoes"]:
                st.caption(f"🔗 {len(p['corroboracoes'])} corroboração(ões) multimodal(is)")
    df_scores = pd.DataFrame([{"paciente": p["paciente_id"], "fonte": f, "score": s}
                              for p in resultado["perfis"] for f, s in p["scores_por_fonte"].items()])
    st.plotly_chart(px.bar(df_scores, x="paciente", y="score", color="fonte", barmode="group", range_y=[0, 1],
                           title="Score de risco por modalidade"), use_container_width=True)

    st.subheader("Feed de alertas")
    alertas = carregar_alertas()
    if len(alertas):
        c1, c2, c3 = st.columns(3)
        sev = c1.multiselect("Severidade", list(CORES_SEV), default=["CRÍTICA", "ALTA"])
        fontes = c2.multiselect("Fonte", sorted(alertas["fonte"].unique()), default=sorted(alertas["fonte"].unique()))
        pacs = c3.multiselect("Paciente", pacientes, default=pacientes)
        f = alertas[alertas["severidade"].isin(sev) & alertas["fonte"].isin(fontes) & alertas["paciente_id"].isin(pacs)]
        st.caption(f"{len(f)} alertas (de {len(alertas)} emitidos)")
        fig = px.scatter(f, x="timestamp", y="paciente_id", color="severidade", symbol="fonte", hover_data=["mensagem"],
                         color_discrete_map=CORES_SEV, title="Linha do tempo dos alertas")
        fig.update_traces(marker_size=10)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(f[["timestamp", "paciente_id", "severidade", "fonte", "tipo", "mensagem", "recomendacao"]],
                     use_container_width=True, hide_index=True, height=380)
    if (OUT / "relatorio_geral.md").exists():
        with st.expander("📄 Relatório geral da execução"):
            st.markdown((OUT / "relatorio_geral.md").read_text(encoding="utf-8").replace("](risco_pacientes.png)", "](app/static/x)"))
            st.image(str(OUT / "risco_pacientes.png"))

# ─── Paciente ────────────────────────────────────────────────────────────────
with aba_pac:
    pid = st.selectbox("Paciente", pacientes)
    perfil = next(p for p in resultado["perfis"] if p["paciente_id"] == pid)
    pasta = pasta_paciente(pid)
    st.markdown(f"## {CORES_NIVEL.get(perfil['nivel'], '')} {pid} — risco **{perfil['nivel'].upper()}** ({perfil['score_global']:.2f})")
    if perfil["corroboracoes"]:
        for c in perfil["corroboracoes"][:3]:
            st.error(f"**Corroboração multimodal** em {c['timestamp'][:16].replace('T', ' ')} — " + " | ".join(c["alertas"]))

    t_sv, t_pr, t_mv, t_au, t_vi, t_rel = st.tabs(["Sinais vitais", "Prescrições", "Movimentação", "Áudio", "Vídeo", "Relatório"])
    with t_sv:
        p = pasta / "sinais_vitais.csv"
        if p.exists():
            sv = pd.read_csv(p, parse_dates=["timestamp"])
            cols_sv = [c for c in config.SINAIS_VITAIS if c in sv]
            escolha = st.multiselect("Variáveis", cols_sv, default=cols_sv[:4])
            fig = go.Figure()
            for c in escolha:
                fig.add_trace(go.Scatter(x=sv["timestamp"], y=sv[c], mode="lines", name=c))
                an = sv[sv["anomalia"]]
                fig.add_trace(go.Scatter(x=an["timestamp"], y=an[c], mode="markers", name=f"{c} (anomalia)",
                                         marker=dict(color="red", size=5), showlegend=False))
            fig.update_layout(height=420, title="Sinais vitais (vermelho = anomalia detectada)")
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.line(sv, x="timestamp", y="ews", title="Escore de alerta precoce (NEWS2 simplificado)")
            fig2.add_hline(y=5, line_dash="dash", line_color="orange")
            fig2.add_hline(y=7, line_dash="dash", line_color="red")
            st.plotly_chart(fig2, use_container_width=True)
            av = resultado["avaliacoes"].get(pid, {}).get("sinais_vitais")
            if av:
                st.markdown("**Avaliação contra anomalias injetadas**")
                st.dataframe(pd.DataFrame([av["metricas"]]), hide_index=True)
                if av.get("episodios"):
                    st.dataframe(pd.DataFrame(av["episodios"]), hide_index=True)
    with t_pr:
        p = pasta / "prescricoes.csv"
        if p.exists():
            pr = pd.read_csv(p, parse_dates=["timestamp"])
            st.dataframe(pr[["timestamp", "medicamento", "dose_mg", "frequencia_dia", "via", "prescritor", "dose_diaria_mg",
                             "dose_max_mg", "anomalia", "motivos"]].style.apply(
                lambda r: ["background-color: #ffd6d6" if r["anomalia"] else "" for _ in r], axis=1),
                use_container_width=True, hide_index=True, height=420)
            if (pasta / "prescricoes.png").exists():
                st.image(str(pasta / "prescricoes.png"))
    with t_mv:
        p = pasta / "movimentacao.csv"
        if p.exists():
            mv = pd.read_csv(p, parse_dates=["timestamp"])
            fig = px.line(mv, x="timestamp", y="indice_atividade", title="Índice de atividade (horário)")
            an = mv[mv["anomalia"]]
            fig.add_trace(go.Scatter(x=an["timestamp"], y=an["indice_atividade"], mode="markers+text", text=an["motivo"],
                                     textposition="top center", marker=dict(color="red", size=9), name="anomalia"))
            st.plotly_chart(fig, use_container_width=True)
            st.bar_chart(mv.set_index("timestamp")["mudancas_decubito"], height=180)
    with t_au:
        p = pasta / "audio.json"
        if p.exists():
            for a in json.loads(p.read_text(encoding="utf-8")):
                st.markdown(f"#### 🎙️ `{Path(a['arquivo']).name}`")
                if Path(a["arquivo"]).exists():
                    st.audio(a["arquivo"])
                c1, c2, c3 = st.columns(3)
                ind = a["indicadores"]
                c1.metric("Fadiga", f"{ind['fadiga']:.2f}")
                c2.metric("Dispneia", f"{ind['dispneia']:.2f}")
                c3.metric("Disartria", f"{ind['disartria']:.2f}")
                tr, tx = a["transcricao"], a["analise_texto"]
                tem_transcricao = bool(tr.get("texto", "").strip()) and tr.get("provedor") != "nenhum" and not tr.get("erro")
                if tem_transcricao:
                    st.markdown(f"**Transcrição** _(provedor: {tr['provedor']})_: {tr['texto']}")
                else:
                    st.markdown("**Transcrição**: _indisponível_")
                if tem_transcricao and tx.get("scores_sentimento") and tx.get("provedor") != "nenhum" and not tx.get("erro"):
                    st.markdown(f"**Sentimento** _(provedor: {tx['provedor']})_: **{tx['sentimento']}** {tx['scores_sentimento']}")
                else:
                    st.markdown("**Sentimento**: _indisponível_")
                    st.warning("O sentimento depende de uma transcrição válida e de uma análise de texto concluída.")
                st.caption("O sentimento é calculado a partir do texto transcrito; os indicadores vocais vêm do áudio.")
                if tx["termos_criticos"]:
                    st.error("Termos críticos: " + ", ".join(tx["termos_criticos"]))
                if tx["entidades"]:
                    st.dataframe(pd.DataFrame(tx["entidades"]), hide_index=True, use_container_width=True)
                with st.expander("Biomarcadores vocais"):
                    st.json({k: v for k, v in a["features"].items() if not k.startswith("mfcc")})
                    st.json(ind.get("componentes", {}))
            if (pasta / "audio.png").exists():
                st.image(str(pasta / "audio.png"))
        else:
            st.info("Sem áudio para este paciente.")
    with t_vi:
        p = pasta / "video.json"
        if p.exists():
            for v in json.loads(p.read_text(encoding="utf-8")):
                st.markdown(f"#### 🎥 `{Path(v['arquivo']).name}` — {v['resumo']['n_eventos']} eventos · pessoas máx. {v['resumo']['pessoas_max']}")
                if v.get("video_anotado") and Path(v["video_anotado"]).exists():
                    st.video(v["video_anotado"])
                if v["eventos"]:
                    st.dataframe(pd.DataFrame(v["eventos"])[["tempo_s", "severidade", "tipo", "descricao", "score"]],
                                 hide_index=True, use_container_width=True)
                frames = [e for e in v["eventos"] if e.get("frame_chave") and Path(e["frame_chave"]).exists()]
                if frames:
                    cols = st.columns(min(3, len(frames)))
                    for i, e in enumerate(frames[:6]):
                        cols[i % 3].image(e["frame_chave"], caption=f"{e['tempo_s']:.1f}s — {e['tipo']} ({e['severidade']})")
                g = OUT / "video" / Path(v["arquivo"]).stem / f"{Path(v['arquivo']).stem}_grafico.png"
                if g.exists():
                    st.image(str(g))
        else:
            st.info("Sem vídeo para este paciente.")
    with t_rel:
        p = pasta / "relatorio.md"
        if p.exists():
            texto = p.read_text(encoding="utf-8")
            st.markdown("\n".join(l for l in texto.splitlines() if not l.startswith("![")))
            for img in sorted(pasta.glob("*.png")):
                st.image(str(img))
        if (pasta / "alertas.csv").exists():
            st.download_button("⬇️ Baixar alertas (CSV)", (pasta / "alertas.csv").read_bytes(), f"alertas_{pid}.csv")

# ─── Tempo real ──────────────────────────────────────────────────────────────
with aba_rt:
    st.subheader("Simulação de monitoramento em tempo real")
    st.caption("Reproduz os sinais vitais do paciente amostra a amostra; o monitor aplica regras clínicas, "
               "z-score deslizante e Isolation Forest online, emitindo alertas com latência de uma amostra.")
    c1, c2, c3 = st.columns(3)
    pid_rt = c1.selectbox("Paciente", pacientes, index=min(1, len(pacientes) - 1), key="rt_pid")
    velocidade = c2.slider("Amostras por segundo", 2, 60, 20)
    max_amostras = c3.slider("Amostras a reproduzir", 50, 600, 300, step=50)
    if st.button("▶️ Iniciar", type="primary"):
        from monitor_multimodal.anomalias.stream import MonitorTempoReal, simular_stream
        sv_all = pd.read_csv(config.DADOS_SINTETICOS_DIR / "sinais_vitais.csv", parse_dates=["timestamp"])
        df = sv_all[sv_all.paciente_id == pid_rt].head(max_amostras)
        g = GerenciadorAlertas(caminho_log=OUT / "alertas_dashboard_rt.jsonl", console=False)
        monitor = MonitorTempoReal(g)
        ph_metricas, ph_graf, ph_feed = st.empty(), st.empty(), st.container()
        feed = []
        hist = []
        for i, (amostra, alertas_novos) in enumerate(simular_stream(df, pid_rt, monitor)):
            hist.append(amostra)
            for a in alertas_novos:
                feed.insert(0, a)
            if i % max(1, velocidade // 10) == 0 or alertas_novos:
                h = pd.DataFrame(hist)
                with ph_metricas.container():
                    cols = st.columns(6)
                    for col, (c, rot) in zip(cols, [("frequencia_cardiaca", "FC"), ("pressao_sistolica", "PAS"), ("pressao_diastolica", "PAD"),
                                                    ("saturacao_o2", "SpO₂"), ("frequencia_respiratoria", "FR"), ("temperatura", "T")]):
                        val = amostra[c]
                        col.metric(rot, f"{val:.1f}" if c == "temperatura" else f"{val:.0f}",
                                   delta=None if len(h) < 2 else f"{val - h[c].iloc[-2]:+.1f}")
                fig = go.Figure()
                for c in ("frequencia_cardiaca", "pressao_sistolica", "saturacao_o2", "frequencia_respiratoria"):
                    fig.add_trace(go.Scatter(x=h["timestamp"], y=h[c], mode="lines", name=c))
                for a in feed[:30]:
                    fig.add_vline(x=pd.Timestamp(a.timestamp), line_color=CORES_SEV.get(a.severidade.rotulo, "gray"), opacity=0.4)
                fig.update_layout(height=360, margin=dict(t=20, b=20), title=f"{pid_rt} — {amostra['timestamp']}")
                ph_graf.plotly_chart(fig, use_container_width=True)
                if alertas_novos:
                    for a in alertas_novos:
                        {4: ph_feed.error, 3: ph_feed.warning, 2: ph_feed.info, 1: ph_feed.info}[int(a.severidade)](
                            f"**{a.severidade.rotulo}** · {str(a.timestamp)[:16].replace('T', ' ')} · {a.mensagem}  \n_{a.recomendacao}_")
            time.sleep(1 / velocidade)
        st.success(f"Fim da simulação: {len(g.alertas)} alertas emitidos, {g.suprimidos} agrupados.")

# ─── Analisar arquivo ────────────────────────────────────────────────────────
with aba_arq:
    st.subheader("Analisar um arquivo de áudio ou vídeo")
    pid_arq = st.text_input("Identificador do paciente", "P999")
    up = st.file_uploader("Áudio (wav/mp3/m4a) ou vídeo (mp4/avi/mov)", type=["wav", "mp3", "m4a", "mp4", "avi", "mov"])
    if up is not None:
        destino = OUT / "uploads" / up.name
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(up.getbuffer())
        if destino.suffix.lower() in (".wav", ".mp3", ".m4a"):
            from monitor_multimodal.audio import analisar_audio
            with st.spinner("Analisando áudio (biomarcadores + transcrição + análise de texto)…"):
                r = analisar_audio(destino, pid_arq)
            st.audio(str(destino))
            c1, c2, c3 = st.columns(3)
            c1.metric("Fadiga", f"{r.indicadores['fadiga']:.2f}")
            c2.metric("Dispneia", f"{r.indicadores['dispneia']:.2f}")
            c3.metric("Disartria", f"{r.indicadores['disartria']:.2f}")
            tem_transcricao = bool(r.transcricao.texto.strip()) and r.transcricao.provedor != "nenhum" and not r.transcricao.erro
            if tem_transcricao:
                st.markdown(f"**Transcrição** ({r.transcricao.provedor}): {r.transcricao.texto}")
            else:
                st.markdown("**Transcrição**: _indisponível_")
            if tem_transcricao and r.texto.scores_sentimento and r.texto.provedor != "nenhum" and not r.texto.erro:
                st.markdown(f"**Sentimento** ({r.texto.provedor}): {r.texto.sentimento} {r.texto.scores_sentimento}")
            else:
                st.markdown("**Sentimento**: _indisponível_")
                st.warning("O sentimento depende de uma transcrição válida e de uma análise de texto concluída.")
            st.caption("O sentimento é calculado a partir do texto transcrito; os indicadores vocais vêm do áudio.")
            if r.texto.entidades:
                st.dataframe(pd.DataFrame(r.texto.entidades), hide_index=True)
            for a in r.alertas:
                {4: st.error, 3: st.warning, 2: st.info, 1: st.info}[int(a.severidade)](a.resumo() + f"  \n_{a.recomendacao}_")
            with st.expander("Biomarcadores vocais"):
                st.json({k: v for k, v in r.features.items() if not k.startswith("mfcc")})
        else:
            from monitor_multimodal.relatorios import gerar_relatorio_video
            from monitor_multimodal.video import analisar_video
            sessao = st.selectbox("Tipo de sessão", ["fisioterapia", "cirurgia", "leito", "enfermaria"])
            maxf = st.slider("Máx. de frames processados (0 = todos)", 0, 2000, 300, step=50)
            if st.button("Analisar vídeo", type="primary"):
                with st.spinner("Processando vídeo (YOLOv8 + pose + movimento)…"):
                    r = analisar_video(destino, pid_arq, tipo_sessao=sessao, max_frames=maxf or None)
                    rel = gerar_relatorio_video(r)
                st.json(r.resumo)
                if r.video_anotado:
                    st.video(r.video_anotado)
                if r.eventos:
                    st.dataframe(pd.DataFrame(r.eventos)[["tempo_s", "severidade", "tipo", "descricao", "score"]], hide_index=True)
                for a in r.alertas[:10]:
                    {4: st.error, 3: st.warning, 2: st.info, 1: st.info}[int(a.severidade)](a.resumo())
                st.markdown(rel.read_text(encoding="utf-8").split("## Frames-chave")[0])
