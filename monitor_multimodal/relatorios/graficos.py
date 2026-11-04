"""graficos.py — Gráficos (matplotlib, backend Agg) para os relatórios automáticos."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import config  # noqa: E402

CORES = {"normal": "#4C78A8", "anomalia": "#E45756", "injetada": "#F58518", "faixa": "#54A24B"}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def grafico_sinais_vitais(df: pd.DataFrame, destino: Path, titulo: str = "") -> Path:
    cols = [c for c in config.SINAIS_VITAIS if c in df.columns]
    fig, axes = plt.subplots(len(cols) + 1, 1, figsize=(12, 2.0 * (len(cols) + 1)), sharex=True)
    t = pd.to_datetime(df["timestamp"])
    for ax, c in zip(axes, cols):
        lo, hi = config.SINAIS_VITAIS[c]["normal"]
        ax.axhspan(lo, hi, color=CORES["faixa"], alpha=0.10, lw=0)
        ax.plot(t, df[c], color=CORES["normal"], lw=1)
        if "anomalia_injetada" in df:
            m = df["anomalia_injetada"].astype(bool)
            ax.fill_between(t, df[c].min(), df[c].max(), where=m, color=CORES["injetada"], alpha=0.12, lw=0,
                            label="anomalia injetada (rótulo)")
        m = df["anomalia"].astype(bool)
        ax.scatter(t[m], df.loc[m, c], color=CORES["anomalia"], s=10, zorder=3, label="detectada")
        ax.set_ylabel(f"{c.replace('_', ' ')}\n({config.SINAIS_VITAIS[c]['unidade']})", fontsize=8)
    ax = axes[-1]
    ax.step(t, df["ews"], color="#B279A2", where="post", lw=1.2)
    ax.axhline(5, color="orange", ls="--", lw=0.8)
    ax.axhline(7, color="red", ls="--", lw=0.8)
    ax.set_ylabel("Escore alerta\nprecoce", fontsize=8)
    axes[0].legend(loc="upper left", fontsize=7, ncol=2)
    fig.suptitle(titulo or "Sinais vitais e anomalias detectadas", fontsize=11)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)
    return destino


def grafico_movimentacao(df: pd.DataFrame, destino: Path, titulo: str = "") -> Path:
    fig, ax1 = plt.subplots(figsize=(12, 3.2))
    t = pd.to_datetime(df["timestamp"])
    noite = df["noturno"].astype(bool) if "noturno" in df else ((t.dt.hour >= 22) | (t.dt.hour < 6))
    ax1.fill_between(t, 0, 1, where=noite, color="#333", alpha=0.07, lw=0, step="post", label="período noturno")
    ax1.plot(t, df["indice_atividade"], color=CORES["normal"], lw=1.2, label="índice de atividade")
    m = df["anomalia"].astype(bool)
    ax1.scatter(t[m], df.loc[m, "indice_atividade"], color=CORES["anomalia"], s=18, zorder=3, label="anomalia")
    if "anomalia_injetada" in df:
        mi = df["anomalia_injetada"].astype(bool)
        ax1.scatter(t[mi], df.loc[mi, "indice_atividade"] + 0.04, marker="v", color=CORES["injetada"], s=18, label="rótulo")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("índice de atividade")
    ax2 = ax1.twinx()
    ax2.bar(t, df["mudancas_decubito"], width=0.03, color="#72B7B2", alpha=0.5, label="mudanças de decúbito/h")
    ax2.set_ylabel("mudanças/h")
    ax2.grid(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7, ncol=3)
    ax1.set_title(titulo or "Padrão de movimentação do paciente")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)
    return destino


def grafico_prescricoes(df: pd.DataFrame, destino: Path, titulo: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(12, 3.4))
    meds = sorted(df["medicamento"].unique())
    ypos = {m: i for i, m in enumerate(meds)}
    t = pd.to_datetime(df["timestamp"])
    rel = df["dose_relativa"].fillna(0.3).clip(0.05, 3)
    ax.scatter(t, df["medicamento"].map(ypos), s=40 + 120 * rel, c=np.where(df["anomalia"], CORES["anomalia"], CORES["normal"]),
               alpha=0.8, edgecolor="k", lw=0.3)
    for _, r in df[df["anomalia"]].iterrows():
        ax.annotate(", ".join(r["motivos"])[:40], (pd.Timestamp(r["timestamp"]), ypos[r["medicamento"]]),
                    xytext=(5, 8), textcoords="offset points", fontsize=6.5, color=CORES["anomalia"])
    ax.set_yticks(range(len(meds)))
    ax.set_yticklabels(meds)
    ax.set_title(titulo or "Evolução das prescrições (tamanho ∝ dose relativa ao máximo; vermelho = anomalia)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)
    return destino


def grafico_audio(features_por_arquivo: dict[str, dict], indicadores_por_arquivo: dict[str, dict], destino: Path) -> Path:
    nomes = list(features_por_arquivo)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    ind = ["fadiga", "dispneia", "disartria"]
    x = np.arange(len(nomes))
    w = 0.26
    for k, nome_ind in enumerate(ind):
        axes[0].bar(x + (k - 1) * w, [indicadores_por_arquivo[n][nome_ind] for n in nomes], w, label=nome_ind)
    axes[0].axhline(0.6, color="red", ls="--", lw=0.8, label="limiar de alerta")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.replace("consulta_", "") for n in nomes], rotation=0)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Indicadores clínicos derivados da voz")
    axes[0].legend(fontsize=7)
    feats = ["taxa_silabica", "razao_pausa", "pausas_por_min", "f0_desvio_semitons", "hnr_db", "duracao_media_frase_s"]
    M = np.array([[features_por_arquivo[n].get(f, 0) for f in feats] for n in nomes], dtype=float)
    M_norm = M / (np.abs(M).max(axis=0) + 1e-9)
    im = axes[1].imshow(M_norm, cmap="viridis", aspect="auto")
    axes[1].set_xticks(range(len(feats)))
    axes[1].set_xticklabels(feats, rotation=30, ha="right", fontsize=7)
    axes[1].set_yticks(range(len(nomes)))
    axes[1].set_yticklabels([n.replace("consulta_", "") for n in nomes])
    for i in range(len(nomes)):
        for j in range(len(feats)):
            axes[1].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="w")
    axes[1].set_title("Biomarcadores vocais (normalizados por coluna)")
    axes[1].grid(False)
    fig.colorbar(im, ax=axes[1], fraction=0.03)
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)
    return destino


def grafico_video(serie: pd.DataFrame, eventos: list[dict], destino: Path, titulo: str = "") -> Path:
    tem_pose = "quadril_y_norm" in serie and serie["quadril_y_norm"].notna().any()
    n = 3 if tem_pose else 1
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.4 * n), sharex=True)
    axes = np.atleast_1d(axes)
    t = serie["tempo_s"]
    axes[0].plot(t, serie["energia_movimento"], color=CORES["normal"], lw=1, label="energia de movimento")
    axes[0].plot(t, serie["fluxo_medio"] * 10, color="#72B7B2", lw=0.8, alpha=0.8, label="fluxo óptico ×10")
    axes[0].set_ylabel("movimento")
    cores_sev = {"CRÍTICA": "red", "ALTA": "orange", "ATENÇÃO": "gold", "INFO": "gray"}
    for ev in eventos:
        for ax in axes:
            ax.axvline(ev["tempo_s"], color=cores_sev.get(ev["severidade"], "gray"), lw=1, alpha=0.7, ls="--")
        axes[0].annotate(ev["tipo"], (ev["tempo_s"], axes[0].get_ylim()[1] * 0.95), rotation=90, fontsize=6,
                         va="top", ha="right", color=cores_sev.get(ev["severidade"], "gray"))
    axes[0].legend(fontsize=7, loc="upper left")
    if tem_pose:
        axes[1].plot(t, serie["quadril_y_norm"], color="#B279A2", lw=1, label="altura do quadril (0=topo)")
        if "angulo_tronco_graus" in serie:
            axes[1].plot(t, serie["angulo_tronco_graus"] / 90, color="#F58518", lw=1, label="ângulo do tronco / 90°")
        axes[1].set_ylabel("pose")
        axes[1].legend(fontsize=7, loc="upper left")
        for col, rot in (("angulo_joelho_e", "joelho esq (°)"), ("angulo_joelho_d", "joelho dir (°)")):
            if col in serie:
                axes[2].plot(t, serie[col], lw=1, label=rot)
        if "n_pessoas" in serie:
            axes[2].plot(t, serie["n_pessoas"].fillna(0) * 30, lw=0.8, color="k", alpha=0.5, label="nº pessoas ×30")
        axes[2].set_ylabel("articulações")
        axes[2].legend(fontsize=7, loc="upper left")
    axes[-1].set_xlabel("tempo (s)")
    fig.suptitle(titulo or "Análise de vídeo", fontsize=11)
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)
    return destino


def grafico_risco(perfis: list, destino: Path) -> Path:
    fontes = sorted({f for p in perfis for f in p.scores_por_fonte})
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = np.arange(len(perfis))
    w = 0.8 / (len(fontes) + 1)
    for k, f in enumerate(fontes):
        ax.bar(x + k * w, [p.scores_por_fonte.get(f, 0) for p in perfis], w, label=f)
    ax.bar(x + len(fontes) * w, [p.score_global for p in perfis], w, color="k", label="score global")
    for xi, p in zip(x, perfis):
        ax.text(xi + 0.4, p.score_global + 0.02, p.nivel, ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x + 0.4)
    ax.set_xticklabels([p.paciente_id for p in perfis])
    ax.set_ylim(0, 1.1)
    ax.set_title("Score de risco por modalidade e fusão multimodal")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)
    return destino
