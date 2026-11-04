"""
analise.py — Pipeline de análise de vídeo clínico (fisioterapia, cirurgia, leito).

  frames ──► energia de movimento / fluxo óptico (OpenCV)            ─┐
         ──► YOLOv8 detecção (pessoas, objetos críticos)              ├─► séries temporais
         ──► YOLOv8-pose (quadril, tronco, joelhos, assimetria)      ─┘        │
                                                                                ▼
                               detectores de anomalia (regras + z-score + Isolation Forest)
                                                                                ▼
                       eventos com timestamp ─► alertas ─► frames-chave + vídeo anotado + relatório

Funciona em modo degradado (só OpenCV) se o ultralytics/pesos não estiverem disponíveis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .. import config
from ..alertas import Alerta, Severidade
from ..anomalias.detectores import DetectorIsolationForest, DetectorZScoreDeslizante
from ..utils import agora_iso, log
from .movimento import calcular_movimento, info_video, iterar_frames, reencodar_h264


@dataclass
class ResultadoVideo:
    paciente_id: str
    arquivo: str
    info: dict[str, Any]
    serie: pd.DataFrame
    eventos: list[dict[str, Any]] = field(default_factory=list)
    alertas: list[Alerta] = field(default_factory=list)
    resumo: dict[str, Any] = field(default_factory=dict)
    frames_chave: list[str] = field(default_factory=list)
    video_anotado: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"paciente_id": self.paciente_id, "arquivo": self.arquivo, "info": self.info,
                "eventos": self.eventos, "alertas": [a.to_dict() for a in self.alertas], "resumo": self.resumo,
                "frames_chave": self.frames_chave, "video_anotado": self.video_anotado}


def _rodar_modelos(caminho: Path, passo: int, usar_yolo: bool, pasta_saida: Path | None,
                   max_frames: int | None) -> tuple[pd.DataFrame, str | None, dict]:
    """Roda detecção + pose frame a frame; opcionalmente grava vídeo anotado."""
    linhas = []
    writer = None
    video_anotado = None
    detector = estimador = None
    pose_anterior = None
    status = {"yolo": False, "erro": None}
    if usar_yolo:
        try:
            from .deteccao import DetectorObjetos
            from .pose import EstimadorPose, desenhar_pose
            detector, estimador = DetectorObjetos(), EstimadorPose()
            status["yolo"] = True
        except Exception as exc:  # sem ultralytics / sem rede para os pesos
            log.warning(f"YOLOv8 indisponível ({exc}); seguindo apenas com energia de movimento")
    for idx, t, frame in iterar_frames(caminho, passo, max_frames):
        linha: dict[str, Any] = {"frame": idx}
        if detector is not None:
            dets = detector.detectar(frame)
            r = detector.resumir(dets)
            linha.update(n_pessoas=r["n_pessoas"], objetos_criticos=";".join(r["objetos_criticos"]),
                         classes=";".join(r["classes"]))
            poses = estimador.estimar(frame)
            diag = float(np.hypot(frame.shape[0], frame.shape[1]))
            p, trocou = estimador.principal(poses, pose_anterior, diag)
            pose_anterior = p
            if p is not None:
                linha.update({k: v for k, v in p.metricas.items()})
                linha["troca_identidade"] = trocou
            if pasta_saida is not None:
                if writer is None:
                    video_anotado = str(pasta_saida / f"{caminho.stem}_anotado.mp4")
                    writer = cv2.VideoWriter(video_anotado, cv2.VideoWriter_fourcc(*"mp4v"),
                                             max(1.0, info_video(caminho)["fps"] / passo), (frame.shape[1], frame.shape[0]))
                anot = frame.copy()
                for d in dets:
                    x1, y1, x2, y2 = map(int, d.bbox)
                    cor = (0, 0, 255) if d.classe in config.CLASSES_CRITICAS_VIDEO else (255, 160, 0)
                    cv2.rectangle(anot, (x1, y1), (x2, y2), cor, 2)
                    cv2.putText(anot, f"{d.classe} {d.confianca:.2f}", (x1, max(12, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)
                for pz in poses:
                    desenhar_pose(anot, pz)
                cv2.putText(anot, f"t={t:6.2f}s pessoas={r['n_pessoas']}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(anot, f"t={t:6.2f}s pessoas={r['n_pessoas']}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                writer.write(anot)
        linhas.append(linha)
    if writer is not None:
        writer.release()
        reencodar_h264(video_anotado)
    return pd.DataFrame(linhas), video_anotado, status


def _salvar_frame(caminho: Path, idx: int, pasta: Path, nome: str, texto: str) -> str | None:
    cap = cv2.VideoCapture(str(caminho))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(frame, texto[:80], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
    destino = pasta / f"{nome}_f{idx:05d}.jpg"
    cv2.imwrite(str(destino), frame)
    return str(destino)


def analisar_video(caminho: Path | str, paciente_id: str, tipo_sessao: str = "fisioterapia",
                   usar_yolo: bool = True, gravar_anotado: bool = True, passo: int = config.VIDEO_PASSO_FRAMES,
                   max_frames: int | None = None, pasta_saida: Path | None = None,
                   timestamp: str | None = None) -> ResultadoVideo:
    caminho = Path(caminho)
    ts_base = pd.Timestamp(timestamp or agora_iso())
    pasta = Path(pasta_saida) if pasta_saida else config.OUTPUTS_DIR / "video" / caminho.stem
    pasta.mkdir(parents=True, exist_ok=True)
    info = info_video(caminho)
    log.info(f"[vídeo] {caminho.name}: {info['n_frames']} frames @ {info['fps']:.1f} fps "
             f"({info['duracao_s']:.1f}s), passo={passo}")

    mov = calcular_movimento(caminho, passo, max_frames=max_frames)
    modelos, video_anotado, status = _rodar_modelos(caminho, passo, usar_yolo, pasta if gravar_anotado else None, max_frames)
    serie = mov.merge(modelos, on="frame", how="left") if len(modelos) else mov
    dt = passo / info["fps"]                      # segundos entre amostras

    eventos: list[dict[str, Any]] = []

    def evento(i: int, tipo: str, severidade: Severidade, descricao: str, score: float, **ev):
        eventos.append({"tempo_s": float(serie.at[i, "tempo_s"]), "frame": int(serie.at[i, "frame"]), "tipo": tipo,
                        "severidade": severidade.rotulo, "severidade_nivel": int(severidade),
                        "descricao": descricao, "score": round(float(score), 3), "evidencias": ev})

    # ── 1. Energia de movimento: pico seguido de imobilidade (queda/colapso) ───
    e = serie["energia_movimento"].to_numpy()
    base = np.median(e[e > 0]) if (e > 0).any() else 1.0
    janela_quieto = max(2, int(2.0 / dt))           # 2 s
    i = 1
    while i < len(e) - janela_quieto:
        if e[i] > 3.5 * base and e[i] > 2.0:
            fim_pico = i
            while fim_pico + 1 < len(e) and e[fim_pico + 1] > 1.5 * base:
                fim_pico += 1
            depois = e[fim_pico + 1: fim_pico + 1 + janela_quieto]
            if len(depois) == janela_quieto and np.all(depois < 0.35 * base):
                evento(i, "pico_seguido_de_imobilidade", Severidade.CRITICA,
                       f"Movimento brusco (energia {e[i]:.1f}, {e[i]/base:.1f}× a mediana) seguido de ≥2 s de imobilidade "
                       f"— padrão compatível com queda/colapso", min(1.0, e[i] / (6 * base)),
                       energia_pico=float(e[i]), mediana=float(base))
                i = fim_pico + janela_quieto
                continue
        i += 1

    # ── 2. Imobilidade prolongada em sessão ativa ──────────────────────────────
    if tipo_sessao in ("fisioterapia", "cirurgia"):
        limite = int(4.0 / dt)
        quieto = (e < 0.25 * base).astype(int)
        grupos = (quieto != pd.Series(quieto).shift()).cumsum()
        run = pd.Series(quieto).groupby(grupos).cumsum()
        for i in np.flatnonzero((run.to_numpy() == limite) & (quieto == 1)):
            evento(i, "imobilidade_prolongada", Severidade.ALTA,
                   f"≥4 s sem movimento durante sessão de {tipo_sessao}", 0.7)

    # ── 3. Quebra de ritmo (sessões de fisioterapia são periódicas) ────────────
    if tipo_sessao == "fisioterapia" and len(e) > 60:
        jan = int(8.0 / dt)
        periodos = []
        for k in range(jan, len(e), jan // 2):
            x = e[k - jan: k] - e[k - jan: k].mean()
            if np.std(x) < 1e-6:
                periodos.append(np.nan)
                continue
            ac = np.correlate(x, x, "full")[jan - 1:]
            ac /= ac[0] + 1e-9
            picos = [j for j in range(2, len(ac) - 1) if ac[j] > ac[j - 1] and ac[j] > ac[j + 1] and ac[j] > 0.3]
            periodos.append(picos[0] * dt if picos else np.nan)
        periodos = np.array(periodos)
        validos = periodos[~np.isnan(periodos)]
        if len(validos) >= 2:
            ritmo = float(np.median(validos))
            for k, p in enumerate(periodos):
                idx = min(len(e) - 1, jan + k * (jan // 2))
                if not np.isnan(p) and abs(p - ritmo) / ritmo > 0.5 and not any(ev["tipo"] == "quebra_de_ritmo" and abs(ev["tempo_s"] - serie.at[idx, "tempo_s"]) < 6 for ev in eventos):
                    evento(idx, "quebra_de_ritmo", Severidade.ATENCAO,
                           f"Cadência do exercício mudou de ~{ritmo:.1f}s para ~{p:.1f}s por repetição", 0.45,
                           periodo_ref=ritmo, periodo=float(p))

    # ── 4. Pose: queda do quadril, tronco horizontal, assimetria ───────────────
    if status["yolo"] and "quadril_y_norm" in serie:
        q = serie["quadril_y_norm"]
        vel = q.diff() / dt                                  # fração do frame por segundo (↓ = corpo caiu)
        if "troca_identidade" in serie:                      # ignora frames em que a pessoa acompanhada mudou
            vel = vel.where(~serie["troca_identidade"].fillna(False).astype(bool), 0.0)
        for i in np.flatnonzero((vel > config.QUEDA_VELOCIDADE_QUADRIL).to_numpy()):
            if not any(ev["tipo"] == "queda_pose" and abs(ev["tempo_s"] - serie.at[i, "tempo_s"]) < 3 for ev in eventos):
                evento(i, "queda_pose", Severidade.CRITICA,
                       f"Quadril desceu {100*vel[i]*dt:.0f}% da altura do frame em {dt:.2f}s (velocidade {vel[i]:.2f}/s)",
                       min(1.0, vel[i] / 1.0), quadril_y=float(q[i]))
        if "angulo_tronco_graus" in serie:
            deitado = ((serie["angulo_tronco_graus"] > 60) | (serie["razao_bbox"] > 1.2))
            if "troca_identidade" in serie:
                deitado &= ~serie["troca_identidade"].fillna(False).astype(bool)
            run = deitado.astype(int).groupby((deitado != deitado.shift()).cumsum()).cumsum()
            for i in np.flatnonzero((run == max(2, int(1.5 / dt))).to_numpy()):
                evento(i, "corpo_horizontal", Severidade.ALTA,
                       f"Tronco a {serie.at[i, 'angulo_tronco_graus']:.0f}° da vertical por ≥1,5 s (paciente deitado/caído?)", 0.75)
        if "assimetria_joelhos" in serie:
            a = serie["assimetria_joelhos"].rolling(max(3, int(2 / dt)), min_periods=2).median()
            run = (a > config.ASSIMETRIA_LIMIAR).astype(int)
            run = run.groupby((run != run.shift()).cumsum()).cumsum()
            for i in np.flatnonzero((run == int(3 / dt)).to_numpy()):
                evento(i, "assimetria_postural", Severidade.ATENCAO,
                       f"Assimetria entre joelhos de {100*a[i]:.0f}% mantida por 3 s (compensação / fraqueza unilateral)",
                       min(1.0, float(a[i]) / 0.5))
        # amplitude de movimento (fatiga: ângulo mínimo do joelho sobe ao longo das repetições)
        if "angulo_joelho_e" in serie and serie["angulo_joelho_e"].notna().sum() > 40:
            ang = serie[["angulo_joelho_e", "angulo_joelho_d"]].mean(axis=1)
            jan = int(6 / dt)
            minimos = ang.rolling(jan).min().dropna()
            if len(minimos) > jan:
                ini, fim = minimos.iloc[:jan].mean(), minimos.iloc[-jan:].mean()
                if fim - ini > 20:
                    i = int(minimos.index[-1])
                    evento(i, "perda_amplitude", Severidade.ATENCAO,
                           f"Amplitude do agachamento reduziu: ângulo mínimo do joelho {ini:.0f}° → {fim:.0f}° (fatiga?)", 0.5)

    # ── 5. Detecção: objetos críticos e pessoas ────────────────────────────────
    if status["yolo"] and "n_pessoas" in serie:
        npess = serie["n_pessoas"].fillna(0).astype(int)
        for i in np.flatnonzero(((npess == 0) & (npess.shift(1) > 0)).to_numpy()):
            evento(i, "pessoa_saiu_de_cena", Severidade.ALTA, "Nenhuma pessoa detectada após presença anterior", 0.6)
        obj = serie["objetos_criticos"].fillna("")
        vistos: set[str] = set()
        for i, s in obj.items():
            novos = set(s.split(";")) - {""} - vistos
            if novos:
                vistos |= novos
                evento(i, "objeto_critico", Severidade.ATENCAO, f"Objeto(s) de atenção em cena: {', '.join(sorted(novos))}", 0.4,
                       objetos=sorted(novos))

    # ── 6. Estatística: z-score + Isolation Forest sobre energia/fluxo ─────────
    # Séries suavizadas em 1 s: picos periódicos normais (ex.: cada repetição do exercício)
    # deixam de parecer outliers; só mudanças sustentadas de regime são sinalizadas.
    jan_suav = max(2, int(2.0 / dt))
    feats = serie[["energia_movimento", "fluxo_medio", "fluxo_vertical"]].rolling(jan_suav, min_periods=1, center=True).mean()
    feats["fluxo_vertical"] = feats["fluxo_vertical"].abs()
    zres = DetectorZScoreDeslizante(janela=int(10 / dt), limiar=5.0).detectar(feats)
    ires = DetectorIsolationForest(contaminacao=0.01).detectar(feats)
    serie["score_zscore"], serie["score_iforest"] = zres.score, ires.score
    for i in np.flatnonzero(zres.anomalia | (ires.anomalia & (ires.score >= 0.7))):
        if i < jan_suav or serie.at[i, "tempo_s"] > serie["tempo_s"].max() - 0.5:
            continue  # bordas da série (primeiro frame tem energia 0 por definição)
        if not any(abs(ev["tempo_s"] - serie.at[i, "tempo_s"]) < 2.0 for ev in eventos):
            evento(i, "movimento_atipico", Severidade.ATENCAO,
                   f"Padrão de movimento estatisticamente atípico (energia {e[i]:.1f}, fluxo vertical {serie.at[i, 'fluxo_vertical']:.2f})",
                   max(zres.score[i], ires.score[i]))

    eventos.sort(key=lambda ev: ev["tempo_s"])
    serie["evento"] = ""
    for ev in eventos:
        j = serie.index[serie["frame"] == ev["frame"]][0]
        serie.at[j, "evento"] = (serie.at[j, "evento"] + ";" + ev["tipo"]).strip(";")

    # ── Alertas + frames-chave ─────────────────────────────────────────────────
    alertas: list[Alerta] = []
    frames_chave: list[str] = []
    for ev in eventos:
        sev = Severidade(ev["severidade_nivel"])
        ts = (ts_base + pd.Timedelta(seconds=ev["tempo_s"])).isoformat()
        alertas.append(Alerta(paciente_id=paciente_id, fonte="video", tipo=ev["tipo"], severidade=sev, timestamp=ts,
                              mensagem=f"[{caminho.name} @ {ev['tempo_s']:.1f}s] {ev['descricao']}",
                              evidencias={"video": str(caminho), "frame": ev["frame"], **ev["evidencias"]},
                              recomendacao={"CRÍTICA": "Verificar o paciente imediatamente.",
                                            "ALTA": "Fisioterapeuta/enfermagem deve checar o paciente.",
                                            "ATENÇÃO": "Revisar trecho do vídeo no relatório."}.get(sev.rotulo, ""),
                              score=ev["score"]))
        if sev >= Severidade.ALTA or len(frames_chave) < 6:
            f = _salvar_frame(caminho, ev["frame"], pasta, ev["tipo"], f"{ev['tempo_s']:.1f}s {ev['tipo']}")
            if f:
                frames_chave.append(f)
                ev["frame_chave"] = f

    resumo = {
        "duracao_s": round(info["duracao_s"], 1), "frames_processados": int(len(serie)), "yolo": status["yolo"],
        "n_eventos": len(eventos), "eventos_por_tipo": pd.Series([ev["tipo"] for ev in eventos]).value_counts().to_dict() if eventos else {},
        "pessoas_max": int(serie["n_pessoas"].max()) if "n_pessoas" in serie and serie["n_pessoas"].notna().any() else None,
        "objetos_detectados": sorted({c for s in serie.get("classes", pd.Series(dtype=str)).dropna() for c in s.split(";") if c}),
        "energia_mediana": round(float(base), 2),
        "score_risco": float(max([ev["score"] for ev in eventos if ev["severidade_nivel"] >= 3], default=0.0)
                             or max([ev["score"] * 0.5 for ev in eventos], default=0.0)),
    }
    serie.to_csv(pasta / f"{caminho.stem}_serie.csv", index=False)
    return ResultadoVideo(paciente_id, str(caminho), info, serie, eventos, alertas, resumo, frames_chave, video_anotado)
