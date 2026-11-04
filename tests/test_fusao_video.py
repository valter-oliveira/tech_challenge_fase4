from types import SimpleNamespace

import cv2
import numpy as np

from monitor_multimodal.alertas import Alerta, Severidade
from monitor_multimodal.fusao import fundir
from monitor_multimodal.video import analisar_video


def _res(score, alertas):
    return SimpleNamespace(resumo={"score_risco": score}, alertas=alertas)


def test_fusao_corroboracao_eleva_risco():
    a1 = Alerta("P", "sinais_vitais", "spo2", Severidade.CRITICA, "m", timestamp="2026-01-01T10:00:00")
    a2 = Alerta("P", "audio", "dispneia", Severidade.ALTA, "m", timestamp="2026-01-01T10:20:00")
    perfil = fundir("P", {"sinais_vitais": _res(0.9, [a1]), "audio": [_res(0.7, [a2])], "prescricoes": _res(0.0, [])})
    assert perfil.corroboracoes and perfil.nivel in ("alto", "crítico")
    assert any(a.tipo == "corroboracao_multimodal" for a in perfil.alertas)
    baixo = fundir("Q", {"sinais_vitais": _res(0.05, []), "prescricoes": _res(0.0, [])})
    assert baixo.nivel == "baixo" and baixo.score_global < 0.25


def test_video_sem_yolo_detecta_pico_e_imobilidade(tmp_path):
    caminho = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), 20, (320, 240))
    for i in range(20 * 14):
        t = i / 20
        frame = np.full((240, 320, 3), 220, np.uint8)
        if t < 6:                       # movimento rítmico suave
            y = int(100 + 20 * np.sin(2 * np.pi * t))
            cv2.circle(frame, (160, y), 25, (50, 50, 200), -1)
        elif t < 6.4:                   # movimento brusco
            cv2.circle(frame, (160 + int(300 * (t - 6)), 100 + int(300 * (t - 6))), 25, (50, 50, 200), -1)
        else:                           # imóvel
            cv2.circle(frame, (280, 220), 25, (50, 50, 200), -1)
        w.write(frame)
    w.release()
    r = analisar_video(caminho, "T", usar_yolo=False, gravar_anotado=False, pasta_saida=tmp_path / "out")
    tipos = {e["tipo"] for e in r.eventos}
    assert "pico_seguido_de_imobilidade" in tipos or "imobilidade_prolongada" in tipos
    assert r.resumo["yolo"] is False and len(r.serie) > 50
