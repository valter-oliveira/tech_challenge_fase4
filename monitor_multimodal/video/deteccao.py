"""
deteccao.py — Detecção de objetos e pessoas com YOLOv8 (ultralytics).

Uso clínico:
  • contar pessoas em cena (paciente sozinho / acompanhado / saiu do campo de visão)
  • identificar objetos em áreas críticas (config.CLASSES_CRITICAS_VIDEO)
  • fornecer caixas para o módulo de pose e para anotação do vídeo

Os pesos (yolov8n.pt, ~6 MB) são baixados automaticamente pelo ultralytics na
primeira execução para a pasta `modelos/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import config
from ..utils import log


@dataclass
class Deteccao:
    classe: str
    confianca: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)

    @property
    def area(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1])


def yolo_disponivel() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache(maxsize=4)
def carregar_modelo(nome: str):
    from ultralytics import YOLO

    caminho = config.MODELOS_DIR / nome
    log.info(f"Carregando YOLOv8: {nome}" + ("" if caminho.exists() else " (download na primeira execução)"))
    return YOLO(str(caminho))


class DetectorObjetos:
    def __init__(self, modelo: str = config.YOLO_MODELO_DETECCAO, confianca: float = config.YOLO_CONFIANCA):
        self.modelo = carregar_modelo(modelo)
        self.confianca = confianca
        self.nomes = self.modelo.names

    def detectar(self, frame: np.ndarray) -> list[Deteccao]:
        res = self.modelo.predict(frame, conf=self.confianca, verbose=False)[0]
        out = []
        if res.boxes is None:
            return out
        for xyxy, conf, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            out.append(Deteccao(self.nomes[int(cls)], float(conf), tuple(float(v) for v in xyxy)))
        return out

    @staticmethod
    def resumir(dets: list[Deteccao]) -> dict:
        pessoas = [d for d in dets if d.classe == "person"]
        criticos = sorted({d.classe for d in dets if d.classe in config.CLASSES_CRITICAS_VIDEO})
        return {"n_pessoas": len(pessoas), "objetos_criticos": criticos,
                "classes": sorted({d.classe for d in dets}),
                "maior_pessoa_bbox": max(pessoas, key=lambda d: d.area).bbox if pessoas else None}
