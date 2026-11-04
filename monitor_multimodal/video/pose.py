"""
pose.py — Estimativa de pose humana (YOLOv8-pose, 17 keypoints COCO) e métricas posturais.

OpenPose foi o modelo sugerido no enunciado; YOLOv8-pose produz o mesmo conjunto de 17
keypoints (COCO) com instalação muito mais simples (pip) e inferência em CPU — por isso
foi adotado aqui. As métricas abaixo são as usadas em análise de fisioterapia/queda:

  quadril_y_norm      : altura do centro do quadril (0=topo, 1=base do frame) → queda = sobe rápido
  angulo_tronco_graus : inclinação do tronco em relação à vertical (0 = em pé, ~90 = deitado)
  angulo_joelho_e/d   : ângulo quadril-joelho-tornozelo (amplitude de agachamento)
  assimetria_joelhos  : |esq − dir| / média  → compensação / fraqueza unilateral
  assimetria_ombros   : diferença de altura dos ombros normalizada pela largura dos ombros
  razao_bbox          : largura/altura da caixa da pessoa (> 1.0 → corpo na horizontal)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .. import config
from .deteccao import carregar_modelo

# índices COCO
NARIZ, OLHO_E, OLHO_D, ORELHA_E, ORELHA_D = 0, 1, 2, 3, 4
OMBRO_E, OMBRO_D, COTOVELO_E, COTOVELO_D, PUNHO_E, PUNHO_D = 5, 6, 7, 8, 9, 10
QUADRIL_E, QUADRIL_D, JOELHO_E, JOELHO_D, TORNOZELO_E, TORNOZELO_D = 11, 12, 13, 14, 15, 16
ESQUELETO = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
             (0, 5), (0, 6)]


@dataclass
class Pose:
    keypoints: np.ndarray            # (17, 2) pixels
    confiancas: np.ndarray           # (17,)
    bbox: tuple[float, float, float, float]
    metricas: dict = field(default_factory=dict)


def _angulo(a, b, c) -> float:
    """Ângulo em graus no vértice b formado por a-b-c."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(math.degrees(math.acos(np.clip(cos, -1, 1))))


def calcular_metricas(kp: np.ndarray, conf: np.ndarray, bbox, altura_frame: int, min_conf: float = 0.3) -> dict:
    ok = conf >= min_conf
    m: dict = {}
    if ok[QUADRIL_E] and ok[QUADRIL_D]:
        quadril = (kp[QUADRIL_E] + kp[QUADRIL_D]) / 2
        m["quadril_y_norm"] = float(quadril[1] / altura_frame)
        if ok[OMBRO_E] and ok[OMBRO_D]:
            ombro = (kp[OMBRO_E] + kp[OMBRO_D]) / 2
            v = ombro - quadril
            m["angulo_tronco_graus"] = float(math.degrees(math.atan2(abs(v[0]), abs(v[1]) + 1e-9)))
    if ok[OMBRO_E] and ok[OMBRO_D]:
        larg = abs(kp[OMBRO_E][0] - kp[OMBRO_D][0]) + 1e-9
        m["assimetria_ombros"] = float(abs(kp[OMBRO_E][1] - kp[OMBRO_D][1]) / larg)
    if ok[QUADRIL_E] and ok[JOELHO_E] and ok[TORNOZELO_E]:
        m["angulo_joelho_e"] = _angulo(kp[QUADRIL_E], kp[JOELHO_E], kp[TORNOZELO_E])
    if ok[QUADRIL_D] and ok[JOELHO_D] and ok[TORNOZELO_D]:
        m["angulo_joelho_d"] = _angulo(kp[QUADRIL_D], kp[JOELHO_D], kp[TORNOZELO_D])
    if "angulo_joelho_e" in m and "angulo_joelho_d" in m:
        media = (m["angulo_joelho_e"] + m["angulo_joelho_d"]) / 2 + 1e-9
        m["assimetria_joelhos"] = float(abs(m["angulo_joelho_e"] - m["angulo_joelho_d"]) / media)
    x1, y1, x2, y2 = bbox
    m["razao_bbox"] = float((x2 - x1) / max(y2 - y1, 1e-9))
    m["keypoints_visiveis"] = int(ok.sum())
    return m


class EstimadorPose:
    def __init__(self, modelo: str = config.YOLO_MODELO_POSE, confianca: float = config.YOLO_CONFIANCA):
        self.modelo = carregar_modelo(modelo)
        self.confianca = confianca

    def estimar(self, frame: np.ndarray) -> list[Pose]:
        res = self.modelo.predict(frame, conf=self.confianca, verbose=False)[0]
        poses: list[Pose] = []
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return poses
        xy = res.keypoints.xy.cpu().numpy()
        kconf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(xy.shape[:2])
        for i, box in enumerate(res.boxes.xyxy.cpu().numpy()):
            bbox = tuple(float(v) for v in box)
            poses.append(Pose(xy[i], kconf[i], bbox, calcular_metricas(xy[i], kconf[i], bbox, frame.shape[0])))
        return poses

    @staticmethod
    def principal(poses: list[Pose], anterior: Pose | None = None, diag_frame: float = 1.0,
                  max_desloc: float = 0.20) -> tuple[Pose | None, bool]:
        """
        Escolhe o paciente acompanhado: a pose cuja caixa está mais próxima da pose anterior
        (continuidade espacial). Se nenhuma estiver a menos de `max_desloc`·diagonal do frame,
        assume a maior caixa e sinaliza troca de identidade (métricas de velocidade são ignoradas
        nesse frame para não gerar falsas quedas em cenas com várias pessoas).
        Devolve (pose, trocou_identidade).
        """
        if not poses:
            return None, False
        maior = max(poses, key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]))
        if anterior is None:
            return maior, False
        ca = np.array([(anterior.bbox[0] + anterior.bbox[2]) / 2, (anterior.bbox[1] + anterior.bbox[3]) / 2])
        dist = [np.linalg.norm(np.array([(p.bbox[0] + p.bbox[2]) / 2, (p.bbox[1] + p.bbox[3]) / 2]) - ca) for p in poses]
        k = int(np.argmin(dist))
        if dist[k] <= max_desloc * diag_frame:
            return poses[k], False
        return maior, True


def desenhar_pose(frame: np.ndarray, pose: Pose, cor=(0, 200, 0), min_conf: float = 0.3) -> np.ndarray:
    import cv2
    for a, b in ESQUELETO:
        if pose.confiancas[a] >= min_conf and pose.confiancas[b] >= min_conf:
            cv2.line(frame, tuple(pose.keypoints[a].astype(int)), tuple(pose.keypoints[b].astype(int)), cor, 2)
    for k, c in zip(pose.keypoints, pose.confiancas):
        if c >= min_conf:
            cv2.circle(frame, tuple(k.astype(int)), 3, (0, 0, 255), -1)
    return frame
