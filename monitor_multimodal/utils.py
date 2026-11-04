"""utils.py — Utilidades compartilhadas (logging, serialização, tempo)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.logging import RichHandler


def configurar_logging(nivel: int = logging.INFO) -> logging.Logger:
    """Configura logging bonito (rich) uma única vez e devolve o logger raiz do projeto."""
    logger = logging.getLogger("monitor")
    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True, show_path=False, markup=True)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%H:%M:%S]"))
        logger.addHandler(handler)
        logger.setLevel(nivel)
        logger.propagate = False
    return logger


log = configurar_logging()


def agora_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class _EncoderJSON(json.JSONEncoder):
    """Serializa tipos numpy/pandas/Path/datetime que o json padrão não aceita."""

    def default(self, o: Any):  # noqa: D401
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, set):
            return sorted(o)
        return super().default(o)


def salvar_json(obj: Any, caminho: Path | str) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(obj, cls=_EncoderJSON, ensure_ascii=False, indent=2), encoding="utf-8")
    return caminho


def carregar_json(caminho: Path | str) -> Any:
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def limitar(valor: float, minimo: float = 0.0, maximo: float = 1.0) -> float:
    return float(max(minimo, min(maximo, valor)))
