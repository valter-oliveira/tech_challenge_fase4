"""
detectores.py — Detectores genéricos de anomalias para séries temporais e tabelas.

Três famílias complementares (combinadas pelo DetectorEnsemble):

  1. DetectorRegras            : limiares clínicos (conhecimento de domínio) — explicável,
                                 detecta valores absolutos perigosos.
  2. DetectorZScoreDeslizante  : estatística robusta (mediana/MAD) em janela deslizante —
                                 detecta mudanças abruptas em relação ao histórico recente
                                 do próprio paciente, funciona em streaming.
  3. DetectorIsolationForest   : aprendizado não supervisionado multivariado — detecta
                                 combinações atípicas de variáveis (ex.: FC alta + PA baixa
                                 + SpO2 caindo), mesmo quando cada uma isoladamente está
                                 dentro do limite.

Todos devolvem um ResultadoDeteccao com máscara booleana e score em [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .. import config


@dataclass
class ResultadoDeteccao:
    metodo: str
    anomalia: np.ndarray                 # bool, shape (n,)
    score: np.ndarray                    # float em [0,1], shape (n,)
    detalhes: dict[str, Any] = field(default_factory=dict)

    @property
    def n_anomalias(self) -> int:
        return int(self.anomalia.sum())

    @property
    def indices(self) -> np.ndarray:
        return np.flatnonzero(self.anomalia)


def _para_dataframe(X, colunas: list[str] | None = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X[colunas] if colunas else X.select_dtypes("number")
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return pd.DataFrame(X, columns=colunas or [f"x{i}" for i in range(X.shape[1])])


# ─── 1. Regras clínicas ──────────────────────────────────────────────────────
class DetectorRegras:
    """
    Limiares por variável. `faixas` tem o formato de config.SINAIS_VITAIS:
        {"coluna": {"normal": (lo, hi), "critico": (lo, hi)}}
    Score: 0 dentro do normal; 0.5 fora do normal; 1.0 fora do crítico
    (interpolado conforme a distância).
    """

    def __init__(self, faixas: dict[str, dict] | None = None):
        self.faixas = faixas or config.SINAIS_VITAIS

    def pontuar_valor(self, coluna: str, valor: float) -> tuple[float, str]:
        """Devolve (score, categoria) para um valor; categoria ∈ {normal, alterado, critico}."""
        f = self.faixas.get(coluna)
        if f is None or valor is None or np.isnan(valor):
            return 0.0, "normal"
        n_lo, n_hi = f["normal"]
        c_lo, c_hi = f["critico"]
        if n_lo <= valor <= n_hi:
            return 0.0, "normal"
        if valor < c_lo or valor > c_hi:
            return 1.0, "critico"
        # Entre normal e crítico → interpola 0.5..0.95
        if valor < n_lo:
            frac = (n_lo - valor) / max(n_lo - c_lo, 1e-9)
        else:
            frac = (valor - n_hi) / max(c_hi - n_hi, 1e-9)
        return float(0.5 + 0.45 * min(frac, 1.0)), "alterado"

    def detectar(self, X: pd.DataFrame) -> ResultadoDeteccao:
        colunas = [c for c in X.columns if c in self.faixas]
        n = len(X)
        score = np.zeros(n)
        por_coluna: dict[str, np.ndarray] = {}
        categorias: dict[str, list[str]] = {}
        for c in colunas:
            vals = X[c].to_numpy(dtype=float)
            sc = np.zeros(n)
            cat = []
            for i, v in enumerate(vals):
                sc[i], k = self.pontuar_valor(c, v)
                cat.append(k)
            por_coluna[c] = sc
            categorias[c] = cat
            score = np.maximum(score, sc)
        return ResultadoDeteccao(
            metodo="regras_clinicas",
            anomalia=score >= 0.5,
            score=score,
            detalhes={"score_por_coluna": por_coluna, "categoria_por_coluna": categorias},
        )


def resolucao(serie: pd.Series) -> float:
    """Menor diferença entre valores distintos da série (≈ resolução do sensor)."""
    vals = np.unique(serie.dropna().to_numpy(dtype=float))
    if len(vals) < 2:
        return 1.0
    d = np.diff(vals)
    d = d[d > 1e-9]
    return float(d.min()) if len(d) else 1.0


# ─── 2. Z-score robusto deslizante ───────────────────────────────────────────
class DetectorZScoreDeslizante:
    """
    z_i = 0.6745 · (x_i − mediana_janela) / MAD_janela   (janela só com o passado)
    Robusto a outliers na própria janela (mediana e MAD em vez de média e desvio).
    """

    def __init__(self, janela: int = config.JANELA_ROLLING, limiar: float = config.ZSCORE_LIMIAR,
                 min_amostras: int = 8):
        self.janela = janela
        self.limiar = limiar
        self.min_amostras = min_amostras

    def z_scores(self, serie: pd.Series) -> pd.Series:
        passado = serie.shift(1)  # usa apenas valores anteriores → sem vazamento
        mediana = passado.rolling(self.janela, min_periods=self.min_amostras).median()
        mad = (passado - mediana).abs().rolling(self.janela, min_periods=self.min_amostras).median()
        # Piso de escala: MAD pode ser 0 em medidas arredondadas (ex.: SpO₂ inteira).
        # Usamos a resolução da medida (menor diferença entre valores distintos) como piso,
        # para que 1 unidade de variação não vire um z-score gigante.
        escala = mad.clip(lower=max(resolucao(serie), 1e-6))
        return 0.6745 * (serie - mediana) / escala

    def detectar(self, X) -> ResultadoDeteccao:
        df = _para_dataframe(X)
        z = pd.DataFrame({c: self.z_scores(df[c]) for c in df.columns})
        zabs = z.abs().fillna(0.0)
        zmax = zabs.max(axis=1).to_numpy()
        score = np.clip(zmax / (2 * self.limiar), 0, 1)
        return ResultadoDeteccao(
            metodo="zscore_deslizante",
            anomalia=zmax > self.limiar,
            score=score,
            detalhes={"z": z, "coluna_responsavel": zabs.idxmax(axis=1).to_numpy()},
        )


# ─── 3. Isolation Forest ─────────────────────────────────────────────────────
class DetectorIsolationForest:
    """Isolation Forest multivariado com padronização prévia das variáveis."""

    def __init__(self, contaminacao: float = config.IFOREST_CONTAMINACAO,
                 n_estimators: int = 200, random_state: int = config.RANDOM_STATE):
        self.scaler = StandardScaler()
        self.modelo = IsolationForest(
            n_estimators=n_estimators, contamination=contaminacao, random_state=random_state
        )
        self.colunas: list[str] = []
        self._ajustado = False

    def ajustar(self, X) -> "DetectorIsolationForest":
        df = _para_dataframe(X).dropna()
        self.colunas = list(df.columns)
        self.modelo.fit(self.scaler.fit_transform(df.to_numpy(dtype=float)))
        self._ajustado = True
        return self

    def detectar(self, X) -> ResultadoDeteccao:
        df = _para_dataframe(X)
        if not self._ajustado:
            self.ajustar(df)
        df = df[self.colunas]
        valido = ~df.isna().any(axis=1).to_numpy()
        n = len(df)
        anomalia = np.zeros(n, dtype=bool)
        score = np.zeros(n)
        if valido.any():
            Z = self.scaler.transform(df[valido].to_numpy(dtype=float))
            # decision_function: negativo = anômalo. Convertemos para [0,1] com sigmoide.
            dec = self.modelo.decision_function(Z)
            anomalia[valido] = self.modelo.predict(Z) == -1
            score[valido] = 1 / (1 + np.exp(dec / 0.05))
        return ResultadoDeteccao(metodo="isolation_forest", anomalia=anomalia, score=score)


# ─── Ensemble ────────────────────────────────────────────────────────────────
class DetectorEnsemble:
    """
    Combina detectores: score final = máximo ponderado; anomalia se qualquer
    detector sinalizar (OR) — em monitoramento clínico preferimos sensibilidade.
    """

    def __init__(self, detectores: list[tuple[Any, float]]):
        self.detectores = detectores  # [(detector, peso)]

    def detectar(self, X) -> ResultadoDeteccao:
        resultados = [(d.detectar(X), w) for d, w in self.detectores]
        anomalia = np.zeros(len(X), dtype=bool)
        score = np.zeros(len(X))
        for r, w in resultados:
            anomalia |= r.anomalia
            score = np.maximum(score, np.clip(r.score * w, 0, 1))
        return ResultadoDeteccao(
            metodo="ensemble",
            anomalia=anomalia,
            score=score,
            detalhes={r.metodo: r for r, _ in resultados},
        )
