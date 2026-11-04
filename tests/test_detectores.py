import numpy as np
import pandas as pd

from monitor_multimodal.anomalias.detectores import (
    DetectorIsolationForest, DetectorRegras, DetectorZScoreDeslizante, resolucao,
)


def test_resolucao_inteira_e_decimal():
    assert resolucao(pd.Series([97.0, 98.0, 96.0, 97.0])) == 1.0
    assert abs(resolucao(pd.Series([36.5, 36.6, 36.8])) - 0.1) < 1e-9


def test_zscore_detecta_pico_isolado():
    rng = np.random.default_rng(0)
    x = 70 + rng.normal(0, 2, 200)
    x[150] = 140
    r = DetectorZScoreDeslizante(janela=30).detectar(pd.DataFrame({"fc": x}))
    assert r.anomalia[150]
    assert r.n_anomalias <= 4  # pouquíssimos falsos positivos no ruído


def test_zscore_nao_explode_em_serie_arredondada():
    x = np.round(97 + np.random.default_rng(1).normal(0, 0.6, 300))
    r = DetectorZScoreDeslizante().detectar(pd.DataFrame({"spo2": x}))
    assert r.n_anomalias <= 6  # MAD=0 não pode virar z infinito


def test_regras_categorias():
    d = DetectorRegras()
    assert d.pontuar_valor("saturacao_o2", 97)[1] == "normal"
    assert d.pontuar_valor("saturacao_o2", 92)[1] == "alterado"
    assert d.pontuar_valor("saturacao_o2", 85)[1] == "critico"
    assert d.pontuar_valor("frequencia_cardiaca", 150) == (1.0, "critico")


def test_isolation_forest_marca_outlier_multivariado():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(0, 1, 500)})
    df.loc[10] = [8, -8]
    r = DetectorIsolationForest(contaminacao=0.01).detectar(df)
    assert r.anomalia[10] and r.score[10] > 0.5
