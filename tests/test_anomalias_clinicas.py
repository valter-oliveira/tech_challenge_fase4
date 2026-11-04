import numpy as np
import pandas as pd

from monitor_multimodal.alertas import Severidade
from monitor_multimodal.anomalias.movimentacao import analisar_movimentacao
from monitor_multimodal.anomalias.prescricoes import analisar_prescricoes
from monitor_multimodal.anomalias.sinais_vitais import analisar_sinais_vitais, escore_alerta_precoce
from monitor_multimodal.dados_sinteticos import gerar_movimentacao, gerar_prescricoes, gerar_sinais_vitais


def test_ews():
    assert escore_alerta_precoce(pd.Series({"frequencia_respiratoria": 16, "saturacao_o2": 97, "pressao_sistolica": 120,
                                            "frequencia_cardiaca": 70, "temperatura": 36.8})) == 0
    assert escore_alerta_precoce(pd.Series({"frequencia_respiratoria": 28, "saturacao_o2": 88, "pressao_sistolica": 85,
                                            "frequencia_cardiaca": 135, "temperatura": 39.5})) >= 12


def test_sinais_vitais_estavel_sem_critico():
    df = gerar_sinais_vitais("T", "estavel", horas=24)
    r = analisar_sinais_vitais(df, "T")
    assert not any(a.severidade == Severidade.CRITICA for a in r.alertas)
    assert r.dados["anomalia"].mean() < 0.08


def test_sinais_vitais_deterioracao_detectada():
    df = gerar_sinais_vitais("T", "deterioracao", horas=48)
    r = analisar_sinais_vitais(df, "T")
    y, yh = df["anomalia_injetada"].to_numpy(), r.dados["anomalia"].to_numpy()
    recall = (y & yh).sum() / y.sum()
    assert recall > 0.85
    assert any(a.tipo == "ews_alto" for a in r.alertas)


def test_prescricoes_regras():
    df = gerar_prescricoes("T", "deterioracao")
    r = analisar_prescricoes(df, "T")
    tipos = {a.tipo for a in r.alertas}
    assert {"dose_acima_maximo", "interacao_medicamentosa", "aumento_abrupto_dose"} <= tipos
    assert r.dados.loc[r.dados["anomalia_injetada"], "anomalia"].all()


def test_prescricoes_estavel_sem_critico():
    r = analisar_prescricoes(gerar_prescricoes("T", "estavel"), "T")
    assert not any(a.severidade == Severidade.CRITICA for a in r.alertas)


def test_movimentacao_imobilidade_e_queda():
    r = analisar_movimentacao(gerar_movimentacao("T", "deterioracao"), "T")
    assert "imobilidade_prolongada" in r.resumo["motivos"]
    r2 = analisar_movimentacao(gerar_movimentacao("T", "episodios"), "T")
    assert {"agitacao_noturna", "padrao_queda"} <= set(r2.resumo["motivos"])
    assert any(a.severidade == Severidade.CRITICA for a in r2.alertas)
