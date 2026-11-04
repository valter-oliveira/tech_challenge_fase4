from monitor_multimodal.anomalias.stream import MonitorTempoReal, simular_stream
from monitor_multimodal.dados_sinteticos import gerar_sinais_vitais


def test_stream_emite_critico_na_deterioracao(gerenciador):
    df = gerar_sinais_vitais("T", "deterioracao", horas=48)
    m = MonitorTempoReal(gerenciador)
    primeiro_critico = None
    for amostra, alertas in simular_stream(df, "T", m):
        for a in alertas:
            if a.severidade.rotulo == "CRÍTICA" and primeiro_critico is None:
                primeiro_critico = amostra["timestamp"]
    assert primeiro_critico is not None
    # deterioração começa na hora 24 (2026-11-02 08:00); o alerta crítico deve vir antes do fim
    assert str(primeiro_critico) < "2026-11-02 20:00"


def test_stream_estavel_poucos_alertas(gerenciador):
    df = gerar_sinais_vitais("T", "estavel", horas=24)
    m = MonitorTempoReal(gerenciador)
    for _ in simular_stream(df, "T", m):
        pass
    assert gerenciador.resumo()["por_severidade"]["CRÍTICA"] == 0
    assert len(gerenciador.alertas) < 40
