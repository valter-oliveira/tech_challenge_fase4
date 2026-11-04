from datetime import timedelta

from monitor_multimodal import config
from monitor_multimodal.alertas import Alerta, Severidade


def _a(ts, sev=Severidade.ATENCAO):
    return Alerta("P1", "sinais_vitais", "x", sev, "msg", timestamp=ts)


def test_dedup_e_escalada(gerenciador):
    gerenciador.janela_dedup = timedelta(minutes=30)
    base = "2026-01-01T10:00:00"
    emitidos = [gerenciador.emitir(_a(f"2026-01-01T10:{m:02d}:00")) for m in (0, 5, 10, 15, 20)]
    # 1º emitido; repetições suprimidas até a PERSISTENCIA_MINIMA-ésima, que escala e é emitida
    assert emitidos[0] is not None and emitidos[0].severidade == Severidade.ATENCAO
    escalados = [e for e in emitidos[1:] if e is not None]
    assert len(escalados) == 1 and escalados[0].severidade == Severidade.ALTA
    assert gerenciador.suprimidos == 3
    # fora da janela volta a emitir
    assert gerenciador.emitir(_a("2026-01-01T11:00:00")) is not None


def test_resumo_e_log(gerenciador):
    gerenciador.emitir(_a("2026-01-01T10:00:00", Severidade.CRITICA))
    gerenciador.emitir(Alerta("P2", "audio", "y", Severidade.ALTA, "m", timestamp="2026-01-01T10:00:00"))
    r = gerenciador.resumo()
    assert r["total"] == 2 and r["por_severidade"]["CRÍTICA"] == 1 and r["por_fonte"] == {"sinais_vitais": 1, "audio": 1}
    assert gerenciador.caminho_log.read_text().count("\n") == 2


def test_timestamps_com_e_sem_fuso_nao_quebram(gerenciador):
    gerenciador.emitir(_a("2026-01-01T10:00:00"))
    assert gerenciador.emitir(_a("2026-01-01T10:01:00-03:00")) is not None
