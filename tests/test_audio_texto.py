import numpy as np
import soundfile as sf

from monitor_multimodal import config
from monitor_multimodal.audio.features import extrair_features_vocais, indicadores_clinicos
from monitor_multimodal.azure_servicos.text_analytics import analisar_texto, termos_criticos
from monitor_multimodal.dados_sinteticos import _tom_sintetico


def test_features_em_audio_sintetico(tmp_path):
    wav = tmp_path / "x.wav"
    _tom_sintetico(wav, "consulta_fadiga")
    f = extrair_features_vocais(wav)
    for k in ("taxa_silabica", "razao_pausa", "pausas_por_min", "f0_mediana_hz", "hnr_db", "mfcc_1"):
        assert k in f
    ind = indicadores_clinicos(f)
    assert all(0 <= ind[k] <= 1 for k in ("fadiga", "dispneia", "disartria"))


def test_silencio_nao_quebra(tmp_path):
    wav = tmp_path / "s.wav"
    sf.write(wav, np.zeros(config.AUDIO_SR * 2, dtype=np.float32), config.AUDIO_SR)
    f = extrair_features_vocais(wav)
    assert f["tempo_fala_s"] == 0.0
    assert indicadores_clinicos(f)["fadiga"] == 0.0


def test_texto_local_termos_e_negacao():
    t = analisar_texto("Estou com falta de ar e dor no peito desde ontem.", modo="local")
    assert t.provedor == "lexico_local" and t.sentimento == "negativo"
    assert {"falta de ar", "dor no peito"} <= set(t.termos_criticos)
    assert any(e["categoria"] == "SymptomOrSign" for e in t.entidades)
    t2 = analisar_texto("Estou bem, não tive dor nem febre.", modo="local")
    assert t2.sentimento == "positivo"
    assert termos_criticos("sem falta de AR") == ["falta de ar"]  # busca é insensível a caixa/acento
