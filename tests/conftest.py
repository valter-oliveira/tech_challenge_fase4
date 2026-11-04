import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def gerenciador(tmp_path):
    from monitor_multimodal.alertas import GerenciadorAlertas
    return GerenciadorAlertas(caminho_log=tmp_path / "alertas.jsonl", console=False, webhook_url="", smtp={})
