#!/bin/bash
# Configuração do ambiente — Tech Challenge Fase 4 (monitoramento multimodal)
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "  Monitoramento Clínico Multimodal — configuração do ambiente"
echo "============================================================"

# 1. Python 3.11 ou 3.12 (torch/ultralytics/librosa ainda não suportam bem 3.13+)
PY=""
for cand in python3.12 python3.11 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  echo "[!] Python 3.11/3.12 não encontrado. No macOS: brew install python@3.12"
  exit 1
fi
echo "[+] Usando $($PY --version) ($PY)"

# 2. Ambiente virtual
if [ ! -d .venv ]; then
  echo "[+] Criando .venv…"
  "$PY" -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip -q

# 3. Dependências (torch/ultralytics podem levar alguns minutos)
echo "[+] Instalando dependências…"
pip install -r requirements.txt

# 4. Arquivo .env
if [ ! -f .env ]; then
  cp .env.example .env
  echo "[+] .env criado a partir de .env.example — preencha as chaves do Azure se tiver (opcional)."
fi

# 5. Dados de demonstração
echo "[+] Gerando dados sintéticos (CSV, áudios TTS, vídeo)…"
python main.py gerar-dados

echo "============================================================"
echo "✅ Pronto. Próximos passos:"
echo "   source .venv/bin/activate"
echo "   python main.py demo        # pipeline completo → outputs/"
echo "   python main.py dashboard   # painel da equipe médica"
echo "============================================================"
