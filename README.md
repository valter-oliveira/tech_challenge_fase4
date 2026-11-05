# Monitoramento Clínico Multimodal com Detecção de Anomalias

**Tech Challenge — Fase 4 · PosTech FIAP (IA para Devs)**

Sistema que monitora pacientes continuamente a partir de dados **multimodais** — vídeo, áudio e dados clínicos
estruturados (sinais vitais, prescrições, movimentação) — detecta anomalias em tempo real, funde as evidências
em um score de risco por paciente e **alerta a equipe médica** com recomendações e relatórios automáticos.
Integra os serviços gerenciados **Azure Speech to Text** e **Azure AI Language (Text Analytics / Text Analytics for health)**,
com fallback local para rodar a demonstração completa sem credenciais.

| Entrega do enunciado | Onde está |
|---|---|
| **1. Análise de vídeo** — YOLOv8 (detecção) + YOLOv8-pose (postura, equivalente aos 17 keypoints do OpenPose), energia de movimento/fluxo óptico, eventos (queda, imobilidade, quebra de ritmo, assimetria, objetos críticos) e **relatório automático** por vídeo | `monitor_multimodal/video/`, `outputs/video/<vídeo>/<vídeo>_relatorio.md` |
| **2. Análise de áudio** — biomarcadores vocais (fadiga, dispneia, disartria), **Azure Speech to Text** (transcrição), **Azure Text Analytics** (sentimento, frases-chave, entidades clínicas, termos críticos) | `monitor_multimodal/audio/`, `monitor_multimodal/azure_servicos/` |
| **3. Detecção de anomalias** — sinais vitais (lote e **tempo real**), prescrições, movimentação; regras clínicas + z-score robusto + Isolation Forest + escore de alerta precoce; **alertas automáticos** (console, JSONL, webhook Teams/Slack, e-mail) | `monitor_multimodal/anomalias/`, `monitor_multimodal/alertas/` |
| Fusão multimodal e painel para a equipe médica | `monitor_multimodal/fusao/`, `dashboard/app.py` |
| Relatório técnico (fluxo multimodal, modelos, resultados, exemplos de anomalias) | [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md) |
| Arquitetura (diagrama) e roteiro do vídeo de demonstração | [`docs/arquitetura.md`](docs/arquitetura.md), [`docs/roteiro_video.md`](docs/roteiro_video.md) |

## Como executar

Requisitos: macOS/Linux, **Python 3.11 ou 3.12**, ~3 GB de disco (torch + ultralytics). Opcional: `ffmpeg`.

```bash
./setup.sh                      # cria .venv, instala dependências, cria .env, gera dados sintéticos
source .venv/bin/activate
python main.py demo             # pipeline completo (3 pacientes) → outputs/
python main.py dashboard        # painel Streamlit da equipe médica (http://localhost:8501)
pytest -q                       # testes automatizados
```

Sem o `setup.sh`:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # preencha as chaves do Azure se tiver (opcional)
python main.py gerar-dados
python main.py demo
```

> Na primeira execução o `ultralytics` baixa os pesos `yolov8n.pt` e `yolov8n-pose.pt` (~6 MB cada) para `modelos/`.
> `python main.py baixar-amostras` baixa o vídeo público `vtest.avi` (OpenCV, pessoas caminhando) para testar a detecção/pose em imagens reais.

### Comandos da CLI

```bash
python main.py gerar-dados                                   # CSVs, áudios (TTS do macOS, voz pt-BR) e vídeo sintético
python main.py demo [--sem-video] [--sem-audio] [--max-frames N] [--modo auto|azure|local]
python main.py sinais-vitais --paciente P002 [--arquivo meu.csv]
python main.py prescricoes    --paciente P002
python main.py movimentacao   --paciente P003
python main.py stream         --paciente P002 --intervalo 0.05 --verboso   # simulação em tempo real
python main.py audio  dados/sinteticos/gerados/audio/consulta_dispneia.wav --paciente P003
python main.py video  dados/sinteticos/gerados/video/fisioterapia_sintetica.mp4 --paciente P003 --sessao fisioterapia
python main.py video  dados/amostras/vtest.avi --paciente P001 --sessao enfermaria --max-frames 200
python main.py dashboard
```

Qualquer CSV com as colunas esperadas (ver `monitor_multimodal/config.py` e os CSVs gerados em `dados/sinteticos/gerados/`)
pode ser passado com `--arquivo`; qualquer `.wav/.mp3` ou `.mp4/.avi/.mov` pode ser analisado com `audio`/`video`.

### Transcrição local de áudios enviados

Para transcrever novos arquivos de áudio sem Azure, instale o provedor local no ambiente virtual:

```bash
.venv/bin/python -m pip install faster-whisper
```

Use `MODO_SERVICOS=local` no `.env` e reinicie o painel. Na primeira transcrição, o Whisper baixa o modelo
`small`; depois, o reconhecimento roda no computador. A análise de sentimento usa regras sobre o texto
transcrito, separadamente dos indicadores acústicos de fadiga e dispneia. Sem transcrição ou quando a
análise de texto falhar, o painel mostra o sentimento como indisponível.

### Configurando o Azure (opcional)

No portal Azure crie um recurso **Speech** e um recurso **Language** e preencha o `.env`:

```ini
AZURE_SPEECH_KEY=...
AZURE_SPEECH_REGION=brazilsouth
AZURE_LANGUAGE_KEY=...
AZURE_LANGUAGE_ENDPOINT=https://<recurso>.cognitiveservices.azure.com/
MODO_SERVICOS=auto      # auto: usa Azure se houver chave, senão fallback local · azure: exige Azure · local: nunca chama Azure
```

Com as chaves presentes, `analisar_audio` passa a usar **Speech to Text** (reconhecimento contínuo pt-BR, com offsets por trecho)
e **AI Language** (`analyze_sentiment`, `extract_key_phrases`, `begin_analyze_healthcare_entities`). Sem chaves, usa a
transcrição de referência gerada junto com o áudio sintético (ou `faster-whisper`, se instalado) e um léxico clínico local.

Canais de alerta opcionais no `.env`: `ALERTA_WEBHOOK_URL` (Teams/Slack/Discord — alertas ≥ ALTA) e `ALERTA_SMTP_*` (e-mail — alertas CRÍTICOS).

## O que a demonstração mostra

Três pacientes sintéticos com 48 h de sinais vitais (a cada 5 min), 5 dias de prescrições, 72 h de movimentação,
um áudio de consulta cada e dois vídeos:

| Paciente | Perfil | Episódios injetados | Resultado da fusão |
|---|---|---|---|
| P001 | estável (controle) | nenhum; áudio de consulta normal; vídeo de enfermaria (`vtest.avi`) | risco **baixo** |
| P002 | deterioração progressiva (sepse) | FC↑ PA↓ T↑ SpO₂↓ FR↑ a partir da hora 24; dipirona 10× a dose; varfarina + heparina; ceftriaxona dobrada; 7 h de imobilidade; fala com fadiga | risco **crítico**, 2 corroborações multimodais |
| P003 | episódios agudos | dessaturação, taquiarritmia, febre, artefatos de sensor; duplicidade e troca de via; agitação noturna seguida de queda; fala com dispneia ("falta de ar", "dor no peito"); vídeo de fisioterapia com queda | risco **crítico**, 6 corroborações (ex.: SpO₂ crítica + dispneia na fala; queda no vídeo + padrão de queda no sensor) |

Métricas contra os rótulos injetados (ver `outputs/relatorio_geral.md`): sinais vitais do P002 **F1 = 0,91** (AUC 0,97) com o
primeiro alerta **85 min antes** do limiar de rotulagem; P003 recall 1,0 em todos os episódios; prescrições anômalas 5/5
detectadas sem falso positivo no paciente estável.

Saídas em `outputs/`: `alertas.jsonl` (todos os alertas despachados), `relatorio_geral.md`, `pacientes/<id>/relatorio.md`
(+ gráficos, CSVs, `perfil.json`), `video/<vídeo>/` (relatório, vídeo anotado, frames-chave, série temporal).

## Estrutura

```
tech_challenge_fase4/
├── main.py                      # CLI
├── monitor_multimodal/
│   ├── config.py                # limiares clínicos, pesos da fusão, credenciais (.env)
│   ├── anomalias/               # detectores, sinais vitais, stream (tempo real), prescrições, movimentação, avaliação
│   ├── audio/                   # biomarcadores vocais (librosa) + pipeline de áudio
│   ├── azure_servicos/          # Azure Speech to Text, Azure AI Language (com fallback local)
│   ├── video/                   # energia de movimento, YOLOv8 detecção, YOLOv8-pose, pipeline de vídeo
│   ├── alertas/                 # Alerta, Severidade, GerenciadorAlertas (dedup, escalada, canais)
│   ├── fusao/                   # fusão multimodal → PerfilRisco
│   ├── relatorios/              # relatórios Markdown/JSON + gráficos
│   ├── dados_sinteticos.py      # geração de dados com anomalias rotuladas (CSV, TTS, vídeo)
│   └── pipeline.py              # orquestração ponta a ponta (demo)
├── dashboard/app.py             # painel Streamlit
├── dados/                       # sinteticos/gerados (gerado) · amostras (vtest.avi)
├── docs/                        # relatorio_tecnico.md · arquitetura.md · roteiro_video.md
├── tests/                       # pytest
├── outputs/                     # resultados (gerado)
├── requirements.txt · requirements-opcional.txt · setup.sh · .env.example
```

## Limitações e próximos passos

- Os dados clínicos são **sintéticos** (PhysioNet/MIMIC exigem credenciamento). O carregamento de registros WFDB (`wfdb`) está previsto em `requirements-opcional.txt`.
- O vídeo sintético valida o pipeline de eventos (queda, imobilidade), mas a figura desenhada não é reconhecida pelo YOLOv8 como pessoa; a detecção/pose real é demonstrada com `vtest.avi` e funciona com qualquer vídeo real de fisioterapia.
- Os indicadores vocais são heurísticas calibradas em literatura de fala patológica; um modelo supervisionado exigiria um corpus clínico anotado.
- Azure Anomaly Detector não foi usado por ter sido descontinuado pela Microsoft; a detecção de anomalias é local (scikit-learn).
- Próximos passos: integração HL7/FHIR com o prontuário, modelos sequenciais (LSTM/Transformer) para sinais vitais, rastreamento multi-pessoa (ByteTrack) no vídeo, validação com dados reais.
