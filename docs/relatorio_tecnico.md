# Relatório técnico — Monitoramento clínico multimodal com detecção de anomalias

**Tech Challenge Fase 4 · PosTech FIAP (IA para Devs)**
Repositório: pasta `tech_challenge_fase4/` · Execução de referência: `python main.py demo` (34,9 s em um Mac arm64, CPU)

---

## 1. Contexto e objetivo

O hospital já usa IA para analisar exames e laudos e deseja **monitorar continuamente os pacientes com dados multimodais**
(vídeo, áudio e texto/dados clínicos) para identificar sinais precoces de risco. O sistema construído:

1. analisa **vídeos clínicos** (fisioterapia, cirurgia, leito) e identifica movimentos/eventos fora do padrão, gerando relatórios automáticos;
2. analisa **áudios de consulta**, detectando alterações vocais (fadiga, dificuldade respiratória, disartria), transcrevendo com **Azure Speech to Text** e extraindo termos críticos, entidades clínicas e sentimento com **Azure AI Language (Text Analytics)**;
3. detecta **anomalias em tempo real** em séries de sinais vitais, na evolução de prescrições e nos padrões de movimentação do paciente;
4. **funde** as evidências em um score de risco por paciente e **alerta a equipe médica** (console, log estruturado, webhook Teams/Slack, e-mail), com painel e relatórios.

## 2. Fluxo multimodal

O diagrama completo está em [`arquitetura.md`](arquitetura.md). Em resumo:

```
 vídeo ──► OpenCV (energia de movimento, fluxo óptico) ─┐
       ──► YOLOv8 (pessoas, objetos críticos)            ├─► eventos de vídeo ─┐
       ──► YOLOv8-pose (quadril, tronco, joelhos)       ─┘                     │
 áudio ──► librosa (biomarcadores vocais) ──────────────┐                      │
       ──► Azure Speech to Text ──► Azure AI Language ──┴─► indicadores + termos│
 sinais vitais ─► regras NEWS2 + z-score robusto + Isolation Forest + EWS ─────┤
 prescrições ───► regras farmacológicas + Isolation Forest ─────────────────────┼─► GerenciadorAlertas ─► Fusão ─► risco/paciente
 movimentação ──► imobilidade / agitação / queda + z-score / Isolation Forest ──┘      (dedup, escalada,      (score ponderado +
                                                                                        canais)              corroboração)
                                                                                                                 │
                                                                   relatórios Markdown/JSON + gráficos + painel Streamlit ◄──┘
```

**Fusão tardia (nível de decisão).** Cada modalidade é analisada por modelos especializados e devolve um score em [0, 1] e uma
lista de alertas tipados (`Alerta`: paciente, fonte, tipo, severidade INFO/ATENÇÃO/ALTA/CRÍTICA, mensagem, evidências,
recomendação, score). A fusão calcula `score_global = Σ peso_fonte · score_fonte` (pesos em `config.PESOS_FUSAO`,
renormalizados para as fontes disponíveis; o score de cada fonte é limitado à faixa compatível com a severidade máxima dos seus
alertas) e adiciona um bônus quando **duas ou mais modalidades distintas** emitem alertas ≥ ALTA na mesma janela de 60 min
(**corroboração multimodal**), que também gera um alerta CRÍTICO próprio. Níveis: baixo (< 0,35), moderado (< 0,55), alto (< 0,75), crítico.

**Fluxo do alerta à equipe.** `GerenciadorAlertas.emitir` deduplica alertas iguais do mesmo paciente numa janela de 30 min,
escala a severidade uma vez quando a mesma anomalia se repete pela 3ª vez (persistência) e suprime o restante — reduzindo a
fadiga de alarme (na demo, 438 detecções brutas → 172 alertas emitidos, 266 agrupados). Cada alerta é gravado em
`outputs/alertas.jsonl`, exibido no console e, conforme a severidade, enviado ao webhook (≥ ALTA) e por e-mail (CRÍTICA);
o painel Streamlit mostra o feed filtrável e o detalhe por paciente com a evidência e a recomendação.

## 3. Modelos aplicados por tipo de dado

### 3.1 Vídeo (`monitor_multimodal/video`)

| Camada | Modelo / técnica | Saída | Uso |
|---|---|---|---|
| Movimento | Diferença absoluta entre frames (escala de cinza, 320 px) + fluxo óptico de Farnebäck (OpenCV) | `energia_movimento`, `fluxo_medio`, `fluxo_vertical` por frame | funciona em qualquer vídeo, sem modelos pesados; base dos eventos de queda/imobilidade/ritmo |
| Detecção | **YOLOv8n** (ultralytics, COCO, 80 classes) | caixas, classes, confiança | nº de pessoas em cena, objetos de atenção (`knife`, `scissors`, `bottle`, …), saída de cena |
| Pose | **YOLOv8n-pose** (17 keypoints COCO — o mesmo esqueleto do OpenPose, com instalação via pip e inferência em CPU) | keypoints + métricas: altura do quadril, ângulo do tronco, ângulos dos joelhos, assimetria joelhos/ombros, razão da caixa | queda (velocidade do quadril), corpo horizontal, assimetria postural, perda de amplitude |
| Rastreamento | continuidade espacial da pessoa principal (menor deslocamento da caixa entre frames) | `troca_identidade` | evita falsas quedas quando a pessoa acompanhada muda (cenas com várias pessoas) |
| Anomalia estatística | z-score robusto deslizante (janela 10 s, séries suavizadas em 2 s) + Isolation Forest (contaminação 1 %) | `movimento_atipico` | desvio do padrão do próprio vídeo |

Eventos gerados (com timestamp, severidade, score, evidências e frame-chave): `pico_seguido_de_imobilidade` (CRÍTICA),
`queda_pose` (CRÍTICA), `corpo_horizontal` (ALTA), `imobilidade_prolongada` (ALTA), `pessoa_saiu_de_cena` (ALTA),
`quebra_de_ritmo`, `perda_amplitude`, `assimetria_postural`, `objeto_critico`, `movimento_atipico` (ATENÇÃO).
Cada vídeo recebe um **relatório automático** (`outputs/video/<vídeo>/<vídeo>_relatorio.md`) com linha do tempo, gráfico,
frames-chave e interpretação, além do **vídeo anotado** com caixas e esqueleto.

### 3.2 Áudio (`monitor_multimodal/audio`, `monitor_multimodal/azure_servicos`)

| Etapa | Técnica | Saída |
|---|---|---|
| Biomarcadores vocais | librosa: VAD por energia; taxa silábica (picos do envelope 300–2500 Hz); razão/duração/frequência de pausas; f0 por pYIN (mediana, desvio em semitons, jitter); shimmer; HNR via separação harmônico-percussivo; centróide espectral; 13 MFCC | 30+ características interpretáveis |
| Indicadores clínicos | combinação ponderada de sigmoides em torno de valores de referência da fala conversacional (taxa 4–6 síl/s, pausas < 25 %, variação de pitch 2–4 st, HNR > 10 dB) | `fadiga`, `dispneia`, `disartria` ∈ [0, 1] |
| Transcrição | **Azure Speech to Text** (SDK `azure-cognitiveservices-speech`, reconhecimento contínuo pt-BR, formato detalhado com confiança e offsets) → fallback `faster-whisper` (opcional) → fallback transcrição de referência | texto, segmentos, provedor |
| Análise de texto | **Azure AI Language**: `analyze_sentiment` (com opinion mining), `extract_key_phrases`, `begin_analyze_healthcare_entities` (**Text Analytics for health**: sintomas, estruturas corporais, medicamentos, negação, relações) → fallback léxico clínico local em português (com tratamento de negação) | sentimento + scores, frases-chave, entidades, **termos críticos** (`config.TERMOS_CRITICOS_AUDIO`) |
| Fusão intra-modal | termos respiratórios na transcrição reforçam o indicador acústico de dispneia; termos de fadiga reforçam fadiga | alertas `termo_emergencia` (CRÍTICA), `termo_critico`, `dispneia_vocal`, `disartria_suspeita` (ALTA), `fadiga_vocal`, `sentimento_negativo` |

### 3.3 Dados clínicos estruturados (`monitor_multimodal/anomalias`)

Três famílias complementares, reutilizadas em todas as séries:

| Detector | Princípio | Para que serve |
|---|---|---|
| `DetectorRegras` | limiares clínicos por variável (faixa normal e crítica, derivadas do NEWS2) | valores absolutos perigosos, explicável |
| `DetectorZScoreDeslizante` | z robusto = 0,6745·(x − mediana)/MAD numa janela só com o passado; piso de escala = resolução do sensor | mudanças abruptas em relação ao histórico **do próprio paciente**; funciona em streaming |
| `DetectorIsolationForest` | floresta de isolamento multivariada (200 árvores) sobre variáveis padronizadas | combinações atípicas (ex.: FC↑ + PA↓ + SpO₂↓) mesmo com cada variável "quase normal" |

- **Sinais vitais** (`analisar_sinais_vitais`): regras + z-score + Isolation Forest + tendência de SpO₂ (regressão em janela) +
  **escore de alerta precoce** (NEWS2 simplificado; ≥ 5 risco médio, ≥ 7 alto). Quando o EWS ≥ 5, as variáveis alteradas vão
  como evidência do alerta de EWS em vez de alertas separados.
- **Tempo real** (`MonitorTempoReal`): recebe uma amostra por vez, mantém janela por paciente (24 h), aplica regras e z-score
  imediatamente e treina/retreina o Isolation Forest online (aquecimento 60 amostras, retreino a cada 120); latência de uma amostra.
- **Prescrições** (`analisar_prescricoes`): dose diária acima do máximo usual (CRÍTICA), aumento abrupto > 50 % (ALTA),
  interação medicamentosa crítica com fármaco ativo nas últimas 48 h (CRÍTICA, uma vez por par), duplicidade terapêutica e troca
  de via (ATENÇÃO) + Isolation Forest sobre (dose relativa, frequência, hora, Δdose).
- **Movimentação** (`analisar_movimentacao`): imobilidade ≥ 4 h sem mudança de decúbito (ALTA, risco de lesão por pressão),
  agitação noturna (ATENÇÃO), pico de atividade seguido de imobilidade fora do leito (CRÍTICA, padrão de queda), saída do leito
  com restrição (ALTA) + z-score / Isolation Forest.

## 4. Dados utilizados

Dados reais de pacientes (PhysioNet/MIMIC) exigem credenciamento e não podem ser redistribuídos, por isso a demonstração usa
**dados sintéticos com anomalias rotuladas** (`monitor_multimodal/dados_sinteticos.py`, semente fixa), o que permite medir
precisão, recall e tempo de antecipação:

| Paciente | Perfil | Sinais vitais (48 h, 5 min) | Prescrições (5 dias) | Movimentação (72 h) | Áudio | Vídeo |
|---|---|---|---|---|---|---|
| P001 | estável | ritmo circadiano + ruído | esquema fixo | diurno/noturno normal | consulta normal (TTS pt-BR) | `vtest.avi` — amostra pública do OpenCV, pessoas caminhando (enfermaria) |
| P002 | deterioração progressiva (sepse) | rampa de 12 h a partir da hora 24: FC +45, PAS −35, SpO₂ −8, FR +12, T +2,6 | dipirona 10× (erro de digitação), varfarina iniciada com heparina ativa, ceftriaxona dobrada | 7 h de imobilidade | fala lenta com pausas longas (fadiga) | — |
| P003 | episódios agudos | dessaturação 40 min, taquiarritmia 30 min, febre 4 h, 3 artefatos de sensor | duplicidade de prescritor, troca de via | agitação noturna + padrão de queda | fala entrecortada, "falta de ar", "dor no peito" (dispneia) | vídeo sintético de fisioterapia com queda aos 14 s e imobilidade |

Os áudios são gerados com o TTS do macOS (`say`, vozes pt-BR) com controle de velocidade e pausas; o vídeo sintético é
renderizado com OpenCV (figura humanoide fazendo agachamentos, queda e objeto entrando em cena). Os horários das mídias
coincidem com episódios dos dados estruturados para exercitar a corroboração multimodal.

## 5. Resultados obtidos

### 5.1 Detecção de anomalias × rótulos injetados

| Paciente | Fonte | Amostras | Positivos reais | Previstos | Precisão | Recall | F1 | FPR | AUC-ROC |
|---|---|---|---|---|---|---|---|---|---|
| P001 | sinais vitais | 576 | 0 | 25 | — | — | — | 4,3 % | — |
| P001 | movimentação | 72 | 0 | 2 | — | — | — | 2,8 % | — |
| P001 | prescrições | 25 | 0 | 2 | — | — | — | 8,0 % | — |
| **P002** | **sinais vitais** | 576 | 251 | 258 | **0,90** | **0,92** | **0,91** | 8,0 % | **0,965** |
| P002 | movimentação | 72 | 7 | 8 | 0,50 | 0,57 | 0,53 | 6,2 % | — |
| P002 | prescrições | 28 | 3 | 3 | **1,00** | **1,00** | **1,00** | 0 % | — |
| **P003** | **sinais vitais** | 576 | 66 | 100 | 0,66 | **1,00** | 0,80 | 6,7 % | **0,976** |
| P003 | movimentação | 72 | 3 | 3 | **1,00** | **1,00** | **1,00** | 0 % | — |
| P003 | prescrições | 27 | 2 | 3 | 0,67 | 1,00 | 0,80 | 4,0 % | — |

No paciente estável os únicos "positivos" são alertas de ATENÇÃO (estatísticos), nenhum CRÍTICO. Na movimentação do P002 a
imobilidade rotulada (7 h) é detectada a partir da 4ª hora por definição do limiar clínico, o que reduz a cobertura por amostra.

### 5.2 Tempo de antecipação (primeiro alerta × início do episódio rotulado)

| Paciente | Episódio | Início | Duração | Detectado | Antecipação | Cobertura |
|---|---|---|---|---|---|---|
| P002 | deterioração progressiva | 02/11 11:05 | 251 amostras (21 h) | sim | **85 min antes** | 92 % |
| P003 | dessaturação aguda | 01/11 18:00 | 9 amostras | sim | 85 min antes (tendência de SpO₂) | 100 % |
| P003 | taquiarritmia | 02/11 14:00 | 6 amostras | sim | 5 min antes | 100 % |
| P003 | febre | 03/11 00:00 | 48 amostras | sim | na 1ª amostra | 100 % |
| P003 | artefatos de sensor (3) | — | 1 amostra cada | sim (3/3) | — | 100 % |

No modo **tempo real** (`python main.py stream --paciente P002`), o monitor emite o primeiro alerta CRÍTICO (EWS ≥ 7) ainda na
manhã do 2º dia, com latência de uma amostra (5 min), e agrupa 352 repetições em 137 alertas.

### 5.3 Áudio

| Áudio | Taxa silábica | Pausas | Pausas/min | Δpitch (st) | HNR | Fadiga | Dispneia | Disartria | Sentimento | Termos críticos | Alertas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| consulta normal (P001) | 4,8 síl/s | 6 % | 15 | 2,0 | 11,7 dB | 0,15 | 0,21 | 0,21 | positivo | — | nenhum |
| consulta fadiga (P002) | 4,0 | 30 % | 17 | 2,1 | 13,4 | **0,72** | 0,27 | 0,29 | negativo | tontura, fraqueza, cansaço extremo | `termo_critico` (ALTA), `fadiga_vocal` |
| consulta dispneia (P003) | 3,3 | 37 % | 15 | 2,0 | **2,4** | 0,71 | **0,69** | 0,39 | negativo | dor no peito, falta de ar, não consigo respirar | `termo_emergencia` (CRÍTICA), `termo_critico`, `dispneia_vocal` (ALTA), `fadiga_vocal` |

As entidades clínicas extraídas (fallback local; com Azure, pelo Text Analytics for health) incluem `falta de ar`, `dor no
peito` (SymptomOrSign), `lábios` (BodyStructure), `desde ontem` (Time). A execução de referência foi feita **sem credenciais
Azure** (fallback local); com `AZURE_SPEECH_KEY`/`AZURE_LANGUAGE_KEY` no `.env`, o mesmo pipeline usa os serviços em nuvem
automaticamente e o relatório registra o provedor usado.

### 5.4 Vídeo

| Vídeo | Duração | Frames | Pessoas (máx.) | Eventos | Score |
|---|---|---|---|---|---|
| `fisioterapia_sintetica.mp4` (P003) | 24 s | 240 | 0 (figura sintética não é reconhecida como pessoa) | `pico_seguido_de_imobilidade` **CRÍTICA @ 14,1 s** (energia 8× a mediana seguida de ≥ 2 s imóvel), `imobilidade_prolongada` ALTA @ 19,0 s, 3× `movimento_atipico` | 1,00 |
| `vtest.avi` (P001, enfermaria) | 79,5 s | 398 | **10** | nenhum (após rastreamento da pessoa principal; antes, 3 falsas quedas por troca de pessoa) | 0,00 |

O YOLOv8 detecta corretamente pessoas e objetos no vídeo real (`person`, `backpack`, `car`, …) e o YOLOv8-pose fornece em
média 14 dos 17 keypoints por pessoa, permitindo as métricas posturais; o vídeo anotado está em `outputs/video/vtest/vtest_anotado.mp4`.

### 5.5 Fusão multimodal e alertas

| Paciente | Sinais vitais | Prescrições | Movimentação | Áudio | Vídeo | Score global | Nível | Corroborações | Alertas emitidos |
|---|---|---|---|---|---|---|---|---|---|
| P001 | 0,45 | 0,20 | 0,45 | 0,00 | 0,00 | **0,26** | **baixo** | 0 | 28 (nenhum CRÍTICO) |
| P002 | 1,00 | 1,00 | 0,75 | 0,75 | — | **1,00** | **crítico** | 2 | 80 (43 CRÍTICOS) |
| P003 | 1,00 | 0,20 | 0,95 | 1,00 | 1,00 | **1,00** | **crítico** | 6 | 66 (13 CRÍTICOS) |

Total: **172 alertas emitidos** (54 CRÍTICOS, 30 ALTA, 88 ATENÇÃO), 266 detecções agrupadas por deduplicação/persistência.

## 6. Exemplos de anomalias detectadas

1. **Deterioração progressiva (P002, sinais vitais)** — 02/11 09:40: `Mudança abrupta em Pressão sistólica (z robusto = 4,7)`
   (ATENÇÃO) → 11:xx: `Escore de alerta precoce = 5 — Frequência respiratória 21 (alterado), Temperatura 38,1 (alterado)` (ALTA)
   → 13:xx: `Escore de alerta precoce = 8 (≥7: risco alto) — FC 112, PAS 95, SpO₂ 92, FR 24, T 38,9` (CRÍTICA, recomendação:
   acionar time de resposta rápida). O primeiro alerta antecede em 85 min o limiar de rotulagem.
2. **Erro de prescrição (P002)** — `Dipirona: dose diária 20000 mg excede o máximo usual (4000 mg) em 400 %` (CRÍTICA,
   bloquear dispensação); `Interação crítica: varfarina prescrito com heparina ativo` (CRÍTICA); `Ceftriaxona: dose diária subiu 100 %
   (2000 → 4000 mg)` (ALTA).
3. **Corroboração sinais vitais + áudio (P003, 01/11 18:00–18:10)** — `Escore de alerta precoce = 5 — Saturação de O₂ 88 (crítico)`
   + `Paciente relata: dor no peito, não consigo respirar` (CRÍTICA) + `Padrão de fala compatível com dificuldade respiratória
   (indicador 0,69): HNR 2 dB, 37 % de pausas` → alerta de fusão `Evidências convergentes de 2 modalidades — risco global crítico`.
4. **Corroboração movimentação + vídeo + sinais vitais (P003, 03/11 02:10–03:00)** — `Pico de atividade seguido de imobilidade
   fora do leito — padrão compatível com queda` (sensor, CRÍTICA) + `[fisioterapia_sintetica.mp4 @ 14,1 s] Movimento brusco
   (8× a mediana) seguido de ≥ 2 s de imobilidade` (vídeo, CRÍTICA) + `Temperatura alterado: 38,6 °C (persistente)` → fusão de 3 modalidades.
5. **Fadiga vocal (P002, áudio)** — `4,0 síl/s, 30 % de pausas (média 1,07 s), variação de pitch 2,1 st` → `fadiga_vocal` 0,72,
   reforçado pelos termos "cansaço extremo", "fraqueza", "tontura" na transcrição (sentimento negativo 0,83).
6. **Imobilidade prolongada (P002, movimentação)** — `4h sem mudança de decúbito` (ALTA, realizar mudança de decúbito e avaliar pele),
   repetido e escalado às 8 h.
7. **Artefato de sensor (P003)** — leitura isolada de SpO₂ = 63 % → `Saturação de O₂ critico` (CRÍTICA) com recomendação de repetir a
   medida; a persistência (alerta único, sem repetição) ajuda a diferenciá-lo de uma dessaturação real.

## 7. Qualidade, reprodutibilidade e limitações

- **Testes**: 21 testes `pytest` (detectores, deduplicação/escalada de alertas, regras clínicas, streaming, áudio/texto, fusão, vídeo sem YOLO).
- **Reprodutibilidade**: semente fixa (`RANDOM_STATE = 42`), dependências fixadas em `requirements.txt`, `setup.sh`.
- **Limitações**: dados sintéticos (padrões plausíveis, mas não substituem validação clínica); indicadores vocais heurísticos;
  o vídeo sintético não é reconhecido pelo YOLO como pessoa (a pose é validada em `vtest.avi`); sem rastreamento multi-pessoa
  robusto (ByteTrack) nem re-identificação; os limiares de prescrição são simplificados; Azure Anomaly Detector não foi usado
  por ter sido descontinuado pela Microsoft.
- **Próximos passos**: ingestão HL7/FHIR do prontuário, validação com MIMIC-IV/PhysioNet (`wfdb`), modelos sequenciais para sinais
  vitais, rastreamento multi-pessoa, avaliação clínica dos limiares e dos pesos da fusão com a equipe médica.

## 8. Como reproduzir

```bash
./setup.sh && source .venv/bin/activate
python main.py demo                 # gera outputs/relatorio_geral.md, outputs/pacientes/*/relatorio.md, outputs/video/*/
python main.py dashboard            # painel
pytest -q
```
