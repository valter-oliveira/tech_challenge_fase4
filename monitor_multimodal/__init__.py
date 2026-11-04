"""
monitor_multimodal — Monitoramento clínico multimodal com detecção de anomalias.

Tech Challenge Fase 4 — PosTech FIAP (IA para Devs)

Módulos:
  video          : análise de vídeos clínicos (YOLOv8 detecção + pose, energia de movimento)
  audio          : biomarcadores vocais + Azure Speech to Text + Azure AI Language
  anomalias      : detectores (z-score, Isolation Forest, regras clínicas) para sinais vitais,
                   prescrições e padrões de movimentação; monitor em tempo real
  alertas        : gerenciador de alertas (níveis, deduplicação, canais)
  fusao          : fusão multimodal → score de risco por paciente
  relatorios     : relatórios automáticos (Markdown / JSON / gráficos)
  azure_servicos : clientes Azure com fallback local
"""

__version__ = "1.0.0"
