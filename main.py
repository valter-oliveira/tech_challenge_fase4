#!/usr/bin/env python
"""
main.py — Interface de linha de comando do monitoramento multimodal.

Exemplos:
  python main.py gerar-dados                      # dados sintéticos (CSV, áudios TTS, vídeo)
  python main.py baixar-amostras                  # vídeo público de amostra (OpenCV) para testar YOLO/pose
  python main.py demo                             # pipeline completo + relatórios em outputs/
  python main.py sinais-vitais --paciente P002    # só sinais vitais
  python main.py stream --paciente P003 --intervalo 0.05   # simulação em tempo real
  python main.py audio dados/sinteticos/gerados/audio/consulta_dispneia.wav --paciente P003
  python main.py video meu_video.mp4 --paciente P001 --sessao fisioterapia
  python main.py dashboard                        # abre o painel Streamlit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from monitor_multimodal import config
from monitor_multimodal.utils import log, salvar_json


def _carregar(nome: str) -> pd.DataFrame:
    caminho = config.DADOS_SINTETICOS_DIR / f"{nome}.csv"
    if not caminho.exists():
        from monitor_multimodal import dados_sinteticos
        dados_sinteticos.gerar_tudo()
    return pd.read_csv(caminho, parse_dates=["timestamp"])


def cmd_gerar_dados(args):
    from monitor_multimodal import dados_sinteticos
    saidas = dados_sinteticos.gerar_tudo(forcar=True)
    for k, v in saidas.items():
        print(f"  {k:22s} → {v}")


def cmd_baixar_amostras(args):
    """Baixa o vídeo de amostra público do OpenCV (pessoas caminhando) para testar detecção/pose."""
    import urllib.request
    destino = config.DADOS_AMOSTRAS_DIR / "vtest.avi"
    if destino.exists():
        print(f"já existe: {destino}")
        return
    url = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"
    print(f"baixando {url} (~8 MB)…")
    urllib.request.urlretrieve(url, destino)
    print(f"salvo em {destino}")


def cmd_demo(args):
    from monitor_multimodal.pipeline import executar_demo
    out = executar_demo(pacientes=args.paciente, com_video=not args.sem_video, com_audio=not args.sem_audio,
                        gerar_dados=not args.sem_gerar, max_frames_video=args.max_frames, modo_servicos=args.modo)
    print("\n" + "═" * 70)
    for p in out["perfis"]:
        print(f"  {p.paciente_id}: risco {p.nivel.upper():9s} ({p.score_global:.2f})  fontes={ {k: round(v, 2) for k, v in p.scores_por_fonte.items()} }")
    print(f"\n  Alertas emitidos: {out['gerenciador'].resumo()['total']} → outputs/alertas.jsonl")
    print(f"  Relatório geral : {out['relatorio']}")
    print(f"  Painel          : python main.py dashboard")


def _emitir_e_mostrar(alertas, resumo):
    from monitor_multimodal.alertas import GerenciadorAlertas
    g = GerenciadorAlertas()
    emitidos = g.emitir_lote(sorted(alertas, key=lambda a: a.timestamp))
    print(f"\n{len(alertas)} detecções → {len(emitidos)} alertas emitidos ({g.suprimidos} agrupados)")
    print("Resumo:", resumo)


def cmd_sinais_vitais(args):
    from monitor_multimodal.anomalias.sinais_vitais import analisar_sinais_vitais
    df = pd.read_csv(args.arquivo, parse_dates=["timestamp"]) if args.arquivo else _carregar("sinais_vitais")
    if "paciente_id" in df:
        df = df[df.paciente_id == args.paciente]
    r = analisar_sinais_vitais(df, args.paciente)
    _emitir_e_mostrar(r.alertas, r.resumo)
    r.dados.to_csv(config.OUTPUTS_DIR / f"sinais_vitais_{args.paciente}.csv", index=False)


def cmd_prescricoes(args):
    from monitor_multimodal.anomalias.prescricoes import analisar_prescricoes
    df = pd.read_csv(args.arquivo, parse_dates=["timestamp"]) if args.arquivo else _carregar("prescricoes")
    if "paciente_id" in df:
        df = df[df.paciente_id == args.paciente]
    r = analisar_prescricoes(df, args.paciente)
    _emitir_e_mostrar(r.alertas, r.resumo)


def cmd_movimentacao(args):
    from monitor_multimodal.anomalias.movimentacao import analisar_movimentacao
    df = pd.read_csv(args.arquivo, parse_dates=["timestamp"]) if args.arquivo else _carregar("movimentacao")
    if "paciente_id" in df:
        df = df[df.paciente_id == args.paciente]
    r = analisar_movimentacao(df, args.paciente, restricao_leito=args.restricao_leito)
    _emitir_e_mostrar(r.alertas, r.resumo)


def cmd_stream(args):
    from monitor_multimodal.alertas import GerenciadorAlertas
    from monitor_multimodal.anomalias.stream import MonitorTempoReal, simular_stream
    df = pd.read_csv(args.arquivo, parse_dates=["timestamp"]) if args.arquivo else _carregar("sinais_vitais")
    if "paciente_id" in df:
        df = df[df.paciente_id == args.paciente]
    g = GerenciadorAlertas(caminho_log=config.OUTPUTS_DIR / "alertas_stream.jsonl")
    monitor = MonitorTempoReal(g)
    n = 0
    for amostra, alertas in simular_stream(df, args.paciente, monitor, intervalo_s=args.intervalo):
        n += 1
        if args.verboso and not alertas:
            print(f"  {amostra['timestamp']}  FC={amostra.get('frequencia_cardiaca'):.0f} PA={amostra.get('pressao_sistolica'):.0f}/"
                  f"{amostra.get('pressao_diastolica'):.0f} SpO2={amostra.get('saturacao_o2'):.0f} FR={amostra.get('frequencia_respiratoria'):.0f} "
                  f"T={amostra.get('temperatura'):.1f}")
    print(f"\n{n} amostras processadas → {len(g.alertas)} alertas ({g.suprimidos} agrupados): {g.resumo()['por_severidade']}")


def cmd_audio(args):
    from monitor_multimodal.audio import analisar_audio
    r = analisar_audio(args.arquivo, args.paciente, modo=args.modo)
    print(f"\nTranscrição [{r.transcricao.provedor}]: {r.transcricao.texto}")
    print(f"Sentimento [{r.texto.provedor}]: {r.texto.sentimento} {r.texto.scores_sentimento}")
    print(f"Entidades: {[(e['texto'], e['categoria']) for e in r.texto.entidades]}")
    print(f"Termos críticos: {r.texto.termos_criticos}")
    print(f"Indicadores vocais: {r.resumo['indicadores']}")
    _emitir_e_mostrar(r.alertas, r.resumo)
    destino = salvar_json(r.to_dict(), config.OUTPUTS_DIR / "audio" / f"{Path(args.arquivo).stem}.json")
    print(f"Resultado: {destino}")


def cmd_video(args):
    from monitor_multimodal.relatorios import gerar_relatorio_video
    from monitor_multimodal.video import analisar_video
    r = analisar_video(args.arquivo, args.paciente, tipo_sessao=args.sessao, usar_yolo=not args.sem_yolo,
                       max_frames=args.max_frames, passo=args.passo)
    rel = gerar_relatorio_video(r)
    for ev in r.eventos:
        print(f"  {ev['tempo_s']:7.2f}s  {ev['severidade']:8s} {ev['tipo']:28s} {ev['descricao'][:90]}")
    _emitir_e_mostrar(r.alertas, r.resumo)
    print(f"Relatório: {rel}\nVídeo anotado: {r.video_anotado}")


def cmd_dashboard(args):
    app = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app), "--server.headless", "false"], check=False)


def main(argv=None):
    p = argparse.ArgumentParser(description="Monitoramento clínico multimodal — Tech Challenge Fase 4")
    sub = p.add_subparsers(dest="comando", required=True)

    sub.add_parser("gerar-dados", help="gera dados sintéticos (CSV, áudios TTS, vídeo)").set_defaults(fn=cmd_gerar_dados)
    sub.add_parser("baixar-amostras", help="baixa vídeo de amostra público (OpenCV vtest.avi)").set_defaults(fn=cmd_baixar_amostras)

    s = sub.add_parser("demo", help="pipeline completo + relatórios")
    s.add_argument("--paciente", nargs="*", default=None)
    s.add_argument("--sem-video", action="store_true")
    s.add_argument("--sem-audio", action="store_true")
    s.add_argument("--sem-gerar", action="store_true", help="não regenera os dados sintéticos")
    s.add_argument("--max-frames", type=int, default=None, help="limita frames por vídeo (execução rápida)")
    s.add_argument("--modo", choices=["auto", "azure", "local"], default=None, help="provedor dos serviços cognitivos")
    s.set_defaults(fn=cmd_demo)

    for nome, fn in (("sinais-vitais", cmd_sinais_vitais), ("prescricoes", cmd_prescricoes), ("movimentacao", cmd_movimentacao)):
        s = sub.add_parser(nome)
        s.add_argument("--paciente", default="P002")
        s.add_argument("--arquivo", default=None, help="CSV próprio (padrão: dados sintéticos)")
        if nome == "movimentacao":
            s.add_argument("--restricao-leito", action="store_true")
        s.set_defaults(fn=fn)

    s = sub.add_parser("stream", help="simula monitoramento em tempo real")
    s.add_argument("--paciente", default="P002")
    s.add_argument("--arquivo", default=None)
    s.add_argument("--intervalo", type=float, default=0.0, help="segundos entre amostras (ex.: 0.1)")
    s.add_argument("--verboso", action="store_true")
    s.set_defaults(fn=cmd_stream)

    s = sub.add_parser("audio", help="analisa um áudio de consulta")
    s.add_argument("arquivo")
    s.add_argument("--paciente", default="P000")
    s.add_argument("--modo", choices=["auto", "azure", "local"], default=None)
    s.set_defaults(fn=cmd_audio)

    s = sub.add_parser("video", help="analisa um vídeo clínico")
    s.add_argument("arquivo")
    s.add_argument("--paciente", default="P000")
    s.add_argument("--sessao", default="fisioterapia", choices=["fisioterapia", "cirurgia", "leito", "enfermaria"])
    s.add_argument("--sem-yolo", action="store_true")
    s.add_argument("--max-frames", type=int, default=None)
    s.add_argument("--passo", type=int, default=config.VIDEO_PASSO_FRAMES)
    s.set_defaults(fn=cmd_video)

    sub.add_parser("dashboard", help="abre o painel Streamlit").set_defaults(fn=cmd_dashboard)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
