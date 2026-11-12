"""Gera o PDF de entrega do Tech Challenge Fase IV no estilo do PDF da Fase III."""
import sys
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

destino = sys.argv[1]
YT = "https://www.youtube.com/watch?v=5hJHO8rMMk4"
GH = "https://github.com/valter-oliveira/tech_challenge_fase4/tree/main"
LEAD = 9.7
st = {
    "h0": ParagraphStyle("h0", fontName="Helvetica-Bold", fontSize=13, leading=16),
    "meta": ParagraphStyle("meta", fontName="Helvetica", fontSize=9, leading=12),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=10, leading=13, spaceBefore=6, spaceAfter=2),
    "p": ParagraphStyle("p", fontName="Helvetica", fontSize=8, leading=LEAD, alignment=TA_JUSTIFY, spaceAfter=2.5),
    "i": ParagraphStyle("i", fontName="Helvetica-Oblique", fontSize=8, leading=LEAD, alignment=TA_JUSTIFY, spaceAfter=2.5),
    "li": ParagraphStyle("li", fontName="Helvetica", fontSize=8, leading=LEAD, alignment=TA_JUSTIFY, spaceAfter=2),
    "nome": ParagraphStyle("nome", fontName="Helvetica", fontSize=9, leading=12, leftIndent=10),
}
P = lambda t, s="p": Paragraph(t, st[s])

doc = SimpleDocTemplate(destino, pagesize=letter, leftMargin=62, rightMargin=62, topMargin=44, bottomMargin=32,
                        title="Tech Challenge Fase IV - Monitoramento Clínico Multimodal", author="Grupo 4 - 9IADT")
f = [P("FIAP", "h0"),
     P("Tech Challenge Fase IV — Monitoramento Clínico Multimodal com Detecção de Anomalias", "h0"),
     Spacer(1, 6),
     P("<b>Curso/Turma:</b> 9IADT", "meta"), P("<b>Grupo:</b> Grupo 4", "meta"),
     P("<b>Projeto:</b> Monitor Multimodal — Hospital Vida Plena (cenário fictício)", "meta"),
     P("Integrantes", "h1")]
for n in ["Alexandre Carneiro do Carmo - rm370980", "Brunno Costa Castigrini - rm371429",
          "Pedro Henrique Azevedo Aragão - rm373481", "Valter Willian de Oliveira Filho - rm370979"]:
    f.append(P(n, "nome"))

f += [P("O Projeto", "h1"),
      P("O Monitor Multimodal acompanha pacientes internados de forma contínua a partir de três tipos de dado: <b>vídeo</b> de "
        "sessões de fisioterapia e do leito, <b>áudio</b> das consultas e <b>dados clínicos</b> (sinais vitais, prescrições e "
        "movimentação). Cada fonte é analisada por modelos próprios; as evidências são fundidas em um score de risco por paciente "
        "e viram alertas com severidade, evidência e recomendação para a equipe médica, entregues no painel, em log estruturado, "
        "por webhook (Teams/Slack) e por e-mail. O sistema não decide por ninguém: aponta o desvio, mostra a evidência (valor, "
        "z-score, trecho da transcrição, frame do vídeo) e devolve a decisão ao profissional. Todo o fluxo roda em um computador "
        "comum, em cerca de 35 segundos para três pacientes, e integra os serviços gerenciados <b>Azure Speech to Text</b> e "
        "<b>Azure AI Language</b>, com fallback local quando não há credenciais."),
      P("Arquitetura do Projeto", "h1"),
      P("O código é um pacote Python modularizado (<b>monitor_multimodal/</b>), com uma CLI (<b>main.py</b>) e um painel Streamlit "
        "(<b>dashboard/</b>). As modalidades são independentes e se encontram apenas na camada de alertas e fusão, de modo que um "
        "paciente sem vídeo ou sem áudio é analisado normalmente com o que existe. Cinco camadas:")]
itens = [
    ("Análise de vídeo (video/):", " energia de movimento e fluxo óptico de Farnebäck com OpenCV; <b>YOLOv8</b> para pessoas e objetos "
     "em áreas críticas; <b>YOLOv8-pose</b> com os 17 keypoints COCO (o mesmo esqueleto do OpenPose, rodando em CPU) para altura do "
     "quadril, ângulo do tronco, ângulos dos joelhos e assimetria, com rastreamento da pessoa acompanhada. Gera eventos com timestamp "
     "(queda, imobilidade, quebra de ritmo, assimetria, objeto crítico), frames-chave, vídeo anotado e um relatório automático por vídeo."),
    ("Análise de áudio e serviços Azure (audio/, azure_servicos/):", " biomarcadores vocais extraídos com librosa (taxa silábica, "
     "pausas, pitch, jitter, shimmer, relação harmônico/ruído) convertidos em indicadores de fadiga, dispneia e disartria. A fala é "
     "transcrita pelo <b>Azure Speech to Text</b> (reconhecimento contínuo pt-BR) e o texto passa pelo <b>Azure AI Language</b>: "
     "sentimento, frases-chave e entidades clínicas do Text Analytics for health, além da busca de termos críticos como “falta de ar” "
     "e “dor no peito”. Sem chaves, usa faster-whisper ou a transcrição de referência e um léxico clínico local."),
    ("Detecção de anomalias (anomalias/):", " três detectores complementares reutilizados em todas as séries — regras clínicas "
     "derivadas do NEWS2 (explicáveis), z-score robusto em janela deslizante sobre o histórico do próprio paciente e Isolation Forest "
     "multivariado —, mais tendência de SpO<sub>2</sub> e escore de alerta precoce. Um <b>monitor em tempo real</b> processa uma amostra por vez "
     "com latência de uma amostra. Prescrições: dose máxima, aumento abrupto, interação, duplicidade e troca de via; movimentação: "
     "imobilidade prolongada, agitação noturna e padrão de queda."),
    ("Alertas e fusão multimodal (alertas/, fusao/):", " o gerenciador deduplica alertas iguais em 30 minutos, escala a severidade "
     "quando a anomalia persiste e despacha para console, JSONL, webhook e e-mail. A fusão tardia pondera o score de cada fonte e "
     "detecta <b>corroboração multimodal</b> — alertas graves de modalidades distintas na mesma janela de tempo —, que eleva o risco "
     "global e gera um alerta crítico próprio."),
    ("Relatórios, painel e dados (relatorios/, dashboard/, dados_sinteticos.py):", " relatórios Markdown/JSON com gráficos por paciente, "
     "por vídeo e da execução; painel com visão geral, detalhe por paciente, simulação em tempo real e análise de arquivos enviados. "
     "Como dados reais de pacientes exigem credenciamento, a demonstração usa dados sintéticos com anomalias rotuladas, o que permite "
     "medir precisão, recall e tempo de antecipação."),
]
f.append(ListFlowable([ListItem(P(f"<b>{a}</b>{b}", "li"), leftIndent=14) for a, b in itens],
                      bulletType="1", bulletFontName="Helvetica-Bold", bulletFontSize=8, leftIndent=14, bulletFormat="%s."))
f += [P("Validação e Resultados", "h1"),
      P("A suíte tem 21 testes, roda em cerca de 30 segundos e não exige GPU: cobre os detectores, a deduplicação e a escalada dos "
        "alertas, as regras clínicas e farmacológicas, o monitor em tempo real, os biomarcadores vocais, a análise de texto e a fusão. "
        "Na execução de referência, o paciente estável ficou em risco baixo sem nenhum alerta crítico; o paciente com deterioração "
        "progressiva teve <b>F1 de 0,91</b> (AUC 0,97) nos sinais vitais, com o primeiro alerta <b>85 minutos antes</b> do limiar de "
        "rotulagem e as três prescrições anômalas detectadas sem falso positivo; o paciente com episódios agudos teve recall 1,0 em "
        "todos os episódios, a queda no vídeo detectada aos 14,1 s e corroborada pelo sensor de movimentação, e a dispneia identificada "
        "tanto na acústica da voz quanto nos termos da transcrição. A documentação do repositório traz o README (instalação e comandos), "
        "o <b>relatório técnico</b> (fluxo multimodal, modelos por tipo de dado, resultados, exemplos de anomalias e limitações), a "
        "arquitetura com diagrama e o roteiro do vídeo."),
      P("Trabalho acadêmico. Não passou por comitê de ética nem por validação clínica, e não deve ser usado em assistência a "
        "pacientes. Nenhum dado real foi utilizado.", "i"),
      P("URLs do Projeto", "h1"),
      P(f'<b>Youtube:</b> <link href="{YT}">{YT}</link>', "meta"),
      P(f'<b>Github:</b> <link href="{GH}">{GH}</link>', "meta")]
doc.build(f)
print("ok", destino)
