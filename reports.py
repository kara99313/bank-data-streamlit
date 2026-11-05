"""Rapports téléchargeables, sans service externe ni décision automatique."""
from datetime import datetime, timezone
from uuid import uuid4
from io import BytesIO
from html import escape
import json
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, LongTable, TableStyle, PageBreak


def report_context(metadata):
    return {'reference':'BANK-'+uuid4().hex[:12].upper(),
            'date_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'metadata':metadata}


def html_report(result, context):
    m=context['metadata']; e=lambda x:escape(str(x))
    # Chaque donnée est échappée par to_html, y compris le contenu des CSV.
    metrics=''.join(f'<tr><th scope="row">{e(k)}</th><td>{float(v):.4f}</td></tr>' for k,v in m['test_metrics'].items())
    limits=''.join('<li>'+e(s)+'</li>' for s in m['limitations'])
    return f'''<!doctype html><html lang="fr"><meta charset="utf-8"><title>Rapport Bank Data</title>
<style>body{{font:15px Arial;margin:35px;color:#17324d}}table{{border-collapse:collapse;margin:16px 0;width:100%}}th,td{{border:1px solid #b8c8d8;padding:7px;text-align:left;overflow-wrap:anywhere}}th{{background:#eaf1f7}}.scroll{{overflow-x:auto}}@media print{{body{{margin:10mm}}}}</style>
<h1>Bank Data - AfriAI Solutions</h1><p>Référence : {e(context['reference'])}<br>Date UTC : {e(context['date_utc'])}</p>
<p>Modèle : {e(m['selected_model'])} ; clients : {len(result)}.<br>Entraînement UTC : {e(m['created_utc'])}</p>
<h2>Données soumises et résultats</h2><p>Oui = souscription prédite ; Non = absence de souscription prédite. Les pourcentages sont des estimations non calibrées.</p>
<div class="scroll">{result.to_html(index=False,escape=True,border=0)}</div>
<h2>Performances de référence sur le test</h2><p>Mesures issues du notebook, distinctes des prédictions ci-dessus.</p>
<table><caption>Évaluation du modèle retenu</caption><thead><tr><th scope="col">Mesure</th><th scope="col">Valeur</th></tr></thead><tbody>{metrics}</tbody></table>
<h2>Limites</h2><ul>{limits}</ul><p>Usage pédagogique. Aucune décision bancaire automatique ni certification de conformité.</p>
<h2>Traçabilité</h2><p>SHA-256 des données : {e(m['data_sha256'])}<br>Versions : {e(json.dumps(m['versions'],ensure_ascii=False))}<br>Découpage : {e(m['split'])}. Sélection : {e(m['selection'])}.</p>
</html>'''.encode('utf-8')


def pdf_report(result, context):
    fonts=Path(__file__).resolve().parent/'fonts'
    if 'BankSans' not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont('BankSans',str(fonts/'DejaVuSans.ttf')))
        pdfmetrics.registerFont(TTFont('BankSansBold',str(fonts/'DejaVuSans-Bold.ttf')))
    m=context['metadata']; stream=BytesIO()
    doc=SimpleDocTemplate(stream,pagesize=landscape(A4),rightMargin=30,leftMargin=30,topMargin=32,bottomMargin=32)
    styles=getSampleStyleSheet()
    for style in styles.byName.values():style.fontName='BankSans'
    styles['Title'].fontName='BankSansBold';styles['Heading2'].fontName='BankSansBold'
    normal=styles['BodyText'];normal.fontSize=9;normal.leading=12
    p=lambda s:Paragraph(escape(str(s)),normal)
    story=[Paragraph('Bank Data - AfriAI Solutions',styles['Title']),p('Référence : '+context['reference']),
           p('Date UTC : '+context['date_utc']),p(f"Modèle : {m['selected_model']} | Clients : {len(result)}"),
           p('Oui = souscription prédite. Probabilités estimées non calibrées. Usage pédagogique.'),Spacer(1,12)]
    rows=[[p('Mesure test'),p('Valeur')]]+[[p(k),p(f'{float(v):.4f}')] for k,v in m['test_metrics'].items()]
    t=LongTable(rows,colWidths=[240,150],repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf1f7')),('GRID',(0,0),(-1,-1),.4,colors.grey),('VALIGN',(0,0),(-1,-1),'TOP')]))
    story.extend([Paragraph('Performances sur le test du notebook',styles['Heading2']),t,Spacer(1,10)])
    for s in m['limitations']:story.append(p('- '+s))
    story.extend([Spacer(1,8),p('Entraînement UTC : '+m['created_utc']),p('SHA-256 données : '+m['data_sha256']),
                  p('Versions : '+json.dumps(m['versions'],ensure_ascii=False)),p('Sélection : '+m['selection'])])
    # Les tableaux larges sont divisés en blocs de colonnes ; chaque ligne garde son identifiant.
    columns=list(result.columns)
    for start in range(0,len(columns),5):
        selected=columns[start:start+5]
        story.extend([PageBreak(),Paragraph('Données et résultats - colonnes '+str(start+1)+' à '+str(start+len(selected)),styles['Heading2'])])
        data=[[p('Ligne')]+[p(c) for c in selected]]
        for i,(_,row) in enumerate(result.iterrows(),1):data.append([p(i)]+[p(row[c]) for c in selected])
        widths=[45]+[(landscape(A4)[0]-105)/len(selected)]*len(selected)
        table=LongTable(data,colWidths=widths,repeatRows=1)
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf1f7')),('GRID',(0,0),(-1,-1),.3,colors.grey),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7fa')])]))
        story.append(table)
    def footer(canvas, doc):
        canvas.setFont('BankSans',8);canvas.drawString(30,18,context['reference']);canvas.drawRightString(landscape(A4)[0]-30,18,f'Page {doc.page}')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return stream.getvalue()
