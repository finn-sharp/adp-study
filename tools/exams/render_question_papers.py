"""Render all question-only PDFs in a consistent Korean administrative style.

Sources are the existing question manifests, never answer notebook outputs.
Run from any directory: python tools/exams/render_question_papers.py.
"""
from pathlib import Path
import argparse
import csv
import html
import json
import re
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
    CondPageBreak, KeepTogether, Image,
)

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '기출문제'
PAGE_W, PAGE_H = A4
MARGIN = 20 * mm
WIDTH = PAGE_W - 2 * MARGIN
INK = colors.HexColor('#151515')
GREY = colors.HexColor('#555555')
RULE = colors.HexColor('#999999')
PALE = colors.HexColor('#F3F3F3')


def styles():
    pdfmetrics.registerFont(TTFont('ExamGothic', str(ROOT / '공통/NanumGothic-Regular.ttf')))
    base = dict(fontName='ExamGothic', textColor=INK, wordWrap='CJK',
                splitLongWords=True, allowWidows=0, allowOrphans=0)
    return {
        'title': ParagraphStyle('title', **base, fontSize=21, leading=30,
                                alignment=TA_CENTER, spaceAfter=5),
        'subtitle': ParagraphStyle('subtitle', **base, fontSize=11, leading=18,
                                   alignment=TA_CENTER, spaceAfter=16),
        'section': ParagraphStyle('section', **base, fontSize=13, leading=21),
        'body': ParagraphStyle('body', **base, fontSize=11, leading=19,
                               spaceAfter=7),
        'small': ParagraphStyle('small', **base, fontSize=9, leading=15,
                                spaceAfter=3),
        'label': ParagraphStyle('label', **base, fontSize=9, leading=15),
        'question': ParagraphStyle('question', **base, fontSize=11, leading=19,
                                   leftIndent=35, firstLineIndent=-35, spaceAfter=15),
        'table': ParagraphStyle('table', **base, fontSize=8, leading=12),
        'caption': ParagraphStyle('caption', **base, fontSize=8, leading=13,
                                  spaceBefore=6, spaceAfter=5),
    }


ST = None


def p(text, style='body'):
    return Paragraph(html.escape(str(text)).replace('\n', '<br/>'), ST[style])


def ruled_table(rows, widths, header=True):
    table = Table([[p(v, 'table') for v in row] for row in rows], colWidths=widths,
                  hAlign='LEFT', repeatRows=1 if header else 0)
    commands = [
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('LINEABOVE', (0, 0), (-1, 0), .7, INK),
        ('LINEBELOW', (0, -1), (-1, -1), .7, INK),
        ('INNERGRID', (0, 0), (-1, -1), .3, RULE),
    ]
    if header:
        commands.extend([('BACKGROUND', (0, 0), (-1, 0), PALE),
                         ('LINEBELOW', (0, 0), (-1, 0), .4, RULE)])
    table.setStyle(TableStyle(commands))
    return table


def paper_header(folder, exam, count):
    parts = folder.split('/')
    if parts[0] == '모의고사':
        subtitle = '모의고사 제' + parts[1]
    else:
        subtitle = '제' + parts[0]
        if len(parts) > 1:
            subtitle += ' · ' + parts[1]
    paper_title = '분석전문가(ADP) 실기'
    rows = [
        [p('구    분', 'label'), p('복원 연습 문제지', 'label'),
         p('문 항 수', 'label'), p(str(count) + '문항', 'label')],
        [p('성    명', 'label'), p('', 'label'), p('작 성 일', 'label'),
         p('______년 ____월 ____일', 'label')],
    ]
    identity = Table(rows, colWidths=[54, WIDTH / 2 - 54, 54, WIDTH / 2 - 54])
    identity.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), .4, RULE),
        ('BACKGROUND', (0, 0), (0, -1), PALE),
        ('BACKGROUND', (2, 0), (2, -1), PALE),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ]))
    notice = ('본 자료는 학습용 복원 문제지이며 공식 시험지가 아닙니다. '
              '별도 풀이 노트북과 해당 회차의 data 폴더를 함께 사용하십시오.')
    return [p(paper_title, 'title'), p(subtitle + '  |  복원 연습 문제지', 'subtitle'),
            identity, Spacer(1, 10), p('응시 안내', 'label'), p(notice, 'small'),
            p(exam.get('note', ''), 'small'),
            p('자료 출처: ' + exam['source'], 'small'), Spacer(1, 17)]


def section_heading(title):
    title = re.sub(r'^(\d+)번\s*\|\s*', r'\1. ', title)
    title = re.sub(r'^(\d+)번$', r'\1. 분석 과제', title)
    bar = Table([[p(title, 'section')]], colWidths=[WIDTH])
    bar.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PALE),
        ('LINEABOVE', (0, 0), (-1, -1), .8, INK),
        ('LINEBELOW', (0, 0), (-1, -1), .4, RULE),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ]))
    bar.keepWithNext = True
    return bar


def data_preview(dest, name):
    path = dest / 'data' / name.split(' + ')[0]
    if not path.is_file() or path.suffix.lower() != '.csv':
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        rows = []
        for _, row in zip(range(4), reader):
            rows.append(row[:6])
    if not rows or not rows[0]:
        return []
    size = len(rows[0])
    for row in rows[1:]:
        for i, value in enumerate(row):
            try:
                row[i] = format(float(value), '.6g')
            except ValueError:
                row[i] = value[:45]
        row.extend([''] * (size - len(row)))
    caption = '〈자료 예시〉 첫 3행·최대 6열, 숫자는 표시용 반올림'
    label = p(caption, 'caption')
    label.keepWithNext = True
    return [label, ruled_table(rows, [WIDTH / size] * size), Spacer(1, 12)]


def render_paper(folder, exam):
    """Write one question PDF, retaining original givens, tables and figures."""
    global ST
    if ST is None:
        ST = styles()
    dest = ROOT / folder
    count = sum(len(s['questions']) for s in exam['sections'])
    story = paper_header(folder, exam, count)
    for index, section in enumerate(exam['sections']):
        if index:
            story.extend([Spacer(1, 18), CondPageBreak(155)])
        section_story = []
        lead = [section_heading(section['title']), Spacer(1, 10)]
        if section.get('data'):
            lead.append(p('제공 자료: ' + section['data'], 'small'))
        for line in section.get('info', []):
            lead.append(p(line, 'small'))
        if section.get('condition'):
            lead.append(p('분석 조건: ' + section['condition'], 'small'))
        # Only heading + first info line are tied, allowing long givens to flow.
        for flowable in lead[:-1]:
            flowable.keepWithNext = True
        section_story.extend(lead)
        if section.get('table'):
            rows = section['table']
            section_story.extend([p('〈제시 자료〉', 'caption'),
                          ruled_table(rows, [WIDTH / len(rows[0])] * len(rows[0])),
                          Spacer(1, 12)])
        if section.get('data'):
            section_story.extend(data_preview(dest, section['data']))
        section_story.append(Spacer(1, 8))
        intro_count = len(section_story)
        for question in section['questions']:
            label, text = question['id'], question['text']
            # Keep identifier on a dedicated line for the longer legacy labels.
            if len(label) > 7:
                block = [p('[' + label + ']', 'label'), p(text)]
            else:
                markup = html.escape(label) + '.  ' + html.escape(text).replace('\n', '<br/>')
                block = [Paragraph(markup, ST['question'])]
            image = section.get('images', {}).get(label)
            if image:
                image_path = dest / image
                if not image_path.exists():
                    image_path = dest / '원문그림' / image
                if not image_path.exists():
                    raise FileNotFoundError(image_path)
                figure = Image(str(image_path))
                scale = min(WIDTH / figure.imageWidth, 160 / figure.imageHeight, 1)
                figure.drawWidth = figure.imageWidth * scale
                figure.drawHeight = figure.imageHeight * scale
                figure.hAlign = 'LEFT'
                block.extend([figure, Spacer(1, 15)])
            question_block = Table([[block]], colWidths=[WIDTH], hAlign='LEFT')
            question_block.setStyle(TableStyle([
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                ('TOPPADDING', (0, 0), (-1, -1), 0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ]))
            section_story.append(question_block)
        group = KeepTogether(section_story)
        group.wrapOn(Canvas(BytesIO()), WIDTH, PAGE_H)
        if index and group._H <= 350:
            story.append(group)
        else:
            story.append(KeepTogether(section_story[:intro_count + 1]))
            story.extend(section_story[intro_count + 1:])

    stem = ('ADP_' + folder.replace('/', '_')) if '모의고사' not in folder else folder.replace('/', '_')
    output = dest / (stem + '_문제지.pdf')

    def page_furniture(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(GREY)
        canvas.setFont('ExamGothic', 8)
        canvas.drawString(MARGIN, PAGE_H - 14 * mm, '분석전문가(ADP) 실기 · ' + folder.replace('/', ' / '))
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 14 * mm, '학습용 복원 자료')
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(.35)
        canvas.line(MARGIN, PAGE_H - 17 * mm, PAGE_W - MARGIN, PAGE_H - 17 * mm)
        canvas.line(MARGIN, 18 * mm, PAGE_W - MARGIN, 18 * mm)
        canvas.drawString(MARGIN, 13 * mm, '복원 연습 문제지')
        canvas.drawCentredString(PAGE_W / 2, 13 * mm, '- %d -' % doc.page)
        canvas.drawRightString(PAGE_W - MARGIN, 13 * mm, '공식 시험지 아님')
        canvas.restoreState()

    doc = BaseDocTemplate(str(output), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=24 * mm, bottomMargin=24 * mm,
                          title='ADP ' + folder + ' 복원 연습 문제지',
                          author='ADP Study', pageCompression=1)
    frame = Frame(MARGIN, doc.bottomMargin, WIDTH, PAGE_H - doc.topMargin - doc.bottomMargin,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id='official', frames=[frame], onPage=page_furniture))
    doc.build(story)
    return output


def all_exams():
    original = json.loads((ROOT / '문항목록.json').read_text())
    for number, sections in original.items():
        copied = []
        for section in sections:
            converted = dict(section)
            converted['questions'] = [dict(id=q[0], text=q[1]) for q in section['questions']]
            copied.append(converted)
        yield number + '회', dict(source='DataManim ADP ' + number + '회 실기 문제 PDF',
                                  note='첨부 자료의 문항·표·그림을 바탕으로 정리한 학습용 자료입니다.',
                                  sections=copied)
    additional = json.loads((ROOT / '추가_문항명세.json').read_text())
    for folder, exam in additional.items():
        yield folder, exam


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', action='append', help='Render selected folder(s) only.')
    args = parser.parse_args()
    for folder, exam in all_exams():
        if not args.folder or folder in args.folder:
            print(render_paper(folder, exam).relative_to(REPO))
