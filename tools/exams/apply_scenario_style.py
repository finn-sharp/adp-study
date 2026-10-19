"""22개 기존 자료를 원문 명세와 분리한 상황형 문제·풀이로 재생성한다.

원문 manifest는 읽기 전용이다. 수작업 blueprint와 문항 ID로 1:1 대응한다.
셀에 원래 source를 보존하여 재실행해도 코드나 가설 보고가 중복되지 않는다.
"""
from pathlib import Path
import copy
import json
import re
import nbformat
from scenario_blueprint import BLUEPRINT
from scenario_answers import EXTRA, RATIONALE

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '기출문제'


def apply():
    from render_question_papers import source_exams, render_paper
    manifest = {}
    for folder, original in source_exams():
        blueprint = BLUEPRINT[folder]
        assert len(blueprint) == len(original['sections']), folder
        exam = copy.deepcopy(original)
        exam['source_note'] = exam.get('note', '')
        exam['note'] = ('기출·모의 자료를 바탕으로 질문과 답안 요구를 재설계한 상황형 연습 자료입니다. '
                        '공식 기출 원문이 아니며 기존 데이터의 출처·합성 여부는 원문 명세를 따릅니다. '
                        '통계 검정은 별도 언급이 없으면 유의수준 5%입니다.')
        exam['note'] += ' 자료 상태: ' + exam['source_note']
        exam['source_original'] = exam['source']
        exam['source'] = exam['source'].split(':', 1)[0]
        exam['type'] = 'scenario_practice'
        exam['title'] = 'ADP ' + folder.replace('/', ' ') + ' 상황형 연습'
        all_items = []
        for section_index, (section, spec) in enumerate(zip(exam['sections'], blueprint)):
            assert len(section['questions']) == len(spec['questions']), (folder, spec['title'])
            section['source_title'] = section['title']
            section['title'] = spec['title']
            # 원문 조건은 독립 명세에 보존. 응시자용 조건에는 정답 검정명을 쓰지 않는다.
            section['source_info'] = section.get('info', [])
            section['source_condition'] = section.get('condition', '')
            section['info'] = [spec['context']]
            legacy_data = {
                '15회': ['problem1.csv', 'problem2_usage.csv + problem2_usage_history.csv + problem2_avg_tem.csv'],
                '17회': ['problem1.csv', 'problem2.csv', 'problem3.csv'],
                '20회': ['problem1.csv', 'problem2.csv', 'problem3.csv'],
                '22회': ['data1.csv', 'data1.csv', 'data1.csv', 'content.csv', 'error.csv'],
                '23회': ['problem1.csv', 'problem2.csv', 'problem3_covid2.csv', '본문 인원수 표'],
            }
            if folder in legacy_data:
                section['data'] = legacy_data[folder][section_index]
            section['condition'] = ''
            for item, prompt in zip(section['questions'], spec['questions']):
                key = (folder, item['id'])
                item['source_text'] = item['text']
                item['text'] = prompt
                item['answer_rationale'] = RATIONALE.get(key, item.get('reason', ''))
                item['inference_report'] = key in EXTRA
                all_items.append(item)
        manifest[folder] = exam
        path = next((ROOT / folder).glob('*풀이.ipynb'))
        book = nbformat.read(path, as_version=4)
        changed_code = False
        book.cells[0].source = ('# ' + exam['title'] + ' - 풀이\n\n'
            '**상황형 재구성 연습·예시 풀이이며 공식 원문/정답이 아닙니다.**\n\n'
            '출처: ' + exam['source'] + '\n\n원 자료 조건: ' + exam.get('source_note', '') + '\n\n'
            '문제는 동봉 상황형 문제지와 같습니다. 원문은 문항목록.json·추가_문항명세.json에 보존합니다. '
            '순서대로 실행하며 각 대문항에서 고정한 자료·전처리·분할을 후속 문항에 재사용합니다. '
            '통계 답안은 선정 근거 → 가정·가설 → 통계량·p-value → 기각 여부·상황 해석 순서로 작성합니다. '
            '확률·구간 추정·관리도 문항에 불필요한 귀무가설을 만들지 않습니다.\n\n'
            '실행: 해당 회차 폴더에서 위에서 아래로 실행합니다. 가설검정_답안.json은 실제 계산한 가설·수치·결론 기록입니다.')
        setup = next(c for c in book.cells if c.cell_type == 'code')
        if 'scenario_base_source' not in setup.metadata:
            setup.metadata.scenario_base_source = setup.source
        new_setup = setup.metadata.scenario_base_source + '''
# 상황형 답안 기록 초기화
_SCENARIO_ROOT=next(p for p in [Path.cwd()]+list(Path.cwd().parents) if (p/'기출문제/공통/scenario_inference.py').is_file())
sys.path.insert(0,str(_SCENARIO_ROOT/'기출문제/공통'))
from scenario_inference import (reset_reports, set_question, report_test, report_anova,
    report_correlations, report_adf, report_residuals, report_regression_diagnostics)
reset_reports(OUT)
'''
        if setup.source != new_setup:
            setup.source = new_setup; changed_code = True
        sections = iter(exam['sections'])
        by_id = {q['id']: q for q in all_items}
        seen = []
        for i, cell in enumerate(book.cells):
            if cell.cell_type != 'markdown':
                continue
            if cell.source.startswith('## '):
                # 마지막 제출 확인은 분석 대문항이 아니다.
                if cell.source.startswith('## 제출'):
                    continue
                sec = next(sections)
                cell.source = '## ' + sec['title'] + '\n\n' + '\n\n'.join(sec['info'])
                if sec.get('data'):
                    cell.source += '\n\n자료: ' + sec['data']
            if not cell.source.startswith('### '):
                continue
            qid = cell.source.splitlines()[0][4:].strip()
            assert qid in by_id, (folder, qid)
            item = by_id[qid]
            old_markdown = cell.metadata.get('scenario_base_source', cell.source)
            cell.metadata.scenario_base_source = old_markdown
            if '**풀이 및 해석**' in old_markdown:
                old_reason = old_markdown.split('**풀이 및 해석**', 1)[1].strip()
            elif '**풀이 근거:**' in old_markdown:
                old_reason = old_markdown.split('**풀이 근거:**', 1)[1].strip()
            else:
                old_reason = item.get('reason', '')
            rationale = RATIONALE.get((folder,qid), old_reason)
            item['answer_rationale'] = rationale
            cell.source = ('### ' + qid + '\n\n' + item['text'] +
                           '\n\n**판단 및 풀이 근거**\n\n' + rationale +
                           ('\n\n**계산·해석 상세**\n\n' + old_reason if old_reason and old_reason != rationale else '') +
                           '\n\n**실행 답안**\n\n아래 코드는 이 문항의 산출물과 수치를 계산합니다. '
                           + ('가설·통계량·p-value·기각 여부·상황 결론을 함께 출력하며 수치는 실행 결과에 따라 결정합니다.'
                              if (folder,qid) in EXTRA else '결과와 위 판단 근거를 연결해 답안을 작성합니다.'))
            code = book.cells[i+1]
            assert code.cell_type == 'code', (folder,qid)
            base = code.metadata.get('scenario_base_source', code.source)
            code.metadata.scenario_base_source = base
            new_code = 'set_question(' + repr(qid) + ')\n' + base
            if (folder,qid) in EXTRA:
                new_code += '\n\n# 상황형 요구: 가설·실제 수치·기각 여부·맥락 결론\n' + EXTRA[(folder,qid)]
            if code.source != new_code:
                code.source = new_code; changed_code = True
            seen.append(qid)
        assert seen == [q['id'] for q in all_items], folder
        book.metadata.adp['type'] = 'scenario_practice'
        book.metadata.adp['scenario_version'] = 1
        if changed_code:
            for c in book.cells:
                if c.cell_type == 'code':
                    c.outputs = []; c.execution_count = None
            book.metadata.pop('execution_validation', None)
        nbformat.validate(book)
        nbformat.write(book,path)
        print(folder, len(all_items))
    assert set(manifest) == set(BLUEPRINT)
    (ROOT/'상황형_문항명세.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    for folder, exam in manifest.items():
        render_paper(folder,exam)
    return manifest


if __name__ == '__main__':
    apply()
