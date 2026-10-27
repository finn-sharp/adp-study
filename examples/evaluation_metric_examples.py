"""
================================================================================
evaluation_metric_examples.py - 모든 독립 평가 모듈의 오프라인 실행·DTO JSON 계약 검증.
================================================================================
설명:
    - 모든 독립 평가 모듈의 오프라인 실행·DTO JSON 계약 검증.
분석 가정:
    - 입력의 형상·타입·자료 순서를 호출자가 확인합니다. 계산과 파일 쓰기의 조건은 각 공개 함수의 계약을 따릅니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""
import importlib
import json
from src.Recipes.loader import _REGISTRY
from src.Evaluation._common import example_inputs
from src.Evaluation.comparison import run_comparison
from src.Evaluation.cv_summary import run_cv_summary


def main():
    """Args: 없음. Returns: 실행한 모듈 수. Raises: 평가/JSON 오류."""
    count=0
    for key,(module_name,_,_,_) in _REGISTRY.items():
        if not key.startswith('metric-') or key in ('metric-comparison','metric-cv-summary'):continue
        group,name=module_name.split('.')[-2:]
        module=importlib.import_module(module_name)
        runner=getattr(module,'run_'+(group+'_metrics' if name=='metrics' else name))
        json.loads(runner(**example_inputs(group)).to_json());count+=1
    json.loads(run_comparison([0,0,1,1],{'baseline':[0,0,0,0],'model':[0,1,1,1]}).to_json());count+=1
    json.loads(run_cv_summary({'f1':[.7,.8,.9]}).to_json());count+=1
    print('evaluation examples:',count)
    return count


if __name__=='__main__':main()
