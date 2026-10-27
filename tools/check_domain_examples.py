"""
================================================================================
check_domain_examples.py - 신규 공개 모듈과 skeleton을 각각 독립 프로세스로 실행합니다.
================================================================================
설명:
    - 신규 공개 모듈과 skeleton을 각각 독립 프로세스로 실행합니다.
분석 가정:
    - 저장소 경로와 입력 명세를 기준으로 실행합니다. 생성·검사 대상과 외부 의존성은 각 실행 함수의 조건을 따릅니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
api=json.loads((ROOT/'docs/DOMAIN_EXTENSION_API.json').read_text())
for key,item in api.items():
    module=item['module'];name=module.rsplit('.',1)[1]
    if module.startswith('src.stats'):skeleton='src.stats.skeleton.'+name
    elif module.startswith('src.TimeSeries'):skeleton='src.TimeSeries.skeleton.'+name
    elif module.startswith('src.Preprocessing'):skeleton='src.Preprocessing.skeleton.resampling_'+name
    else:skeleton='src.MachineLearning.skeleton.nonlinear_regression'
    for target in [module,skeleton]:
        result=subprocess.run([sys.executable,'-m',target],cwd=str(ROOT),capture_output=True,universal_newlines=True)
        if result.returncode:
            print(result.stdout);print(result.stderr);raise SystemExit(target+' failed')
    print(key,'module + skeleton passed',flush=True)
