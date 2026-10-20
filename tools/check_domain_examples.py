"""신규 공개 모듈과 skeleton을 각각 독립 프로세스로 실행합니다."""
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
