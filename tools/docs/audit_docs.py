"""등록 모듈 전수 수록·실제 DTO 필드·문서 링크·삭제 문서 참조를 검사합니다."""
import ast
import re
from pathlib import Path
from urllib.parse import unquote
import build_module_catalog

ROOT=Path(__file__).resolve().parents[2]
content,count=build_module_catalog.build()
assert (ROOT/'docs/ANALYSIS_CATALOG.md').read_text()==content,'기능 카탈로그를 재생성하세요.'
reg=build_module_catalog.registry()
keys=re.findall(r'^\| \[`([^`]+)`\]\(',content,re.M)
assert len(keys)==len(set(keys))==count and set(keys)==set(reg),'모듈 누락/중복'
for key,(module,profile,_,_) in reg.items():
    tree=ast.parse((ROOT/(module.replace('.','/')+'.py')).read_text())
    assert any(isinstance(n,ast.ClassDef) and n.name==profile for n in tree.body),(key,profile)
paths=[ROOT/'README.md',*sorted((ROOT/'docs').rglob('*.md')),ROOT/'기출문제/모의고사/응용10회/README.md']
links=0
for path in paths:
    text=path.read_text()
    prose=re.sub(r'```.*?```','',text,flags=re.S)
    for raw in re.findall(r'\]\(([^)]+)\)',prose):
        target=raw.split(' "')[0].strip('<>')
        if target.startswith(('http:','https:','mailto:','codex:','#')):continue
        target=unquote(target.split('#',1)[0])
        assert (path.parent/target).exists(),(str(path.relative_to(ROOT)),target)
        links+=1
    for obsolete in ('APPLIED_REGRESSION_WORKFLOWS.md','STATISTICAL_TESTS.md','ANOVA_MODULES.md','TENSORFLOW_MODULES.md','STATISTICS_PDF_AUDIT.md','proposals/architecture-update-demo'):
        assert obsolete not in text,(str(path.relative_to(ROOT)),obsolete)
print('문서 검증 통과: %d개 모듈, %d개 문서, %d개 로컬 링크'%(count,len(paths),links))
