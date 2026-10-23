"""생성 통합 목록·문서 수치·계층 의존 방향을 검사합니다."""
from pathlib import Path
import ast
import json
import subprocess
import sys
import build_catalog

ROOT=Path(__file__).resolve().parents[2]
expected_json=(ROOT/'기출문제/전체목록.json').read_bytes()
expected_md=(ROOT/'기출문제/전체목록.md').read_bytes()
# Rebuilding must be deterministic. Restore files even if comparison fails.
try:
    result=build_catalog.build()
    assert (ROOT/'기출문제/전체목록.json').read_bytes()==expected_json,'전체목록.json을 재생성하세요.'
    assert (ROOT/'기출문제/전체목록.md').read_bytes()==expected_md,'전체목록.md를 재생성하세요.'
finally:
    (ROOT/'기출문제/전체목록.json').write_bytes(expected_json)
    (ROOT/'기출문제/전체목록.md').write_bytes(expected_md)
subprocess.run([sys.executable,str(ROOT/'tools/docs/audit_docs.py')],check=True)
counts=build_catalog.registry_counts()
for bundle in result['bundles']:
    for q in bundle['questions']:
        for module in q['module_paths']:
            path=ROOT/module.replace('.','/')
            assert path.is_dir() or path.with_suffix('.py').exists(),(bundle['folder'],q['id'],module)
readme=(ROOT/'README.md').read_text();catalog=(ROOT/'docs/ANALYSIS_CATALOG.md').read_text();recipes=(ROOT/'docs/RECIPES.md').read_text()
assert '**'+str(counts['modules'])+'개**' in readme
assert '**'+str(counts['recipe_files'])+'개**' in readme
assert '모듈은 '+str(counts['modules'])+'개' in catalog
assert '레시피는 '+str(counts['recipe_files'])+'개' in catalog
assert '전체 JSON '+str(counts['recipe_files'])+'개' in recipes
assert '조합은 '+str(counts['default_recipe_combinations'])+'개' in recipes
for path in (ROOT/'src').rglob('*.py'):
    tree=ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith(('기출문제','tools')),(path,node.module)
api=json.loads((ROOT/'docs/DOMAIN_EXTENSION_API.json').read_text())
for key,item in api.items():
    path=ROOT/(item['module'].replace('.','/')+'.py');tree=ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node,ast.ClassDef):
            doc=ast.get_docstring(node) or ''
            for field in node.body:
                if isinstance(field,ast.AnnAssign):assert field.target.id+' (' in doc,(path,node.name,field.target.id)
print(json.dumps(dict(status='passed',analysis=counts,summary=result['summary']),ensure_ascii=False))
