"""해당 파일 폴더 또는 임의 cwd에서 실행 가능한 실제 모의 풀이."""
from pathlib import Path
import sys
FOLDER=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
sys.path.insert(0,str(FOLDER.parent/'공통'))
from analysis import Context
ctx=Context(FOLDER)


# 1-1
ctx.run('1-1','reg_design')

# 1-2
ctx.run('1-2','reg_baseline')

# 1-3
ctx.run('1-3','reg_diagnostics')

# 1-4
ctx.run('1-4','reg_selection')

# 1-5
ctx.run('1-5','reg_tuning')

# 1-6
ctx.run('1-6','reg_final')

# 2-1
ctx.run('2-1','cls_design')

# 2-2
ctx.run('2-2','cls_tuning')

# 2-3
ctx.run('2-3','cls_logit')

# 2-4
ctx.run('2-4','cls_final')

# 2-5
ctx.run('2-5','cls_decision')

# 3-1
ctx.run('3-1','ts_design')

# 3-2
ctx.run('3-2','ts_tuning')

# 3-3
ctx.run('3-3','ts_diagnostics')

# 3-4
ctx.run('3-4','ts_final')
