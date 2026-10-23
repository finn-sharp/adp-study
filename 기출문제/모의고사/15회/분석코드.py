"""이 파일 위치를 기준으로 위에서 아래로 실행하는 모의 풀이."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)

from pathlib import Path
import sys

FOLDER = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(FOLDER.parent / "공통"))
from analysis import Context

ctx = Context(FOLDER)

# 문항 1-1

ctx.run("1-1", "reg_design")

# 문항 1-2

ctx.run("1-2", "reg_baseline")

# 문항 1-3

ctx.run("1-3", "reg_diagnostics")

# 문항 1-4

ctx.run("1-4", "reg_selection")

# 문항 1-5

ctx.run("1-5", "reg_tuning")

# 문항 1-6

ctx.run("1-6", "reg_final")

# 문항 2-1

ctx.run("2-1", "cls_design")

# 문항 2-2

ctx.run("2-2", "cls_tuning")

# 문항 2-3

ctx.run("2-3", "cls_logit")

# 문항 2-4

ctx.run("2-4", "cls_final")

# 문항 2-5

ctx.run("2-5", "cls_decision")

# 문항 3-1

ctx.run("3-1", "ts_design")

# 문항 3-2

ctx.run("3-2", "ts_tuning")

# 문항 3-3

ctx.run("3-3", "ts_diagnostics")

# 문항 3-4

ctx.run("3-4", "ts_final")
