"""이 파일 위치를 기준으로 위에서 아래로 실행하는 모의 풀이."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)

from pathlib import Path
import sys

FOLDER = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(FOLDER.parent / "공통"))
from researched_core import ResearchContext

ctx = ResearchContext(FOLDER)

# 문항 1-1

ctx.run("1-1", "cls_1")

# 문항 1-2

ctx.run("1-2", "cls_2")

# 문항 1-3

ctx.run("1-3", "cls_3")

# 문항 1-4

ctx.run("1-4", "cls_4")

# 문항 2-1

ctx.run("2-1", "ts_1")

# 문항 2-2

ctx.run("2-2", "ts_2")

# 문항 2-3

ctx.run("2-3", "ts_3")

# 문항 2-4

ctx.run("2-4", "ts_4")

# 문항 3-1

ctx.run("3-1", "text_1")

# 문항 3-2

ctx.run("3-2", "text_2")

# 문항 3-3

ctx.run("3-3", "text_3")

# 문항 3-4

ctx.run("3-4", "text_4")
