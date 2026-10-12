"""이전 확률분포 예제 경로의 호환 export; 정식 위치는 src.stats.skeleton.distributions."""
from src.stats.skeleton.distributions import main, run_pdf_examples

if __name__ == '__main__':
    import json
    print(json.dumps(main(), ensure_ascii=False, allow_nan=False, indent=2))
