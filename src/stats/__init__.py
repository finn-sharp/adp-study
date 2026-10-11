"""ANOVA·MANOVA·사후검정의 공개 API. 선택 통계 엔진은 실행 시 로드합니다."""
from .anova import ANOVAProfile, run_anova
from .manova import MANOVAProfile, MANOVAResult, run_manova
from .posthoc import PostHocProfile, PostHocResult, run_posthoc

from .anova import ANOVAOutputDTO
from .manova import MANOVAOutputDTO
from .posthoc import PostHocOutputDTO
from .analysis import DescriptiveOutputDTO, StatisticalOutputDTO, run_descriptive
