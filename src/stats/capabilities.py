"""
================================================================================
capabilities.py - PDF 표본 관계·척도 지원 목록
================================================================================
설명:
    - 기법별 Profile·Output DTO·공개 실행기와 독립 JSON 레시피를 제공합니다.
분석 가정:
    - 지원 안내용이며 척도/표본수만으로 검정을 자동 선택하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    통계 PDF 전체 범위 모듈화·입력 검증·명세
================================================================================
"""
__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

# 각 표본 설계는 공개 기법의 추가 가정을 확인해야 합니다.
TEST_CAPABILITIES = {'one-sample-mean': {'sample_designs': ('one_sample',),
                     'measurement_level': 'numeric'},
 'two-sample-mean': {'sample_designs': ('independent',),
                     'measurement_level': 'numeric'},
 'paired-t': {'sample_designs': ('paired',), 'measurement_level': 'numeric'},
 'one-sample-variance': {'sample_designs': ('one_sample',),
                         'measurement_level': 'numeric'},
 'one-sample-proportion': {'sample_designs': ('one_sample',),
                           'measurement_level': 'binary'},
 'two-sample-proportion': {'sample_designs': ('independent',),
                           'measurement_level': 'binary'},
 'sign': {'sample_designs': ('one_sample', 'paired'),
          'measurement_level': 'ordinal_or_numeric'},
 'wilcoxon': {'sample_designs': ('one_sample', 'paired'),
              'measurement_level': 'ordinal_or_numeric'},
 'runs': {'sample_designs': ('ordered',), 'measurement_level': 'binary'},
 'mann-whitney': {'sample_designs': ('independent',),
                  'measurement_level': 'ordinal_or_numeric'},
 'kruskal-wallis': {'sample_designs': ('independent',),
                    'measurement_level': 'ordinal_or_numeric'},
 'mood-median': {'sample_designs': ('independent',),
                 'measurement_level': 'ordinal_or_numeric'},
 'friedman': {'sample_designs': ('paired',),
              'measurement_level': 'ordinal_or_numeric'},
 'repeated-measures-anova': {'sample_designs': ('paired',),
                             'measurement_level': 'numeric'},
 'anderson-darling': {'sample_designs': ('one_sample',),
                      'measurement_level': 'numeric'},
 'kolmogorov-smirnov': {'sample_designs': ('one_sample',),
                        'measurement_level': 'numeric'},
 'hotelling': {'sample_designs': ('independent',),
               'measurement_level': 'numeric'},
 'box-m': {'sample_designs': ('independent',), 'measurement_level': 'numeric'},
 'odds-ratio': {'sample_designs': ('independent',),
                'measurement_level': 'nominal'},
 'risk-ratio': {'sample_designs': ('independent',),
                'measurement_level': 'nominal'},
 'fisher-exact': {'sample_designs': ('independent',),
                  'measurement_level': 'nominal'},
 'cmh': {'sample_designs': ('independent',), 'measurement_level': 'nominal'},
 'bowker': {'sample_designs': ('paired',), 'measurement_level': 'nominal'},
 'cochran-q': {'sample_designs': ('paired',), 'measurement_level': 'binary'},
 'adf': {'sample_designs': ('ordered',), 'measurement_level': 'numeric'},
 'chi-square-homogeneity': {'sample_designs': ('independent',),
                            'measurement_level': 'nominal'}}
