"""
================================================================================
scenario_answers.py - 상황형 문항별 추가 해석·실행 답안. 기존 계산과 같은 값을 보고한다.
================================================================================
설명:
    - 상황형 문항별 추가 해석·실행 답안. 기존 계산과 같은 값을 보고한다.
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
EXTRA = {}
RATIONALE = {}
def a(folder, question, rationale, code):
    EXTRA[(folder, question)] = code.strip()
    RATIONALE[(folder, question)] = rationale

a('15회','2-3', '독립 일별 표본을 작업 가정으로 한 Welch 일원 평균 비교를 유형별로 수행하고 Holm 보정한다. 실제 연속 일별 자료의 자기상관은 별도 점검이 필요하므로 이 p값을 확증 근거로 단정하지 않는다.', '''
for i,t in enumerate(types):
    row=weekday_tests.iloc[i]
    report_test('Welch 일원 평균 비교: '+t,'모든 요일의 평균 일일 사용량이 같다','적어도 한 요일의 평균이 다르다',row.Welch_F,row.p_value,t+' 유형 사용량에 요일 차이가 있다','요일 간 독립을 작업 가정으로 계산. 시계열 의존성을 무시한 p값은 탐색적이다.',adjusted_p=row.p_Holm)
''')
a('15회','2-4', '기온·사용량은 수치형이고 선형 관계를 탐색하므로 Pearson을 주 분석으로, 순위 관계는 보조로 본다. H0: rho=0, H1: rho≠0이며 5개 비교는 Holm 보정한다.', '''
for _,row in correlations.iterrows():
    report_test('Pearson 기온-사용량: '+row['type'],'모상관 rho=0','모상관 rho != 0',row.Pearson_r,row.p_value,row['type']+' 사용량과 기온에 선형 관계가 있다','독립·이변량 정규 추론을 작업 가정. 날짜 의존성·다중 비교·인과 해석에 주의.',adjusted_p=row.p_Holm)
''')
a('17회','3-2', '독립 4집단의 평균 비교를 작업 가정으로 ANOVA·Welch와 비교한다. 5점 척도는 연속 정규 반응과 다르므로 순위 기반 민감도 분석도 보고한다. 주 분석과 보조 결과를 골라 유리한 p값만 보고하지 않는다.', '''
report_test('일원 ANOVA 주 분석','A~D의 Q1-1 평균이 모두 같다','하나 이상의 집단 평균이 다르다',F,pvalue,'집단에 따라 Q1-1 평균이 다르다','독립 관측·정규 오차·등분산을 작업 가정. 5점 척도와 분산 진단의 한계를 함께 보고.',df=[len(groups)-1,sum(map(len,groups))-len(groups)])
report_test('Welch ANOVA 민감도 분석','A~D의 Q1-1 평균이 모두 같다','하나 이상의 집단 평균이 다르다',welch.statistic,welch.pvalue,'집단에 따라 Q1-1 평균이 다르다','독립 관측·정규 오차, 등분산은 요구하지 않음.',df=welch.df)
report_test('Kruskal-Wallis 민감도 분석','집단별 응답 분포가 같다','하나 이상의 집단 응답 분포가 다르다',kw.statistic,kw.pvalue,'집단에 따라 Q1-1 응답 분포가 다르다','독립 표본·순위 및 동률 보정. 형태가 다르면 중앙값만의 차이로 해석하지 않음.',df=len(groups)-1)
''')
a('17회','3-3', '문항 간 공통 구조를 찾는 탐색적 요인분석이다. KMO는 적합성 지표이고 Bartlett 구형성 검정의 H0는 모집단 상관행렬이 단위행렬이라는 것이다. 유의성만으로 유지 요인 수를 결정하지 않는다.', '''
report_test('Bartlett 구형성 진단','문항의 모상관행렬이 단위행렬이다','문항 간 0이 아닌 상관이 존재한다',bartlett,stats.chi2.sf(bartlett,df_b),'문항 간 공통 구조를 검토할 상관이 있다','독립 관측·다변량 정규 근사. 리커트 척도와 탐색적 사용의 한계.',df=df_b)
''')
a('22회','4.3', '정규 모집단 한 표본의 분산은 카이제곱 분포로 추론한다. 기본 양측 H0: σ²=1.3, H1: σ²≠1.3과 불량 초과 방향 H0: σ²≤1.3, H1: σ²>1.3을 구분한다.', '''
report_test('정규 한 표본 분산 양측 검정','sigma^2=1.3','sigma^2 != 1.3',statistic,p_two,'이 공정의 성분 분산이 기준 1.3과 다르다','단일 정규 모집단·독립 표본. 제조사 간 비교 자료는 아님.',df=df)
report_test('보조: 분산 상측 검정','sigma^2 <= 1.3','sigma^2 > 1.3',statistic,right,'이 공정의 성분 분산이 1.3을 초과한다','양측 결론과 목적이 다름. 방향은 결과 관측 전에 정해야 한다.',df=df)
''')
a('23회','통계 2-3', '정규성·대칭성을 요구하지 않는 독립 수명 표본의 중앙값 검정이므로 정확 부호 검정을 선택한다. 동점 제외 후 S+~Binomial(n_eff,.5)이며 기본 양측과 미달 단측을 분리한다.', '''
report_test('정확 부호 검정 양측','수명 중앙값=10000','수명 중앙값 != 10000',positive,two.pvalue,'진공관 수명 중앙값이 10000시간과 다르다','독립 관측·기준값 동점 제외. 평균 수명의 검정이 아니다.')
report_test('보조: 정확 부호 검정 미달 단측','수명 중앙값 >= 10000','수명 중앙값 < 10000',positive,lower.pvalue,'수명 중앙값이 10000시간 미만이다','방향은 사전 지정; 기본 양측 검정과 별개.')
''')
a('23회','통계 4-3', '서로 다른 학생의 두 범주 변수 교차 빈도이므로 독립성 카이제곱 검정을 선택한다. H0: 학과와 성적 구간은 독립, H1: 연관성이 존재. 기대도수 조건과 효과 크기를 함께 본다.', '''
report_test('범주 독립성 카이제곱','학과와 성적 구간이 독립이다','학과와 성적 구간이 연관된다',chi2,pvalue,'학과와 성적 구간 사이에 연관성이 있다','독립 학생·1명 1셀·기대도수 조건 충족. 인과가 아님.',df=degrees)
''')
for f,q,df,target,features in [
('모의고사/2회','1-1','admit','Chance_of_Admit',['GRE','TOEFL','Univ_Rating','SOP','LOR','CGPA']),
('모의고사/3회','1-2','car','Sales',['CompPrice','Income','Advertising','Population','Price','Age','Education'])]:
    a(f,q,'연속 변수의 선형 관계를 Pearson으로 평가한다. 각 H0: rho=0, H1: rho≠0이며 상관 크기와 Holm 보정 p값을 구분한다. 명목 범주 코드에 수치 간격 의미를 부여하지 않는다.',
      'report_correlations(%s,%r,%r)' % (df,target,features))
for f,q,df,target,group,second in [
('모의고사/1회','2-3','fifa','Value','Position_Class',None),
('모의고사/1회','2-4','fifa','Value','Position_Class','Preferred_Foot'),
('모의고사/4회','2-1','bike','revenues','pop_density',None),
('21회','4-1','aov','y','group','band'),
('21회/보충','4-1','travel','points','sex','travel')]:
    a(f,q,'독립 집단의 수치 반응을 설명하는 요인 모형을 선택한다. 한 요인이면 H0: 모든 집단 평균이 같다, H1: 적어도 하나 다르다. 두 요인이면 주효과와 상호작용의 가설·p값을 각각 구분한다. 정규 오차·등분산은 작업 가정이며 적합성을 검토한다.',
      '_aov_model=anova(%s,%r,%r%s)\nreport_anova(_aov_model,%r,%r)' % (df,target,group,','+repr(second) if second else '',target,[group,second] if second else [group]))
a('모의고사/3회','1-1', '독립 두 집단의 정규 평균 비교이며 모분산 동일성은 알려져 있지 않아 Welch t검정을 선택한다. H0: μ도시=μ비도시, H1: μ도시≠μ비도시. 분산 검정 결과로 유리한 평균 검정을 사후 선택하지 않는다.', '''
_w=stats.ttest_ind(a,b_,equal_var=False)
_va=a.var(ddof=1)/len(a);_vb=b_.var(ddof=1)/len(b_)
_df=(_va+_vb)**2/(_va**2/(len(a)-1)+_vb**2/(len(b_)-1))
report_test('Welch 독립 두 표본 t','도시/비도시 평균 매출이 같다','도시/비도시 평균 매출이 다르다',_w.statistic,_w.pvalue,'도시 여부에 따라 평균 매출이 다르다','독립 매장·집단별 정규성 가정·등분산 불필요. 지역의 인과 효과로 단정하지 않음.',df=_df)
''')
a('모의고사/4회','2-3','잔차의 정규성은 Shapiro-Wilk, 등분산은 Breusch-Pagan으로 점검한다. H0의 내용이 다르므로 p값·기각 여부를 따로 보고한다. 선택된 회귀식의 진단은 확증 인과 검정이 아니다.','report_regression_diagnostics(bike_model)')
a('모의고사/2회','1-2','AIC 선택 후 잔차의 정규성·등분산을 진단한다. 선택 후 계수 p값은 탐색적이며 전체 선택 절차를 무시한 확증 근거로 쓰지 않는다.','report_regression_diagnostics(admit_model)')
a('18회','3-1','수준 안정성을 위해 ADF 단위근 검정을 선택한다. H0: 단위근이 있다, H1: 없다. 수준·일반 차분·계절 차분을 구분하며 기각으로 모든 정상성 조건을 입증하지 않는다.', '''
report_adf(train18,'월별 원자료')
report_adf(train18.diff().dropna(),'일반 차분')
report_adf(train18.diff(12).dropna(),'12개월 차분')
''')
a('18회','3-3','적합 잔차의 정규성은 Shapiro-Wilk, 시차 무상관은 Ljung-Box로 평가한다. 후자는 선택 모형의 비계절·계절 AR/MA 계수 개수를 자유도에서 차감한다.', '''
_order=traffic18_model.model.order;_seasonal=traffic18_model.model.seasonal_order
_model_df=_order[0]+_order[2]+_seasonal[0]+_seasonal[2]
report_residuals(res18,'월별 예측 모형 잔차',[12,24],_model_df)
''')
for f in ['19회','19회/보충']:
    a(f,'2-1','분기 자료를 4계절로 보고 ADF 검정의 H0(단위근 존재)를 원자료·계절 차분에서 평가한다. 분포의 정규성 검정과는 목적이 다르다.',"report_adf(train_q,'분기 원자료')\nreport_adf(train_q.diff(4).dropna(),'4분기 차분')")
    a(f,'2-4' if f=='19회' else '2-2','잔차의 정규성과 시차 무상관을 별도 가설로 평가하고, 선택한 비계절·계절 AR/MA 차수를 Ljung-Box 자유도에서 차감한다.', '''
_order=quarter_model.model.order;_seasonal=quarter_model.model.seasonal_order
_model_df=_order[0]+_order[2]+_seasonal[0]+_seasonal[2]
report_residuals(quarter_model.resid,'분기 예측 모형 잔차',[4,8],_model_df)
''')
a('22회/보충','4-1', '순서의 무작위성을 평가하는 runs 검정을 선택한다. H0: 주어진 두 상태 개수 하 순서가 무작위, H1: 비무작위다. 짧은 이진 순서의 정규 근사 결과이므로 정확 검정과 차이가 날 수 있다.', '''
report_test('Runs 순서 무작위성 정규 근사','상태 발생 순서가 무작위이다','상태 발생 순서가 비무작위이다',z,p,'공정 상태의 순서가 무작위에서 벗어난다','시간 순서를 보존·이진 상태·짧은 표본 정규 근사 및 연속성 보정의 한계.')
''')
a('24회','2-1','절편 포함 두 수치 특성 회귀의 전체 F검정을 선택한다. H0: β광고=β연구=0, H1: 적어도 한 계수≠0. 개별 t검정 및 인과 효과와 구분한다.', '''
report_test('전체 회귀 F','beta_advertising=beta_rd=0','두 기울기 중 하나 이상 0이 아니다',ad_model.fvalue,ad_model.f_pvalue,'광고비와 연구개발비의 전체 설명력이 있다','독립 관측·정규 오차·등분산·선형성·공선성 없음. n=10의 한계.',df=[ad_model.df_model,ad_model.df_resid])
''')
a('24회','3-1','독립 두 표본의 알려진 모분산과 정규 모형이므로 두 평균 Z검정을 선택한다. H0: μA=μB, H1: μA≠μB. 표본 표준편차였다면 Welch가 필요하다.', '''
report_test('알려진 모분산 두 평균 Z','mu_A=mu_B','mu_A != mu_B',z,p,'두 공정의 평균 수준이 다르다','독립 정규 표본·모표준편차 .03/.04가 알려짐. 양측 사전 지정.')
''')
a('25회','4-1', '같은 사람 전후의 차이에 대한 대응 t검정이다. d=after-before, H0: μd≥0, H1: μd<0. 독립 두 표본 검정을 사용하지 않으며 차이의 정규성을 확인한다.', '''
report_test('대응 t 감소 단측','mean(after-before) >= 0','mean(after-before) < 0',t,p,'복용 후 평균 혈압이 감소한다','개인 간 독립·전후 짝 유지·차이 정규성 작업 가정. 원인 효과에는 대조군 등 추가 설계 필요.',df=len(diff)-1)
''')
a('25회','5-1','정규성을 가정하지 않는 세 독립 집단의 분포 비교이므로 Kruskal-Wallis 순위 검정을 선택한다. H0: 세 분포 동일, H1: 적어도 한 분포 다름. 중앙값 차이에는 같은 분포 형태 가정이 추가로 필요하다.', '''
report_test('Kruskal-Wallis 독립 순위 비교','세 공장 생산량 분포가 같다','하나 이상의 공장 분포가 다르다',h,p,'공장별 생산량 분포가 다르다','독립 표본·동률 보정·순위 비교. 후속 쌍 차이는 별도 다중 비교가 필요.',df=len(groups)-1)
''')
a('26회','4-1','독립 지역 표본의 이진 반응 비율 비교이므로 카이제곱 동질성 검정을 선택한다. H0: p1=p2=p3, H1: 하나 이상 다름. 기대도수와 근사 조건을 확인한다.', '''
report_test('지역 비율 동질성 카이제곱','세 지역의 지지 비율이 같다','하나 이상의 지역 지지 비율이 다르다',chi,p,'지역별 지지 비율이 다르다','독립 사람·지역별 표집·기대도수 점검. 인과 또는 특정 쌍 차이로 단정하지 않음.',df=df_)
''')
a('26회','5-1','독립 두 집단의 정규 평균 비교이되 모분산 동일성이 알려지지 않았으므로 Welch t검정을 선택한다. H0: μM=μF, H1: μM≠μF. 같은 평균차의 Welch 구간과 결론을 대조한다.', '''
report_test('Welch 독립 혈압 평균','mu_M=mu_F','mu_M != mu_F',test_b.statistic,test_b.pvalue,'남녀 집단의 평균 혈압이 다르다','독립 관측·집단별 정규성 작업 가정·등분산 불필요. 두 집단 각각의 CI 겹침으로 대체하지 않음.',df=df_)
''')
a('26회','6-1','시간 기울기에 대한 회귀 t 단측 검정이다. H0: βmonth≤0, H1: βmonth>0. 관측된 월만 사용하고 시계열 의존성·작은 n 때문에 작업 가정이 강함을 명시한다.', '''
report_test('회귀 기울기 증가 단측','beta_month <= 0','beta_month > 0',trend.tvalues['month'],stats.t.sf(trend.tvalues['month'],trend.df_resid),'은 가격의 선형 증가 추세가 있다','실제 관측만 사용·선형성·독립 정규 오차·등분산 작업 가정. 9개월 시계열의 추론 취약성.',df=trend.df_resid)
''')
a('28회','3-1','독립 두 집단이고 정규성을 가정하지 않아 Mann-Whitney U를 선택한다. H0: 두 학교 점수 분포 동일, H1: 다름. 서로 짝이 아니므로 signed-rank가 아니며 분포 형태가 다를 때 중앙값만의 검정이 아니다.', '''
_u=stats.mannwhitneyu(a,b_,alternative='two-sided')
report_test('Mann-Whitney U 양측','두 학교의 성적 분포가 같다','두 학교의 성적 분포가 다르다',_u.statistic,_u.pvalue,'두 학교의 성적 분포가 다르다','독립 관측·순위 비교·동률 고려. p값 계산 방식은 scipy auto이며 형태가 같아야 위치 차이 해석 가능.')
''')
a('28회','4-1','같은 사람의 이진 전후 반응은 대응 범주 자료다. McNemar 정확 검정에서 불일치 두 방향 확률의 같음을 평가한다. H0: p01=p10, H1: p01≠p10.', '''
report_test('McNemar 정확 양측','p01=p10 (전후 주변 선호율 동일)','p01 != p10',result.statistic,result.pvalue,'시식 전후 선호 비율이 변했다','개인 간 독립·짝 유지·불일치 셀 합 조건부 정확 이항 검정. 독립성 카이제곱과 다름.')
''')
a('28회','5-1','z에 선형 회귀한 두 잔차의 상관으로 편상관을 구한다. H0: ρxy.z=0, H1: ρxy.z≠0. 검정 자유도는 n-3이며 잔차의 단순 Pearson p값(n-2)을 재사용하지 않는다.', '''
report_test('1개 변수 보정 편상관 t','rho_xy.z=0','rho_xy.z != 0',t,p,'z를 선형 통제한 x와 y에 관계가 남는다','독립 표본·선형 정규 모형·보정변수 1개. 관측 관계이며 인과가 아님.',df=df_)
''')
