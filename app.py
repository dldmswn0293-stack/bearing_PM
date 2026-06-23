# app.py — 베어링 열화 판정 시연 (실행: 터미널에서  streamlit run app.py)
import streamlit as st
import numpy as np
import joblib

# 1. 모델 불러오기
model = joblib.load('bearing_model.pkl')
features = ['h_rms', 'h_kurt', 'h_skew', 'h_crest']

# 2. 페이지 기본 설정
st.set_page_config(page_title='베어링 열화 진단', page_icon='⚙️')
st.title('⚙️ 베어링 열화 진단')
st.caption('진동 데이터로 베어링 열화를 조기 감지하는 예지보전 데모')

tab1, tab2, tab3, tab4 = st.tabs(['📖 설명', '📊 데이터 탐색', '🎯 모델 성능', '⚙️ 열화 진단'])

# ── 탭1: 비전문가용 설명 ──
with tab1:
    st.header('베어링이란?')
    c1, c2 = st.columns([1, 1])
    with c1:
        st.image('bearing.PNG', caption='볼 베어링 구조와 마모 원리')
    with c2:
        st.markdown('''
**베어링**은 회전하는 축을 받쳐주는 부품으로, 리니어 가이드·볼스크류 등
리니어 모션 부품의 핵심입니다.

- **외륜**: 고정된 바깥 링
- **내륜**: 축과 함께 회전하는 안쪽 링
- **볼**: 외륜·내륜 사이를 굴러 마찰을 줄임

**왜 마모될까?** 볼이 하중을 받으며 궤도 위를 반복적으로 구르면 접촉 지점에
피로가 쌓여 표면이 거칠어집니다. 마모가 진행되면 **진동이 점점 커지고**,
결국 고장으로 이어집니다. 그래서 **진동을 측정하면 마모 상태를 미리 알 수 있습니다.**
        ''')
    st.divider()
    st.header('진동에서 뽑는 4가지 특성')
    st.markdown('''
| 특성 | 의미 | 쉽게 말하면 |
|---|---|---|
| **RMS** | 진동의 평균 크기 | 얼마나 크게 떨리나 (열화 핵심 지표) |
| **첨도** | 신호의 뾰족함 | 날카로운 충격이 많나 |
| **왜도** | 분포의 치우침 | 한쪽으로 쏠렸나 |
| **파고율** | 최대÷평균 비율 | 튀는 스파이크가 큰가 |
    ''')

# ── 탭2: 데이터 탐색 ──
with tab2:
    st.header('데이터 탐색')
    st.subheader('① 베어링별 진동(RMS) 추이 — 열화 곡선')
    st.image('fig_rms.PNG')
    st.caption('정상 구간에선 평탄하다가, 열화가 시작되면 RMS가 급상승합니다.')
    st.subheader('② 특성 간 상관관계 — 다중공선성 확인')
    st.image('fig_corr.PNG')
    st.caption('선정한 4개 특성은 서로 상관이 낮아 독립적입니다.')

# ── 탭3: 모델 성능 ──
with tab3:
    st.header('모델 성능')
    c1, c2 = st.columns(2)
    with c1:
        st.subheader('변수 중요도')
        st.image('fig_importance.PNG')
        st.caption('RMS가 약 82%로 열화 판정의 핵심입니다.')
    with c2:
        st.subheader('혼동행렬')
        st.image('fig_cm.PNG')
        st.caption('열화 126건 전부 탐지 (재현율 1.0).')
    st.divider()
    cols = st.columns(4)
    cols[0].metric('정확도', '0.994')
    cols[1].metric('정밀도', '0.992')
    cols[2].metric('재현율', '1.000')
    cols[3].metric('F1', '0.996')

# ── 탭4: 열화 진단 (시연) ──
with tab4:
    st.header('열화 진단')
    st.caption('진동 특성 4종을 입력하면 정상/열화를 판정합니다')
    c1, c2 = st.columns(2)
    with c1:
        h_rms = st.slider('RMS (진동 크기)', 0.3, 9.5, 0.8, 0.05)
        h_skew = st.slider('왜도 (분포 치우침)', -0.6, 0.4, 0.0, 0.01)
    with c2:
        h_kurt = st.slider('첨도 (뾰족함)', -0.4, 50.0, 0.5, 0.1)
        h_crest = st.slider('파고율 (스파이크 비율)', 3.5, 26.5, 5.8, 0.1)
    if st.button('진단하기', type='primary', use_container_width=True):
        X = np.array([[h_rms, h_kurt, h_skew, h_crest]])
        pred = model.predict(X)[0]; proba = model.predict_proba(X)[0]
        if pred == 1:
            st.error('### 🔴 열화 감지 — 점검/교체 권장')
            st.metric('열화 확률', f'{proba[1]*100:.1f}%')
        else:
            st.success('### 🟢 정상')
            st.metric('정상 확률', f'{proba[0]*100:.1f}%')