# =====================================================================
#  실습 2 — 진동으로 압연 모터 전류를 맞혀 보기 (02 다변수 선형회귀)
# =====================================================================
#  실행: python 실습2_문제.py      (이 파일이 있는 폴더에서)
#  필요한 것: numpy, pandas  /  데이터는 옆의 데이터/ 폴더에 들어 있습니다.
#
#  [상황]
#    P제철 열간압연기(SPM01)에 진동센서 두 개(TOP·BOT)와 모터 전류계가 붙어 있습니다.
#    그런데 전류계가 자주 고장 납니다. 진동만 있을 때 전류를 추정할 수 있을까요?
#    맞힐 대상(정답) = CUR-MTR_RMS  (모터 전류의 실효값)
#
#  [쓰는 도구]  02 에서 배운 것 전부. 새 라이브러리 없습니다.
#    다변수 X / train·test 분할 / 열별 표준화(학습용 통계로만) / 경사하강 / R2
#    ※ 02_선형회귀_다변수_train_test.py 를 옆에 띄워 놓고 베껴 쓰세요. 그게 정상입니다.
#
#  [푸는 법]  TODO 를 위에서부터 하나씩 채우고, 그때그때 실행해서 숫자를 확인하세요.
#             한 번에 다 짜고 실행하면 어디서 틀렸는지 못 찾습니다.
# =====================================================================

import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "데이터")
d = pd.read_csv(os.path.join(DATA, "T-CR1-SPM01_압연특징.csv"), encoding="utf-8-sig")
# ↑ encoding="utf-8-sig" 빠뜨리면 첫 열 이름이 깨져서 KeyError 납니다.


# =====================================================================
# A. 데이터부터 본다  (모델 얘기는 아직 이르다)
# =====================================================================
# [A1] 표의 모양과 열 이름, 결측 개수를 찍으세요.
#      힌트: d.shape / list(d.columns) / d.isna().sum().sum()
# TODO
print("표의 모양:", d.shape)
print("열 이름:", list(d.columns))
print("결측 개수:", d.isna().sum().sum())
print()

# [A2] 정답으로 쓸 CUR-MTR_RMS 의 요약통계를 보세요. (describe)
#      → 이 값이 대략 몇에서 몇 사이인지 말할 수 있어야 합니다.
#        나중에 "MSE 500" 이 큰 건지 작은 건지 판단하는 기준이 됩니다.
# TODO
print("CUR-MTR_RMS의 요약통계:\n", d["CUR-MTR_RMS"].describe())
print()

# [A3] 숫자 열들의 상관계수 중, CUR-MTR_RMS 와의 상관만 크기순으로 보세요.
#      힌트: d.corr(numeric_only=True)["CUR-MTR_RMS"].sort_values()
#
#      ★ 보고 나서 답하세요 (주석으로 적어 두기) ★
#        (1) 상관이 0.98, 0.99 로 말도 안 되게 높은 열이 몇 개 보입니다. 이름이 뭔가요?
#        (2) 그 열들을 입력으로 쓰면 안 되는 이유가 있습니다. 뭘까요?
#            힌트: 열 이름의 앞부분을 보세요. CUR-MTR-... 로 시작하죠.
#                 전류계가 고장 나서 전류를 추정하려는 건데, 그 입력은 어디서 옵니까?
#      내 답: (1)
#             (2)
# TODO
print(d.corr(numeric_only=True)["CUR-MTR_RMS"].sort_values())
print()
# (1) CUR-MTR_PTP    0.982105
#     CUR-MTR_STD    0.998795
#     CUR-MTR_RMS    1.000000
# (2) 전류계가 고장 나서 전류를 추정하려는 상황(목적)인데, 입력값 자체가 타겟(전류)의 통계값(표준편차, 최댓값 등)이라면
# 정답을 미리 보고 맞히는 것과 같으므로 입력으로 사용할 수 없습니다.

# =====================================================================
# B. 입력 고르고 train / test 나누기
# =====================================================================
# [B1] 입력(특징) 4개를 아래 이름 그대로 쓰세요. 정답은 CUR-MTR_RMS.
특징이름 = ["VIB-BOT_RMS", "VIB-BOT_PTP", "VIB-BOT_KUR", "VIB-TOP_RMS"]
# X = ...   (d[특징이름].values.astype(float))
# y = ...   (d["CUR-MTR_RMS"].values.astype(float))
# X.shape, y.shape 를 찍어서 (570, 4) 와 (570,) 인지 확인하세요.
# TODO
X = d[특징이름].values.astype(float)
y = d["CUR-MTR_RMS"].values.astype(float)
print("X.shape:", X.shape, "/ y.shape:", y.shape)
print()

# [B2] 7:3 으로 나누세요. 반드시 '섞은 다음에' 자릅니다.
#      RandomState(42) 를 쓰면 정답지와 숫자가 똑같이 나옵니다.
#      힌트: 순서 = np.random.RandomState(42).permutation(len(X))
#            n_train = int(len(X) * 0.7)
#      학습용 몇 대 / 시험용 몇 대인지 찍으세요.
# TODO
rng = np.random.RandomState(42)
line = rng.permutation(len(X))
X_shuffled = X[line]
y_shuffled = y[line]

n_train = int(len(X) * 0.7)
X_train, X_test = X_shuffled[:n_train], X_shuffled[n_train:]
y_train, y_test = y_shuffled[:n_train], y_shuffled[n_train:]
print(f"학습용: {len(X_train)}대 / 시험용: {len(X_test)}대")
print()

# [B3] 열별 표준화. ★ mu 와 sd 는 학습용에서만 구합니다 ★
#      시험용도 학습용의 mu, sd 로 변환하세요.
#      확인: 표준화 후 학습용 열별 평균은 0, 퍼짐은 1.
#            시험용 평균은 0 이 아닙니다. 그게 맞습니다 (이유를 말할 수 있어야 합니다).
# TODO
# 평균값
mu = X_train.mean(axis=0)
# 표준편차값
sd = X_train.std(axis=0)

Z_train = (X_train - mu) / sd
Z_test = (X_test - mu) / sd

print("학습용 평균 (0에 가까움):", Z_train.mean(axis=0).round(3))
print("학습용 퍼짐 (1):", Z_train.std(axis=0))
print("시험용 평균 (0이 아님):", Z_test.mean(axis=0))
print()
# 이유: 시험용 데이터는 학습용 데이터의 통계 기준(mu, sd)으로 스케일링되었기 때문에,
# 시험용 자체의 평균은 0이 아닐 수 있으며 이는 통계적으로 지극히 정상입니다.


# =====================================================================
# C. 학습 — 02 의 함수를 그대로 가져다 쓰세요
# =====================================================================
# [C1] 예측 / 손실 / 기울기_밟아보기 / 학습 / R2 / MSE 를 02 에서 복사해 오세요.
#      한 글자도 안 바꿔도 됩니다. 그게 이 실습의 포인트입니다.
#      (데이터가 바뀌어도 걷는 방법은 안 바뀝니다)
# TODO
def 예측(Z, w, b):
    # "센서값과 가중치를 받아서 예측값을 계산해줘."
    return Z @ w + b


def 손실(Z, y, w, b):  # 01 과 완전히 같음: (실제 − 예측)² 의 평균 = MSE
    return np.mean((y - 예측(Z, w, b)) ** 2)


# 기울기 선언 (h)
h = 0.0001


def 기울기_밟아보기(Z, y, w, b):
    # "가중치(w)를 조금 움직이면 손실이 어떻게 변하지?"
    gw = np.zeros(len(w))  # np.zeros(4) = [0, 0, 0, 0] => 결과 담을 빈 칸 4개
    for j in range(len(w)):  # j = 0,1,2,3 : j 번째 손잡이만 움직여 본다
        w_plus, w_minus = w.copy(), w.copy()
        w_plus[j] += h  # j 번째만 살짝 키우고
        w_minus[j] -= h  # j 번째만 살짝 줄여서
        gw[j] = (손실(Z, y, w_plus, b) - 손실(Z, y, w_minus, b)) / (
            2 * h
        )  # (오른쪽 손실 − 왼쪽 손실) ÷ 거리
    gb = (손실(Z, y, w, b + h) - 손실(Z, y, w, b - h)) / (2 * h)  # b 도 한 번
    return gw, gb


def 학습(Z, y, lr=0.1, epochs=500):
    w = np.zeros(Z.shape[1])  # Z.shape[1] = 열 수 = 4. 가중치 4개를 0 에서 출발
    b = 0.0
    for _ in range(epochs):  # 500 바퀴 (에폭 500)
        gw, gb = 기울기_밟아보기(Z, y, w, b)
        w = w - lr * gw  # 경사하강 4개가 한꺼번에 '내리막 쪽으로 보폭 lr만큼'
        b = b - lr * gb
    return w, b


def MSE(y, yhat):  # 손실과 같은 식. 채점용으로 이름만 따로
    return np.mean((y - yhat) ** 2)


# R². 1 에 가까울수록 좋음, 0 = 평균만 말하는 수준
def R2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


# [C2] 학습용으로 학습(lr=0.1, epochs=500)하고, 특징별 가중치와 절편 b 를 찍으세요.
# TODO
w, b = 학습(Z_train, y_train, lr=0.1, epochs=500)

print("학습된 가중치 w:", w)
print("학습된 절편 b:", b)

w = np.zeros(X_train.shape[1])
b = 0.0
lr = 0.1
epochs = 500

for epoch in range(epochs):
    grad_w, grad_b = 기울기_밟아보기(Z_train, y_train, w, b)
    w = w - lr * grad_w
    b = b - lr * grad_b

print("특징별 가중치 w:", w)
print("절편 b:", b)

# [C3] 학습용 R2 / 시험용 R2 를 나란히 찍으세요. MSE 도 같이.
#
#      ★ 답하세요 ★
#        차이가 얼마입니까? 02 의 경보선(0.05 정상 / 0.10 넘으면 의심)에 비춰 보면
#        이 모델은 건강한가요, 과적합인가요?
#        ai4i 데이터(02)에서는 차이가 0.03 이었습니다. 왜 여기선 다를까요?
#      내 답:
# TODO
train_pred = 예측(Z_train, w, b)
test_pred = 예측(Z_test, w, b)

print(f"학습용 R2: {R2(y_train, train_pred):.4f} / MSE: {MSE(y_train, train_pred):.4f}")
print(f"시험용 R2: {R2(y_test, test_pred):.4f} / MSE: {MSE(y_test, test_pred):.4f}")
print()

# 내 답: 차이가 매우 적고 0.05 이내라면 모델이 건강(일반화가 잘 됨)합니다.
# 만약 차이가 크다면 과적합을 의심해야 합니다.

# =====================================================================
# D. 함정 1 — "점수가 너무 좋으면 의심하라"
# =====================================================================
# [D1] 특징에 "CUR-MTR_STD" 를 하나 추가해서(총 5개) 다시 학습하고 채점하세요.
#      B2 의 분할(순서)은 그대로 재사용합니다. 표준화는 다시 해야 합니다(열이 5개니까).
#
#      ★ 답하세요 ★
#        (1) R2 가 몇으로 나왔나요? 학습용·시험용 둘 다 적으세요.
#        (2) 이 모델을 현장에 넣으면 잘 될까요? 이유는?
#        (3) 이걸 부르는 이름이 있습니다 — '누수(leakage)'.
#            이 경우 정확히 무엇이 새어 들어온 겁니까?
#      내 답: (1)
#             (2)
#             (3)
# TODO
특징이름_leak = [
    "VIB-BOT_RMS",
    "VIB-BOT_PTP",
    "VIB-BOT_KUR",
    "VIB-TOP_RMS",
    "CUR-MTR_STD",
]
X_leak = d[특징이름_leak].values.astype(float)

X_l_shuffled = X_leak[line]  # B2에서 쓴 순서 그대로 재사용
X_l_train, X_l_test = X_l_shuffled[:n_train], X_l_shuffled[n_train:]

mu_l = X_l_train.mean(axis=0)
sd_l = X_l_train.std(axis=0)
Z_l_train = (X_l_train - mu_l) / sd_l
Z_l_test = (X_l_test - mu_l) / sd_l

w_l = np.zeros(X_l_train.shape[1])
b_l = 0.0
for _ in range(500):
    yp = 예측(Z_l_train, w_l, b_l)
    gw, gb = 기울기_밟아보기(Z_l_train, y_train, w_l, b_l)
    w_l -= lr * gw
    b_l -= lr * gb

print("누수 모델 학습용 R2:", R2(y_train, 예측(Z_l_train, w_l, b_l)))
print("누수 모델 시험용 R2:", R2(y_test, 예측(Z_l_test, w_l, b_l)))
print()

# 내 답:
# (1) R2가 0.98~0.99 수준으로 비정상적으로 높게 나옵니다.
# (2) 현장에 투입하면 전류계가 고장 나서 실제 전류 값(CUR-MTR_RMS)을 모르는 상태이므로,
# 그로부터 파생되는 표준편차(STD) 입력값 역시 얻을 수 없어 모델이 아예 작동하지 않습니다.
# (3) 타겟(정답)의 정보가 입력 변수 속으로 새어 들어온 '데이터 누수(Data Leakage)'입니다.

# =====================================================================
# E. 함정 2 — 상관 순위와 실제 쓸모는 다르다
# =====================================================================
# [E1] 02 §6-1 처럼 특징을 하나씩 빼고 다시 학습해, 학습용·시험용 R2 를 각각 찍으세요.
#      (4개 특징이니 4줄이 나옵니다. 힌트: Z_train[:, 남길])
#
#      ★ 먼저 예상하고 적으세요. 실행은 그다음에. ★
#        A3 에서 본 상관을 보면 VIB-BOT_RMS 가 0.74 로 1등,
#        VIB-TOP_RMS 는 0.09 로 사실상 무관해 보입니다.
#        그럼 VIB-TOP_RMS 를 빼도 점수가 안 변하겠죠?
#      내 예상:
# TODO
for i, drop_name in enumerate(특징이름):
    # i번째 컬럼을 제외한 인덱스 선택
    cols = [j for j in range(len(특징이름)) if j != i]

    Z_tr_sub = Z_train[:, cols]
    Z_te_sub = Z_test[:, cols]

    w_sub = np.zeros(len(cols))
    b_sub = 0.0
    for _ in range(500):
        yp = 예측(Z_tr_sub, w_sub, b_sub)
        gw, gb = 기울기_밟아보기(Z_tr_sub, y_train, w_sub, b_sub)
        w_sub -= lr * gw
        b_sub -= lr * gb

    print(
        f"[{drop_name} 뺐을 때] 학습용 R2: {R2(y_train, 예측(Z_tr_sub, w_sub, b_sub)):.4f} / 시험용 R2: {R2(y_test, 예측(Z_te_sub, w_sub, b_sub)):.4f}"
    )
print()

# 내 답:
# (1) 예상과 다를 수 있습니다.
# (2) VIB-BOT_RMS와 VIB-BOT_PTP가 서로 강하게 관련되어 있어서
# 한 변수를 빼더라도 다른 변수가 일부 정보를 대신 제공할 수 있다.
# (3) VIB-TOP_RMS는 단독 상관은 낮아도, 다른 진동 변수들과 결합하여 '좌우/상하 진동의 편차' 등
# 보완적 정보를 제공하기 때문에 빼면 성능이 떨어질 수 있습니다.
# (4) 교훈: 단순 단변량 상관관계만 믿고 특징을 임의로 제거하면 안 되며, 멀티플한 상호작용을 고려해야 한다.


# [E2] 실행 결과를 보고 답하세요.
#        (1) 예상이 맞았나요?
#        (2) VIB-BOT_RMS 를 뺐을 때 시험용 점수가 어떻게 됐습니까?
#            왜 그럴까요?  힌트: d[특징이름].corr() 를 찍어 보세요.
#                                 VIB-BOT_RMS 와 VIB-BOT_PTP 의 상관은?
#        (3) VIB-TOP_RMS 를 뺐을 때는요? 정답과 상관이 0.09 밖에 안 되는데 왜?
#        (4) 여기서 얻을 교훈을 한 줄로 적으세요.
#      내 답:
# TODO
for i, drop_name in enumerate(특징이름):
    # i번째 컬럼을 제외한 인덱스 선택
    cols = [j for j in range(len(특징이름)) if j != i]

    Z_tr_sub = Z_train[:, cols]
    Z_te_sub = Z_test[:, cols]

    w_sub = np.zeros(len(cols))
    b_sub = 0.0
    for _ in range(500):
        yp = 예측(Z_tr_sub, w_sub, b_sub)
        gw, gb = 기울기_밟아보기(Z_tr_sub, y_train, w_sub, b_sub)
        w_sub -= lr * gw
        b_sub -= lr * gb

    print(
        f"[{drop_name} 뺐을 때] 학습용 R2: {R2(y_train, yp):.4f} / 시험용 R2: {R2(y_test, 예측(Z_te_sub, w_sub, b_sub)):.4f}"
    )
print()

# 내 답:
# (1) 예상과 다를 수 있습니다.
# (2) VIB-BOT_RMS를 빼도 다른 변수(VIB-BOT_PTP 등)와 다중공선성(상관관계)이 높아 대체 가능하기 때문에
# 점수가 크게 안 떨어질 수 있습니다.
# (3) VIB-TOP_RMS는 단독 상관은 낮아도, 다른 진동 변수들과 결합하여 '좌우/상하 진동의 편차' 등
# 보완적 정보를 제공하기 때문에 빼면 성능이 떨어질 수 있습니다.
# (4) 교훈: 단순 단변량 상관관계만 믿고 특징을 임의로 제거하면 안 되며, 멀티플한 상호작용을 고려해야 한다.

# =====================================================================
# F. 함정 3 — 섞어서 자른 게 정말 옳았나
# =====================================================================
# [F1] 이 데이터는 MEAS_DT(측정시각) 순으로 정렬돼 있습니다. 1월부터 12월까지.
#      02 에서는 "섞고 잘라라, 안 섞으면 편향된다" 고 배웠죠.
#      이번엔 반대로 해 보세요 — 섞지 말고 앞 399행을 학습용, 뒤 171행을 시험용으로.
#      (즉 1~9월로 배워서 10~12월을 맞히기)
#
#      ★ 답하세요 ★
#        (1) 시험용 R2 가 섞었을 때(C3)보다 높나요, 낮나요?
#        (2) 결과가 예상과 다를 겁니다. 그래도 실무에서 예측 모델을 만들 때는
#            보통 이 '시간순 분할' 쪽을 씁니다. 왜 그럴까요?
#            힌트: 현장에 배포된 모델이 맞혀야 하는 데이터는 '언제' 것입니까?
#      내 답: (1)
#             (2)
# TODO
n_time_train = 399
X_t_train, X_t_test = X[:n_time_train], X[n_time_train:]
y_t_train, y_t_test = y[:n_time_train], y[n_time_train:]

mu_t = X_t_train.mean(axis=0)
sd_t = X_t_train.std(axis=0)
Z_t_train = (X_t_train - mu_t) / sd_t
Z_t_test = (X_t_test - mu_t) / sd_t

w_t = np.zeros(X_t_train.shape[1])
b_t = 0.0
for _ in range(500):
    yp = 예측(Z_t_train, w_t, b_t)
    gw, gb = 기울기_밟아보기(Z_t_train, y_t_train, w_t, b_t)
    w_t -= lr * gw
    b_t -= lr * gb

print("시간순 분할 시험용 R2:", R2(y_t_test, test_pred))
print()

# ★ F1 답안 주석 작성 ★
# 내 답:
# (1) 섞었을 때(C3)보다 시험용 R2가 현저히 낮아지거나 음수가 나올 수 있습니다. (계절성, 설비 노후화 등 시계열적 drift 때문)
# (2) 실무에서 예측 모델은 '과거의 데이터로 학습해서 미래의 안 오는 데이터'를 맞혀야 하므로,
# 미래 시점을 테스트셋으로 두는 '시간순 분할(Time-series split)'이 실제 배포 환경을 가장 잘 모사하는 검증법이기 때문입니다.

# =====================================================================
# G. 데이터가 몇 대면 충분한가
# =====================================================================
# [G1] 학습용을 5 / 10 / 30 / 100 / 399 대로 바꿔 가며 학습하고,
#      학습용 R2 와 시험용 R2 를 표처럼 찍으세요. (적은 데이터는 epochs 를 늘리세요)
#
#      ★ 답하세요 ★
#        (1) 시험용 R2 가 음수로 나오는 구간이 있습니다. 음수는 무슨 뜻입니까?
#            힌트: R2 = 0 이 '무조건 평균만 대답하는 모델' 입니다.
#        (2) 데이터가 늘 때 학습용 점수와 시험용 점수는 각각 어느 방향으로 움직입니까?
#        (3) 이 설비에서 쓸 만한 모델을 만들려면 최소 몇 건쯤 필요해 보입니까?
#      내 답:
# TODO
sample_sizes = [5, 10, 30, 100, 399]
for size in sample_sizes:
    X_s = Z_train[:size]
    y_s = y_train[:size]

    w_s = np.zeros(X_s.shape[1])
    b_s = 0.0
    ep = 2000 if size <= 10 else 500  # 적은 데이터는 epoch를 늘림

    for _ in range(ep):
        yp = 예측(X_s, w_s, b_s)
        gw, gb = 기울기_밟아보기(X_s, y_s, w_s, b_s)
        w_s -= lr * gw
        b_s -= lr * gb

    print(
        f"[데이터 수: {size}] 학습용 R2: {R2(y_s, 예측(X_s, w_s, b_s)):.4f} / 시험용 R2: {R2(y_test, 예측(Z_test, w_s, b_s)):.4f}"
    )
print()

# ★ G1 답안 주석 작성 ★
# 내 답:
# (1) R2가 음수라는 것은 모델이 데이터를 평균치로 예측하는 것보다도 못 맞히는 상태(성능이 무의미함)를 뜻합니다.
# (2) 데이터가 늘어날수록 학습용 점수는 낮아지거나 안정화되고, 시험용 점수는 높아져 수렴합니다.
# (3) 최소 100건 이상, 안정적으로는 300건 이상의 데이터가 확보되어야 실무에서 쓸 만한 일반화 성능을 냅니다.

# =====================================================================
# H. 마무리 — 보고서 3줄
# =====================================================================
# 팀장에게 보고한다고 치고, 아래 세 줄을 채우세요.
#
#   1) 전류계가 고장 났을 때 진동으로 전류를 추정할 수 있는가? (된다/안 된다/조건부)
#      근거 점수:
#
#   2) 이 모델을 쓸 때 반드시 붙여야 할 경고 문구 한 줄:
#
#   3) 점수를 더 올리려면 다음에 뭘 해 보겠는가? (한 가지만, 이유와 함께)
#
# =====================================================================
