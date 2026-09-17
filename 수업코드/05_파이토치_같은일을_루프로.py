# =====================================================================
#  05. 파이토치 — 01~03 의 '걷기'를 그대로, 미분만 자동으로. 그리고 언제 쓰나
# =====================================================================
#
# 파이토치(PyTorch, 코드에선 torch) = 딥러닝 도구 상자.
# - 사이킷런이 'fit 한 줄에 다 해 주는 완성품"이라면, 파이토치는 우리가 01 에서 짠 루프를 그대로 쓰되
#   '밟아보기(미분)'와 '한 걸음(갱신)'만 대신해 주는 부품 상자.
#
# 언제쓰나?
# 1. 데이터가 아주 커서 조금씩 나눠 학습해야 할 때
# 2. 학습 과정(손실 보폭 걸음 수)을 내 손으로 조절해야 할 때
# 3. 다음 과정에서 배울 '층을 여러 겹 쌓는 모델'(신경망)을 만들 때 - 오늘은 언급만
#
# 표 데이터 + 선형모델이면 사이킷런이 더 편합니다. 여기선 '같은 답이 나온다'를 확인해서
# "파이토치가 별개의 마법이 아니라 01의 그 루프"라는 걸 아는 게 목적

import os
import numpy as np
import pandas as pd
import torch

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "수업용데이터")
df = pd.read_csv(os.path.join(DATA, "11_설비센서_ai4i.csv"), encoding="utf-8-sig")
특징이름 = ["공기온도", "회전수", "토크", "공구마모"]
torch.manual_seed(0)
# 파이토치는 w, b 시작값을 난수로 잡습니다 (01에선 0에서 출발). 고정해야 매번 같은 결과

# =====================================================================
# 1. 텐서 — numpy 배열의 파이토치 판 (딱 하나만 조심: float32)
# =====================================================================
# 텐서(tensor) = 파이토치의 배열. numpy 배열과 거의 같고, '미분 추적 기능'이 붙어 있는 게 차이.
x = df["공기온도"].values
y = df["공정온도"].values
x_t = torch.tensor(x, dtype=torch.float32)
# numpy -> 텐서. dtype=torch.float32를 습관처럼 붙입니다.

y_t = torch.tensor(y, dtype=torch.float32)
# 왜? 파이토치 부품들은 전부 float32를 기대. 정수(int)나 float64 가 섞이면 에러
print("[1] 텐서:", x_t[:3], "/ 모양", x_t.shape, "/ 타입", x_t.dtype)
print("    다시 numpy 로:", x_t[:3].numpy())
# .numpy()로 언제든 되돌립니다. 둘은 자유롭게 오갑니다.
# float32는 소수 7자리 정도만 정확합니다
# numpy 기본 float64 는 15자리 그래서 파일의 결과 끝자리가 01 04 와 0.0002 쯤 다를 수 있음.

# =====================================================================
# 2. 자동 미분 — 01 의 '밟아보기'를 파이토치가 대신
# =====================================================================
# 01 에선 w를 +-0.001 밀어 손실 차이로 기울기(편미분)를 쟀습니다.
# 손잡이 5개면 5번, 100만 개면 100만 번.
#
# 파이토치는 계산 과정을 '기록'해 뒀다가 backward() 한 번에 모든 손잡이의 기울기를 내놓습니다.
# autoagrad(자동 미분) = 계산 과정을 기록해 두고 backward() 한 번으로 모든 손잡이의 편미분을 구해 주는 기능.
# 파이토치를 쓰는 첫 번째 이유. 손잡이가 몇백만 개인 신경망에선 밟아보기가 불가능해서 이게 필수.

m, s = (
    x_t.mean(),
    x_t.std(unbiased=False),
)
# 표준화 (unbiased=False = numpy와 같은 방식의 표준편차. 안 붙이면 살짝 다른 값)
z_t = (x_t - m) / s

w = torch.tensor(0.0, requires_grad=True)
# requires_grad=True  = "이 값은 손잡이다. 기울기를 추적해라"
b = torch.tensor(0.0, requires_grad=True)

손실 = ((y_t - (w * z_t + b)) ** 2).mean()
# 01 의 MSE 그래도. 이 계산이 '기록'됩니다
손실.backward()  # <- 미분! 이 한 줄이 01의 기울기_밟아보기(). 결과는 w.grad, b.grad에 담김
print(
    "\n[2] autograd 가 구한 기울기: w 방향",
    round(w.grad.item(), 3),
    "/ b 방향",
    round(b.grad.item(), 3),
)
# .item() = 숫자 하나짜리 텐서를 그냥 파이썬 숫자로. 출력하거나 round 할 때 필요


# 정말 맞나? 01 방식(밟아보기)으로 대조 (밟아보기는 아주 작은 차이를 재니, 정밀한 numpy float64로 합니다.)

h = 1e-4
z64, y64 = z_t.numpy().astype(np.float64), y_t.numpy().astype(np.float64)
L = lambda wv, bv: np.mean((y64 - (wv * z64 + bv)) ** 2)
# lambda = 이름 없는 한 줄 함수. L(w, b) = 손실

print(
    "    밟아보기로 잰 기울기:      w 방향",
    round((L(h, 0) - L(-h, 0)) / (2 * h), 3),
    "/ b 방향",
    round((L(0, h) - L(0, -h)) / (2 * h), 3),
    "← 같다",
)
# 주의 .grad는 backward  할 때마다 '더해집니다'. 매 걸음 전에 비워야 합니다

# =====================================================================
# 3. 파이토치 표준 루프 — 부품 세 개 + 네 줄
# =====================================================================
# 부품 (01 의 무엇에 해당하는지 보세요):
#   torch.nn.Linear(입력 수, 출력 수) = w, b 를 가진 '직선 한 층'  (01 의 w, b + 예측())
#   torch.nn.MSELoss()               = 손실 함수                    (01 의 손실())
#   torch.optim.SGD(…, lr)           = 갱신 담당. step() 이 "w = w − lr × 기울기"  (01 의 한 걸음)
#   옵티마이저(optimizer) = "기울기를 받아 손잡이를 어떻게 갱신할지" 규칙을 맡은 부품.
#           SGD = 01 의 규칙 그대로 (w − lr×기울기). Adam = 손잡이마다 보폭을 알아서 조절하는 인기 대안.
Z = z_t.reshape(
    -1, 1
)  # nn.Linear 는 (설비 수, 센서 수) 세로 표를 받습니다 (사이킷런 규칙 1 과 같음)
Y = y_t.reshape(
    -1, 1
)  # 정답도 (설비 수, 1) 로 맞춥니다. 안 맞추면 경고만 뜨고 엉뚱한 손실이 조용히 계산됩니다

torch.manual_seed(0)
model = torch.nn.Linear(1, 1)  # 입력 개(공기온도) -> 출력 1개(공정온도). w 1개 + b 1개
loss_fn = torch.nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
# model.parameters() =  이 모델의 손잡이들(w, b). "얘네를 갱신해라"

print("\n[3] 학습 루프 — 이 네 줄이 01 의 '한 걸음'")
for epoch in range(300):  # 01 과 같은 300 에폭
    opt.zero_grad()  # 1 기울기 비우기(누적 방지)
    loss = loss_fn(model(Z), Y)  # 2 예측 -> 손실 (model(Z) = wz + b를 200대에 대해)
    loss.backward()  # 3 기울기 자동 계산 (01의 밟아보기)
    opt.step()  # 4 한 걸음 (01 의 w = w - lr 기울기)
    if epoch in (0, 10, 30, 100, 299):
        print(f"    epoch {epoch:3d}  손실 {loss.item():.4f}")
# 95900 -> 1106 -> 1.04 -> 0.892 -> 0.892. 01 과 같은 모양으로 내려와 같은 바닥(0.892)에 멈춥니다.
# 이 네줄(zero_grad -> loss -> backward -> step)은 파이토치 코드 어디서나 똑같이 나옵니다. 외우세요.

# 결과를 원래 눈금으로 되돌려 01 04 와 대조
w_z, b_z = (
    model.weight.item(),
    model.bias.item(),
    # 학습된 손잡이. 사이킷런의 coef_, intercept_ 에 해당
)
print(
    f"    원래 눈금: 공정온도 ≈ {w_z / s.item():.4f} × 공기온도 + {b_z - w_z * m.item() / s.item():.4f}"
)
print(
    "    01 (손) / 04 (사이킷런): 0.9840 × 공기온도 + 14.7799  ← 같다 (끝자리 0.0002 차이는 float32 탓)"
)


# =====================================================================
# 4. 다변수 + train/test — 02 를 파이토치로 (나누기·표준화는 사이킷런을 빌려 씀)
# =====================================================================
# 파이토치와 사이킷런은 '같이' 씁니다. 나누고 표준화하고 채점하는 건 사이킷런이 편하고,
# 학습 루프만 파이토치로. numpy <-> 텐서 변환 한 줄이 둘을 잇습니다. 실무 코드도 이 조합이 보통.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[특징이름].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)  # 04 같은 분할

scaler = StandardScaler().fit(X_train)
to_t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
# numpy -> float32 텐서 도우미. 매번 길게 안 쓰려고

Ztr, Zte = to_t(scaler.transform(X_train)), to_t(scaler.transform(X_test))
Ytr, Yte = to_t(y_train).reshape(-1, 1), to_t(y_test).reshape(-1, 1)

torch.manual_seed(0)
model4 = torch.nn.Linear(4, 1)
# 입력 4개(센서 4개) -> 출력 1개. 숫자 하나만 바뀜. w 4개 + b 1개

opt = torch.optim.SGD(model4.parameters(), lr=0.1)
for epoch in range(500):
    opt.zero_grad()
    loss = loss_fn(model4(Ztr), Ytr)
    loss.backward()
    opt.step()  # 네 줄을 세미콜론으로 한 줄에


def R2(y, yhat):  # 02의 알제곱 을 텐서용
    return (1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()).item()


# with torch.no_grad(): 이 블록 안에선 기록(autograd)을 끕니다. 예측만 할 땐 관례
with torch.no_grad():
    print(
        "\n[4] 다변수 회귀 — train R²",
        round(R2(Ytr, model4(Ztr)), 4),
        "/ test R²",
        round(R2(Yte, model4(Zte)), 4),
    )
    print(
        "    표준화 가중치:",
        {k: round(v, 3) for k, v in zip(특징이름, model4.weight.squeeze(0).tolist())},
    )
    # weight 는 (1, 4) 표 모양. .squeeze(0) = 크기 1인 겉껍질을 벗겨 (4,)로. .tolist() = 파이썬 리스트로

print("    04 사이킷런: train 0.8155 / test 0.7425, 공기온도 1.986 ← 같은 자리")
# 사이킷런 공식으로, 파이토치는 500 걸음 걸어서. 같은 바닥에 도착. 과적합 판단도 04 같습니다.

# =====================================================================
# 5. 분류 — 03 을 파이토치로: 손실 이름 하나만 바뀐다
# =====================================================================
# 03 에서 배운 두 가지 변화(sigmoid, 로그손실)를 파이토치는 손실 함수 하나에 담아 둡니다:
# torch.nn.BCEWithLogitsLoss() = "직선값(logit)d을 받아 안에서 sigmoid를 씌우고 로그손실(BCE) 계산"
# (BCE = Binary Cross-Entropy) = 03의 로그손실. 같은 것의 다른 이름
# 층(nn.Linear) 옵티마이저 네 줄 루프는 회귀와 완전히 같습니다. "분류 = 회귀 + sigmoid + 손실 교체"가 여기서도 그대로.

yc = df["고장여부"].values
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X, yc, test_size=0.3, random_state=3, stratify=yc
)  # 04 와 같은 분할

scaler_c = StandardScaler().fit(Xc_train)
Zc_tr, Zc_te = to_t(scaler_c.transform(Xc_train)), to_t(scaler_c.transform(Xc_test))
Yc_tr = to_t(yc_train).reshape(-1, 1)
# 0/1  정답도 float32 로! (정수면 손실 함수가 에러)

torch.manual_seed(0)
clf = torch.nn.Linear(4, 1)
loss_c = torch.nn.BCEWithLogitsLoss()  # <- 회귀와 다른 유일한 줄
opt = torch.optim.SGD(clf.parameters(), lr=0.5)  # 03과 같은 lr, 같은 걸음 수
for epoch in range(2000):
    opt.zero_grad()
    loss = loss_c(clf(Zc_tr), Yc_tr)
    loss.backward()
    opt.step()

with torch.no_grad():
    p = torch.sigmoid(clf(Zc_te)).squeeze(1).numpy()
    # 직선값 -> sigmoid -> 고장확률. 04 의 predict_proba[:, 1]을 직접 만든 것

print("\n[5] 분류 — 시험용 고장 확률 상위 5:", np.sort(p)[::-1][:5].round(3))
from sklearn.metrics import (
    recall_score,
    precision_score,
)  # 채점은 사이킷런 함수를 그대로 빌려 씀

for th in [0.5, 0.2]:
    판정 = (p >= th).astype(int)
    print(
        f"    임계값 {th}: 재현율 {recall_score(yc_test, 판정, zero_division=0):.2f}  정밀도 {precision_score(yc_test, 판정, zero_division=0):.2f}"
    )
# 04 사이킷런(같은 분할)과 비교: 04 는 1등 확률 0.366, 여기선 0.539.
# 파이토치엔 규제가 없어 확률이 더 벌어집니다.
# 하지만 1등(0.539)은 진짜 고장이 아니라 헛경보고, 임계값 0.5 에선 여전히 고장을 못 잡습니다. 0.2 로 내려야 1대.
# 도구가 달라도 결론은 같습니다. 불균형 데이터는 입계값 class_weight 로 손 봐야함 (04)
# 파이토치에서 class_weight 는 BCEWithLogitsLoss(pos_weight=torch.tensor([22.]))로 줍니다.


# =====================================================================
# 6. 미니배치 — 데이터가 클 때 조금씩 나눠 걷기 (파이토치를 쓰는 두 번째 이유)
# =====================================================================
# 지금까진 매 걸음 140대 전부로 기울기를 쟀습니다. 데이터가 100만 건이면 한 걸음이 너무 무겁고 메모리에 안 들어감.
# 그래서 16대씩 뽑아 한 걸음, 다음 16대로 한 걸음... 이렇게 갑니다. DataLoader 가 섞고 나눠 줍니다.
# 배치 = 한 걸음에 보는 데이터 묶음. 미니배치 = 그 묶음이 전체의 일부일 때.
# 에폭 = 전체를 한 바퀴 다 보는 것. 배치 16이면 140대는 한 에폭에 9걸음(16x8=128, 남은 12대가 9번째)
# DataLoader = "섞어서 배치 크기만큼 잘라 하나씩 건네주는" 부품.
from torch.utils.data import TensorDataset, DataLoader
#
# 습관 하나 추가: 정답(y)도 표준화합니다. 공정온도가 310 근처라 0에서 출발하면 310 까지 가는 데 걸음이 많이 들고,
# 특히 Adam은 한 걸음이 lr 크기(0.01)로 제한돼 310 까지 3만 걸음이 필요
# R2는 눈금을 바꿔도 같은 값이라 채점엔 영향 없음)

y_m, y_s = Ytr.mean(), Ytr.std()
Ytr_s, Yte_s = (Ytr - y_m) / y_s, (Yte - y_m) / y_s

loader = DataLoader(TensorDataset(Ztr, Ytr_s), batch_size=16, shuffle=True)
# 16대씩, 매 에폭 새로 섞어서
print("\n[6] 미니배치 — 한 에폭에", len(loader), "걸음 (140대 ÷ 16)")

torch.manual_seed(0)
model_mb = torch.nn.Linear(4, 1)
opt = torch.optim.Adam(model_mb.parameters(), lr=0.01)
# Adam: 손잡이마다 보폭을 알아서 조절. 실무에서 가장 흔한 선택. lr은 0.001~0.01이 관례
for epoch in range(100):
    for (
        zb,
        yb,
    ) in (
        loader
    ):  # <- 미니배치 루프. zb = 이번 16대의 센서, yb = 그 정답. 안쪽 네 줄은 그대로
        opt.zero_grad()
        loss = loss_fn(model_mb(zb), yb)
        loss.backward()
        opt.step()
with torch.no_grad():
    print(
        "    미니배치 + Adam — test R²:",
        round(R2(Yte_s, model_mb(Zte)), 4),
        "(같은 바닥에 도착)",
    )

# 언제 뭘 쓰냐?
#
# 표 데이터에 선형모델이나 트리 -> 사이킷런
# 데이터가 너무 커서 나눠 넣어야 하거나, 손실을 직접 만들어야 하거나, 층을 쌓는 신경망이면 -> 파이토치
# 파이토치를 배운 이유?
# 신경망 할 때 이 네줄이 그대로 나옴.
# 140대 한꺼번에 500걸음이나, 16대씩 900걸음(100에폭 x9)이나 같은 자리. 데이터가 커도 메모리엔 16대만 올리면 됨

# [메모리?]
# 메모리(RAM) = 작업 책상, 하드디스크 = 책장. 계산은 책상에 펼쳐 놓은 것만 할 수 있습니다.
# 1) 파일이 컴퓨터에 저장돼 있는 건 '책장에 꽂힌' 상태.
#   read_csv로 읽어서 변수에 담으면 '책상에 펼친' 상태.
# 2) 우리 데이터는 140행이라 책상에 다 펼쳐도 자리가 남습니다. 그래서 04까지는 한꺼번에 넣었음.
# 3) 100만 행이면 책상이 모자랍니다. 다 펼치려는 순간 컴퓨터가 멈춥니다.
# 4) 그래서 16행만 펼쳤다 치우고, 다음 16행 펼쳤다 치우고 -- 이게 미니배치입니다.
#    책장에는 100만 행이 있어도 책상 위에는 항상 16행만 있습니다.
#
# 숫자로 확인해 보면:
#   숫자 하나 = 8바이트.  100만 행 x 컬럼 100개 = 약 0.8GB. 램이 16GB면 될 것 같습니다.
#   그런데 학습을 시작하면 기울기와 중간 계산값 사본이 몇 개씩 생겨서 3~5배가 됩니다. 그래서 터집니다.

# =====================================================================
# 7. 다음 과정 예고 — 층을 쌓으면 신경망 (지금은 한 줄만, 코드 없음)
# =====================================================================
# 지금 model4는 '직선 한 겹'(nn.Linear 하나)입니다. 다음 과정에서는 이 층을 여러 겹 쌓아 '신경망'을 만드는데,
# 놀랍게도 위의 네 줄 루프(zero_grad -> loss -> backward -> step)는 한 글자도 안 바뀝니다. 층만 바뀜.
# 오늘 이 루프를 손에 익혀 두는 게 그 준비다.

# =====================================================================
# 8. 정리 — 언제 무엇을 쓰나
# =====================================================================
print("""
[8] 손코드 ↔ 사이킷런 ↔ 파이토치
    밟아보기(기울기)     ↔ (내부)               ↔ loss.backward()   (autograd)
    w = w − lr·기울기    ↔ (내부)               ↔ opt.step()        (옵티마이저)
    for epoch 루프       ↔ (내부, fit 한 줄)    ↔ 직접 씀 (zero_grad → loss → backward → step)
    sigmoid + 로그손실   ↔ LogisticRegression  ↔ BCEWithLogitsLoss
    (없음)               ↔ (없음)               ↔ DataLoader 미니배치
    섞고 나누기·표준화   ↔ train_test_split·StandardScaler ↔ (사이킷런 것을 빌려 씀)
    채점                 ↔ metrics             ↔ (사이킷런 것을 빌려 씀)

    선택 기준 한 줄:
      표 데이터, 선형모델/트리, 빨리 결과                → 사이킷런  (fit 한 줄)   ← 이 과정의 데이터는 전부 여기
      아주 큰 데이터, 학습 과정을 직접 조절, (다음 과정) 층 쌓기 → 파이토치  (루프를 내 손에)
      둘 다 심장은 01 의 경사하강. 도구는 원리를 빠르게 할 뿐, 대신하지 않습니다.
""")

# =====================================================================
# 실습 — 몇 걸음이면 충분한가
# =====================================================================
# [문제] 4번에서 500걸음을 걸었습니다. 5걸음, 50걸음, 500걸음을 비교해 보세요.
#        각각 test R2 가 얼마인가요?
#
#   힌트: 매번 torch.manual_seed(0) 으로 시작 위치를 같게 맞춰야 공정한 비교가 됩니다.
