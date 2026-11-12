#!/usr/bin/env python3
"""
김프 로직 검증 - 답을 아는 데이터를 넣는다.

제일 중요한 건 [3] '어느 쪽이 되돌리나'를 구분하는가다.
  UPSIDE  : 김프가 '업비트가 내려서' 되돌아감  -> 먹을 수 있음(예) 이 나와야
  GLOBAL  : 김프가 '글로벌이 올라서' 되돌아감  -> 먹을 수 없음(아니오) 가 나와야
  NOREV   : 김프가 안 되돌아감                 -> 회귀 안 함
"""
import numpy as np, pandas as pd
import upbit_leadlag_check as U
import kimchi_premium_check as K

K.HORIZONS_MIN = [60, 240]
rng = np.random.default_rng(9)
N = 8000
idx = pd.date_range("2026-04-01", periods=N, freq="5min", tz="UTC")

usdt = pd.Series(1400.0 + rng.normal(0, 1, N).cumsum() * 0.01, index=idx)
base = 100_000_000 * np.exp(np.cumsum(rng.normal(0, 0.001, N)))   # 글로벌 USD 가격

def make(mode):
    prem = np.zeros(N)
    for t in range(1, N):
        shock = rng.normal(0, 0.002)
        if mode == "NOREV":
            prem[t] = prem[t-1] + shock            # 랜덤워크 = 안 되돌아옴
        else:
            prem[t] = prem[t-1] * 0.97 + shock     # 평균회귀
    g = base.copy()
    if mode == "GLOBAL":
        g = base * (1 - prem)                       # 글로벌이 움직여서 김프가 해소됨
    up = g * usdt.values * (1 + prem)
    if mode == "GLOBAL":
        up = base * usdt.values                     # 업비트는 가만히
    return pd.Series(up, index=idx), pd.Series(g, index=idx)

data = {}
for mode in ["UPSIDE", "GLOBAL", "NOREV"]:
    up, g = make(mode)
    df = pd.DataFrame({"upbit": up, "binance": g, "usdt": usdt}, index=idx)
    df["global_krw"] = df["binance"] * df["usdt"]
    df["premium"] = df["upbit"] / df["global_krw"] - 1
    data[mode] = df

print("="*80); print("[1] 김프 통계"); print("="*80)
print(K.premium_stats(data).to_string(index=False))

print("\n"+"="*80); print("[2] 평균회귀 감지"); print("="*80)
rv = K.reversion_test(data); print(rv.to_string(index=False))
ok = True
for m, expect in [("UPSIDE","평균회귀"), ("NOREV","랜덤워크와 구분 안 됨")]:
    got = rv[rv["코인"]==m].iloc[0]["판정"]
    h = got == expect; ok &= h
    print(f"  {m}: {got} (기대 {expect}) -> {'OK' if h else '실패'}")

print("\n"+"="*80); print("[3] 어느 쪽이 되돌리나 - 제일 중요"); print("="*80)
ch = K.channel_test(data); print(ch.to_string(index=False))
u = ch[ch["코인"]=="UPSIDE"].iloc[0]; g = ch[ch["코인"]=="GLOBAL"].iloc[0]
h = u["먹을 수 있나"]=="예"; ok &= h
print(f"\n  UPSIDE: 먹을 수 있나={u['먹을 수 있나']} (업비트가 내려서 해소) -> {'OK' if h else '실패'}")
h = g["먹을 수 있나"]=="아니오"; ok &= h
print(f"  GLOBAL: 먹을 수 있나={g['먹을 수 있나']} (글로벌이 올라서 해소) -> {'OK (못 먹는 걸 걸러냄)' if h else '실패'}")

print("\n"+"="*80)
print("[4] 실제 돈 - 이게 최종 판정이어야 한다")
print("  ([3]은 랜덤워크 착시를 완전히 못 피한다. 김프가 정의상 업비트 가격에 들어있어서다.")
print("   그래서 baseline 대비 실제 수익이 최종 심판이다)")
print("="*80)
q = K.quantile_test(data)
print(q.to_string(index=False))

u4 = q[q["코인"] == "UPSIDE"]["판정"].tolist()
g4 = q[q["코인"] == "GLOBAL"]["판정"].tolist()
n4 = q[q["코인"] == "NOREV"]["판정"].tolist()
print(f"\n  UPSIDE 통과 {u4.count('통과')}/{len(u4)}   (진짜 엣지 -> 통과해야)")
print(f"  GLOBAL 통과 {g4.count('통과')}/{len(g4)}   (못 먹는 것 -> 탈락해야)")
print(f"  NOREV  통과 {n4.count('통과')}/{len(n4)}   (랜덤워크 -> 탈락해야)")

h = u4.count("통과") > 0; ok &= h
print(f"\n  진짜 엣지를 통과시키나: {'OK' if h else '실패'}")
h = n4.count("통과") == 0; ok &= h
print(f"  랜덤워크를 걸러내나:   {'OK' if h else '실패'}")

print("\n"+"="*80); print("최종:", "김프 로직 검증 통과" if ok else "문제 있음"); print("="*80)
