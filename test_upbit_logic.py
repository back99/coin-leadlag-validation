#!/usr/bin/env python3
"""
업비트 v2 로직 검증 - 네트워크 없이 합성 데이터로.

v1 버그(구멍 뚫린 캔들 -> 행 기준 shift -> horizon 이 거짓말)가 실제로 고쳐졌는지가 핵심.

심어놓는 것:
  1. 정확히 30분 시차로 따라오는 알트        -> best_lag = 30분
  2. 캔들에 구멍이 뚫린 알트 (얇은 코인 흉내) -> 재색인으로 시간축이 살아야 함
  3. 한국 낮에만 따라오는 알트               -> 세션 분석이 구분해야 함
  4. 무관계 알트                            -> 0
"""

import numpy as np
import pandas as pd

import upbit_leadlag_check as U

# 5분봉 기준으로 맞춘다
U.INTERVAL_MIN = 5
U.BARS = lambda m: max(1, m // U.INTERVAL_MIN)
U.MAX_LAG = U.BARS(U.MAX_LAG_MIN)
U.EVENT_WINDOW = U.BARS(U.EVENT_WINDOW_MIN)
U.HORIZONS = [U.BARS(m) for m in U.HORIZONS_MIN]
U.LEADERS = ["KRW-BTC"]
U.EVENT_THRESHOLD = 0.003
U.MIN_EVENTS = 30

rng = np.random.default_rng(11)
N = 8000                      # 5분봉 8000개 = 약 28일
LAG_BARS = 6                  # 6바 = 30분

idx = pd.date_range("2026-01-01", periods=N, freq="5min", tz="UTC")
kst_hour = idx.tz_convert(U.KST).hour
is_day = (kst_hour >= U.KR_SESSION_START) & (kst_hour < U.KR_SESSION_END)

btc = rng.normal(0, 0.0025, N)

follow = np.zeros(N)
follow[LAG_BARS:] = 0.9 * btc[:-LAG_BARS] + rng.normal(0, 0.0008, N - LAG_BARS)

dayonly = rng.normal(0, 0.001, N)
for t in range(LAG_BARS, N):
    if is_day[t]:
        dayonly[t] = 0.9 * btc[t - LAG_BARS] + rng.normal(0, 0.0008)

none = rng.normal(0, 0.0025, N)


def to_close(r, start=1000.0):
    return start * np.exp(np.cumsum(r))


full = pd.DataFrame({
    "KRW-BTC": to_close(btc),
    "FOLLOW": to_close(follow),
    "DAYONLY": to_close(dayonly),
    "NONE": to_close(none),
}, index=idx)

# ── 얇은 코인 흉내: FOLLOW 캔들의 40% 를 무작위로 지운다 (거래 없는 분) ──
holes = rng.random(N) < 0.40
sparse_follow = full["FOLLOW"].copy()
sparse_follow[holes] = np.nan

print("=" * 76)
print("[검증 0] v1 버그 재현 - 구멍 뚫린 캔들을 dropna 로 처리하면?")
print("=" * 76)
v1_style = pd.concat([full["KRW-BTC"], sparse_follow], axis=1).dropna()
print(f"  원본 바: {N}개")
print(f"  v1 방식(dropna) 후: {len(v1_style)}개   <- {(1-len(v1_style)/N)*100:.0f}% 소실")
print("  이 상태에서 shift(6) 은 '30분 뒤'가 아니라 '남은 데이터 6번째 뒤'가 된다.")

# v2 방식: 재색인 + ffill
v2 = pd.DataFrame({
    "KRW-BTC": full["KRW-BTC"],
    "FOLLOW": sparse_follow.reindex(idx).ffill(),
    "DAYONLY": full["DAYONLY"],
    "NONE": full["NONE"],
}, index=idx).dropna()
print(f"  v2 방식(재색인+ffill) 후: {len(v2)}개   <- 시간축 보존")

ok = True
alts = ["FOLLOW", "DAYONLY", "NONE"]

print("\n" + "=" * 76)
print(f"[검증 1] 지연 상관 - 심어둔 시차 = {LAG_BARS * U.INTERVAL_MIN}분 (구멍 40% 뚫린 상태)")
print("=" * 76)
corr = U.lagged_corr_table(v2, alts)
print(corr.to_string(index=False))

r = corr[corr["alt"] == "FOLLOW"].iloc[0]
expected_min = LAG_BARS * U.INTERVAL_MIN
hit = r["best_lag(분)"] == expected_min and r["corr@best"] > 0.3
ok &= hit
print(f"\n  FOLLOW : best_lag={r['best_lag(분)']}분 (기대 {expected_min}분), "
      f"corr={r['corr@best']}  -> {'OK' if hit else '실패'}")
print("    (구멍을 뚫어놨는데도 시차를 정확히 잡으면 재색인이 제대로 된 것)")

r = corr[corr["alt"] == "NONE"].iloc[0]
hit = abs(r["corr@best"]) < 0.1
ok &= hit
print(f"  NONE   : corr@best={r['corr@best']}  -> {'OK' if hit else '실패'}")

print("\n" + "=" * 76)
print("[검증 2] 세션 분석")
print("=" * 76)
ss = U.session_study(v2, alts, 30)
print(ss.to_string(index=False))
d = ss[(ss["alt"] == "DAYONLY") & (ss["session"].str.contains("낮"))].iloc[0]
n = ss[(ss["alt"] == "DAYONLY") & (ss["session"].str.contains("밤"))].iloc[0]
if d["mean%"] is not None and n["mean%"] is not None:
    hit = d["mean%"] > n["mean%"]
    ok &= hit
    print(f"\n  DAYONLY: 낮 {d['mean%']}% vs 밤 {n['mean%']}%  -> {'OK' if hit else '실패'}")

print("\n" + "=" * 76)
print("[검증 3] 판정 기준 - 진짜 엣지는 통과시키고 극단값 사기는 걸러내는가")
print("=" * 76)
ev = U.event_study(v2, alts)
print(ev.to_string(index=False))

f = ev[(ev["alt"] == "FOLLOW") & (ev["verdict"] == "통과")]
hit = len(f) > 0
ok &= hit
print(f"\n  FOLLOW 통과 조합 {len(f)}개  -> {'OK (진짜 엣지는 통과시킴)' if hit else '실패'}")

nn = ev[(ev["alt"] == "NONE") & (ev["verdict"] == "통과")]
hit = len(nn) == 0
ok &= hit
print(f"  NONE 통과 조합 {len(nn)}개  -> {'OK (무관계는 안 통과시킴)' if hit else '실패'}")

# 극단값 사기 테스트: 중앙값은 음수인데 평균만 양수인 케이스를 만들어본다
fake = pd.Series([-0.001] * 95 + [0.5] * 5)  # 95번 소폭 손실, 5번 대박
mean_f, med_f, beat_f = fake.mean(), fake.median(), (fake > U.ROUNDTRIP_COST).mean()
verdict_f = U._verdict(mean_f, med_f, beat_f, 0.0, len(fake))
hit = verdict_f != "통과"
ok &= hit
print(f"  극단값 사기 (평균 {mean_f*100:.2f}% 인데 중앙값 {med_f*100:.2f}%): "
      f"판정={verdict_f}  -> {'OK (v1 이 속았던 패턴을 걸러냄)' if hit else '실패'}")

print("\n" + "=" * 76)
print("최종:", "v2 로직 검증 통과" if ok else "아직 문제 있음")
print("=" * 76)
