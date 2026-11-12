#!/usr/bin/env python3
"""
분석 로직 검증 - 네트워크 없이 합성 데이터로 돌린다.

알려진 시차(3분)를 심은 가짜 알트를 만들고, 스크립트가 그 3분을 찾아내는지 확인한다.
찾아내면 로직이 맞는 것이고, 못 찾으면 분석 코드가 틀린 것이다.
"""

import numpy as np
import pandas as pd

import leadlag_check as L

rng = np.random.default_rng(42)
N = 20000
LAG = 3  # 심어둔 진짜 시차

# 리더: 랜덤워크 수익률
btc_ret = rng.normal(0, 0.001, N)

# FOLLOW: 리더를 3분 뒤에 따라간다  -> best_lag 가 3으로 나와야 한다
follow_ret = np.zeros(N)
follow_ret[LAG:] = 0.8 * btc_ret[:-LAG] + rng.normal(0, 0.0005, N - LAG)

# SYNC: 리더와 동시에 움직인다  -> corr@0 만 높고 lag>=1 은 낮아야 한다 (먹을 시차 없음)
sync_ret = 0.8 * btc_ret + rng.normal(0, 0.0005, N)

# NOISE: 아무 관계 없다  -> 전부 0 근처여야 한다
noise_ret = rng.normal(0, 0.001, N)


def to_close(ret, start=100.0):
    return start * np.exp(np.cumsum(ret))


idx = pd.date_range("2026-01-01", periods=N, freq="1min", tz="UTC")
closes = pd.DataFrame({
    "BTCUSDT": to_close(btc_ret),
    "FOLLOW": to_close(follow_ret),
    "SYNC": to_close(sync_ret),
    "NOISE": to_close(noise_ret),
}, index=idx)

# 모듈 설정을 합성 데이터에 맞게 교체
L.LEADERS = ["BTCUSDT"]
L.ALTS = ["FOLLOW", "SYNC", "NOISE"]

print("=" * 70)
print(f"[검증] 심어둔 진짜 시차 = {LAG}분")
print("=" * 70)

corr = L.lagged_corr_table(closes)
print(corr.to_string(index=False))

print("\n판정:")
ok = True

row = corr[corr["alt"] == "FOLLOW"].iloc[0]
hit = row["best_lag"] == LAG and row["corr@best_lag"] > 0.5
ok &= hit
print(f"  FOLLOW: best_lag={row['best_lag']} (기대 {LAG}), corr={row['corr@best_lag']}"
      f"  -> {'OK' if hit else '실패'}")

row = corr[corr["alt"] == "SYNC"].iloc[0]
hit = row["corr@0"] > 0.5 and abs(row["corr@1"]) < 0.2
ok &= hit
print(f"  SYNC  : corr@0={row['corr@0']}, corr@1={row['corr@1']}"
      f"  -> {'OK (동시 이동, 시차 없음을 정확히 구분)' if hit else '실패'}")

row = corr[corr["alt"] == "NOISE"].iloc[0]
hit = abs(row["corr@0"]) < 0.1 and abs(row["corr@best_lag"]) < 0.1
ok &= hit
print(f"  NOISE : corr@0={row['corr@0']}, corr@best={row['corr@best_lag']}"
      f"  -> {'OK (무관계를 무관계로 판정)' if hit else '실패'}")

print("\n" + "=" * 70)
print("[검증] 이벤트 스터디 동작 확인")
print("=" * 70)
L.EVENT_THRESHOLD = 0.002  # 합성 데이터는 변동성이 작아 임계치를 낮춘다
ev = L.event_study(closes)
print(ev.to_string(index=False))

has_events = ev["events"].max() > 0
ok &= has_events
print(f"\n  이벤트 탐지: {'OK' if has_events else '실패 - 이벤트가 하나도 안 잡힘'}")

print("\n" + "=" * 70)
print("최종:", "로직 검증 통과" if ok else "로직에 문제 있음")
print("=" * 70)
