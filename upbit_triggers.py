#!/usr/bin/env python3
"""
진입 트리거 비교 - 어떤 타이밍이 비용을 넘는가

리드-래그(갭)만 재는 게 아니라, 우리가 얘기한 타이밍 아이디어를 전부 같은 잣대로 줄 세운다.
데이터와 분석 로직은 upbit_leadlag_check.py 것을 그대로 재사용한다 (캐시도 공유).

테스트하는 트리거:
  1. gap          - 리더가 급등했는데 알트가 안 따라옴 (원래 리드-래그)
  2. bounce_raw   - 리더가 하락하다 반등  <- 원래 아이디어 1, 필터 없음
  3. pullback     - 리더가 상승추세 중에 눌렸다 반등  <- 원래 아이디어 + 추세 필터
  4. breakout     - 리더가 N시간 신고가 돌파
  5. vol_expand   - 리더 변동성 급증
  6. random       - 무작위 진입 (대조군. 이걸 못 이기면 그 트리거는 의미 없다)

bounce_raw 와 pullback 을 나란히 두는 게 핵심이다.
같은 "반등"이지만 추세 필터 하나로 결과가 갈리는지 보려는 것.

실행:
    python upbit_triggers.py
"""

import sys
import numpy as np
import pandas as pd

import upbit_leadlag_check as U

BARS = U.BARS


# ─────────────────────────────────────────────────────────────
# 트리거 정의  - 전부 "리더를 보고 알트를 산다"
# ─────────────────────────────────────────────────────────────

def trig_gap(closes, leader, alt, threshold=0.005):
    """리더가 급등했는데 알트는 절반도 안 따라옴 -> 따라잡을 것이다."""
    lw = closes[leader].pct_change(U.EVENT_WINDOW)
    aw = closes[alt].pct_change(U.EVENT_WINDOW)
    return (lw >= threshold) & (aw < U.GAP_RATIO * lw)


def trig_bounce_raw(closes, leader, alt, drop=0.005, win_min=30):
    """
    원래 아이디어 1 그대로: 리더가 하락하다가 방금 상승으로 돌았다.
    필터가 없어서 '폭락 중 잠깐 튀는 것'도 다 걸린다 (떨어지는 칼 잡기).
    """
    w = BARS(win_min)
    fell = closes[leader].pct_change(w) <= -drop
    turned = closes[leader].pct_change(1) > 0
    return fell & turned


def trig_pullback(closes, leader, alt, drop=0.003, win_min=30, trend_min=1440):
    """
    원래 아이디어 1 + 상위 추세 필터: 리더가 '큰 흐름상 위'에 있을 때만 반등을 산다.
    bounce_raw 와의 차이는 uptrend 조건 하나뿐. 그 하나가 (a)폭락 중 반등을 걸러낸다.
    """
    w, tw = BARS(win_min), BARS(trend_min)
    ma = closes[leader].rolling(tw).mean()
    uptrend = closes[leader] > ma
    dipped = closes[leader].pct_change(w) <= -drop
    turned = closes[leader].pct_change(1) > 0
    return uptrend & dipped & turned


def trig_breakout(closes, leader, alt, win_min=240):
    """리더가 직전 N시간 고점을 뚫었다 -> 추세 시작에 알트도 따라간다."""
    w = BARS(win_min)
    prev_high = closes[leader].shift(1).rolling(w).max()
    return closes[leader] > prev_high


def trig_vol_expand(closes, leader, alt, short_min=60, long_min=1440, mult=1.8):
    """
    리더 변동성 급증 = 뭔가 터졌다 -> 알트가 증폭해서 따라온다.

    장기 std 와 비교하면 장기 안에 단기가 포함돼서 비율이 잘 안 벌어진다.
    그래서 '지금 단기 변동성'을 '평소 단기 변동성(중앙값)'과 비교한다.
    """
    sw, lw = BARS(short_min), BARS(long_min)
    r = np.log(closes[leader]).diff()
    s = r.rolling(sw).std()
    return s > mult * s.rolling(lw).median()


def trig_random(closes, leader, alt, rate=0.02, seed=0):
    """대조군. 아무 때나 산다. 다른 트리거가 이걸 못 이기면 그 트리거는 의미 없다."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.random(len(closes)) < rate, index=closes.index)


TRIGGERS = [
    ("gap",         trig_gap,         "리더 급등 + 알트 미추종"),
    ("bounce_raw",  trig_bounce_raw,  "하락하다 반등 (필터 없음)"),
    ("pullback",    trig_pullback,    "상승추세 중 눌림목 반등"),
    ("breakout",    trig_breakout,    "리더 4시간 신고가 돌파"),
    ("vol_expand",  trig_vol_expand,  "리더 변동성 급증"),
    ("random",      trig_random,      "무작위 진입 (대조군)"),
]


# ─────────────────────────────────────────────────────────────
# 평가
# ─────────────────────────────────────────────────────────────

def evaluate(closes, alts, name, fn):
    rows = []
    for leader in U.LEADERS:
        for alt in alts:
            try:
                mask = fn(closes, leader, alt)
            except Exception as e:
                print(f"    [{name}] {leader}->{alt} 실패: {e}")
                continue

            for h, hmin in zip(U.HORIZONS, U.HORIZONS_MIN):
                fwd = closes[alt].shift(-h) / closes[alt] - 1.0
                ev = fwd[mask].dropna()
                base = fwd.dropna()

                if len(ev) < U.MIN_EVENTS:
                    continue

                mean, median = ev.mean(), ev.median()
                beat = (ev > U.ROUNDTRIP_COST).mean()
                rows.append({
                    "trigger": name, "leader": leader, "alt": alt, "horizon(분)": hmin,
                    "events": len(ev),
                    "mean%": round(mean * 100, 4),
                    "median%": round(median * 100, 4),
                    "beat_cost%": round(beat * 100, 1),
                    "baseline%": round(base.mean() * 100, 4),
                    "net%": round((mean - U.ROUNDTRIP_COST) * 100, 4),
                    "verdict": U._verdict(mean, median, beat, base.mean(), len(ev)),
                })
    return pd.DataFrame(rows)


def sweep_gap_threshold(closes, alts):
    """
    갭 트리거의 임계치를 올리면 어떻게 되나.
    이론: 트리거를 크게 잡으면 횟수는 줄지만 한 번의 폭이 커져서 수수료를 넘길 수 있다.
    """
    rows = []
    for th in [0.003, 0.005, 0.01, 0.02, 0.03]:
        df = evaluate(closes, alts, f"gap@{th*100:.1f}%",
                      lambda c, l, a, t=th: trig_gap(c, l, a, threshold=t))
        if len(df) == 0:
            rows.append({"임계치%": th * 100, "조합수": 0, "통과": 0,
                         "최고 median%": None, "최고 beat_cost%": None})
            continue
        rows.append({
            "임계치%": th * 100,
            "조합수": len(df),
            "통과": int((df["verdict"] == "통과").sum()),
            "최고 median%": df["median%"].max(),
            "최고 beat_cost%": df["beat_cost%"].max(),
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 88)
    print("진입 트리거 비교")
    print(f"  기간: 최근 {U.DAYS}일, {U.INTERVAL_MIN}분봉")
    print(f"  왕복 비용 가정: {U.ROUNDTRIP_COST*100:.2f}%")
    print(f"  통과 조건: mean% > 비용  AND  median% > 비용  AND  beat_cost% >= 50"
          f"  AND  mean% > baseline  AND  표본 >= {U.MIN_EVENTS}")
    print("=" * 88)

    alts, volumes = U.get_top_krw_alts(U.TOP_N_ALTS)
    closes, comp, spans = U.build_close_frame(U.LEADERS + alts)
    grid_n = len(closes)

    dropped = [a for a in alts if comp[a] < U.MIN_COMPLETENESS]
    alts = [a for a in alts if comp[a] >= U.MIN_COMPLETENESS]
    if not alts:
        sys.exit("완전성 기준을 통과한 알트가 없다. upbit_leadlag_check.py 를 먼저 확인할 것.")

    closes = closes[U.LEADERS + alts].dropna()
    if dropped:
        print(f"\n제외된 알트 (이력 부족/거래 뜸함): "
              f"{', '.join(f'{d}({comp[d]*100:.0f}%)' for d in dropped)}")
    print(f"대상 알트: {', '.join(alts)}")
    print(f"정렬된 바: {len(closes)}개 / 리더 그리드 {grid_n}개")
    if len(closes) < grid_n * 0.8:
        print("  !!! 경고: 리더 그리드보다 20% 이상 짧다. 정렬 문제일 수 있다.")
    print()

    all_df = []
    summary = []

    for name, fn, desc in TRIGGERS:
        print(f"  평가 중: {name:12s} - {desc}")
        df = evaluate(closes, alts, name, fn)
        if len(df) == 0:
            summary.append({"trigger": name, "설명": desc, "조합수": 0, "통과": 0,
                            "최고 median%": None, "최고 beat_cost%": None})
            continue
        all_df.append(df)
        summary.append({
            "trigger": name, "설명": desc,
            "조합수": len(df),
            "통과": int((df["verdict"] == "통과").sum()),
            "최고 median%": df["median%"].max(),
            "최고 beat_cost%": df["beat_cost%"].max(),
        })

    print("\n" + "─" * 88)
    print("[1] 트리거별 요약")
    print("─" * 88)
    sm = pd.DataFrame(summary)
    print(sm.to_string(index=False))
    print("\n  random(대조군)보다 나은 게 없으면, 그 트리거들은 전부 의미 없는 것이다.")
    print("  bounce_raw 와 pullback 을 비교해 볼 것 - 추세 필터가 실제로 효과가 있는지.")

    if all_df:
        full = pd.concat(all_df, ignore_index=True)

        print("\n" + "─" * 88)
        print("[2] 통과 조합 (있다면)")
        print("─" * 88)
        passed = full[full["verdict"] == "통과"]
        if len(passed) == 0:
            print("  없음.")
        else:
            print(passed.sort_values("median%", ascending=False).to_string(index=False))
            print("\n  표본 수를 반드시 볼 것. 그리고 random 대조군도 통과했는지 확인할 것.")
            print("  (random 이 통과했다면 그건 시장이 오른 것이지 트리거 덕이 아니다)")

        print("\n" + "─" * 88)
        print("[3] 상위 15개 (판정 무관, median% 순)")
        print("─" * 88)
        print(full.sort_values("median%", ascending=False).head(15).to_string(index=False))

        full.to_csv("results_triggers.csv", index=False)

    print("\n" + "─" * 88)
    print("[4] 갭 임계치 스윕  -  크게 잡으면 폭이 커져서 수수료를 넘을까?")
    print("─" * 88)
    sw = sweep_gap_threshold(closes, alts)
    print(sw.to_string(index=False))
    sw.to_csv("results_gap_sweep.csv", index=False)

    print("\n" + "=" * 88)
    print("저장: results_triggers.csv , results_gap_sweep.csv")
    print("=" * 88)


if __name__ == "__main__":
    main()
