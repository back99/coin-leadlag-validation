#!/usr/bin/env python3
"""
청산 로직 포함 백테스트

지금까지의 테스트는 "사고 나서 무조건 N분 뒤에 판다" 였다. 그건 전략이 아니라 대용품이다.
실제로는 이런 일이 벌어진다:

    10:00  비트 +1%, 알트 +0.2%  -> 갭 0.8% -> 산다
    10:05  알트가 +1.0% 따라잡음     <- 여기서 팔면 +0.8% 번다
    10:30  알트가 도로 +0.1%로 빠짐

  고정 30분 청산은 이 거래를 "-0.1%, 손해"로 기록한다. 번 걸 손해로 적는 것이다.
  그래서 "엣지 없음"이 나왔을 수 있다.

이 스크립트는 진짜 청산 로직을 넣는다:

    살 때:  리더가 창 안에서 +X% 갔는데 알트는 그 절반도 안 따라옴
    팔 때:  넷 중 먼저 오는 것
            1. 익절   알트가 +A%
            2. 손절   알트가 -B%
            3. 탈출   리더가 진입가 대비 -C%   (산 이유가 사라짐)
            4. 포기   Z분 경과

한계 (정직하게):
  캐시에 종가만 있어서 5분봉 종가마다 조건을 확인한다.
  즉 '5분마다 확인하는 봇'을 시뮬레이션하는 것이다.
  실시간으로 계속 보는 봇이면 익절은 조금 더 잘 맞고 손절은 조금 더 자주 맞을 것이다.
  어느 쪽이 유리한지는 상쇄된다.

다중검정 방어:
  조합을 여러 개 돌리므로 제일 좋은 걸 고르면 우연일 수 있다.
  그래서 앞 45일에서 고르고 뒤 45일에서 확인한다.

실행:
    python exit_backtest.py
"""

import glob
import os
import sys
import numpy as np
import pandas as pd

import upbit_leadlag_check as U

KST = U.KST

LEADERS = ["KRW-BTC", "KRW-ETH"]

# 진입
ENTRY_BARS = 1          # 5분봉 1개 = 5분 창
ENTRY_THRESH = 0.005    # 리더가 그 창에서 +0.5% 이상
GAP_RATIO = 0.5         # 알트가 리더 움직임의 50% 미만만 따라왔으면 갭 있음

# 청산 - 이 조합들을 다 돌린다
TP_LIST = [0.005, 0.01, 0.02, 0.03]    # 익절 0.5% ~ 3%
SL_LIST = [0.005, 0.01, 0.02]          # 손절 0.5% ~ 2%
LEADER_DROP = 0.003     # 탈출: 리더가 진입가 대비 -0.3%
MAX_BARS = 12           # 포기: 12바 = 60분

# 비용 (spread_collector.py 실측 + 수수료)
MEASURED_SPREAD = {
    "KRW-BTC": 0.000152, "KRW-ETH": 0.000354, "KRW-XRP": 0.000610,
    "KRW-SOL": 0.000879, "KRW-XLM": 0.003591, "KRW-DOGE": 0.009217,
}
DEFAULT_SPREAD = 0.0015
FEE_PER_SIDE = 0.0005


def cost_of(m):
    return MEASURED_SPREAD.get(m, DEFAULT_SPREAD) + FEE_PER_SIDE * 2


_DATE_CACHE = {}


def _dates_of(closes):
    """인덱스의 KST 날짜 배열. 한 번만 계산해서 재사용한다."""
    k = id(closes)
    if k not in _DATE_CACHE:
        _DATE_CACHE[k] = np.array([d.date() for d in closes.index.tz_convert(KST)])
    return _DATE_CACHE[k]


def simulate(closes, leader, alt, tp, sl, day_filter=None):
    """
    바 단위로 걸어가며 실제 매매를 흉내낸다.
    포지션은 한 번에 하나만 (중복 진입 없음).
    """
    lead = closes[leader].values
    a = closes[alt].values
    idx = closes.index
    n = len(closes)
    w = ENTRY_BARS
    cost = cost_of(alt)

    # 날짜는 미리 계산해둔다. 안쪽 루프에서 tz_convert 를 부르면 수백만 번 돌아 엄청 느려진다.
    dates = _dates_of(closes)

    trades = []
    i = w
    while i < n - 1:
        if day_filter is not None and dates[i] not in day_filter:
            i += 1
            continue

        # 진입 조건: 리더가 창 안에서 올랐는데 알트는 절반도 안 따라옴
        lead_move = lead[i] / lead[i - w] - 1
        alt_move = a[i] / a[i - w] - 1
        if not (lead_move >= ENTRY_THRESH and alt_move < GAP_RATIO * lead_move):
            i += 1
            continue

        entry_alt = a[i]
        entry_lead = lead[i]
        exit_reason, exit_ret = None, None

        # 진입 다음 바부터 걸어가며 청산 조건 확인
        for j in range(i + 1, min(i + 1 + MAX_BARS, n)):
            r = a[j] / entry_alt - 1
            lr = lead[j] / entry_lead - 1

            if r >= tp:
                exit_reason, exit_ret = "익절", r
                break
            if r <= -sl:
                exit_reason, exit_ret = "손절", r
                break
            if lr <= -LEADER_DROP:
                exit_reason, exit_ret = "탈출", r
                break
        else:
            j = min(i + MAX_BARS, n - 1)
            exit_reason, exit_ret = "포기", a[j] / entry_alt - 1

        trades.append({
            "진입시각": idx[i], "청산사유": exit_reason,
            "수익%": exit_ret * 100, "순수익%": (exit_ret - cost) * 100,
            "보유바": j - i,
        })
        i = j + 1   # 청산한 다음 바부터 다시 탐색 (중복 진입 방지)

    return pd.DataFrame(trades)


def summarize(tr, leader, alt, tp, sl):
    if len(tr) == 0:
        return None
    net = tr["순수익%"]
    reasons = tr["청산사유"].value_counts()
    return {
        "리더": leader, "알트": alt,
        "익절%": tp * 100, "손절%": sl * 100,
        "거래수": len(tr),
        "총수익%": round(net.sum(), 3),
        "평균%": round(net.mean(), 4),
        "중앙값%": round(net.median(), 4),
        "승률%": round((net > 0).mean() * 100, 1),
        "익절": int(reasons.get("익절", 0)),
        "손절": int(reasons.get("손절", 0)),
        "탈출": int(reasons.get("탈출", 0)),
        "포기": int(reasons.get("포기", 0)),
    }


def run_grid(closes, alts, days):
    rows = []
    for leader in LEADERS:
        for alt in alts:
            for tp in TP_LIST:
                for sl in SL_LIST:
                    tr = simulate(closes, leader, alt, tp, sl, day_filter=days)
                    s = summarize(tr, leader, alt, tp, sl)
                    if s:
                        rows.append(s)
    return pd.DataFrame(rows)


def main():
    files = sorted(glob.glob("data_upbit/*_5m_90d.csv"))
    mk = [os.path.basename(f).split("_5m_")[0] for f in files]
    mk = [m for m in mk if m != "KRW-USDT"]
    if not mk:
        sys.exit("data_upbit 캐시가 없다. upbit_leadlag_check.py 를 먼저 돌려라.")

    closes, comp, spans = U.build_close_frame(mk)
    keep = [m for m in mk if comp[m] >= 0.90]
    alts = [m for m in keep if m not in LEADERS]
    closes = closes[keep].dropna()

    all_days = sorted(set(d.tz_convert(KST).date() for d in closes.index))
    mid = len(all_days) // 2
    ins_days = set(all_days[:mid])
    out_days = set(all_days[mid:])

    print("=" * 100)
    print("청산 로직 포함 백테스트")
    print(f"  진입: 리더가 {ENTRY_BARS*5}분에 +{ENTRY_THRESH*100:.1f}% & 알트가 절반도 안 따라옴")
    print(f"  청산: 익절 / 손절 / 탈출(리더 -{LEADER_DROP*100:.1f}%) / 포기({MAX_BARS*5}분)")
    print(f"  조합: 익절 {len(TP_LIST)} x 손절 {len(SL_LIST)} x 리더 {len(LEADERS)} x 알트 {len(alts)}")
    print(f"  기간: {len(all_days)}일  ->  인샘플 {mid}일 / 아웃샘플 {len(all_days)-mid}일")
    print("=" * 100)
    print(f"\n알트: {', '.join(alts)}\n")

    # ── 인샘플 ──
    ins = run_grid(closes, alts, ins_days)
    if len(ins) == 0:
        sys.exit("인샘플 거래 없음")
    ins = ins[ins["거래수"] >= 10]
    if len(ins) == 0:
        sys.exit("거래수 10건 이상인 조합이 없음")

    print("─" * 100)
    print(f"[1] 인샘플 (앞 {mid}일)  -  총수익 상위 10개")
    print("─" * 100)
    top = ins.sort_values("총수익%", ascending=False).head(10)
    print(top.to_string(index=False))
    print("\n  여기서 좋은 건 당연하다. 수백 개 조합 중 제일 좋은 걸 고른 거니까.")

    # ── 아웃샘플 ──
    outs = run_grid(closes, alts, out_days)
    key = ["리더", "알트", "익절%", "손절%"]
    m = top[key + ["거래수", "총수익%", "평균%"]].merge(
        outs[key + ["거래수", "총수익%", "평균%", "승률%"]],
        on=key, how="left", suffixes=("_인", "_아웃"))

    print("\n" + "─" * 100)
    print(f"[2] 아웃샘플 검증 (뒤 {len(all_days)-mid}일)  -  위 10개를 그대로 적용")
    print("─" * 100)
    print(m.to_string(index=False))

    ia, oa = m["평균%_인"].mean(), m["평균%_아웃"].mean()
    print(f"\n  인샘플 평균 순수익:  {ia:+.4f}%")
    print(f"  아웃샘플 평균 순수익: {oa:+.4f}%")
    if ia:
        print(f"  축소율: {oa/ia*100:.0f}%   (노이즈면 10% 근처로 무너진다)")

    # ── 전체 기간 청산 사유 분포 ──
    print("\n" + "─" * 100)
    print("[3] 청산 사유 분포 (전체 기간, 익절 1% / 손절 1% 기준)")
    print("─" * 100)
    rows = []
    for leader in LEADERS:
        for alt in alts:
            tr = simulate(closes, leader, alt, 0.01, 0.01)
            s = summarize(tr, leader, alt, 0.01, 0.01)
            if s:
                rows.append(s)
    dist = pd.DataFrame(rows)
    if len(dist):
        print(dist.to_string(index=False))
        tot = dist[["익절", "손절", "탈출", "포기"]].sum()
        print(f"\n  전체 청산 사유:  익절 {tot['익절']} / 손절 {tot['손절']} / "
              f"탈출 {tot['탈출']} / 포기 {tot['포기']}")
        print("  익절이 손절보다 많은데 총수익이 마이너스면 = 손절 한 방이 익절 여러 번을 먹는 것.")

    print("\n" + "=" * 100)
    print("판정")
    print("=" * 100)
    surv = m[m["총수익%_아웃"] > 0]
    print(f"  아웃샘플에서 수익 난 조합: {len(surv)}/{len(m)}")
    if len(surv) == 0 or (ia and oa / ia < 0.3):
        print("  -> 청산 로직을 넣어도 안 된다.")
        print("     고정 시간 청산이 문제가 아니었다는 뜻이다.")
    else:
        print(f"\n{surv.to_string(index=False)}")
        print("\n  -> 살아남은 게 있다. 거래수와 축소율을 같이 볼 것.")

    ins.to_csv("results_exit_insample.csv", index=False)
    outs.to_csv("results_exit_outsample.csv", index=False)
    m.to_csv("results_exit_validation.csv", index=False)
    print("\n저장: results_exit_*.csv")


if __name__ == "__main__":
    main()
