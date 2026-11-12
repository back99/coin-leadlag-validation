#!/usr/bin/env python3
"""
스프레드 / 거래소간 차익 분석

이건 통계가 아니라 산수다. 답이 흐릿할 수가 없다.

  [1] 마켓메이킹:  스프레드 > 왕복 수수료 인가?
      호가 양쪽에 걸어서 스프레드를 먹는 게 마켓메이킹이다.
      스프레드가 왕복 수수료보다 얇으면, 양쪽 다 체결돼도 손해다. 전략 이전의 문제.

  [2] 거래소간 차익: (빗썸 매수호가 - 업비트 매도호가) > 양쪽 수수료 합 인가?
      싼 데서 사서 비싼 데서 판다. 양방향 다 본다.

실행:
    python spread_analyze.py
"""

import sys
import numpy as np
import pandas as pd

# 수수료 - 실제 값 확인 필요. 계정 등급/이벤트/쿠폰에 따라 다르다.
# 업비트 원화마켓은 메이커/테이커 구분 없이 같은 요율로 알려져 있다 (확인할 것).
# 메이커 할인이나 리베이트가 있다면 마켓메이킹 판정이 달라진다.
UPBIT_FEE = 0.0005      # 0.05%
BITHUMB_FEE = 0.0025    # 0.25% (쿠폰 미적용 기준. 쿠폰 쓰면 훨씬 낮아짐 - 확인 필요)

IN = "spread_data.csv"


def load():
    try:
        df = pd.read_csv(IN)
    except FileNotFoundError:
        sys.exit(f"{IN} 가 없다. 먼저 수집:  python spread_collector.py")
    if len(df) == 0:
        sys.exit("수집된 데이터가 없다.")
    df["ts_kst"] = pd.to_datetime(df["ts_kst"])
    for c in ["up_bid", "up_ask", "bt_bid", "bt_ask"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def spread_test(df):
    """
    [1] 업비트 스프레드 vs 왕복 수수료.

    함정 주의: 스프레드% 가 커 보여도 그게 '기회'가 아닐 수 있다.

    업비트는 최소 호가 단위(틱)가 정해져 있다. 예: 100~1,000원 구간은 1원 단위.
    DOGE 가 108원이면 1틱 = 1원 = 0.92%. 스프레드가 넓은 게 아니라 가격이 낮은 것이다.
    이때 매수호가와 매도호가는 이미 서로 붙어 있어서 그 사이에 들어갈 자리가 없다.
    호가를 개선할 방법이 없으니 남는 경쟁은 '줄 서기(큐 포지션)' 뿐이고, 그건 속도 싸움이다.

    감지법: 스프레드의 절대값(원)이 거의 항상 같은 값이면 = 틱에 붙어 있는 것.
    """
    d = df.dropna(subset=["up_bid", "up_ask"]).copy()
    d = d[d["up_bid"] > 0]
    mid = (d["up_bid"] + d["up_ask"]) / 2
    d["spread%"] = (d["up_ask"] - d["up_bid"]) / mid * 100
    d["spread_krw"] = d["up_ask"] - d["up_bid"]

    roundtrip = UPBIT_FEE * 2 * 100

    rows = []
    for coin, g in d.groupby("coin"):
        s = g["spread%"]
        krw = g["spread_krw"]

        # 절대 스프레드가 거의 항상 최소값이면 틱에 붙어 있는 것
        tick = krw.min()
        at_tick = (krw <= tick * 1.0001).mean()
        tick_pinned = at_tick >= 0.8

        if s.median() <= roundtrip:
            verdict = "불가 (수수료 미만)"
        elif tick_pinned:
            verdict = "착시 (틱에 붙음)"
        else:
            verdict = "검토 대상"

        rows.append({
            "코인": coin,
            "표본": len(s),
            "스프레드 중앙값%": round(s.median(), 4),
            "스프레드(원)": round(krw.median(), 4),
            "틱에 붙은 비율%": round(at_tick * 100, 1),
            "왕복수수료%": round(roundtrip, 4),
            "판정": verdict,
        })
    return pd.DataFrame(rows).sort_values("스프레드 중앙값%"), roundtrip


def arb_test(df):
    """[2] 업비트 <-> 빗썸 차익. 양방향 다 본다."""
    d = df.dropna(subset=["up_bid", "up_ask", "bt_bid", "bt_ask"]).copy()
    d = d[(d["up_bid"] > 0) & (d["bt_bid"] > 0)]
    if len(d) == 0:
        return None, None

    cost = (UPBIT_FEE + BITHUMB_FEE) * 100  # 편도씩 한 번 = 한 사이클

    # 방향 A: 업비트에서 사고(ask) 빗썸에서 판다(bid)
    d["A_수익%"] = (d["bt_bid"] - d["up_ask"]) / d["up_ask"] * 100 - cost
    # 방향 B: 빗썸에서 사고 업비트에서 판다
    d["B_수익%"] = (d["up_bid"] - d["bt_ask"]) / d["bt_ask"] * 100 - cost
    d["최선%"] = d[["A_수익%", "B_수익%"]].max(axis=1)

    rows = []
    for coin, g in d.groupby("coin"):
        rows.append({
            "코인": coin,
            "표본": len(g),
            "가격차 중앙값%": round(((g["bt_bid"] - g["up_ask"]) / g["up_ask"] * 100).median(), 4),
            "A(업→빗) 최고%": round(g["A_수익%"].max(), 4),
            "B(빗→업) 최고%": round(g["B_수익%"].max(), 4),
            "수익 기회 비율%": round((g["최선%"] > 0).mean() * 100, 2),
            "판정": "기회있음" if (g["최선%"] > 0).mean() > 0.01 else "없음",
        })
    return pd.DataFrame(rows), cost


def main():
    df = load()
    span = df["ts_kst"].max() - df["ts_kst"].min()
    print("=" * 84)
    print("스프레드 / 거래소간 차익 분석")
    print(f"  표본: {len(df)}행   기간: {df['ts_kst'].min():%m-%d %H:%M} ~ "
          f"{df['ts_kst'].max():%m-%d %H:%M} KST ({span})")
    print(f"  수수료 가정: 업비트 {UPBIT_FEE*100:.3f}% / 빗썸 {BITHUMB_FEE*100:.3f}%  (확인 필요)")
    print("=" * 84)

    print("\n" + "─" * 84)
    print("[1] 마켓메이킹  -  스프레드가 왕복 수수료를 넘는가")
    print("─" * 84)
    sp, roundtrip = spread_test(df)
    print(sp.to_string(index=False))
    print(f"\n  왕복 수수료 = {roundtrip:.3f}%  (매수 {UPBIT_FEE*100:.3f}% + 매도 {UPBIT_FEE*100:.3f}%)")
    print("  스프레드 중앙값이 이보다 작으면, 양쪽 다 체결돼도 손해 = 마켓메이킹 불가.")
    print("  (업비트에 메이커 할인/리베이트가 있다면 이 판정이 달라진다. 확인할 것)")

    pinned = sp[sp["판정"] == "착시 (틱에 붙음)"]
    ok = sp[sp["판정"] == "검토 대상"]

    if len(pinned):
        print(f"\n  '착시' {len(pinned)}개: 스프레드%가 커 보이지만 절대값이 최소 호가 단위(틱)에 박혀 있다.")
        print("     매수/매도 호가가 이미 붙어 있어 그 사이에 낄 자리가 없다.")
        print("     남는 경쟁은 큐 포지션(줄 서기) = 속도 싸움 = 리테일이 지는 판.")
        print("     게다가 체결되는 순간은 보통 가격이 내 반대로 갈 때다 (역선택).")

    print(f"\n  -> 검토 대상 코인: {len(ok)}/{len(sp)}개")
    if len(ok):
        print(ok.to_string(index=False))
        print("\n  주의: 스프레드가 진짜 벌어져 있어도 거래가 뜸하면 체결이 안 된다.")
        print("        체결 빈도를 따로 확인할 것.")

    print("\n" + "─" * 84)
    print("[2] 거래소간 차익  -  업비트 vs 빗썸")
    print("─" * 84)
    ar, cost = arb_test(df)
    if ar is None:
        print("  빗썸 데이터 없음 (수집기에서 빗썸 연결 실패했거나 아직 안 모임)")
    else:
        print(ar.to_string(index=False))
        print(f"\n  한 사이클 비용 = {cost:.3f}%  (업비트 {UPBIT_FEE*100:.3f}% + 빗썸 {BITHUMB_FEE*100:.3f}%)")
        print("  '수익 기회 비율%' 는 비용 빼고도 남는 순간의 비율이다.")
        print("  이게 0에 가까우면 두 거래소가 이미 촘촘히 붙어 있다는 뜻 = 먹을 게 없다.")

        chance = ar[ar["판정"] == "기회있음"]
        print(f"\n  -> 기회 있는 코인: {len(chance)}/{len(ar)}개")
        if len(chance):
            print(chance.to_string(index=False))
            print("\n  주의: 기회가 보여도 실제로는 (1) 양쪽에 미리 자금이 있어야 하고")
            print("        (2) 두 주문이 동시에 체결돼야 하고 (3) 남들도 같은 걸 본다.")

    print("\n" + "=" * 84)
    print("판정")
    print("=" * 84)
    mm = len(sp[sp["판정"] == "검토 대상"])
    ab = 0 if ar is None else len(ar[ar["판정"] == "기회있음"])
    if mm == 0 and ab == 0:
        print("  둘 다 없음.")
        print("  -> 현물 + 업비트 제약 안에서 구조형 엣지는 없다.")
        print("     정직하게 다 털어본 것이다. 이제 판이나 목적을 바꿔야 한다.")
    else:
        print(f"  마켓메이킹 검토 대상 {mm}개 / 거래소간 차익 기회 {ab}개")
        print("  -> 다음 단계: 그 코인들의 체결 빈도와 실제 슬리피지를 확인한다.")

    sp.to_csv("results_spread.csv", index=False)
    if ar is not None:
        ar.to_csv("results_arb.csv", index=False)
    print("\n저장: results_spread.csv" + (" , results_arb.csv" if ar is not None else ""))


if __name__ == "__main__":
    main()
