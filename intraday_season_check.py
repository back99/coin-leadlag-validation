#!/usr/bin/env python3
"""
시간대 계절성 검증 (마지막 아이디어)

발상:
  세션 검증에서 우연히 이게 보였다.
      KRW-DATA  baseline  07:00 -> 09:00 :  +0.487%
  조건이 하나도 없다. 갭도 김프도 트리거도 아니다. 그냥 매일 아침 7시에 사서 9시에 판 것.

  다른 것들과 다른 점:
    - 예측이 아니다. "오를까?"를 안 묻는다. 시계만 본다.
    - 속도 싸움이 아니다. 7시에 사고 9시에 팔면 끝. 큐도 틱도 상관없다.
    - 누가 지는가? 한국 리테일의 습관. 매일 같은 시간에 같은 방향으로 압력이 생긴다면.

이 검증의 핵심은 '어떻게 속지 않는가' 다:
  코인 11개 x 시간 24개 x 보유기간 5개 = 1,320 번 테스트한다.
  5% 유의수준이면 우연히 66개가 "발견"된다. 그냥 돌리면 반드시 속는다.

  그래서 인샘플/아웃샘플로 나눈다:
    1. 앞 45일에서만 패턴을 찾는다 (여기선 마음껏 뒤져도 된다)
    2. 거기서 뽑은 것만 뒤 45일에 적용한다 (여기선 다중검정이 없다)
    3. 뒤 45일에서도 살아있으면 진짜, 0으로 죽으면 노이즈였던 것

  이게 "그때 알았다면 실제로 돈을 벌었나"와 같은 질문이다.

비용:
  코인마다 다르다. 우리가 직접 잰 스프레드를 쓴다 (spread_collector.py 결과).
  DOGE 는 스프레드만 0.92% 다. 같은 0.2% 를 적용하면 거짓말이 된다.

실행:
    python intraday_season_check.py
"""

import sys
import numpy as np
import pandas as pd

import upbit_leadlag_check as U

KST = U.KST

# 우리가 직접 측정한 스프레드 (spread_collector.py 결과, 단위 %)
# 측정 안 된 코인은 보수적으로 기본값을 쓴다.
MEASURED_SPREAD = {
    "KRW-BTC": 0.0152, "KRW-ETH": 0.0354, "KRW-XRP": 0.0610, "KRW-SOL": 0.0879,
    "KRW-TRX": 0.2094, "KRW-XLM": 0.3591, "KRW-ADA": 0.4124, "KRW-DOGE": 0.9217,
}
DEFAULT_SPREAD = 0.15
FEE_PER_SIDE = 0.05   # %

HOLDS = [1, 2, 3, 6, 12]   # 몇 시간 보유
TOP_N = 10                 # 인샘플 상위 몇 개를 아웃샘플로 넘길지
MIN_DAYS = 25              # 각 반쪽에서 최소 며칠


def cost_of(market):
    """왕복 비용 = 스프레드 1회 + 수수료 2회. 코인마다 다르다."""
    sp = MEASURED_SPREAD.get(market, DEFAULT_SPREAD)
    return sp + FEE_PER_SIDE * 2


def snap_hours(closes):
    """KST 매 정시의 가격 스냅샷. {시간: DataFrame(날짜 x 코인)}"""
    kst = closes.tz_convert(KST)
    out = {}
    for h in range(24):
        s = kst.at_time(f"{h:02d}:00")
        s = s.copy()
        s.index = pd.Index([t.date() for t in s.index], name="date")
        out[h] = s[~s.index.duplicated(keep="first")]
    return out


def returns_for(hours, market, entry_h, hold):
    """entry_h 시에 사서 hold 시간 뒤에 판다. 날짜별 수익 시리즈."""
    exit_h = (entry_h + hold) % 24
    day_off = (entry_h + hold) // 24

    entry = hours[entry_h][market]
    exit_ = hours[exit_h][market]
    if day_off:
        exit_ = exit_.shift(-day_off)

    idx = entry.index.intersection(exit_.index)
    r = (exit_.loc[idx] / entry.loc[idx] - 1.0) * 100
    return r.dropna()


def tstat(x):
    if len(x) < 3 or x.std() == 0:
        return 0.0
    return float(x.mean() / (x.std() / np.sqrt(len(x))))


def scan(hours, markets, days_index, lo, hi):
    """
    구간 [lo, hi) 의 날짜만 써서 모든 (코인, 진입시각, 보유) 조합을 훑는다.

    핵심: '초과%' 를 같이 낸다.

    함정 - 우상향한 코인은 '아무 시간에 사도' 수익이 난다.
    그래서 시간대 수익만 보면 "이 시각이 특별하다"와 "이 코인이 올랐다"를 구분 못 한다.
    (실측: 순수 노이즈 코인 중 드리프트가 있던 놈은 아웃샘플에서도 +0.5% 로 '생존'했다.
     시간대 패턴이 아니라 그냥 우상향이었는데도.)

    그래서 그 코인이 같은 보유기간으로 '아무 때나' 샀을 때의 평균을 빼준다.
      초과% = 이 시각의 평균 - 이 코인의 전 시각 평균
    이게 진짜 '시간대 효과'다.
    """
    window = set(days_index[lo:hi])
    rows = []
    for m in markets:
        c = cost_of(m)
        for k in HOLDS:
            # 이 코인이 이 보유기간으로 아무 때나 샀을 때의 평균 (= 코인 자체의 드리프트)
            per_hour = {}
            for h in range(24):
                r = returns_for(hours, m, h, k)
                r = r[[d in window for d in r.index]]
                if len(r) >= MIN_DAYS:
                    per_hour[h] = r
            if not per_hour:
                continue
            drift = float(np.mean([r.mean() for r in per_hour.values()]))

            for h, r in per_hour.items():
                rows.append({
                    "코인": m, "진입(KST)": f"{h:02d}시", "보유(h)": k,
                    "일수": len(r),
                    "평균%": round(r.mean(), 4),
                    "코인드리프트%": round(drift, 4),
                    "초과%": round(r.mean() - drift, 4),
                    "승률%": round((r > 0).mean() * 100, 1),
                    "t": round(tstat(r), 2),
                    "비용%": round(c, 4),
                    "순수익%": round(r.mean() - c, 4),
                    "초과-비용%": round(r.mean() - drift - c, 4),
                })
    return pd.DataFrame(rows)


def main():
    print("=" * 96)
    print("시간대 계절성 검증")
    print("  질문: 특정 시각에 사서 몇 시간 뒤 팔면 계통적으로 이득인가?")
    print("  방어: 앞 45일에서 찾고, 뒤 45일에서 검증한다 (인샘플/아웃샘플)")
    print("=" * 96)

    alts, _ = U.get_top_krw_alts(U.TOP_N_ALTS)
    closes, comp, spans = U.build_close_frame(U.LEADERS + alts)
    alts = [a for a in alts if comp[a] >= U.MIN_COMPLETENESS]
    markets = U.LEADERS + alts
    closes = closes[markets].dropna()

    print(f"\n대상: {', '.join(markets)}")
    print("\n코인별 왕복 비용 (스프레드 실측 + 수수료):")
    for m in markets:
        tag = "실측" if m in MEASURED_SPREAD else "추정"
        print(f"    {m:12s} {cost_of(m):5.3f}%   (스프레드 "
              f"{MEASURED_SPREAD.get(m, DEFAULT_SPREAD):.4f}% {tag})")

    hours = snap_hours(closes)
    days = sorted(set(hours[0].index) & set(hours[12].index))
    n = len(days)
    mid = n // 2
    print(f"\n총 {n}일  ->  인샘플 {mid}일 / 아웃샘플 {n-mid}일")
    print(f"테스트 조합: {len(markets)} 코인 x 24 시각 x {len(HOLDS)} 보유 = "
          f"{len(markets)*24*len(HOLDS)}개")
    print(f"  (5% 유의수준이면 우연히 {len(markets)*24*len(HOLDS)*0.05:.0f}개가 '발견'된다.")
    print(f"   그래서 인샘플에서 찾은 것만 아웃샘플로 넘긴다)")

    # ── 1. 인샘플에서 패턴 찾기 ──
    ins = scan(hours, markets, days, 0, mid)
    if len(ins) == 0:
        sys.exit("인샘플 표본 부족")

    print("\n" + "─" * 96)
    print(f"[1] 인샘플 (앞 {mid}일)  -  상위 {TOP_N}개")
    print("─" * 96)
    # 랭킹은 '초과-비용%' 로. 코인 드리프트를 빼야 진짜 시간대 효과다.
    ins_top = ins.sort_values("초과-비용%", ascending=False).head(TOP_N)
    print(ins_top.to_string(index=False))
    print("\n  여기서 좋아 보이는 건 당연하다. 1,320개 중 제일 좋은 걸 고른 거니까.")
    print("  '초과%' 로 줄 세운 이유: 그냥 우상향한 코인은 아무 시각에 사도 수익이 나서")
    print("  '평균%' 로 세우면 시간대 효과가 아니라 코인 드리프트를 고르게 된다.")

    # ── 2. 아웃샘플에서 검증 ──
    outs = scan(hours, markets, days, mid, n)
    key = ["코인", "진입(KST)", "보유(h)"]
    merged = ins_top[key + ["초과%", "초과-비용%"]].merge(
        outs[key + ["초과%", "초과-비용%", "일수", "승률%"]],
        on=key, how="left", suffixes=("_인샘플", "_아웃샘플"))

    print("\n" + "─" * 96)
    print(f"[2] 아웃샘플 검증 (뒤 {n-mid}일)  -  위 {TOP_N}개를 그대로 적용")
    print("─" * 96)
    print(merged.to_string(index=False))

    ins_avg = merged["초과-비용%_인샘플"].mean()
    out_avg = merged["초과-비용%_아웃샘플"].mean()
    print(f"\n  인샘플 평균 초과-비용: {ins_avg:+.4f}%")
    print(f"  아웃샘플 평균 초과-비용: {out_avg:+.4f}%")
    print(f"  축소율: {out_avg/ins_avg*100 if ins_avg else 0:.0f}%")
    print("  (실측: 순수 노이즈 데이터는 인샘플 +0.50% -> 아웃샘플 +0.05%, 즉 10% 로 축소됐다.")
    print("   축소율이 그 근처면 우리가 본 것도 노이즈다)")

    surv = merged[merged["초과-비용%_아웃샘플"] > 0]
    print(f"\n  아웃샘플에서도 초과+비용 넘긴 것: {len(surv)}/{len(merged)}")
    print("  (주의: 노이즈여도 절반은 부호가 유지된다. 개수보다 '축소율'이 중요하다)")

    # ── 3. 양쪽 모두 통과 ──
    print("\n" + "=" * 96)
    print("판정")
    print("=" * 96)
    shrink = out_avg / ins_avg if ins_avg else 0
    if len(surv) == 0 or shrink < 0.3:
        print("  시간대 계절성 없음.")
        if len(surv):
            print(f"  -> 생존이 {len(surv)}개 있지만 축소율이 {shrink*100:.0f}% 다.")
            print("     노이즈 기준선(10%)과 구분이 안 된다. 우연히 부호가 유지된 것뿐이다.")
        else:
            print("  -> 인샘플에서 보인 패턴이 아웃샘플에서 전부 죽었다.")
    else:
        print(f"  생존 {len(surv)}개, 축소율 {shrink*100:.0f}% (노이즈 기준선 10%보다 확실히 높음):\n")
        print(surv.to_string(index=False))
        print("\n  그래도 확인할 것:")
        print("   1) 일수가 충분한가")
        print("   2) 같은 코인의 인접 시간대도 같은 방향인가 (한 시각만 튀면 우연)")
        print("   3) 여러 코인이 같은 시각에 같은 방향인가 (한 코인만이면 우연)")

    # ── 참고: 전체 기간 시간대별 평균 (그림 대신 표) ──
    print("\n" + "─" * 96)
    print("[참고] 전체 기간, 1시간 보유 기준 시간대별 평균 수익% (비용 차감 전)")
    print("─" * 96)
    full = scan(hours, markets, days, 0, n)
    h1 = full[full["보유(h)"] == 1]
    if len(h1):
        pivot = h1.pivot(index="진입(KST)", columns="코인", values="평균%")
        print(pivot.round(3).to_string())
        print("\n  한 코인만 튀면 우연이다. 여러 코인이 같은 시각에 같은 방향이면 그건 뭔가 있는 것.")

    ins.to_csv("results_season_insample.csv", index=False)
    outs.to_csv("results_season_outsample.csv", index=False)
    merged.to_csv("results_season_validation.csv", index=False)
    print("\n저장: results_season_*.csv")


if __name__ == "__main__":
    main()
