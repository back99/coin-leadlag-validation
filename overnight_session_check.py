#!/usr/bin/env python3
"""
밤사이 -> 한국 낮 세션 검증 (원래 아이디어의 마지막 조각)

앞선 검증들이 죽인 것:
  - 초 단위 시차 (바이낸스)      : 없음 + 수수료 벽
  - 5~60분 시차 (바이낸스/업비트) : corr@1 이 0. 알트는 리더를 '따라오는' 게 아니라 '같이' 움직임
  - 5분 급등 트리거 6종           : random 대조군과 구분 안 됨

아직 안 죽은 것:
  원래 아이디어는 5분짜리 시차가 아니었다.
  "밤사이 미국 뉴스가 쌓이고 -> 아침에 일어난 한국 리테일이 뒤늦게 반영한다" 였다.
  이건 12시간짜리 정보 흡수지 분 단위 시차가 아니다.
  앞선 상관표는 최대 60분까지만 봤으므로 이 가설을 반증하지 못한다.

검정력 설계 (중요):
  90일 = 일별 표본 90개뿐이다. 여기에 임계치를 걸면 20개로 쪼그라들어 아무것도 못 본다.
  그래서 조건 없이 90일 전부 쓰는 '상관' 으로 잰다. 임계치도 없고 다중검정도 없다.

  핵심 질문 한 줄:
    "밤사이 리더가 올랐는데 알트가 덜 따라왔으면, 한국 낮에 그 알트가 따라잡는가?"
    -> corr(밤사이 갭, 한국 낮 알트 수익) > 0 인가?

실행:
    python overnight_session_check.py
"""

import sys
import numpy as np
import pandas as pd

import upbit_leadlag_check as U

KST = U.KST

# 세션 경계 (한국시간)
MORNING = "07:00"   # 한국 리테일이 깨어나는 시각 = 진입
EVENING = "19:00"   # 미국이 깨어나는 시각

# 한국 낮 중간 청산 후보
DAY_EXITS = ["09:00", "12:00", "15:00", "19:00"]


def snap(kst_frame, hhmm):
    """매일 특정 한국시간의 가격 스냅샷. 인덱스를 날짜로 바꾼다."""
    s = kst_frame.at_time(hhmm)
    s = s.copy()
    s.index = pd.Index([t.date() for t in s.index], name="date")
    return s[~s.index.duplicated(keep="first")]


def tstat(r, n):
    """상관계수의 t 통계량. |t| < 2 면 0과 구분 안 된다고 본다."""
    if n <= 2 or abs(r) >= 1:
        return 0.0
    return r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2)


def build_sessions(closes, leader, alt):
    """
    하루를 두 토막으로 나눈다.
      밤사이 (전날 19시 -> 당일 07시)  = 미국 시간. 여기서 뉴스가 나온다.
      한국 낮 (당일 07시 -> 당일 19시) = 한국 리테일이 반응할 시간.
    """
    kst = closes.tz_convert(KST)
    p07 = snap(kst, MORNING)
    p19 = snap(kst, EVENING)

    dates = p07.index.intersection(p19.index)
    p07, p19 = p07.loc[dates], p19.loc[dates]

    prev19 = p19.shift(1)  # 전날 저녁

    df = pd.DataFrame(index=dates)
    # 밤사이 움직임
    df["overnight_leader"] = p07[leader] / prev19[leader] - 1
    df["overnight_alt"] = p07[alt] / prev19[alt] - 1
    # 갭 = 리더는 갔는데 알트가 덜 간 정도. 원래 아이디어의 정수.
    df["overnight_gap"] = df["overnight_leader"] - df["overnight_alt"]
    # 한국 낮 수익 (07시 진입 -> 19시 청산)
    df["day_alt"] = p19[alt] / p07[alt] - 1
    df["day_leader"] = p19[leader] / p07[leader] - 1

    return df.dropna()


def correlation_test(closes, alts):
    """
    임계치 없이 90일 전부 쓰는 상관 테스트. 최대 검정력.

    보는 것:
      A. corr(밤사이 갭, 한국 낮 알트)   <- 원래 아이디어. 양수여야 가설이 산다.
      B. corr(밤사이 리더, 한국 낮 알트) <- 밤사이 상승이 낮까지 이어지나 (모멘텀)
      C. corr(밤사이 리더, 밤사이 알트)  <- 대조군. 알트가 이미 밤사이 따라갔나
    """
    rows = []
    for leader in U.LEADERS:
        for alt in alts:
            d = build_sessions(closes, leader, alt)
            n = len(d)
            if n < 30:
                rows.append({"leader": leader, "alt": alt, "일수": n,
                             "A: 갭→낮알트": None, "t(A)": None,
                             "B: 밤리더→낮알트": None,
                             "C: 밤리더→밤알트": None, "판정": "표본부족"})
                continue

            a = d["overnight_gap"].corr(d["day_alt"])
            b = d["overnight_leader"].corr(d["day_alt"])
            c = d["overnight_leader"].corr(d["overnight_alt"])
            ta = tstat(a, n)

            rows.append({
                "leader": leader, "alt": alt, "일수": n,
                "A: 갭→낮알트": round(a, 4),
                "t(A)": round(ta, 2),
                "B: 밤리더→낮알트": round(b, 4),
                "C: 밤리더→밤알트": round(c, 4),
                "판정": "유의" if abs(ta) >= 2 else "0과 구분 안 됨",
            })
    return pd.DataFrame(rows)


def day_exit_study(closes, alts):
    """
    07시에 사서 언제 파는 게 제일 나은가.
    갭이 큰 날(상위 3분의 1)만 골라 샀을 때 vs 매일 샀을 때(baseline) 비교.
    """
    rows = []
    for leader in U.LEADERS:
        for alt in alts:
            kst = closes.tz_convert(KST)
            p07 = snap(kst, MORNING)
            p19 = snap(kst, EVENING)
            dates = p07.index.intersection(p19.index)
            p07, p19 = p07.loc[dates], p19.loc[dates]
            prev19 = p19.shift(1)

            gap = ((p07[leader] / prev19[leader]) - (p07[alt] / prev19[alt])).dropna()
            if len(gap) < 30:
                continue
            # 갭이 큰 날 = 알트가 밤사이 제일 덜 따라온 날
            big_gap_days = gap[gap >= gap.quantile(2 / 3)].index

            for exit_t in DAY_EXITS:
                px = snap(kst, exit_t)
                common = p07.index.intersection(px.index)
                ret = (px[alt].loc[common] / p07[alt].loc[common] - 1).dropna()

                sel = ret.loc[ret.index.intersection(big_gap_days)]
                if len(sel) < 20:
                    continue

                rows.append({
                    "leader": leader, "alt": alt, "청산": exit_t, "일수": len(sel),
                    "mean%": round(sel.mean() * 100, 4),
                    "median%": round(sel.median() * 100, 4),
                    "beat_cost%": round((sel > U.ROUNDTRIP_COST).mean() * 100, 1),
                    "baseline%": round(ret.mean() * 100, 4),
                    "net%": round((sel.mean() - U.ROUNDTRIP_COST) * 100, 4),
                })
    return pd.DataFrame(rows)


def reverse_test(closes, alts):
    """
    대칭 검증: 한국 낮 -> 미국 밤.
    원래 아이디어대로면 이쪽도 같은 논리로 성립해야 한다 (미국 사람들이 아시아 뉴스에 반응).
    한쪽만 되면 그건 메커니즘이 아니라 우연이다.
    """
    rows = []
    for leader in U.LEADERS:
        for alt in alts:
            kst = closes.tz_convert(KST)
            p07 = snap(kst, MORNING)
            p19 = snap(kst, EVENING)
            dates = p07.index.intersection(p19.index)
            p07, p19 = p07.loc[dates], p19.loc[dates]
            next07 = p07.shift(-1)

            day_gap = (p19[leader] / p07[leader]) - (p19[alt] / p07[alt])
            night_alt = next07[alt] / p19[alt] - 1

            d = pd.DataFrame({"gap": day_gap, "night": night_alt}).dropna()
            if len(d) < 30:
                continue
            r = d["gap"].corr(d["night"])
            rows.append({
                "leader": leader, "alt": alt, "일수": len(d),
                "낮갭→밤알트": round(r, 4),
                "t": round(tstat(r, len(d)), 2),
                "판정": "유의" if abs(tstat(r, len(d))) >= 2 else "0과 구분 안 됨",
            })
    return pd.DataFrame(rows)


def main():
    print("=" * 88)
    print("밤사이 -> 한국 낮 세션 검증")
    print(f"  가설: 밤사이(19-07시 KST) 미국발 움직임을 한국 리테일이 낮(07-19시)에 뒤늦게 반영한다")
    print(f"  진입 {MORNING} KST / 청산 {EVENING} KST")
    print(f"  왕복 비용 가정: {U.ROUNDTRIP_COST*100:.2f}%")
    print("=" * 88)

    alts, _ = U.get_top_krw_alts(U.TOP_N_ALTS)
    closes, comp, spans = U.build_close_frame(U.LEADERS + alts)
    alts = [a for a in alts if comp[a] >= U.MIN_COMPLETENESS]
    if not alts:
        sys.exit("완전성 통과 알트 없음")
    closes = closes[U.LEADERS + alts].dropna()

    print(f"\n대상 알트: {', '.join(alts)}")
    print(f"바: {len(closes)}개\n")

    print("─" * 88)
    print("[1] 핵심 상관 테스트  -  조건 없이 90일 전부 사용 (최대 검정력)")
    print("─" * 88)
    ct = correlation_test(closes, alts)
    print(ct.to_string(index=False))
    print("\n  A: 갭→낮알트   = 밤사이 덜 따라온 알트가 한국 낮에 따라잡는가  <- 원래 아이디어")
    print("                    이게 뚜렷한 양수여야 가설이 산다.")
    print("  B: 밤리더→낮알트 = 밤사이 상승이 낮까지 이어지나 (모멘텀)")
    print("  C: 밤리더→밤알트 = 알트가 이미 밤사이 따라가버렸나 (대조. 높으면 낮에 먹을 게 없다)")
    print("  t(A) 가 |2| 미만이면 0과 구분이 안 되는 것 = 없는 것")

    print("\n" + "─" * 88)
    print("[2] 청산 시각별  -  밤사이 갭이 큰 날(상위 1/3)만 07시에 매수")
    print("─" * 88)
    de = day_exit_study(closes, alts)
    if len(de) == 0:
        print("  표본 부족")
    else:
        print(de.to_string(index=False))
        print("\n  mean% 가 baseline% 보다 확실히 커야 '갭 때문'이다. 비슷하면 그냥 시장이 오른 것.")

    print("\n" + "─" * 88)
    print("[3] 대칭 검증  -  한국 낮 -> 미국 밤 (같은 논리면 이쪽도 성립해야 한다)")
    print("─" * 88)
    rt = reverse_test(closes, alts)
    if len(rt) == 0:
        print("  표본 부족")
    else:
        print(rt.to_string(index=False))
        print("\n  한쪽만 되고 반대쪽은 안 되면, 메커니즘이 아니라 우연일 가능성이 높다.")

    print("\n" + "=" * 88)
    print("[4] 판정")
    print("=" * 88)
    valid = ct[ct["A: 갭→낮알트"].notna()]
    if len(valid) == 0:
        print("  표본 부족으로 판정 불가")
        return

    sig = valid[valid["판정"] == "유의"]
    print(f"  A(갭→낮알트) 상관 범위: {valid['A: 갭→낮알트'].min():+.4f} ~ "
          f"{valid['A: 갭→낮알트'].max():+.4f}")
    print(f"  |t| >= 2 인 조합: {len(sig)} / {len(valid)}")
    print(f"  C(밤리더→밤알트) 평균: {valid['C: 밤리더→밤알트'].mean():.4f}"
          f"   <- 이게 높으면 알트는 이미 밤사이 다 따라간 것")
    print()
    if len(sig) == 0:
        print("  유의한 조합 없음.")
        print("  -> 밤사이 갭과 한국 낮 수익 사이에 관계가 없다.")
        print("     한국 리테일이 아침에 뒤늦게 반영한다는 가설은 기각.")
        print("     리드-래그 계열은 여기서 접는다.")
    else:
        print(f"  유의한 조합 {len(sig)}개:")
        print(sig.to_string(index=False))
        print(f"\n  주의: {len(valid)}개를 테스트했으므로 우연히 "
              f"{len(valid)*0.05:.1f}개는 유의하게 나온다.")
        print("  그보다 확실히 많고, [3] 대칭 검증도 같은 방향이어야 진짜다.")

    ct.to_csv("results_overnight_corr.csv", index=False)
    if len(de):
        de.to_csv("results_overnight_exits.csv", index=False)
    if len(rt):
        rt.to_csv("results_overnight_reverse.csv", index=False)
    print("\n저장: results_overnight_*.csv")


if __name__ == "__main__":
    main()
