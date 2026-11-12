#!/usr/bin/env python3
"""
김치 프리미엄 신호 검증

발상:
  김프를 '먹는' 건 불가능하다 (외환 규제로 자본을 못 옮긴다).
  하지만 '신호'로는 쓸 수 있다. 이건 완전히 다른 얘기다.

  김프 = 한국 사람들이 지금 얼마나 흥분했나를 재는 온도계.
    김프 높음 = 한국이 글로벌보다 웃돈 = 과열  -> 되돌림?
    김프 낮음 = 한국이 글로벌보다 쌈   = 투매  -> 반등?

  "누가 나에게 돈을 잃는가?" = FOMO 로 웃돈 주고 사는 한국 리테일.
  이 질문에 답이 있는 건 펀딩비 이후 처음이다.

  그리고 이건 한국 판에서만 보이는 현상이다. 글로벌 봇들은 못 먹으니 관심이 없다.
  지금까지 우리가 진 상대들이 없는 자리라는 뜻이다.

계산:
  김프 = 업비트 KRW 가격 / (바이낸스 USD 가격 x 업비트 KRW-USDT) - 1

핵심 함정 (반드시 확인):
  김프가 되돌아가는 방법은 두 가지다.
    (a) 업비트가 내린다   <- 현물 롱만 하는 우리에게 의미 있음
    (b) 바이낸스가 오른다 <- 우리는 못 먹음
  (b) 로만 되돌아가면 상관이 아무리 예뻐도 돈은 못 번다. 그래서 [3]에서 이걸 쪼갠다.

실행:
    python kimchi_premium_check.py
"""

import os
import sys
import json
import time
import urllib.request
from datetime import datetime, timezone, timedelta

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("pip install pandas numpy")

import upbit_leadlag_check as U

COINS = ["BTC", "ETH", "XRP", "SOL", "DOGE"]

# 진입 후 몇 분 뒤 수익을 볼지. 김프는 몇 시간~며칠 단위로 움직인다.
HORIZONS_MIN = [60, 240, 720, 1440, 4320]   # 1시간, 4시간, 12시간, 1일, 3일

QUANTILE = 0.2      # 하위/상위 20% 를 '싸다/비싸다'로 본다

BINANCE_URL = "https://api.binance.com/api/v3/klines"
BINANCE_CACHE = "data_binance"
KST = U.KST


# ─────────────────────────────────────────────────────────────
# 바이낸스 (글로벌 가격)
# ─────────────────────────────────────────────────────────────

def _get(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kimchi-check/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2)


def load_binance(symbol):
    """바이낸스 5분봉. 업비트 캐시와 같은 기간/간격으로 맞춘다."""
    os.makedirs(BINANCE_CACHE, exist_ok=True)
    path = os.path.join(BINANCE_CACHE, f"{symbol}_{U.INTERVAL_MIN}m_{U.DAYS}d.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df["close"].rename(symbol)

    print(f"  받는 중 (바이낸스): {symbol} ...", flush=True)
    step = U.INTERVAL_MIN * 60_000
    end = int(time.time() * 1000)
    cursor = end - U.DAYS * 86_400_000
    rows = []
    while cursor < end:
        batch = _get(f"{BINANCE_URL}?symbol={symbol}&interval={U.INTERVAL_MIN}m"
                     f"&startTime={cursor}&limit=1000")
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + step
        time.sleep(0.12)
        if len(batch) < 1000:
            break

    if not rows:
        raise RuntimeError(f"{symbol} 데이터 없음")
    df = pd.DataFrame(rows, columns=["open_time", "o", "h", "l", "close", "v",
                                     "ct", "qv", "n", "tb", "tq", "ig"])
    df["t"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("t")[["close"]].astype(float)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.to_csv(path)
    print(f"    저장: {path} ({len(df)}개)")
    return df["close"].rename(symbol)


# ─────────────────────────────────────────────────────────────
# 김프 계산
# ─────────────────────────────────────────────────────────────

def build_premium(coin, usdt_krw):
    """
    김프 = 업비트 KRW / (바이낸스 USD x KRW-USDT) - 1

    세 시계열을 하나의 5분 그리드에 올려 맞춘다. 앞서 두 번 물린 정렬 문제를
    피하려고 그리드를 명시적으로 만들고 ffill 한다.
    """
    up = U.load_market(f"KRW-{coin}")
    bn = load_binance(f"{coin}USDT")

    start = max(up.index[0], bn.index[0], usdt_krw.index[0])
    end = min(up.index[-1], bn.index[-1], usdt_krw.index[-1])
    grid = pd.date_range(start, end, freq=f"{U.INTERVAL_MIN}min", tz="UTC")

    df = pd.DataFrame({
        "upbit": up.reindex(grid).ffill(),
        "binance": bn.reindex(grid).ffill(),
        "usdt": usdt_krw.reindex(grid).ffill(),
    }, index=grid).dropna()

    df["global_krw"] = df["binance"] * df["usdt"]      # 글로벌 가격을 원화로 환산
    df["premium"] = df["upbit"] / df["global_krw"] - 1
    return df


def tstat(r, n):
    if n <= 2 or abs(r) >= 1:
        return 0.0
    return r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2)


# ─────────────────────────────────────────────────────────────
# 검증
# ─────────────────────────────────────────────────────────────

def premium_stats(data):
    """[1] 김프가 실제로 존재하고 움직이는가. 안 움직이면 신호가 될 수 없다."""
    rows = []
    for coin, df in data.items():
        p = df["premium"] * 100
        rows.append({
            "코인": coin, "바": len(df),
            "김프 중앙값%": round(p.median(), 4),
            "최소%": round(p.min(), 4),
            "최대%": round(p.max(), 4),
            "표준편차%": round(p.std(), 4),
        })
    return pd.DataFrame(rows)


def _rw_benchmark(n, h, sigma, trials=30, seed=0):
    """
    랜덤워크 기준선.

    함정: 유한 표본에서 랜덤워크도 corr(수준, 변화)가 음수로 나온다 (단위근 편향).
    즉 corr < 0 이고 t < -2 라는 것만으로는 평균회귀의 증거가 못 된다. 랜덤워크도 그렇다.
    그래서 '같은 길이의 랜덤워크였다면 얼마나 나왔을까'를 구해서 그것보다 확실히
    더 음수여야 진짜 평균회귀로 본다.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(trials):
        w = np.cumsum(rng.normal(0, sigma, n))
        s = pd.Series(w)
        d = pd.DataFrame({"p": s, "dp": s.shift(-h) - s}).dropna()
        out.append(d["p"].corr(d["dp"]))
    return float(np.mean(out))


def reversion_test(data):
    """
    [2] 김프가 평균회귀하는가 (필요조건).

    판정은 절대 기준이 아니라 '랜덤워크였다면?' 과의 비교로 한다.
    """
    rows = []
    for coin, df in data.items():
        p = df["premium"]
        sigma = p.diff().std()
        for hmin in HORIZONS_MIN:
            h = hmin // U.INTERVAL_MIN
            dp = p.shift(-h) - p
            d = pd.DataFrame({"p": p, "dp": dp}).dropna()
            if len(d) < 100:
                continue
            r = d["p"].corr(d["dp"])
            bench = _rw_benchmark(len(d), h, sigma)
            # 랜덤워크 기준선보다 확실히(여유 0.1) 더 음수여야 진짜 회귀
            real = r < bench - 0.1
            rows.append({
                "코인": coin, "기간(분)": hmin, "표본": len(d),
                "corr(김프,김프변화)": round(r, 4),
                "랜덤워크였다면": round(bench, 4),
                "판정": "평균회귀" if real else "랜덤워크와 구분 안 됨",
            })
    return pd.DataFrame(rows)


def channel_test(data):
    """
    [3] 핵심 함정. 김프가 되돌아갈 때 '어느 쪽'이 움직이는가.

    corr(김프, 이후 업비트 수익)   <- 음수여야 우리가 먹을 수 있다
    corr(김프, 이후 바이낸스 수익) <- 이쪽이 주범이면 우리는 못 먹는다
    """
    rows = []
    for coin, df in data.items():
        p = df["premium"]
        for hmin in HORIZONS_MIN:
            h = hmin // U.INTERVAL_MIN
            up_ret = df["upbit"].shift(-h) / df["upbit"] - 1
            bn_ret = df["global_krw"].shift(-h) / df["global_krw"] - 1
            d = pd.DataFrame({"p": p, "up": up_ret, "bn": bn_ret}).dropna()
            if len(d) < 100:
                continue
            r_up = d["p"].corr(d["up"])
            r_bn = d["p"].corr(d["bn"])

            # 표본이 2만개가 넘으면 상관 0.05 도 t=7 이 된다. 통계적 유의성은 필터가 못 된다.
            # 그래서 두 가지를 요구한다:
            #   (1) 크기가 의미 있을 것 (r_up < -0.10)
            #   (2) 업비트 채널이 글로벌 채널보다 지배적일 것
            #       - 글로벌이 주범이면 김프가 해소돼도 우리는 못 먹는다
            meaningful = r_up < -0.10
            dominant = abs(r_up) > abs(r_bn)
            rows.append({
                "코인": coin, "기간(분)": hmin, "표본": len(d),
                "corr(김프,업비트수익)": round(r_up, 4),
                "corr(김프,글로벌수익)": round(r_bn, 4),
                "t(업비트)": round(tstat(r_up, len(d)), 1),
                "먹을 수 있나": "예" if (meaningful and dominant) else "아니오",
            })
    return pd.DataFrame(rows)


def _quantile_null(df, hmin, trials=30, seed=0):
    """
    랜덤워크 기준선 - 이 테스트가 '아무것도 없을 때' 만들어내는 가짜 엣지의 크기.

    왜 필요한가:
      김프는 정의상 업비트 가격 안에 들어있다.  업비트 = 글로벌 x (1 + 김프)
      그래서 김프가 순수 랜덤워크여도, 유한 표본에서는 극단값 뒤에 중앙으로 돌아오는
      착시(평균으로의 회귀)가 생기고, '쌀 때 사면 오른다'가 공짜로 만들어진다.

      실측으로 확인했다: 랜덤워크 김프를 심었더니 쌀때-baseline 이 +0.39% 나왔다.
      즉 이 기준선을 안 넘으면 '통과'는 아무 의미가 없다.

    방법: 실제 글로벌 가격은 그대로 두고 김프만 같은 변동성의 랜덤워크로 갈아끼운다.
    """
    rng = np.random.default_rng(seed)
    sigma = df["premium"].diff().std()
    n = len(df)
    h = hmin // U.INTERVAL_MIN
    g = df["global_krw"].values
    out = []
    for _ in range(trials):
        prem = np.cumsum(rng.normal(0, sigma, n))
        up = pd.Series(g * (1 + prem))
        r = up.shift(-h) / up - 1
        d = pd.DataFrame({"p": pd.Series(prem), "r": r}).dropna()
        if len(d) < 100:
            continue
        cheap = d[d["p"] <= d["p"].quantile(QUANTILE)]["r"]
        out.append(cheap.mean() - d["r"].mean())
    if not out:
        return 0.0
    # 평균이 아니라 상위 95% 를 쓴다.
    # 개별 랜덤워크는 평균 주위로 흩어지므로, 평균과 비교하면 절반은 그냥 통과해버린다.
    # 95 퍼센타일 = "랜덤워크 20번 중 19번은 이보다 작다" = 이걸 넘어야 우연이 아니다.
    return float(np.percentile(out, 95))


def quantile_test(data):
    """
    [4] 실제 매매로 옮기면. 롱만 가능하므로 '김프 쌀 때 사기'만 테스트한다.

    두 개를 넘어야 통과다:
      1) 비용 (수수료)
      2) 랜덤워크 기준선 (아무것도 없어도 생기는 가짜 엣지)
    """
    rows = []
    cost = U.ROUNDTRIP_COST
    for coin, df in data.items():
        p = df["premium"]
        lo, hi = p.quantile(QUANTILE), p.quantile(1 - QUANTILE)
        for hmin in HORIZONS_MIN:
            h = hmin // U.INTERVAL_MIN
            ret = df["upbit"].shift(-h) / df["upbit"] - 1
            d = pd.DataFrame({"p": p, "r": ret}).dropna()
            if len(d) < 100:
                continue
            cheap = d[d["p"] <= lo]["r"]
            rich = d[d["p"] >= hi]["r"]
            base = d["r"]

            edge = cheap.mean() - base.mean()
            null = _quantile_null(df, hmin)

            beats_cost = cheap.mean() - cost > 0
            beats_null = edge > null              # 랜덤워크 상위 95% 를 넘어야
            better_than_rich = cheap.mean() > rich.mean()

            rows.append({
                "코인": coin, "기간(분)": hmin,
                "쌀때 표본": len(cheap),
                "쌀때 평균%": round(cheap.mean() * 100, 4),
                "비쌀때 평균%": round(rich.mean() * 100, 4),
                "baseline%": round(base.mean() * 100, 4),
                "쌀때-baseline%": round(edge * 100, 4),
                "랜덤워크 기준선%": round(null * 100, 4),
                "비용차감후%": round((cheap.mean() - cost) * 100, 4),
                "판정": "통과" if (beats_cost and beats_null and better_than_rich) else "탈락",
            })
    return pd.DataFrame(rows)


def main():
    print("=" * 92)
    print("김치 프리미엄 신호 검증")
    print(f"  김프 = 업비트 KRW / (바이낸스 USD x 업비트 KRW-USDT) - 1")
    print(f"  코인: {', '.join(COINS)}   기간: {U.DAYS}일 {U.INTERVAL_MIN}분봉")
    print(f"  왕복 비용 가정: {U.ROUNDTRIP_COST*100:.2f}%")
    print("=" * 92)

    print("\n원달러 대용으로 업비트 KRW-USDT 사용")
    usdt = U.load_market("KRW-USDT")

    data = {}
    for c in COINS:
        try:
            data[c] = build_premium(c, usdt)
        except Exception as e:
            print(f"  {c} 건너뜀: {e}")
    if not data:
        sys.exit("데이터 없음")

    print("\n" + "─" * 92)
    print("[1] 김프가 존재하고 움직이는가")
    print("─" * 92)
    print(premium_stats(data).to_string(index=False))
    print("\n  표준편차가 0에 가까우면 김프가 안 움직인다는 뜻 = 신호가 될 수 없다.")

    print("\n" + "─" * 92)
    print("[2] 김프가 평균회귀하는가 (필요조건)")
    print("─" * 92)
    rv = reversion_test(data)
    print(rv.to_string(index=False))
    print("\n  corr 이 음수 = 김프가 높으면 이후 내려온다. 이게 안 되면 여기서 끝.")

    print("\n" + "─" * 92)
    print("[3] 핵심 - 되돌릴 때 어느 쪽이 움직이나")
    print("─" * 92)
    ch = channel_test(data)
    print(ch.to_string(index=False))
    print("\n  corr(김프, 업비트수익) 이 음수여야 우리가 먹는다 (김프 높으면 업비트가 내린다).")
    print("  글로벌 쪽만 움직여서 되돌아가면 우리는 못 먹는다. '먹을 수 있나' 컬럼을 볼 것.")

    print("\n" + "─" * 92)
    print(f"[4] 매매로 옮기면 - 김프 하위 {QUANTILE*100:.0f}% 일 때 매수 (롱만 가능하므로)")
    print("─" * 92)
    q = quantile_test(data)
    print(q.to_string(index=False))
    print("\n  '쌀때-baseline%' 가 핵심. 이게 0 근처면 김프 때문이 아니라 그냥 시장이 오른 것.")

    print("\n" + "=" * 92)
    print("판정")
    print("=" * 92)
    tradeable = ch[ch["먹을 수 있나"] == "예"]
    passed = q[q["판정"] == "통과"]
    print(f"  [3] 업비트 쪽으로 되돌아가는 조합: {len(tradeable)}/{len(ch)}")
    print(f"  [4] 비용까지 넘긴 조합:          {len(passed)}/{len(q)}")
    print()
    if len(passed) == 0:
        print("  통과 없음.")
        if len(tradeable) == 0:
            print("  -> 김프가 되돌아가더라도 업비트 쪽이 아니라 글로벌 쪽이 움직여서 그렇다.")
            print("     현물 롱만으로는 못 먹는다.")
        else:
            print("  -> 방향은 맞지만 폭이 수수료를 못 넘는다.")
    else:
        print(f"  통과 {len(passed)}개:")
        print(passed.to_string(index=False))
        print(f"\n  주의: {len(q)}개 테스트 중이므로 우연히 {len(q)*0.05:.1f}개는 나온다.")
        print("  그리고 [3] 에서 '먹을 수 있나'가 '예'인 조합과 겹쳐야 진짜다.")

    premium_stats(data).to_csv("results_kimchi_stats.csv", index=False)
    rv.to_csv("results_kimchi_reversion.csv", index=False)
    ch.to_csv("results_kimchi_channel.csv", index=False)
    q.to_csv("results_kimchi_quantile.csv", index=False)
    print("\n저장: results_kimchi_*.csv")


if __name__ == "__main__":
    main()
