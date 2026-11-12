#!/usr/bin/env python3
"""
리드-래그 엣지 확인 (설계 1단계)

질문: "비트/이더가 움직인 뒤, 알트가 시차를 두고 따라오는가?
       그리고 그 따라오는 폭이 수수료를 넘는가?"

이 스크립트는 그 질문에만 답한다. 매매는 하지 않는다.

실행:
    pip install pandas numpy
    python leadlag_check.py

출력:
    [1] 지연 상관표    - 알트가 리더를 몇 분 시차에서 따라가는가
    [2] 이벤트 스터디  - 리더 급등 + 알트 미추종 상황에서 이후 수익
    [3] 판정          - 엣지가 왕복 비용을 넘는가

결과는 콘솔에 출력되고 results_leadlag.csv 로도 저장된다.
받은 캔들은 data/ 에 캐시되므로 재실행은 빠르다.
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
from datetime import datetime, timezone

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("pandas, numpy 가 필요하다:  pip install pandas numpy")


# ─────────────────────────────────────────────────────────────
# 설정 - 여기만 만지면 된다
# ─────────────────────────────────────────────────────────────

# 어느 시간대를 노리는가.
#   "fast" = 초 단위 시차. 1초봉. HFT와 같은 판. 수수료가 움직임보다 클 위험이 크다.
#   "slow" = 분 단위 과소반응. 1분봉. 속도 경쟁 아님.
MODE = "slow"

LEADERS = ["BTCUSDT", "ETHUSDT"]
ALTS = ["SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"]

PRESETS = {
    "fast": dict(
        interval="1s",
        days=2,               # 1초봉은 하루 86,400개라 길게 못 받는다
        max_lag=60,           # 60초까지 시차를 본다
        event_window=5,       # 리더 5초 움직임으로 판정
        event_threshold=0.001,  # 5초 안에 +0.1% 이상이면 급등
        horizons=[5, 10, 30, 60],  # 진입 후 5/10/30/60초
        unit="초",
    ),
    "slow": dict(
        interval="1m",
        days=30,
        max_lag=15,
        event_window=5,
        event_threshold=0.005,  # 리더가 5분 안에 +0.5% 이상
        # fast 모드에서 "시간이 길수록 엣지가 자란다"가 확인돼서 2시간까지 본다
        horizons=[5, 15, 30, 60, 120],
        unit="분",
    ),
}

_p = PRESETS[MODE]
INTERVAL = _p["interval"]
DAYS = _p["days"]
MAX_LAG = _p["max_lag"]
EVENT_WINDOW = _p["event_window"]
EVENT_THRESHOLD = _p["event_threshold"]
HORIZONS = _p["horizons"]
UNIT = _p["unit"]

GAP_RATIO = 0.5           # 알트가 리더 움직임의 50% 미만만 따라왔으면 "아직 갭 있음"

# 비용 가정 - 실제 값 확인 필요. 바이낸스 현물 테이커 수수료는 계정 등급/BNB 할인에 따라 다르다.
# 초 단위 전략은 무조건 시장가(테이커)라 이 비용을 반드시 문다.
FEE_PER_SIDE = 0.001      # 0.1%
SLIPPAGE_PER_SIDE = 0.0005  # 0.05% (얇은 알트일수록, 급할수록 커진다)
ROUNDTRIP_COST = 2 * (FEE_PER_SIDE + SLIPPAGE_PER_SIDE)  # 매수+매도 왕복

BASE_URL = "https://api.binance.com/api/v3/klines"
CACHE_DIR = "data"
INTERVAL_MS = {"1s": 1_000, "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000}


# ─────────────────────────────────────────────────────────────
# 데이터 수집
# ─────────────────────────────────────────────────────────────

def _get(url, retries=3):
    """단순 GET + 재시도."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "leadlag-check/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            # 429 = 레이트 리밋. 기다렸다 재시도.
            if e.code in (418, 429):
                wait = 10 * (attempt + 1)
                print(f"    레이트 리밋. {wait}초 대기...")
                time.sleep(wait)
                continue
            raise
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2)
    raise RuntimeError(f"요청 실패: {url}")


def fetch_klines(symbol, interval, days):
    """바이낸스에서 캔들을 페이지 단위로 받아온다 (1회 최대 1000개)."""
    step = INTERVAL_MS[interval]
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000

    rows = []
    cursor = start_ms
    while cursor < end_ms:
        url = f"{BASE_URL}?symbol={symbol}&interval={interval}&startTime={cursor}&limit=1000"
        batch = _get(url)
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + step
        time.sleep(0.15)  # 레이트 리밋 여유
        if len(batch) < 1000:
            break
    return rows


def load_symbol(symbol):
    """캐시가 있으면 쓰고, 없으면 받아서 캐시한다. close 시리즈를 반환."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{symbol}_{INTERVAL}_{DAYS}d.csv")

    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        print(f"  받는 중: {symbol} ...", flush=True)
        rows = fetch_klines(symbol, INTERVAL, DAYS)
        if not rows:
            raise RuntimeError(f"{symbol} 데이터가 비었다")
        df = pd.DataFrame(rows, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_base", "taker_quote", "ignore",
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("open_time")[["close"]].astype(float)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        df.to_csv(path)
        print(f"    저장: {path}  ({len(df)}개 캔들)")

    return df["close"].rename(symbol)


def build_close_frame(symbols):
    """모든 심볼의 종가를 같은 시간축으로 정렬한다."""
    series = [load_symbol(s) for s in symbols]
    frame = pd.concat(series, axis=1).dropna()
    return frame


# ─────────────────────────────────────────────────────────────
# [1] 지연 상관
# ─────────────────────────────────────────────────────────────

def lagged_corr_table(closes):
    """
    corr( 리더의 t 시점 수익률 , 알트의 t+L 시점 수익률 ) 을 L=0..MAX_LAG 로 계산.

    핵심은 L=0 이 아니라 L>=1 이다.
    L=0 만 높고 L>=1 이 0에 가까우면 -> 동시에 움직이는 것이지 따라오는 게 아니다 = 먹을 시차가 없다.
    """
    rets = np.log(closes).diff().dropna()
    out = []

    for leader in LEADERS:
        for alt in ALTS:
            corrs = {}
            for lag in range(0, MAX_LAG + 1):
                # 알트를 -lag 만큼 당기면 = 알트의 미래 수익률을 리더의 현재와 맞춘다
                corrs[lag] = rets[leader].corr(rets[alt].shift(-lag))

            positive_lags = {L: c for L, c in corrs.items() if L >= 1}
            best_lag = max(positive_lags, key=lambda L: positive_lags[L])

            out.append({
                "leader": leader,
                "alt": alt,
                "corr@0": round(corrs[0], 4),
                "best_lag": best_lag,
                "corr@best_lag": round(corrs[best_lag], 4),
                "corr@1": round(corrs[1], 4),
            })

    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────
# [2] 이벤트 스터디
# ─────────────────────────────────────────────────────────────

def event_study(closes):
    """
    실제 전략 조건 그대로 재본다:
      리더가 EVENT_WINDOW 안에 EVENT_THRESHOLD 이상 급등했고,
      알트는 아직 그 움직임의 GAP_RATIO 미만만 따라온 상황.
    그 시점에 알트를 샀다면 H분 뒤 수익이 얼마였나?

    baseline(무조건 매수)과 비교해야 의미가 있다.
    이벤트 수익이 baseline 과 같으면 그건 엣지가 아니라 그냥 시장이 오른 것.
    """
    out = []

    for leader in LEADERS:
        lead_w = closes[leader].pct_change(EVENT_WINDOW)

        for alt in ALTS:
            alt_w = closes[alt].pct_change(EVENT_WINDOW)

            # 리더 급등 + 알트 아직 미추종
            event = (lead_w >= EVENT_THRESHOLD) & (alt_w < GAP_RATIO * lead_w)

            for h in HORIZONS:
                fwd = closes[alt].shift(-h) / closes[alt] - 1.0

                ev = fwd[event].dropna()
                base = fwd.dropna()

                if len(ev) < 30:  # 표본이 너무 적으면 의미 없음
                    out.append({
                        "leader": leader, "alt": alt, "horizon": h,
                        "events": len(ev), "mean%": None, "median%": None,
                        "hit%": None, "beat_cost%": None,
                        "baseline%": None, "net%": None, "verdict": "표본부족",
                    })
                    continue

                mean = ev.mean()
                net = mean - ROUNDTRIP_COST

                # 핵심 지표: 개별 거래가 실제로 왕복 비용을 넘긴 비율.
                # 평균이 겨우 양수여도 이 비율이 낮으면 대부분의 거래는 손해다.
                beat_cost = (ev > ROUNDTRIP_COST).mean()

                out.append({
                    "leader": leader,
                    "alt": alt,
                    "horizon": h,
                    "events": len(ev),
                    "mean%": round(mean * 100, 4),
                    "median%": round(ev.median() * 100, 4),
                    "hit%": round((ev > 0).mean() * 100, 1),
                    "beat_cost%": round(beat_cost * 100, 1),
                    "baseline%": round(base.mean() * 100, 4),
                    "net%": round(net * 100, 4),
                    "verdict": "통과" if net > 0 and mean > base.mean() else "탈락",
                })

    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────

def main():
    symbols = LEADERS + ALTS
    print("=" * 78)
    print(f"리드-래그 엣지 확인   [MODE = {MODE}]")
    print(f"  리더: {', '.join(LEADERS)}")
    print(f"  알트: {', '.join(ALTS)}")
    print(f"  기간: 최근 {DAYS}일, {INTERVAL} 캔들  (시차/기간 단위 = {UNIT})")
    print(f"  왕복 비용 가정: {ROUNDTRIP_COST*100:.2f}%  (수수료 {FEE_PER_SIDE*100:.2f}% + 슬리피지 {SLIPPAGE_PER_SIDE*100:.2f}%, 편도 기준)")
    print("=" * 78)
    if MODE == "fast":
        print("  * 1초봉은 하루 86,400개다. 심볼당 수백 번 요청하므로 첫 실행은 오래 걸린다.")
        print("    (data/ 에 캐시되므로 재실행은 즉시)")
        print("=" * 78)

    closes = build_close_frame(symbols)
    print(f"\n정렬된 캔들: {len(closes)}개  "
          f"({closes.index[0]:%Y-%m-%d %H:%M} ~ {closes.index[-1]:%Y-%m-%d %H:%M} UTC)\n")

    print("─" * 78)
    print(f"[1] 지연 상관  -  알트가 리더를 몇 {UNIT} 뒤에 따라가는가")
    print("─" * 78)
    corr_df = lagged_corr_table(closes)
    print(corr_df.to_string(index=False))
    print("\n  읽는 법: corr@0 이 높고 corr@1 이하가 0에 가까우면 = 동시에 움직임 = 먹을 시차 없음.")
    print("           corr@best_lag 이 corr@0 대비 의미 있게 커야 시차가 존재한다.")

    print("\n" + "─" * 78)
    print(f"[2] 이벤트 스터디  -  리더 {EVENT_WINDOW}{UNIT}간 +{EVENT_THRESHOLD*100:.2f}% 이상 & 알트 미추종 시점 진입")
    print("─" * 78)
    ev_df = event_study(closes)
    print(ev_df.to_string(index=False))
    print(f"\n  horizon    = 진입 후 몇 {UNIT} 뒤")
    print(f"  mean%      = 이벤트 후 알트의 평균 수익 (비용 차감 전)")
    print(f"  hit%       = 그냥 오른 비율 (비용 무시)")
    print(f"  beat_cost% = 왕복비용 {ROUNDTRIP_COST*100:.2f}% 를 실제로 넘긴 거래 비율  <- 진짜 봐야 할 숫자")
    print(f"  baseline%  = 아무 때나 샀을 때 평균 수익 (비교군)")
    print(f"  net%       = mean% - 왕복비용")
    print(f"  통과 조건  = net% > 0  이고  mean% > baseline%")

    print("\n" + "=" * 78)
    print("[3] 판정")
    print("=" * 78)
    valid = ev_df[ev_df["mean%"].notna()]
    if len(valid):
        print(f"  이벤트 후 움직임 중앙값 최대: {valid['median%'].max():.4f}%")
        print(f"  넘어야 할 왕복 비용:        {ROUNDTRIP_COST*100:.2f}%")
        print(f"  비용 넘긴 거래 비율 최대:    {valid['beat_cost%'].max():.1f}%\n")

    passed = ev_df[ev_df["verdict"] == "통과"]
    if len(passed) == 0:
        print("  통과 조합 없음.")
        print("  -> 이 파라미터에서는 리드-래그 엣지가 비용을 못 넘는다.")
        if MODE == "fast":
            print("     초 단위는 움직임 자체가 작아서 수수료가 다 먹는 게 정상이다.")
            print("     MODE = 'slow' 로 바꿔서 분 단위 과소반응을 재보는 걸 권한다.")
        else:
            print("     EVENT_THRESHOLD 를 올리거나(더 큰 급등만), HORIZONS 를 늘리거나,")
            print("     다른 알트/기간으로 다시 재본다. 그래도 안 나오면 전략을 갈아탄다.")
    else:
        print(f"  통과 조합 {len(passed)}개:\n")
        print(passed.to_string(index=False))
        print("\n  -> 이 조합들로 2단계(대상/모드 확정)로 넘어간다.")
        print("     단, 표본 수(events)가 적고 beat_cost% 가 낮으면 우연일 수 있으니")
        print("     기간을 늘려 재확인할 것.")

    corr_df.to_csv("results_leadlag_corr.csv", index=False)
    ev_df.to_csv("results_leadlag_events.csv", index=False)
    print("\n저장: results_leadlag_corr.csv , results_leadlag_events.csv")


if __name__ == "__main__":
    main()
