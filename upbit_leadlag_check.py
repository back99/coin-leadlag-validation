#!/usr/bin/env python3
"""
업비트 리드-래그 + 세션 검증 (v2 - 정렬 버그 수정)

v1 의 치명적 버그:
  업비트는 거래가 없는 분에 캔들을 주지 않는다. 얇은 코인은 빈 분이 많다.
  그런데 concat().dropna() 로 "모든 코인에 캔들이 있는 분"만 남겼더니
  43,200분 -> 3,230분 으로 줄었고, 남은 행들이 시간상 띄엄띄엄해졌다.
  그 상태에서 shift(-30) 은 "30분 뒤"가 아니라 "30번째 행 뒤"(= 몇 시간 뒤일 수도)가 된다.
  결과적으로 horizon 이 전부 거짓말이 됐다.

v2 수정:
  1. 전체 시간 그리드에 재색인 후 forward-fill  -> shift(N) 이 진짜 N 바를 의미
  2. 스테이블코인 제외 (USDT 등은 리더를 따라가는 알트가 아니다)
  3. 데이터 완전성 필터 - 실제 캔들이 부족한 코인은 애초에 거래도 못 한다
  4. 판정 강화 - 평균만 보면 극단값에 속는다. 중앙값과 beat_cost% 도 같이 요구
  5. 비용 민감도 범위 확대 - 얇은 코인 슬리피지는 0.025% 가 아니다

실행:
    pip install pandas numpy
    python upbit_leadlag_check.py
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("pandas, numpy 가 필요하다:  pip install pandas numpy")


# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────

LEADERS = ["KRW-BTC", "KRW-ETH"]
# 24시간 거래량 순위는 '오늘 펌핑 중인 잡코인'을 위로 올린다 (평소엔 거래가 거의 없는 것들).
# 그래서 후보를 넉넉히 뽑고, 완전성 필터로 실제로 거래되는 코인만 남긴다.
TOP_N_ALTS = 20

INTERVAL_MIN = 5          # 분봉 단위. 얇은 코인은 1분봉에 구멍이 많아 5분봉이 안전하다.
DAYS = 90                 # 표본 부족이 계속 발목을 잡아서 90일로 늘림

# 아래는 전부 "분" 단위로 쓴다. 내부에서 바 개수로 변환한다.
MAX_LAG_MIN = 60
EVENT_WINDOW_MIN = 5
HORIZONS_MIN = [5, 15, 30, 60, 120]

EVENT_THRESHOLD = 0.005   # 리더가 EVENT_WINDOW 안에 +0.5% 이상
GAP_RATIO = 0.5           # 알트가 리더 움직임의 50% 미만만 따라왔으면 갭 있음

MIN_COMPLETENESS = 0.90   # 실제 캔들이 이 비율 미만인 코인은 제외 (거래가 너무 뜸함)
MIN_EVENTS = 50           # 이보다 표본이 적으면 판정하지 않는다

# 비용. 업비트 수수료 0.05% 수준(확인 필요).
# 슬리피지는 코인이 얇을수록 커진다. 잡알트는 0.1% 도 낙관적일 수 있다.
FEE_PER_SIDE = 0.0005
SLIPPAGE_PER_SIDE = 0.0005
ROUNDTRIP_COST = 2 * (FEE_PER_SIDE + SLIPPAGE_PER_SIDE)  # = 0.20%

KR_SESSION_START = 7
KR_SESSION_END = 19

# 리더를 따라가는 알트가 아니므로 제외
STABLE_SYMBOLS = {"USDT", "USDC", "DAI", "TUSD", "BUSD", "USDP", "FDUSD", "PYUSD"}

API = "https://api.upbit.com/v1"
CACHE_DIR = "data_upbit"
KST = timezone(timedelta(hours=9))

# 분 -> 바 개수
BARS = lambda minutes: max(1, minutes // INTERVAL_MIN)
MAX_LAG = BARS(MAX_LAG_MIN)
EVENT_WINDOW = BARS(EVENT_WINDOW_MIN)
HORIZONS = [BARS(m) for m in HORIZONS_MIN]


# ─────────────────────────────────────────────────────────────
# 업비트 API
# ─────────────────────────────────────────────────────────────

def _get(path, retries=4):
    url = f"{API}{path}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "Accept": "application/json",
                "User-Agent": "upbit-leadlag-check/2.0",
            })
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2)
    raise RuntimeError(f"요청 실패: {url}")


def get_top_krw_alts(n):
    """원화마켓을 24시간 거래대금 순으로 정렬해 상위 n개 알트를 고른다. 스테이블코인은 제외."""
    markets = [m["market"] for m in _get("/market/all?isDetails=false")
               if m["market"].startswith("KRW-")]

    tickers = []
    for i in range(0, len(markets), 50):
        tickers.extend(_get(f"/ticker?markets={','.join(markets[i:i+50])}"))
        time.sleep(0.15)

    tickers.sort(key=lambda t: t.get("acc_trade_price_24h", 0), reverse=True)

    alts = []
    for t in tickers:
        m = t["market"]
        if m in LEADERS:
            continue
        if m.split("-")[1] in STABLE_SYMBOLS:
            continue
        alts.append(m)
        if len(alts) >= n:
            break

    volumes = {t["market"]: t.get("acc_trade_price_24h", 0) for t in tickers}
    return alts, volumes


def fetch_candles(market, unit, days):
    """
    개수가 아니라 '시각'으로 멈춘다.

    개수로 멈추면, 거래가 뜸한 코인은 캔들이 듬성듬성해서 같은 개수를 채우려고
    훨씬 더 먼 과거까지 파고든다 (필요한 90일이 아니라 138일치를 받는 식).
    쓸데없는 요청이고, 어차피 그리드 밖이라 버려진다.
    """
    target_start = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    to = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    while True:
        batch = _get(f"/candles/minutes/{unit}?market={market}&count=200&to={to}")
        if not batch:
            break
        rows.extend(batch)
        oldest = pd.to_datetime(batch[-1]["candle_date_time_utc"], utc=True)
        if oldest <= target_start:
            break
        to = batch[-1]["candle_date_time_utc"] + "Z"
        time.sleep(0.12)
    return rows


def load_market(market):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{market}_{INTERVAL_MIN}m_{DAYS}d.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        print(f"  받는 중: {market} ...", flush=True)
        rows = fetch_candles(market, INTERVAL_MIN, DAYS)
        if not rows:
            raise RuntimeError(f"{market} 데이터가 비었다")
        df = pd.DataFrame(rows)
        df["t"] = pd.to_datetime(df["candle_date_time_utc"], utc=True)
        df = df.set_index("t")[["trade_price"]].rename(columns={"trade_price": "close"})
        df = df[~df.index.duplicated(keep="first")].sort_index().astype(float)
        df.to_csv(path)
        print(f"    저장: {path}  ({len(df)}개 캔들)")
    return df["close"].rename(market)


def build_close_frame(markets):
    """
    시간축을 만드는 곳. 여기서 두 번 틀렸으니 원칙을 박아둔다.

    원칙: 그리드는 '리더' 기준으로만 만든다.
          알트는 그 그리드에 맞추는 쪽이지, 그리드를 정하는 쪽이 아니다.

    왜:
      v1 - 모든 코인에 캔들이 있는 분만 남김(dropna) -> 시간축이 띄엄띄엄해져 shift 가 거짓말
      v2 - 그리드를 '모든 코인의 공통 구간'으로 잡음 -> 최근 상장 코인(KRW-DRV, 이력 1.5일)
           하나가 90일 분석을 1.5일로 잘라버림. 게다가 완전성을 그 잘린 구간 안에서 재니
           DRV 가 98.8% 로 나와서 필터도 못 잡음 (자기 기준으로 자기를 평가하는 순환논리).
      v3 - 리더(BTC/ETH)는 항상 전체 이력이 있다. 그걸 기준으로 그리드를 고정하고,
           알트의 완전성을 '전체 그리드 대비'로 재면 DRV 는 1.6% 로 드러나 정상적으로 걸러진다.

    반환하는 df 는 dropna 하지 않는다. 알트를 필터링한 뒤에 하는 게 맞다.
    """
    series = {m: load_market(m) for m in markets}
    for m in series:
        series[m] = series[m][~series[m].index.duplicated(keep="first")].sort_index()

    # 그리드는 리더만으로 결정한다
    leader_series = [series[m] for m in LEADERS]
    start = max(s.index[0] for s in leader_series)
    end = min(s.index[-1] for s in leader_series)
    grid = pd.date_range(start, end, freq=f"{INTERVAL_MIN}min", tz="UTC")

    frame, completeness, spans = {}, {}, {}
    for m, s in series.items():
        on_grid = s.reindex(grid)
        completeness[m] = on_grid.notna().mean()   # 전체 그리드 대비 (이게 핵심)
        spans[m] = (s.index[0], s.index[-1], len(s))
        frame[m] = on_grid.ffill()                 # 거래 없음 = 가격 유지

    df = pd.DataFrame(frame, index=grid)
    return df, completeness, spans


# ─────────────────────────────────────────────────────────────
# [1] 지연 상관
# ─────────────────────────────────────────────────────────────

def lagged_corr_table(closes, alts):
    rets = np.log(closes).diff().dropna()
    out = []
    for leader in LEADERS:
        for alt in alts:
            corrs = {L: rets[leader].corr(rets[alt].shift(-L)) for L in range(MAX_LAG + 1)}
            pos = {L: c for L, c in corrs.items() if L >= 1}
            best = max(pos, key=lambda L: pos[L])
            out.append({
                "leader": leader, "alt": alt,
                "corr@0": round(corrs[0], 4),
                f"corr@{INTERVAL_MIN}m": round(corrs[1], 4),
                "best_lag(분)": best * INTERVAL_MIN,
                "corr@best": round(corrs[best], 4),
            })
    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────
# [2] 이벤트 스터디
# ─────────────────────────────────────────────────────────────

def _event_mask(closes, leader, alt):
    lead_w = closes[leader].pct_change(EVENT_WINDOW)
    alt_w = closes[alt].pct_change(EVENT_WINDOW)
    return (lead_w >= EVENT_THRESHOLD) & (alt_w < GAP_RATIO * lead_w)


def _verdict(mean, median, beat, baseline, n):
    """
    v1 은 평균만 봐서 극단값에 속았다.
    진짜 엣지라면 평균과 중앙값이 같은 방향이고, 절반 이상의 거래가 비용을 넘어야 한다.
    """
    if n < MIN_EVENTS:
        return "표본부족"
    if mean > ROUNDTRIP_COST and median > ROUNDTRIP_COST and beat >= 0.5 and mean > baseline:
        return "통과"
    return "탈락"


def event_study(closes, alts):
    out = []
    for leader in LEADERS:
        for alt in alts:
            event = _event_mask(closes, leader, alt)
            for h, hmin in zip(HORIZONS, HORIZONS_MIN):
                fwd = closes[alt].shift(-h) / closes[alt] - 1.0
                ev = fwd[event].dropna()
                base = fwd.dropna()

                if len(ev) < MIN_EVENTS:
                    out.append({"leader": leader, "alt": alt, "horizon(분)": hmin,
                                "events": len(ev), "mean%": None, "median%": None,
                                "hit%": None, "beat_cost%": None, "baseline%": None,
                                "net%": None, "verdict": "표본부족"})
                    continue

                mean, median = ev.mean(), ev.median()
                beat = (ev > ROUNDTRIP_COST).mean()
                out.append({
                    "leader": leader, "alt": alt, "horizon(분)": hmin,
                    "events": len(ev),
                    "mean%": round(mean * 100, 4),
                    "median%": round(median * 100, 4),
                    "hit%": round((ev > 0).mean() * 100, 1),
                    "beat_cost%": round(beat * 100, 1),
                    "baseline%": round(base.mean() * 100, 4),
                    "net%": round((mean - ROUNDTRIP_COST) * 100, 4),
                    "verdict": _verdict(mean, median, beat, base.mean(), len(ev)),
                })
    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────
# [3] 세션 분석
# ─────────────────────────────────────────────────────────────

def session_study(closes, alts, horizon_min=30):
    h = BARS(horizon_min)
    kst_hour = closes.index.tz_convert(KST).hour
    day_mask = pd.Series((kst_hour >= KR_SESSION_START) & (kst_hour < KR_SESSION_END),
                         index=closes.index)
    out = []
    for leader in LEADERS:
        for alt in alts:
            event = _event_mask(closes, leader, alt)
            fwd = closes[alt].shift(-h) / closes[alt] - 1.0
            for label, mask in [("한국낮 07-19", event & day_mask),
                                ("한국밤 19-07", event & ~day_mask)]:
                ev = fwd[mask].dropna()
                if len(ev) < 20:
                    out.append({"leader": leader, "alt": alt, "session": label,
                                "events": len(ev), "mean%": None, "median%": None,
                                "beat_cost%": None})
                    continue
                out.append({
                    "leader": leader, "alt": alt, "session": label,
                    "events": len(ev),
                    "mean%": round(ev.mean() * 100, 4),
                    "median%": round(ev.median() * 100, 4),
                    "beat_cost%": round((ev > ROUNDTRIP_COST).mean() * 100, 1),
                })
    return pd.DataFrame(out)


def cost_sensitivity(ev_df):
    """얇은 코인 슬리피지는 추정이 크게 틀릴 수 있다. 비용을 바꿔가며 통과 개수를 본다."""
    rows = []
    v = ev_df[ev_df["mean%"].notna()]
    for c in [0.20, 0.30, 0.50, 1.00]:
        n = ((v["mean%"] > c) & (v["median%"] > c) & (v["mean%"] > v["baseline%"])).sum()
        rows.append({"왕복비용 가정%": c, "통과 조합 수": int(n)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 84)
    print("업비트 리드-래그 + 세션 검증  [v2 - 정렬 버그 수정]")
    print(f"  리더: {', '.join(LEADERS)}")
    print(f"  기간: 최근 {DAYS}일, {INTERVAL_MIN}분봉")
    print(f"  왕복 비용 가정: {ROUNDTRIP_COST*100:.2f}%  "
          f"(수수료 {FEE_PER_SIDE*100:.3f}% + 슬리피지 {SLIPPAGE_PER_SIDE*100:.3f}%, 편도)")
    print("=" * 84)

    print("\n원화마켓 거래량 상위 조회 중... (스테이블코인 제외)")
    alts, volumes = get_top_krw_alts(TOP_N_ALTS)
    for a in alts:
        print(f"    {a:14s}  {volumes[a]/1e8:>10,.0f} 억원")

    closes, comp, spans = build_close_frame(LEADERS + alts)
    expected = len(closes)

    print("\n" + "─" * 84)
    print("데이터 완전성  -  리더 기준 전체 그리드 대비 실제 캔들 비율")
    print("  (신규 상장이라 이력이 짧거나, 거래가 뜸해서 캔들이 빈 코인을 여기서 걸러낸다)")
    print("─" * 84)
    print(f"    {'마켓':14s} {'완전성':>7s}  {'첫 캔들(KST)':16s} {'캔들수':>7s}   판정")
    for m in LEADERS + alts:
        first, last, n = spans[m]
        flag = ""
        if m in alts and comp[m] < MIN_COMPLETENESS:
            flag = "제외 (이력 부족/거래 뜸함)"
        print(f"    {m:14s} {comp[m]*100:6.1f}%  {first.tz_convert(KST):%Y-%m-%d %H:%M}  "
              f"{n:7d}   {flag}")

    alts = [a for a in alts if comp[a] >= MIN_COMPLETENESS]
    if not alts:
        print("\n  완전성 기준을 통과한 알트가 없다.")
        print(f"  -> DAYS 를 줄이거나(신규 상장이 많으면), MIN_COMPLETENESS 를 낮춰라.")
        return

    closes = closes[LEADERS + alts].dropna()  # 알트 필터 후에 dropna (앞부분 ffill 불가 구간만 제거)

    print(f"\n정렬된 바: {len(closes)}개 / 리더 그리드 {expected}개  "
          f"({closes.index[0].tz_convert(KST):%Y-%m-%d %H:%M} ~ "
          f"{closes.index[-1].tz_convert(KST):%Y-%m-%d %H:%M} KST)")
    if len(closes) < expected * 0.8:
        print("  !!! 경고: 리더 그리드보다 20% 이상 짧다. 정렬을 다시 의심할 것.")
    else:
        print("  (리더 그리드와 비슷하면 정상)")

    print("\n" + "─" * 84)
    print("[1] 지연 상관  -  알트가 리더를 몇 분 뒤에 따라가는가")
    print("─" * 84)
    corr_df = lagged_corr_table(closes, alts)
    print(corr_df.to_string(index=False))
    print(f"\n  corr@0 이 높고 corr@{INTERVAL_MIN}m 이 0 근처면 = 같이 움직임 = 먹을 시차 없음.")
    print("  바이낸스가 정확히 그랬다. 업비트에서 다르게 나오는지가 이 검증의 핵심.")

    print("\n" + "─" * 84)
    print(f"[2] 이벤트 스터디  -  리더 {EVENT_WINDOW_MIN}분간 +{EVENT_THRESHOLD*100:.1f}% & 알트 미추종 진입")
    print("─" * 84)
    ev_df = event_study(closes, alts)
    print(ev_df.to_string(index=False))
    print(f"\n  통과 조건 (v1 보다 강화):")
    print(f"    mean% > {ROUNDTRIP_COST*100:.2f}  AND  median% > {ROUNDTRIP_COST*100:.2f}"
          f"  AND  beat_cost% >= 50  AND  mean% > baseline%  AND  표본 >= {MIN_EVENTS}")
    print(f"  (v1 은 평균만 봐서 극단값 몇 개에 속았다. 중앙값이 같이 양수여야 진짜다)")

    print("\n" + "─" * 84)
    print("[3] 세션 분석  -  한국 시간대별로 다른가 (horizon 30분)")
    print("─" * 84)
    ss_df = session_study(closes, alts, 30)
    print(ss_df.to_string(index=False))
    print("\n  가설: 한국 낮에 리테일이 뒤늦게 반응 -> 낮 수익이 밤보다 커야 한다.")

    print("\n" + "─" * 84)
    print("[4] 비용 민감도")
    print("─" * 84)
    print(cost_sensitivity(ev_df).to_string(index=False))

    print("\n" + "=" * 84)
    print("[5] 판정")
    print("=" * 84)
    v = ev_df[ev_df["mean%"].notna()]
    lag_col = f"corr@{INTERVAL_MIN}m"
    if len(v):
        print(f"  중앙값 최대:        {v['median%'].max():.4f}%")
        print(f"  넘어야 할 비용:     {ROUNDTRIP_COST*100:.2f}%")
        print(f"  beat_cost% 최대:    {v['beat_cost%'].max():.1f}%")
    print(f"  {lag_col} 최대:      {corr_df[lag_col].max():.4f}   (바이낸스는 0 근처였음)")

    passed = ev_df[ev_df["verdict"] == "통과"]
    print()
    if len(passed) == 0:
        print("  통과 조합 없음.")
        print(f"  -> {lag_col} 도 0 근처라면 시차 자체가 없는 것이고, 이 전략은 접는다.")
    else:
        print(f"  통과 조합 {len(passed)}개:\n")
        print(passed.to_string(index=False))
        print("\n  -> 표본 수를 반드시 같이 볼 것. 100개 미만이면 우연일 수 있다.")

    corr_df.to_csv("results_upbit_corr.csv", index=False)
    ev_df.to_csv("results_upbit_events.csv", index=False)
    ss_df.to_csv("results_upbit_sessions.csv", index=False)
    print("\n저장: results_upbit_corr.csv , results_upbit_events.csv , results_upbit_sessions.csv")


if __name__ == "__main__":
    main()
