#!/usr/bin/env python3
"""
호가 수집기 - 마켓메이킹 / 거래소간 차익 검증용 데이터 모으기

왜 수집기가 따로 필요한가:
  캔들(과거 가격)은 API 가 주지만, 매수/매도 호가(bid/ask)는 과거를 안 준다.
  스프레드와 거래소간 가격차는 '지금' 찍어서 쌓는 수밖에 없다.

무엇을 재는가:
  [1] 업비트 스프레드   -> 마켓메이킹이 되나?
      호가 양쪽에 지정가를 걸어 스프레드를 먹는 게 마켓메이킹이다.
      스프레드가 왕복 수수료보다 작으면 산수적으로 불가능하다. 전략 이전의 문제.

  [2] 업비트 vs 빗썸 가격차 -> 거래소간 차익이 되나?
      싼 데서 사서 비싼 데서 판다. 국내끼리라 외환 규제 벽이 없다.
      가격차가 양쪽 수수료 합보다 커야 성립한다.

실행:
    python spread_collector.py            # 기본 2시간 수집
    python spread_collector.py 8          # 8시간 수집
    (Ctrl+C 로 중단해도 그때까지 모은 건 저장돼 있다)

결과: spread_data.csv (append 되므로 여러 번 나눠 돌려도 된다)
"""

import os
import sys
import csv
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

# 재볼 코인. 유동성 큰 것부터 얇은 것까지 섞는다.
# (유동성이 클수록 스프레드는 얇고, 얇을수록 스프레드는 넓지만 못 산다. 그 트레이드오프를 본다)
COINS = ["BTC", "ETH", "XRP", "SOL", "DOGE", "XLM", "ADA", "TRX"]

POLL_SEC = 3          # 몇 초마다 찍을지
DEFAULT_HOURS = 2

OUT = "spread_data.csv"
KST = timezone(timedelta(hours=9))

UPBIT_ORDERBOOK = "https://api.upbit.com/v1/orderbook?markets="
BITHUMB_ORDERBOOK = "https://api.bithumb.com/public/orderbook/ALL_KRW"


def _get(url, timeout=8):
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "spread-collector/1.0",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def fetch_upbit():
    """업비트 호가. 여러 마켓을 한 번에 받는다."""
    markets = ",".join(f"KRW-{c}" for c in COINS)
    data = _get(UPBIT_ORDERBOOK + markets)
    out = {}
    for item in data:
        coin = item["market"].split("-")[1]
        unit = item["orderbook_units"][0]     # 최우선 호가
        out[coin] = (float(unit["bid_price"]), float(unit["ask_price"]))
    return out


def fetch_bithumb():
    """빗썸 호가. 실패해도 치명적이지 않다 (업비트 스프레드만으로도 [1]은 답이 나온다)."""
    data = _get(BITHUMB_ORDERBOOK)
    if data.get("status") != "0000":
        raise RuntimeError(f"빗썸 응답 이상: status={data.get('status')}")
    out = {}
    for coin, v in data.get("data", {}).items():
        if coin in ("timestamp", "payment_currency"):
            continue
        if coin not in COINS:
            continue
        try:
            bid = float(v["bids"][0]["price"])
            ask = float(v["asks"][0]["price"])
            out[coin] = (bid, ask)
        except (KeyError, IndexError, TypeError):
            continue
    return out


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_HOURS
    end_at = time.time() + hours * 3600

    new_file = not os.path.exists(OUT)
    f = open(OUT, "a", newline="")
    w = csv.writer(f)
    if new_file:
        w.writerow(["ts_kst", "coin", "up_bid", "up_ask", "bt_bid", "bt_ask"])

    print("=" * 74)
    print("호가 수집기")
    print(f"  코인: {', '.join(COINS)}")
    print(f"  주기: {POLL_SEC}초   기간: {hours}시간")
    print(f"  출력: {OUT} (append)")
    print("=" * 74)

    bithumb_ok = True
    try:
        bt = fetch_bithumb()
        print(f"  빗썸 연결 OK ({len(bt)}개 코인)")
    except Exception as e:
        bithumb_ok = False
        print(f"  빗썸 연결 실패: {e}")
        print("  -> 업비트 스프레드([1])만 수집한다. 거래소간 차익([2])은 건너뛴다.")

    try:
        up = fetch_upbit()
        print(f"  업비트 연결 OK ({len(up)}개 코인)")
    except Exception as e:
        sys.exit(f"  업비트 연결 실패로 중단: {e}")

    print(f"\n수집 시작. Ctrl+C 로 중단해도 그때까지는 저장된다.\n")

    n, errors = 0, 0
    last_report = time.time()

    try:
        while time.time() < end_at:
            ts = datetime.now(KST).isoformat(timespec="seconds")
            try:
                up = fetch_upbit()
            except Exception:
                errors += 1
                time.sleep(POLL_SEC)
                continue

            bt = {}
            if bithumb_ok:
                try:
                    bt = fetch_bithumb()
                except Exception:
                    errors += 1

            for coin in COINS:
                if coin not in up:
                    continue
                ub, ua = up[coin]
                bb, ba = bt.get(coin, ("", ""))
                w.writerow([ts, coin, ub, ua, bb, ba])
            f.flush()
            n += 1

            if time.time() - last_report > 60:
                left = (end_at - time.time()) / 60
                print(f"  [{datetime.now(KST):%H:%M:%S}] {n}회 수집, 오류 {errors}회, "
                      f"남은 시간 {left:.0f}분")
                last_report = time.time()

            time.sleep(POLL_SEC)

    except KeyboardInterrupt:
        print("\n  중단됨 (지금까지 모은 건 저장됨)")
    finally:
        f.close()

    print(f"\n완료: {n}회 수집 -> {OUT}")
    print("이제 분석:  python spread_analyze.py")


if __name__ == "__main__":
    main()
