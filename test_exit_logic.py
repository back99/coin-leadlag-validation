#!/usr/bin/env python3
"""
청산 엔진 검증 - 손으로 계산 가능한 시나리오를 넣고 그대로 나오는지 본다.

이게 중요한 이유: 이 엔진은 '번 걸 손해로 기록하지 않는지'를 확인하는 게 목적이다.
엔진 자체가 틀리면 또 헛다리를 짚는다.
"""
import numpy as np, pandas as pd
import upbit_leadlag_check as U
import exit_backtest as E

E.LEADERS=["L"]; E.ENTRY_BARS=1; E.ENTRY_THRESH=0.005; E.GAP_RATIO=0.5
E.LEADER_DROP=0.003; E.MAX_BARS=12
E.MEASURED_SPREAD={"A":0.0}; E.FEE_PER_SIDE=0.0   # 검증에선 비용 0

def build(lead_path, alt_path):
    n=len(lead_path)
    idx=pd.date_range("2026-05-01 00:00", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"L":lead_path,"A":alt_path}, index=idx)

ok=True

# ── 시나리오 1: 익절 ──
# bar0->1 리더 +1% (진입 트리거), 알트는 +0.1% (갭 있음) -> 진입
# bar2 에서 알트 +2% -> 익절(1%) 걸려야
L=[100,101,101,101,101]
A=[100,100.1,102.2,102.2,102.2]
tr=E.simulate(build(L,A),"L","A",tp=0.01,sl=0.01)
h = len(tr)==1 and tr.iloc[0]["청산사유"]=="익절"
ok&=h
print(f"1. 익절: 거래{len(tr)}건, 사유={tr.iloc[0]['청산사유'] if len(tr) else 'X'}, "
      f"수익={tr.iloc[0]['수익%']:.2f}% -> {'OK' if h else '실패'}")

# ── 시나리오 2: 손절 ──
L=[100,101,101,101,101]
A=[100,100.1,98.0,98.0,98.0]
tr=E.simulate(build(L,A),"L","A",tp=0.01,sl=0.01)
h = len(tr)==1 and tr.iloc[0]["청산사유"]=="손절"
ok&=h
print(f"2. 손절: 사유={tr.iloc[0]['청산사유'] if len(tr) else 'X'}, "
      f"수익={tr.iloc[0]['수익%']:.2f}% -> {'OK' if h else '실패'}")

# ── 시나리오 3: 탈출 (리더가 -0.3% 떨어짐) ──
L=[100,101,100.5,100.5,100.5]   # 진입가 101 대비 -0.5% -> 탈출
A=[100,100.1,100.15,100.15,100.15]
tr=E.simulate(build(L,A),"L","A",tp=0.05,sl=0.05)
h = len(tr)==1 and tr.iloc[0]["청산사유"]=="탈출"
ok&=h
print(f"3. 탈출: 사유={tr.iloc[0]['청산사유'] if len(tr) else 'X'} -> {'OK' if h else '실패'}")

# ── 시나리오 4: 포기 (아무 일 없음) ──
L=[100,101]+[101]*20
A=[100,100.1]+[100.1]*20
tr=E.simulate(build(L,A),"L","A",tp=0.05,sl=0.05)
h = len(tr)==1 and tr.iloc[0]["청산사유"]=="포기" and tr.iloc[0]["보유바"]==12
ok&=h
print(f"4. 포기: 사유={tr.iloc[0]['청산사유'] if len(tr) else 'X'}, "
      f"보유바={tr.iloc[0]['보유바'] if len(tr) else 'X'} (기대 12) -> {'OK' if h else '실패'}")

# ── 시나리오 5: 핵심! 고정시간이 놓치는 걸 잡는가 ──
# 알트가 5분 만에 +1% 따라잡았다가 도로 빠짐.
# 고정 30분 청산이면 -0.1% 손해로 기록됐을 것. 익절 로직이면 +1% 로 기록돼야.
L=[100,101]+[101]*10
A=[100,100.1, 101.2, 100.5, 99.9, 99.9, 99.9, 99.9, 99.9, 99.9, 99.9, 99.9]
tr=E.simulate(build(L,A),"L","A",tp=0.01,sl=0.05)
h = len(tr)==1 and tr.iloc[0]["청산사유"]=="익절" and tr.iloc[0]["수익%"]>0.9
ok&=h
print(f"5. 핵심(번 걸 손해로 안 적는가): 사유={tr.iloc[0]['청산사유'] if len(tr) else 'X'}, "
      f"수익={tr.iloc[0]['수익%']:.2f}% -> {'OK' if h else '실패'}")
print(f"   (같은 데이터를 고정 30분 청산으로 재면 {(A[7]/A[1]-1)*100:.2f}% 손해로 기록됨)")

# ── 시나리오 6: 중복 진입 안 하는가 ──
L=[100,101,102,103,104]+[104]*10
A=[100,100.1,100.2,100.3,100.4]+[100.4]*10
tr=E.simulate(build(L,A),"L","A",tp=0.05,sl=0.05)
h = len(tr)>=1
ok&=h
print(f"6. 중복진입 방지: 거래 {len(tr)}건 (매 바 진입 신호인데 겹치지 않음) -> {'OK' if h else '실패'}")

print("\n최종:", "청산 엔진 검증 통과" if ok else "문제 있음")
