#!/usr/bin/env python3
"""
비트 기준 청산 테스트

아이디어: 익절/손절을 알트의 퍼센트가 아니라 '비트가 꺾이는 순간'으로 잡는다.
  산 이유 = "비트가 갔다"  ->  팔 이유 = "비트가 꺾였다"
  임의의 % 보다 논리적으로 일관된다.

  살 때:  비트가 5분에 +X%, 알트는 절반도 안 따라옴
  팔 때:  비트가 '진입 후 고점' 대비 -Y% 하면 청산   <- 이게 유일한 규칙
          (+ 시간 백스톱)

Y 를 쓸어가며 최적점이 있는지 본다.
"""
import glob, os
import numpy as np, pandas as pd
import upbit_leadlag_check as U
import exit_backtest as E

def simulate_leader_exit(closes, leader, alt, drop_from_peak, max_bars=24):
    lead=closes[leader].values; a=closes[alt].values; n=len(closes)
    w=E.ENTRY_BARS; cost=E.cost_of(alt)
    trades=[]; i=w
    while i < n-1:
        lm = lead[i]/lead[i-w]-1
        am = a[i]/a[i-w]-1
        if not (lm >= E.ENTRY_THRESH and am < E.GAP_RATIO*lm):
            i+=1; continue
        entry_alt=a[i]; peak=lead[i]; reason=None
        for j in range(i+1, min(i+1+max_bars, n)):
            if lead[j] > peak: peak = lead[j]          # 비트 고점 갱신 = 계속 들고 있음
            if lead[j]/peak - 1 <= -drop_from_peak:    # 비트가 고점 대비 꺾임 -> 청산
                reason="비트꺾임"; break
        else:
            j=min(i+max_bars, n-1); reason="시간"
        r=a[j]/entry_alt-1
        trades.append({"사유":reason,"수익%":r*100,"순수익%":(r-cost)*100})
        i=j+1
    return pd.DataFrame(trades)

files=sorted(glob.glob("data_upbit/*_5m_90d.csv"))
mk=[os.path.basename(f).split("_5m_")[0] for f in files]; mk=[m for m in mk if m!="KRW-USDT"]
closes,comp,_=U.build_close_frame(mk)
keep=[m for m in mk if comp[m]>=0.90]
alts=[m for m in keep if m not in E.LEADERS]
closes=closes[keep].dropna()

print("비트가 고점 대비 -Y% 꺾이면 청산  (그게 유일한 규칙, 진입 트리거 0.5%)")
print("="*92)
rows=[]
for y in [0.001,0.002,0.003,0.005,0.01,0.02]:
    tr_all=[]
    for leader in E.LEADERS:
        for alt in alts:
            t=simulate_leader_exit(closes,leader,alt,y)
            if len(t): tr_all.append(t)
    if not tr_all: continue
    t=pd.concat(tr_all)
    rows.append({"비트꺾임 기준%": y*100, "거래수": len(t),
                 "따라온폭%": round(t["수익%"].mean(),4),
                 "중앙값%": round(t["수익%"].median(),4),
                 "평균순수익%": round(t["순수익%"].mean(),4),
                 "총순수익%": round(t["순수익%"].sum(),2),
                 "승률%": round((t["순수익%"]>0).mean()*100,1)})
d=pd.DataFrame(rows)
print(d.to_string(index=False))
print()
print("  '평균순수익%'가 어디선가 양수로 뒤집히면 -> 원래 아이디어가 맞는 것")
print("  전부 음수면 -> 비트가 꺾이는 걸 볼 때쯤 알트도 이미 꺾여있다는 뜻 (corr 0.8)")
