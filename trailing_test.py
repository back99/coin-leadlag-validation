#!/usr/bin/env python3
"""
트레일링 스탑 테스트

아이디어: 고정 익절이 아니라 "오르다가 꺾이는 순간"에 판다.
  오르는 동안 계속 들고 있다가, 알트가 '진입 후 고점' 대비 -X% 하면 청산.
  고정 익절의 문제(낮으면 일찍 자름, 높으면 안 걸림)를 피한다.

대조군 필수: 같은 규칙으로 아무 때나 산 경우와 비교한다.
"""
import glob, os
import numpy as np, pandas as pd
import upbit_leadlag_check as U
import exit_backtest as E

rng=np.random.default_rng(1)

def sim(closes, leader, alt, trail, max_bars, mode="signal"):
    lead=closes[leader].values; a=closes[alt].values; n=len(closes)
    w=E.ENTRY_BARS; cost=E.cost_of(alt)
    trades=[]; i=w
    while i<n-1:
        if mode=="signal":
            lm=lead[i]/lead[i-w]-1; am=a[i]/a[i-w]-1
            hit=(lm>=E.ENTRY_THRESH and am<E.GAP_RATIO*lm)
        else:
            hit=rng.random()<0.004
        if not hit: i+=1; continue
        entry=a[i]; peak=a[i]; reason=None
        for j in range(i+1, min(i+1+max_bars,n)):
            if a[j]>peak: peak=a[j]                 # 고점 갱신 = 계속 보유
            if a[j]/peak-1 <= -trail:               # 고점 대비 꺾임 -> 청산
                reason="트레일"; break
        else:
            j=min(i+max_bars,n-1); reason="시간"
        r=a[j]/entry-1
        trades.append({"사유":reason,"수익%":r*100,"순수익%":(r-cost)*100,
                       "고점도달%":(peak/entry-1)*100})
        i=j+1
    return pd.DataFrame(trades)

files=sorted(glob.glob("data_upbit/*_5m_90d.csv"))
mk=[os.path.basename(f).split("_5m_")[0] for f in files]; mk=[m for m in mk if m!="KRW-USDT"]
closes,comp,_=U.build_close_frame(mk)
keep=[m for m in mk if comp[m]>=0.90]
alts=[m for m in keep if m not in ["KRW-BTC","KRW-ETH"]]
closes=closes[keep].dropna()

print("트레일링 스탑: 알트가 고점 대비 -X% 꺾이면 청산")
print("="*96)
rows=[]
for mb,lab in [(12,"1시간"),(72,"6시간")]:
    for tr_ in [0.002,0.003,0.005,0.01,0.02]:
        for mode in ["signal","random"]:
            ts=[]
            for alt in alts:
                t=sim(closes,"KRW-ETH",alt,tr_,mb,mode)
                if len(t): ts.append(t)
            if not ts: continue
            t=pd.concat(ts)
            rows.append({"보유":lab,"트레일%":tr_*100,"종류":"신호" if mode=="signal" else "대조군",
                         "거래수":len(t),
                         "평균고점%":round(t["고점도달%"].mean(),3),
                         "평균수익%":round(t["수익%"].mean(),4),
                         "평균순수익%":round(t["순수익%"].mean(),4),
                         "승률%":round((t["순수익%"]>0).mean()*100,1)})
d=pd.DataFrame(rows)
print(d.to_string(index=False))
print()
print("="*96)
s=d[d["종류"]=="신호"].set_index(["보유","트레일%"])["평균순수익%"]
r=d[d["종류"]=="대조군"].set_index(["보유","트레일%"])["평균순수익%"]
print(pd.DataFrame({"신호":s,"대조군":r,"차이":(s-r).round(4)}).to_string())
print()
print("  '평균고점%' = 진입 후 알트가 최대 얼마나 올랐나 (여기서 트레일만큼 빼고 나옴)")
