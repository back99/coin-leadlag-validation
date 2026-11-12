#!/usr/bin/env python3
"""
익절을 크게, 보유를 길게 잡으면?

앞선 백테스트는 익절 최대 3%, 보유 최대 60분이었다. 그건 임의로 정한 값이다.
이기는 거래를 너무 일찍 잘랐을 수 있다.

그래서 익절 1~10%, 보유 1시간~24시간까지 넓혀서 다시 잰다.

대조군(random)이 핵심이다:
  같은 익절/손절/보유로 '아무 때나' 산 경우와 비교한다.
  신호 있는 쪽이 대조군보다 나아야 '신호 덕분'이다.
  비슷하면 그냥 오래 들고 있어서 나온 숫자다.
"""
import glob, os
import numpy as np, pandas as pd
import upbit_leadlag_check as U
import exit_backtest as E

rng = np.random.default_rng(0)

def sim(closes, leader, alt, tp, sl, max_bars, mode="signal"):
    lead=closes[leader].values; a=closes[alt].values; n=len(closes)
    w=E.ENTRY_BARS; cost=E.cost_of(alt)
    trades=[]; i=w
    while i < n-1:
        if mode=="signal":
            lm=lead[i]/lead[i-w]-1; am=a[i]/a[i-w]-1
            hit = (lm>=E.ENTRY_THRESH and am < E.GAP_RATIO*lm)
        else:
            hit = rng.random() < 0.004      # 대조군: 아무 때나 (비슷한 빈도)
        if not hit:
            i+=1; continue
        entry=a[i]; reason=None
        for j in range(i+1, min(i+1+max_bars, n)):
            r=a[j]/entry-1
            if r>=tp: reason="익절"; break
            if r<=-sl: reason="손절"; break
        else:
            j=min(i+max_bars, n-1); reason="포기"
        r=a[j]/entry-1
        trades.append({"사유":reason,"순수익%":(r-cost)*100})
        i=j+1
    return pd.DataFrame(trades)

files=sorted(glob.glob("data_upbit/*_5m_90d.csv"))
mk=[os.path.basename(f).split("_5m_")[0] for f in files]; mk=[m for m in mk if m!="KRW-USDT"]
closes,comp,_=U.build_close_frame(mk)
keep=[m for m in mk if comp[m]>=0.90]
alts=[m for m in keep if m not in E.LEADERS]
closes=closes[keep].dropna()
E.LEADERS=["KRW-ETH"]   # 신호가 제일 많이 나오는 리더로

print("익절 크게 + 보유 길게.  손절 3% 고정.  (신호 vs 대조군)")
print("="*100)
rows=[]
for mb,label in [(12,"1시간"),(72,"6시간"),(288,"24시간")]:
    for tp in [0.01,0.03,0.05,0.10]:
        for mode in ["signal","random"]:
            ts=[]
            for alt in alts:
                t=sim(closes,"KRW-ETH",alt,tp,0.03,mb,mode)
                if len(t): ts.append(t)
            if not ts: continue
            t=pd.concat(ts)
            rows.append({"보유":label,"익절%":tp*100,"종류":"신호" if mode=="signal" else "대조군",
                         "거래수":len(t),"평균순수익%":round(t["순수익%"].mean(),4),
                         "총순수익%":round(t["순수익%"].sum(),1),
                         "승률%":round((t["순수익%"]>0).mean()*100,1)})
d=pd.DataFrame(rows)
piv=d.pivot_table(index=["보유","익절%"],columns="종류",values=["평균순수익%","거래수"])
print(d.to_string(index=False))
print()
print("="*100)
print("신호 vs 대조군 비교 (평균순수익%)")
print("="*100)
s=d[d["종류"]=="신호"].set_index(["보유","익절%"])["평균순수익%"]
r=d[d["종류"]=="대조군"].set_index(["보유","익절%"])["평균순수익%"]
cmp=pd.DataFrame({"신호":s,"대조군":r,"차이(신호-대조군)":(s-r).round(4)})
print(cmp.to_string())
print()
print("  '차이'가 확실히 양수여야 신호가 값어치 있는 것.")
print("  0 근처면 -> 신호가 있으나 없으나 같다 = 신호가 무의미.")
