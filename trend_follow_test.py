#!/usr/bin/env python3
"""
순수 추세추종: 갭 조건 없이 그냥 비트 따라가기

지금까지 진입 조건에 항상 '갭'(알트가 아직 안 따라옴)이 있었다. 그건 지연을 전제한 것이다.
이건 다르다:
    비트가 오른다 -> 알트 산다   (갭 상관없음)
    비트가 꺾인다 -> 알트 판다

지연이 필요 없다. 비트가 오르는 동안 알트도 오르니 그 추세를 타는 것.

대조군: 그냥 알트 사서 90일 들고 있기 (buy & hold)
  추세추종의 진짜 값어치는 '하락장을 피하는 것'이라 이게 맞는 비교다.
"""
import glob, os
import numpy as np, pandas as pd
import upbit_leadlag_check as U

COST={"KRW-BTC":0.115,"KRW-ETH":0.135,"KRW-XRP":0.161,"KRW-SOL":0.188,
      "KRW-XLM":0.459,"KRW-DOGE":1.022}
DEF=0.25

files=sorted(glob.glob("data_upbit/*_5m_90d.csv"))
mk=[os.path.basename(f).split("_5m_")[0] for f in files]; mk=[m for m in mk if m!="KRW-USDT"]
closes,comp,_=U.build_close_frame(mk)
keep=[m for m in mk if comp[m]>=0.90]
closes=closes[keep].dropna()
alts=[m for m in keep if m not in ["KRW-BTC","KRW-ETH"]]

def trend(closes, L, a, up, dn, look):
    """비트가 look바 동안 up% 오르면 진입 / 진입후 비트 고점 대비 dn% 꺾이면 청산"""
    lead=closes[L].values; al=closes[a].values; n=len(closes)
    cost=COST.get(a,DEF)/100
    pos=False; entry=0; peak=0; eq=1.0; ntr=0
    for i in range(look, n):
        if not pos:
            if lead[i]/lead[i-look]-1 >= up:
                pos=True; entry=al[i]; peak=lead[i]; ntr+=1
        else:
            if lead[i]>peak: peak=lead[i]
            if lead[i]/peak-1 <= -dn:
                eq *= (al[i]/entry) * (1-cost)
                pos=False
    if pos: eq *= (al[-1]/entry)*(1-cost)
    return eq-1, ntr

print("순수 추세추종:  비트 오르면 알트 산다 / 비트 꺾이면 판다  (갭 조건 없음)")
print("="*96)
rows=[]
for a in alts:
    bh = closes[a].iloc[-1]/closes[a].iloc[0]-1        # 그냥 들고 있기
    best=None
    for L in ["KRW-BTC","KRW-ETH"]:
        for up in [0.003,0.005,0.01]:
            for dn in [0.003,0.005,0.01]:
                for look in [3,12,36]:
                    r,ntr=trend(closes,L,a,up,dn,look)
                    if best is None or r>best[0]: best=(r,ntr,L,up,dn,look)
    rows.append({"알트":a,"들고있기%":round(bh*100,1),
                 "추세추종 최고%":round(best[0]*100,1),"거래수":best[1],
                 "리더":best[2],"진입%":best[3]*100,"청산%":best[4]*100,"창(바)":best[5],
                 "차이":round((best[0]-bh)*100,1)})
d=pd.DataFrame(rows)
print(d.to_string(index=False))
print()
print(f"  추세추종이 들고있기를 이긴 코인: {(d['차이']>0).sum()}/{len(d)}")
print(f"  평균: 들고있기 {d['들고있기%'].mean():+.1f}%  vs  추세추종 {d['추세추종 최고%'].mean():+.1f}%")
print()
print("  주의: '추세추종 최고%'는 54개 조합 중 제일 좋은 걸 고른 값이다 (=인샘플 최적).")
print("        진짜인지 보려면 아웃샘플이 필요하다.")
