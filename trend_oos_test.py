#!/usr/bin/env python3
"""
"떨어질 때 얼마나 떨어지면 팔아야 하나?" - 최적 조건을 찾고, 그게 미래에 통하는지 본다.

이게 실제로 할 일이다:
    오늘까지 데이터로 조건을 정한다  ->  내일부터 그 조건으로 돌린다

그래서 정확히 그렇게 재본다:
    앞 45일에서 최적 조건을 찾는다  (여기선 마음껏 뒤져도 된다)
    뒤 45일에 그 조건을 그대로 적용한다  (여기가 진짜)

넓게 뒤진다: 진입 3 x 청산 6 x 창 3 x 리더 2 = 108 조합
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
n=len(closes); mid=n//2

def trend(lead, al, up, dn, look, cost, s, e):
    pos=False; entry=0; peak=0; eq=1.0; ntr=0
    for i in range(max(look,s), e):
        if not pos:
            if lead[i]/lead[i-look]-1 >= up:
                pos=True; entry=al[i]; peak=lead[i]; ntr+=1
        else:
            if lead[i]>peak: peak=lead[i]
            if lead[i]/peak-1 <= -dn:
                eq *= (al[i]/entry)*(1-cost); pos=False
    if pos: eq *= (al[e-1]/entry)*(1-cost)
    return eq-1, ntr

UPS=[0.003,0.005,0.01]; DNS=[0.003,0.005,0.01,0.02,0.03,0.05]; LOOKS=[3,12,36]
print("앞 45일에서 최적 조건 찾기  ->  뒤 45일에 적용   (108 조합 전수 탐색)")
print("="*104)
rows=[]
for a in alts:
    cost=COST.get(a,DEF)/100
    al=closes[a].values
    best=None
    for L in ["KRW-BTC","KRW-ETH"]:
        lead=closes[L].values
        for up in UPS:
            for dn in DNS:
                for look in LOOKS:
                    r,_=trend(lead,al,up,dn,look,cost,0,mid)      # 인샘플
                    if best is None or r>best[0]: best=(r,L,up,dn,look)
    r_in,L,up,dn,look = best
    r_out,ntr = trend(closes[L].values,al,up,dn,look,cost,mid,n)  # 아웃샘플
    bh_in  = al[mid-1]/al[0]-1
    bh_out = al[n-1]/al[mid]-1
    rows.append({"알트":a,"최적 리더":L,"진입%":up*100,"청산%":dn*100,"창":look,
                 "인샘플%":round(r_in*100,1),"인샘플 들고있기%":round(bh_in*100,1),
                 "아웃샘플%":round(r_out*100,1),"아웃샘플 들고있기%":round(bh_out*100,1),
                 "아웃 거래수":ntr})
d=pd.DataFrame(rows)
print(d.to_string(index=False))
print()
print("="*104)
print(f"  인샘플   추세추종 {d['인샘플%'].mean():+.1f}%   vs  들고있기 {d['인샘플 들고있기%'].mean():+.1f}%")
print(f"  아웃샘플 추세추종 {d['아웃샘플%'].mean():+.1f}%   vs  들고있기 {d['아웃샘플 들고있기%'].mean():+.1f}%")
print()
print(f"  인샘플에서 들고있기를 이긴 코인:   {(d['인샘플%']>d['인샘플 들고있기%']).sum()}/{len(d)}")
print(f"  아웃샘플에서 들고있기를 이긴 코인: {(d['아웃샘플%']>d['아웃샘플 들고있기%']).sum()}/{len(d)}")
print()
print("  인샘플에서 좋은 건 당연하다 (108개 중 최고를 골랐으니).")
print("  아웃샘플이 진짜다. 여기서 무너지면 '조건 찾기'가 답이 아니라는 뜻.")
