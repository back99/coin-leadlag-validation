#!/usr/bin/env python3
"""
쌍별 관계(베타)를 먼저 구하고, 그걸로 갭을 다시 정의한다.

지금까지의 갭 정의:  "알트가 리더 움직임의 절반도 안 따라옴"
  -> 모든 알트가 리더를 1:1 로 따라간다고 가정한 것. 틀렸다.

알트마다 배율(베타)이 다르다:
  비트 +1%, SOL(베타 1.5) +0.5%  ->  기대 1.5%인데 0.5% 왔으니 진짜 갭 = 1.0%
  비트 +1%, DOGE(베타 0.5) +0.5% ->  기대 0.5%인데 0.5% 왔으니 갭 = 0  (다 따라옴)

베타로 보정한 갭 = (베타 x 리더움직임) - 알트움직임
"""
import glob, os
import numpy as np, pandas as pd
import upbit_leadlag_check as U

LEADERS=["KRW-BTC","KRW-ETH"]
COST={"KRW-BTC":0.115,"KRW-ETH":0.135,"KRW-XRP":0.161,"KRW-SOL":0.188,
      "KRW-XLM":0.459,"KRW-DOGE":1.022}
DEF=0.25

files=sorted(glob.glob("data_upbit/*_5m_90d.csv"))
mk=[os.path.basename(f).split("_5m_")[0] for f in files]; mk=[m for m in mk if m!="KRW-USDT"]
closes,comp,_=U.build_close_frame(mk)
keep=[m for m in mk if comp[m]>=0.90]
closes=closes[keep].dropna()
alts=[m for m in keep if m not in LEADERS]
rets=np.log(closes).diff().dropna()

print("[1] 쌍별 관계 - 베타와 설명력")
print("="*92)
rows=[]
for L in LEADERS:
    for a in alts:
        x=rets[L]; y=rets[a]
        beta=np.polyfit(x,y,1)[0]
        r=x.corr(y)
        rows.append({"리더":L,"알트":a,"베타":round(beta,3),
                     "corr@0":round(r,3),"설명력R²":round(r**2,3),
                     "corr@5m":round(x.corr(y.shift(-1)),4)})
d=pd.DataFrame(rows).sort_values("설명력R²",ascending=False)
print(d.to_string(index=False))
print("\n  베타 = 리더가 1% 가면 알트는 몇 % 가나")
print("  설명력R² = 알트 움직임 중 리더로 설명되는 비율")

print()
print("[2] 베타 보정 갭으로 재진입 - 관계 강한 쌍만")
print("="*92)
strong=d[d["설명력R²"]>=0.3]        # 관계가 확실한 쌍만
print(f"  설명력 30% 이상인 쌍: {len(strong)}개")
out=[]
for _,row in strong.iterrows():
    L,a,beta=row["리더"],row["알트"],row["베타"]
    lm=closes[L].pct_change(1); am=closes[a].pct_change(1)
    exp=beta*lm                                  # 베타로 기대되는 알트 움직임
    gap=exp-am                                   # 진짜 갭
    fwd=(closes[a].shift(-12)/closes[a]-1)*100
    for g in [0.005,0.01,0.02]:
        sig=(lm>=0.005)&(gap>=g)                 # 리더 0.5%+ & 베타보정 갭 g 이상
        s=fwd[sig].dropna(); b=fwd.dropna()
        if len(s)<30: continue
        c=COST.get(a,DEF)
        out.append({"리더":L,"알트":a,"베타":round(beta,2),"갭기준%":g*100,
                    "신호수":len(s),"평균%":round(s.mean(),4),
                    "기준선%":round(b.mean(),4),"차이":round(s.mean()-b.mean(),4),
                    "비용%":c,"차이-비용":round(s.mean()-b.mean()-c,4)})
o=pd.DataFrame(out)
if len(o):
    print(o.to_string(index=False))
    print(f"\n  차이-비용 > 0 : {(o['차이-비용']>0).sum()}/{len(o)}")
else:
    print("  신호 30건 이상인 조합 없음")
