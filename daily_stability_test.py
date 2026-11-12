#!/usr/bin/env python3
"""
날마다 관계가 바뀌나? 바뀐다면 미리 알 수 있나?

질문 3개:
  [1] 베타/상관이 안정적인가, 날마다 바뀌나?
  [2] 변동성이 큰 알트가 날마다 바뀌나? 어제 걸로 오늘을 알 수 있나?
  [3] 핵심 - 90일을 뭉치면 시차가 0인데, 날마다 재면 시차가 큰 날이 있나?
      있다면 그런 날을 미리(어제 데이터로) 알 수 있나?

  [3]에서 '어제 시차 -> 오늘 시차'가 예측되면 원래 아이디어가 산다.
  0이면 -> 시차가 큰 날은 사후에만 보인다 = 못 쓴다.
"""
import glob, os
import numpy as np, pandas as pd
import upbit_leadlag_check as U

LEADERS=["KRW-BTC","KRW-ETH"]
files=sorted(glob.glob("data_upbit/*_5m_90d.csv"))
mk=[os.path.basename(f).split("_5m_")[0] for f in files]; mk=[m for m in mk if m!="KRW-USDT"]
closes,comp,_=U.build_close_frame(mk)
keep=[m for m in mk if comp[m]>=0.90]
closes=closes[keep].dropna()
alts=[m for m in keep if m not in LEADERS]
rets=np.log(closes).diff().dropna()
day=np.array([d.date() for d in rets.index.tz_convert(U.KST)])
rets=rets.assign(_d=day)

print("[1] 관계가 날마다 바뀌나?  (일별 베타/상관의 안정성)")
print("="*88)
rows=[]
for L in LEADERS:
    for a in alts:
        betas=[]; cors=[]; lags=[]
        for d,g in rets.groupby("_d"):
            if len(g)<100: continue
            x=g[L].values; y=g[a].values
            if x.std()==0 or y.std()==0: continue
            betas.append(np.polyfit(x,y,1)[0])
            cors.append(np.corrcoef(x,y)[0,1])
            lags.append(np.corrcoef(x[:-1], y[1:])[0,1])   # 그날의 5분 시차
        if len(betas)<30: continue
        b=pd.Series(betas); c=pd.Series(cors); lg=pd.Series(lags)
        rows.append({
            "리더":L,"알트":a,"일수":len(b),
            "베타 평균":round(b.mean(),2),"베타 표준편차":round(b.std(),2),
            "상관 평균":round(c.mean(),3),
            "베타 지속성":round(b.autocorr(1),3),      # 어제 베타 -> 오늘 베타
            "시차 평균":round(lg.mean(),4),
            "시차 최대":round(lg.max(),3),
            "시차 지속성":round(lg.autocorr(1),3),     # 어제 시차 -> 오늘 시차  <- 핵심
        })
d=pd.DataFrame(rows).sort_values("상관 평균",ascending=False)
print(d.to_string(index=False))
print()
print("  '베타 지속성' = 어제 베타로 오늘 베타를 알 수 있나 (1이면 완전 예측, 0이면 불가)")
print("  '시차 지속성' = 어제 시차로 오늘 시차를 알 수 있나  <- 원래 아이디어의 핵심")
print()
print("="*88)
print(f"  베타 지속성 평균:  {d['베타 지속성'].mean():+.3f}")
print(f"  시차 지속성 평균:  {d['시차 지속성'].mean():+.3f}")
print(f"  시차 평균:        {d['시차 평균'].mean():+.4f}   (최대 {d['시차 최대'].max():.3f})")
print()
print("[2] 변동성 큰 알트가 날마다 바뀌나?")
print("="*88)
vrows=[]
for a in alts:
    v=[]
    for dd,g in rets.groupby("_d"):
        if len(g)<100: continue
        v.append(g[a].std())
    s=pd.Series(v)
    vrows.append({"알트":a,"일변동성 평균":round(s.mean()*100,3),
                  "변동성 지속성":round(s.autocorr(1),3)})
v=pd.DataFrame(vrows).sort_values("일변동성 평균",ascending=False)
print(v.to_string(index=False))
print(f"\n  변동성 지속성 평균: {v['변동성 지속성'].mean():+.3f}")
print("  (높으면 어제 변동성 컸던 알트가 오늘도 큼 = 미리 고를 수 있음)")
