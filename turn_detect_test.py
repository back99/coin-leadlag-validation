#!/usr/bin/env python3
"""
"꺾이는 구간"을 실시간에 알 수 있나?

원래 아이디어 그대로 규칙으로 만든다:
  "고점 찍고 떨어지기 시작했다"  =  최근 1시간 고점 대비 -X% 하락
  "저점 찍고 올라가기 시작했다"  =  최근 1시간 저점 대비 +X% 상승

그리고 그게 맞았는지 센다:
  하락 신호 후 -> 진짜 더 떨어졌나? (맞음)  아니면 다시 올랐나? (틀림)
  상승 신호 후 -> 진짜 더 올랐나?  (맞음)  아니면 다시 떨어졌나? (틀림)

50% 면 동전던지기 = 그 '구간'을 실시간엔 알 수 없다는 뜻.
"""
import glob, os
import numpy as np, pandas as pd
import upbit_leadlag_check as U

files=sorted(glob.glob("data_upbit/*_5m_90d.csv"))
mk=[os.path.basename(f).split("_5m_")[0] for f in files]; mk=[m for m in mk if m!="KRW-USDT"]
closes,comp,_=U.build_close_frame(mk)
keep=[m for m in mk if comp[m]>=0.90]
closes=closes[keep].dropna()

LOOK=12    # 최근 1시간(12바) 고점/저점
AHEAD=12   # 그 다음 1시간을 본다

print("원래 규칙을 그대로 만들어서, 실시간에 '꺾임'을 맞힐 수 있는지 센다")
print("="*92)
rows=[]
for m in keep:
    p=closes[m]
    peak=p.rolling(LOOK).max()
    trough=p.rolling(LOOK).min()
    fwd=p.shift(-AHEAD)/p-1          # 신호 이후 1시간 수익

    for x in [0.005, 0.01, 0.02]:
        # "고점 찍고 꺾였다" -> 팔아야 함 -> 그 뒤에 더 떨어져야 맞음
        sell_sig = (p/peak-1 <= -x)
        s = fwd[sell_sig].dropna()
        # "저점 찍고 반등했다" -> 사야 함 -> 그 뒤에 더 올라야 맞음
        buy_sig = (p/trough-1 >= x)
        b = fwd[buy_sig].dropna()
        if len(s)<100 or len(b)<100: continue
        rows.append({
            "코인": m, "기준%": x*100,
            "하락신호 수": len(s),
            "하락신호 적중%": round((s<0).mean()*100,1),     # 팔라 했는데 진짜 떨어짐
            "상승신호 수": len(b),
            "상승신호 적중%": round((b>0).mean()*100,1),     # 사라 했는데 진짜 오름
        })
d=pd.DataFrame(rows)
print(d.to_string(index=False))
print()
print("="*92)
print("전체 평균")
print("="*92)
for x in [0.5,1.0,2.0]:
    g=d[d["기준%"]==x]
    if len(g)==0: continue
    print(f"  기준 {x}%:   하락신호 적중 {g['하락신호 적중%'].mean():.1f}%   "
          f"상승신호 적중 {g['상승신호 적중%'].mean():.1f}%")
print()
print("  50% = 동전던지기 = 실시간엔 '꺾임'을 알 수 없다")
