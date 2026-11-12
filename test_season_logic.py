#!/usr/bin/env python3
"""
계절성 로직 검증 - 제일 중요한 건 '아웃샘플이 노이즈를 죽이는가'다.

  REAL  : 매일 07시에 오르는 진짜 패턴을 심음 -> 아웃샘플에서도 살아야
  NOISE : 아무 패턴 없음                      -> 인샘플에선 뭔가 발견되지만
                                                아웃샘플에서 죽어야 (이게 핵심)
"""
import numpy as np, pandas as pd
import upbit_leadlag_check as U
import intraday_season_check as S

S.MIN_DAYS = 20
S.MEASURED_SPREAD = {"REAL": 0.05, "NOISE": 0.05}
S.DEFAULT_SPREAD = 0.05
S.FEE_PER_SIDE = 0.0   # 검증에선 비용을 빼고 패턴 자체만 본다

rng = np.random.default_rng(4)
DAYS, BPD = 90, 288
N = DAYS * BPD
idx = pd.date_range("2026-04-01 00:00", periods=N, freq="5min", tz=U.KST).tz_convert("UTC")
kst_h = idx.tz_convert(U.KST).hour

# REAL: 07시~08시 구간에만 계통적으로 +0.3%/시간 드리프트를 심는다
real = rng.normal(0, 0.0015, N)
real[kst_h == 7] += 0.003 / 12      # 07시대 12개 바에 걸쳐 +0.3%

noise = rng.normal(0, 0.0015, N)

closes = pd.DataFrame({
    "REAL": 1000*np.exp(np.cumsum(real)),
    "NOISE": 1000*np.exp(np.cumsum(noise)),
}, index=idx)

hours = S.snap_hours(closes)
days = sorted(set(hours[0].index) & set(hours[12].index))
n = len(days); mid = n//2
print(f"총 {n}일 -> 인샘플 {mid} / 아웃샘플 {n-mid}")

ins = S.scan(hours, ["REAL","NOISE"], days, 0, mid)
outs = S.scan(hours, ["REAL","NOISE"], days, mid, n)

print("\n[인샘플 상위 6개]")
top = ins.sort_values("순수익%", ascending=False).head(6)
print(top.to_string(index=False))

key=["코인","진입(KST)","보유(h)"]
m = top[key+["순수익%"]].merge(outs[key+["순수익%"]], on=key, how="left", suffixes=("_인","_아웃"))
print("\n[아웃샘플 검증]")
print(m.to_string(index=False))

ok=True
# REAL 의 07시 진입이 인샘플 상위에 잡히나
r = ins[(ins["코인"]=="REAL")&(ins["진입(KST)"]=="07시")&(ins["보유(h)"]==1)]
h = len(r)>0 and r.iloc[0]["평균%"]>0.15; ok&=h
print(f"\n  REAL 07시 1h 인샘플 평균: {r.iloc[0]['평균%'] if len(r) else 'NA'}% (심은 값 +0.3%) -> {'OK' if h else '실패'}")
r2 = outs[(outs["코인"]=="REAL")&(outs["진입(KST)"]=="07시")&(outs["보유(h)"]==1)]
h = len(r2)>0 and r2.iloc[0]["평균%"]>0.15; ok&=h
print(f"  REAL 07시 1h 아웃샘플 평균: {r2.iloc[0]['평균%'] if len(r2) else 'NA'}% -> {'OK (진짜는 살아남음)' if h else '실패'}")

# NOISE 는 인샘플에서 뭔가 발견되지만 아웃샘플에서 죽어야
nz_in = ins[ins["코인"]=="NOISE"]["순수익%"].max()
nz_top = m[m["코인"]=="NOISE"]
print(f"\n  NOISE 인샘플 최고: +{nz_in:.4f}%  (아무것도 없는데 이만큼 '발견'된다)")
if len(nz_top):
    surv = (nz_top["순수익%_아웃"]>0).sum()
    h = surv < len(nz_top); ok&=h
    print(f"  NOISE 아웃샘플 생존: {surv}/{len(nz_top)} -> {'OK (노이즈가 죽음)' if h else '실패'}")
else:
    print("  NOISE 는 인샘플 상위에 못 들어옴 (REAL 이 더 강해서) - 정상")

print("\n최종:", "계절성 로직 검증 통과" if ok else "문제 있음")
