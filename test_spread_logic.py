#!/usr/bin/env python3
"""분석기 산수 검증 - 답을 아는 데이터를 넣고 그대로 나오는지 본다."""
import pandas as pd, numpy as np, spread_analyze as S

rows=[]
# TIGHT: 스프레드 0.02% (수수료 0.1%보다 얇음)     -> 마켓메이킹 불가여야
# WIDE : 스프레드 0.50% (수수료보다 두꺼움)        -> 가능이어야
# ARB  : 빗썸이 업비트보다 1.0% 비쌈 (비용 0.3%)   -> 기회있음이어야
# NOARB: 두 거래소 가격 동일                       -> 없음이어야
for i in range(300):
    ts=f"2026-07-15T10:{i//60:02d}:{i%60:02d}+09:00"
    # TIGHT
    mid=100_000_000; s=mid*0.0002
    rows.append([ts,"TIGHT",mid-s/2,mid+s/2,"",""])
    # WIDE
    mid=1000; s=mid*0.005
    rows.append([ts,"WIDE",mid-s/2,mid+s/2,"",""])
    # ARB: 업비트 1000, 빗썸 1010 (1.0% 비쌈)
    rows.append([ts,"ARB",999.5,1000.5,1010.0,1011.0])
    # NOARB: 동일
    rows.append([ts,"NOARB",999.5,1000.5,999.5,1000.5])

pd.DataFrame(rows,columns=["ts_kst","coin","up_bid","up_ask","bt_bid","bt_ask"]).to_csv("_test_spread.csv",index=False)
S.IN="_test_spread.csv"
df=S.load()

sp,rt=S.spread_test(df)
print("[1] 스프레드 (왕복 수수료 %.3f%%)"%rt)
print(sp.to_string(index=False))
ok=True
t=sp[sp["코인"]=="TIGHT"].iloc[0]; w=sp[sp["코인"]=="WIDE"].iloc[0]
h=abs(t["스프레드 중앙값%"]-0.02)<0.001 and t["판정"]=="불가"; ok&=h
print(f"\n  TIGHT: 스프레드 {t['스프레드 중앙값%']}% (심어둔 값 0.02%), 판정={t['판정']} -> {'OK' if h else '실패'}")
h=abs(w["스프레드 중앙값%"]-0.50)<0.01 and w["판정"]=="가능"; ok&=h
print(f"  WIDE : 스프레드 {w['스프레드 중앙값%']}% (심어둔 값 0.50%), 판정={w['판정']} -> {'OK' if h else '실패'}")

ar,cost=S.arb_test(df)
print(f"\n[2] 거래소간 차익 (사이클 비용 {cost:.3f}%)")
print(ar.to_string(index=False))
a=ar[ar["코인"]=="ARB"].iloc[0]; n=ar[ar["코인"]=="NOARB"].iloc[0]
h=a["판정"]=="기회있음" and a["A(업→빗) 최고%"]>0; ok&=h
print(f"\n  ARB  : A수익 {a['A(업→빗) 최고%']}% (1.0% 차이 - 0.3% 비용 = 0.7% 기대), 판정={a['판정']} -> {'OK' if h else '실패'}")
h=n["판정"]=="없음"; ok&=h
print(f"  NOARB: 판정={n['판정']} -> {'OK (차이 없으면 없다고 함)' if h else '실패'}")
print("\n최종:", "분석기 산수 검증 통과" if ok else "문제 있음")
