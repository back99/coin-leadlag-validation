import numpy as np, pandas as pd
import upbit_leadlag_check as U
import overnight_session_check as O

U.LEADERS = ["KRW-BTC"]
rng = np.random.default_rng(5)
DAYS = 90
bars_per_day = 288                      # 5분봉 하루치
N = DAYS * bars_per_day
idx = pd.date_range("2026-04-01 00:00", periods=N, freq="5min", tz=U.KST).tz_convert("UTC")

btc = rng.normal(0, 0.0018, N)
kst_h = idx.tz_convert(U.KST).hour
is_night = (kst_h >= 19) | (kst_h < 7)   # 밤사이 = 미국 시간

# CATCHUP: 밤사이엔 리더를 절반만 따라가고, 한국 낮에 나머지를 따라잡는다  <- 원래 가설 그대로
catch = np.zeros(N)
day_start = [i for i in range(N) if kst_h[i] == 7 and idx[i].tz_convert(U.KST).minute == 0]
for i in range(N):
    catch[i] = 0.5 * btc[i] if is_night[i] else 0.5 * btc[i]
for ds in day_start:                     # 아침 7시에 밤사이 못 따라간 만큼을 낮 동안 메운다
    ns, ne = max(0, ds - 144), ds        # 직전 밤사이 12시간
    debt = 0.5 * btc[ns:ne].sum()        # 안 따라간 절반
    span = catch[ds:ds + 144]            # 낮 12시간에 걸쳐 분배
    if len(span) > 0:
        catch[ds:ds + 144] += debt / len(span)

# NOCATCH: 밤이든 낮이든 그냥 같이 움직임 (따라잡기 없음)
nocatch = 0.9 * btc + rng.normal(0, 0.0006, N)

def cl(r, s=1000.0): return s * np.exp(np.cumsum(r))
closes = pd.DataFrame({"KRW-BTC": cl(btc), "CATCHUP": cl(catch), "NOCATCH": cl(nocatch)}, index=idx)

print("=" * 74)
print("[검증] 심어둔 것: CATCHUP = 밤사이 절반만 따라가고 낮에 나머지 따라잡음")
print("                  NOCATCH = 그냥 같이 움직임 (따라잡기 없음)")
print("=" * 74)
ct = O.correlation_test(closes, ["CATCHUP", "NOCATCH"])
print(ct.to_string(index=False))

a_c = ct[ct["alt"]=="CATCHUP"].iloc[0]
a_n = ct[ct["alt"]=="NOCATCH"].iloc[0]
ok = True
hit = a_c["A: 갭→낮알트"] > 0.3 and a_c["판정"] == "유의"
ok &= hit
print(f"\n  CATCHUP: A={a_c['A: 갭→낮알트']}, t={a_c['t(A)']} -> {'OK (따라잡기를 잡아냄)' if hit else '실패'}")
hit = abs(a_n["A: 갭→낮알트"]) < 0.3
ok &= hit
print(f"  NOCATCH: A={a_n['A: 갭→낮알트']}, t={a_n['t(A)']} -> {'OK (없는 걸 없다고 함)' if hit else '실패'}")
print(f"  NOCATCH C(밤리더→밤알트)={a_n['C: 밤리더→밤알트']} (이미 밤에 따라갔음을 보여줌)")
print("\n최종:", "세션 로직 검증 통과" if ok else "문제 있음")
