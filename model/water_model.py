"""25 cm² 포켓 셀: 액상 물을 넣은 2D 깊이 평균 모델 (2편 준비용 1차 판).

    python water_model.py check                # 물을 끈 계산이 port_sweep.py 와 같은지, 격자·반복 수렴
    python water_model.py designs              # 1편의 대표 설계 × 입구 가습 조건
    python water_model.py sens                 # 잔류 포화도·임계 트래핑 수 민감도 (설계 순위가 바뀌는 범위)

기체 유동·산소 식과 형상은 port_sweep.py 와 같다 (70 °C, 대기압, 건공기 기준 0.532 L/min, 0.76 V).
더한 것:
  1) 물 수지: 입구 수증기(RH) + 생성수(i / 2F, 전부 캐소드로 나온다고 가정)를 기체 유동이 실어 나른다.
     포화 농도를 넘는 양이 액상이다(등온, 평형 응축).
  2) 액상 포화도 s: 액상이 기체와 같은 압력 구배로 흐른다고 보고(전단 구동, 모세관 확산 없음)
     상대 투과도 k_rl = s_e^n, k_rg = (1 - s)^n 의 유량 비에서 s_e 를 구한다.
     여기에 젖은 곳의 잔류 포화도 s_r = s_r0 / (1 + N / N_c) 를 더한다. N = K |∇p| / σ (트래핑 수).
  3) 되먹임: 기체 투과도 × (1 - s)^n, 폼 확산계수 × (1 - s)^1.5, GDL 면 가림 × (1 - s)^cover.
s_r0, N_c, n 은 폼에 대한 측정값이 없는 가정값이다. 실험(젖은 압력강하 / 마른 압력강하)으로 정해야 한다.
산소 소모·수증기에 따른 기체 유량 변화, 온도 분포, GDL 안의 물, 헤더(빈 공간) 안의 물은 없다.
"""
import csv, json, os, sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from graded_porosity_opt import (F_, R_, T, P, MU, NU, D_GAS, C_IN, H_F, H_GDL, K_GDL, F_GDL, R_OTHER, C_F, Q,
                                 EPS_BASE, SOLID, FOAM, OUT, k_of_eps, hmean, metrics)
from port_sweep import grid, G_PORT, BASE_IN, BASE_OUT, LAYOUTS

P_SAT = 31.2e3                      # 70 °C 포화 수증기압 (Pa)
C_SAT = P_SAT / (R_ * T)
MU_L, RHO_L, SIGMA, M_W = 4.04e-4, 977.8, 0.0644, 0.018015
S_MAX = 0.9
DEFAULT = dict(s_r0=0.4, n_c=1e-4, n_kr=3.0, cover=1.0)


def props(t_c):
    """온도 t_c (°C) 의 물성: 공기 점성(Sutherland), 동점성, 포화 수증기압(Buck), 물 점성(Vogel), 표면장력, 물 밀도."""
    tk = t_c + 273.15
    mu = 1.716e-5 * (tk / 273.15) ** 1.5 * (273.15 + 110.4) / (tk + 110.4)
    p_sat = 611.21 * np.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))
    return dict(tk=tk, mu=mu, nu=mu / (P * 0.02897 / (R_ * tk)), p_sat=p_sat, c_sat=p_sat / (R_ * tk),
                mu_l=2.414e-5 * 10 ** (247.8 / (tk - 140)), sigma=0.0756 - 1.48e-4 * t_c,
                rho_l=1000 * (1 - (t_c + 288.94) * (t_c - 3.99) ** 2 / (508929.2 * (t_c + 68.13))))


BASE_PROPS = dict(tk=T, mu=MU, nu=NU, p_sat=P_SAT, c_sat=C_SAT, mu_l=MU_L, sigma=SIGMA, rho_l=RHO_L)


TAFEL = (0.85, 0.070, 0.1)         # 기존 모델의 전압 대응: V = 0.85 − 0.070 log10(ik / 0.1) (옴 손실 없음일 때 ik = 2 ↔ 0.76 V)


def r_mem(a, tk, t_mem=50.8e-6, lam_an=14.0):
    """막 저항 (Ω·cm²). 캐소드 쪽 λ 는 활동도 a 에서 Springer(1991), 애노드 쪽 λ = 14(포화), 두께 방향 선형 분포."""
    lam_c = np.maximum(0.043 + 17.18 * a - 39.85 * a ** 2 + 36 * a ** 3, 1.0)
    z = np.linspace(0.0, 1.0, 11)
    lam = lam_c[..., None] * (1 - z) + lam_an * z
    sig = (0.005139 * lam - 0.00326) * np.exp(1268 * (1 / 303 - 1 / tk))   # S/cm
    return t_mem * 100 * np.trapz(1 / sig, z, axis=-1)


def solve_wet(g, rh_in=1.0, ik=2.0, wet=True, s_r0=0.4, n_c=1e-4, n_kr=3.0, cover=1.0, eps=EPS_BASE,
              n_it=80, relax=0.3, tol=1e-4, t_c=None, q_slpm=0.532, w_inj=0.0, k_scale=1.0, c_f=C_F,
              v_cell=None, asr_c=0.05, t_mem=50.8e-6, compress=False):
    """compress=True 면 국소 절대압력 P + p (출구 = 대기압 P) 를 쓴다: 기체 밀도(질량 보존), 산소·수증기 농도,
    포화 몰분율 P_SAT / (P + p), 입구 RH 는 입구 절대압력 기준. 분자 확산 저항(GDL·폼)은 압력에 반비례하지만
    전체 농도에 비례하는 몫과 상쇄되어 몰분율 식에서는 그대로이고, 반응·R_OTHER(이오노머 쪽)는 농도 기준이라 밀도비로 나눈다.
    False 면 이전과 같다(모든 곳 대기압).
    v_cell (V) 을 주면 전압 고정 계산: 국소 반응 속도가 Tafel(TAFEL) + 옴 손실(접촉 asr_c Ω·cm² + 막 r_mem)로 정해진다.
    막 저항은 국소 수분 활동도(캐소드 기체, 애노드 포화 가정)에서 Springer 식으로 구한다. 이때 ik 는 쓰지 않는다.
    v_cell 이 None 이면 이전과 같다(ik 고정, 옴 손실 없음).
    """
    """t_c 를 주면 그 온도의 물성을 쓴다(없으면 70 °C 기본값, 이전 결과와 같다). q_slpm 은 건공기 표준 유량(L/min, 0 °C 기준).
    w_inj 는 입구에 넣는 액상 물 (mL/min, 셀 밖 2단계 시험). ik = 0 이면 반응이 없다. k_scale, c_f 는 1단계 측정으로 맞출 폼 투과도 배수와 관성 계수."""
    pr = BASE_PROPS if t_c is None else props(t_c)
    MU, NU, P_SAT, C_SAT, MU_L, SIGMA, RHO_L, T_ = (pr[k] for k in ("mu", "nu", "p_sat", "c_sat", "mu_l", "sigma", "rho_l", "tk"))
    kind, dx, act = g["kind"], g["dx"], g["act"]
    fl = kind != SOLID
    idx = -np.ones(kind.shape, int)
    idx[fl] = np.arange(fl.sum())
    n = fl.sum()
    foam = kind == FOAM
    K0 = np.where(foam, k_scale * k_of_eps(eps), H_F ** 2 / 12)
    Tg = np.where(act, H_GDL * K_GDL / MU, 0.0)
    ax = fl[:, :-1] & fl[:, 1:]
    ay = fl[:-1, :] & fl[1:, :]
    ia, ib = np.concatenate([idx[:, :-1][ax], idx[:-1, :][ay]]), np.concatenate([idx[:, 1:][ax], idx[1:, :][ay]])
    k_in, k_out = idx[g["p_in"]], idx[g["p_out"]]
    gport = np.zeros(n)
    gport[k_in] = G_PORT
    gport[k_out] = G_PORT
    rhs = np.zeros(n)
    rhs[k_in] = G_PORT
    all_n = np.arange(n)
    nxf = ax.sum()

    def faces(field):
        return np.concatenate([hmean(field[:, :-1], field[:, 1:])[ax], hmean(field[:-1, :], field[1:, :])[ay]])

    rho = np.ones(kind.shape)                                       # 국소 절대압력 / P (compress 일 때만 바뀐다)
    p_in = P                                                        # 입구 절대압력

    def inlet():
        x_v = rh_in * P_SAT / p_in                                  # 입구 수증기 몰분율
        q_gas = q_slpm / 6e4 * (T_ / 273.15) * (101325 / P) / (1 - x_v)   # 건공기 유량 고정, 수증기만큼 총 유량 증가 (P 기준 부피)
        return x_v, q_gas, C_IN * (1 - x_v)

    x_v, q_gas, c_o2 = inlet()
    if v_cell is None:
        k_rxn = ik * 1e4 / (4 * F_ * C_IN)
    else:
        e0, b_t, i_ref = TAFEL
        ik_f = np.full(kind.shape, i_ref * 10 ** ((e0 - v_cell) / b_t))      # A/cm², 옴 손실 0 에서 시작
        k_rxn = ik_f * 1e4 / (4 * F_ * C_IN)
        hist_v, a_w = [], None
    react = ik > 0 or v_cell is not None
    w_mol = w_inj * 1e-6 / 60 * RHO_L / M_W                        # 주입 물 mol/s

    s = np.zeros(kind.shape)
    Kapp = K0.copy()
    hist, d_rho = [], 0.0
    for it in range(n_it):
        krg = np.where(foam, (1 - s) ** n_kr, 1.0)
        for _ in range(8 if it == 0 else 2):                        # Forchheimer 는 바깥 반복과 함께 수렴시킨다
            # 유량은 질량 기준(P 에서의 부피)이다: 컨덕턴스 × 밀도비. Forchheimer 항 c_f ρ u² 도 질량 유속으로 쓰면 NU(P) 그대로다
            Tc = rho * (H_F * Kapp * krg / MU + Tg)
            Tf = faces(Tc)
            A = sp.coo_matrix((np.concatenate([Tf, Tf, -Tf, -Tf, gport]),
                               (np.concatenate([ia, ib, ia, ib, all_n]), np.concatenate([ia, ib, ib, ia, all_n]))),
                              shape=(n, n)).tocsc()
            p = spsolve(A, rhs)
            q_in = G_PORT * (1.0 - p[k_in])
            scale = q_gas / q_in.sum()
            p, q_in = p * scale, q_in * scale
            Ff = Tf * (p[ia] - p[ib])
            qx, qy = np.zeros(kind.shape), np.zeros(kind.shape)
            fx, fy = np.zeros(ax.shape), np.zeros(ay.shape)
            fx[ax], fy[ay] = Ff[:nxf], Ff[nxf:]
            qx[:, :-1] += fx / 2; qx[:, 1:] += fx / 2
            qy[:-1, :] += fy / 2; qy[1:, :] += fy / 2
            u = np.hypot(qx, qy) / (dx * H_F) * rho * (H_F * Kapp * krg / MU) / (Tc + 1e-300)   # P 기준 속도
            new = np.where(foam, 1 / (1 / K0 + c_f * u / (NU * np.sqrt(K0))), K0)
            Kapp = 0.5 * Kapp + 0.5 * new
            if compress:
                rho_new = np.ones(kind.shape)
                rho_new[fl] = 1 + p / P
                d_rho = np.abs(rho_new - rho).max()
                rho = 0.5 * rho + 0.5 * rho_new
                p_in = P + scale
                x_v, q_gas, c_o2 = inlet()
        u = u / rho                                                 # 실제 속도
        gradp = MU * u / (Kapp * krg + 1e-300)

        Dc = np.where(foam, D_GAS * eps ** 1.5 * (1 - s) ** 1.5, D_GAS)
        Df = faces(H_F * Dc)
        r_fixed = H_GDL / (D_GAS * F_GDL) + H_F / (3 * Dc)
        k_tot = (np.where(act, np.where(foam, (1 - s) ** cover, 1.0) * rho / (1 / k_rxn + R_OTHER + rho * r_fixed), 0.0)
                 if react else np.zeros(kind.shape))                # c 는 몰분율 / 입구 몰분율, 농도 = c × c_o2 × rho
        Fp, Fm = np.maximum(Ff, 0), np.maximum(-Ff, 0)
        diag0 = np.bincount(ia, Fp + Df, n) + np.bincount(ib, Fm + Df, n)
        diag0[k_out] += G_PORT * p[k_out]
        off = (np.concatenate([-(Fm + Df), -(Fp + Df)]), (np.concatenate([ia, ib]), np.concatenate([ib, ia])))
        src = np.zeros(n)
        src[k_in] = q_in
        M = (sp.coo_matrix(off, shape=(n, n)) + sp.diags(diag0 + k_tot[fl] * dx * dx)).tocsc()
        c = spsolve(M, src)                                         # 산소, 입구 농도로 나눈 값
        i_cell = 4 * F_ * c_o2 * k_tot[fl] * c                      # A/m²
        if not wet and v_cell is None:
            break
        Mw = (sp.coo_matrix(off, shape=(n, n)) + sp.diags(diag0)).tocsc()
        src_w = src * x_v * P / (R_ * T_)
        src_w[k_in] += w_mol * q_in / q_in.sum()
        cw = spsolve(Mw, src_w + i_cell / (2 * F_) * dx * dx)   # 물 전체, 기체 부피당 mol
        liq = np.zeros(kind.shape)
        c_sat = C_SAT / rho                                         # 포화 몰분율 P_SAT / (P + p) 를 P 기준 농도로
        liq[fl] = np.maximum(cw - c_sat[fl], 0.0)
        t = (MU_L / MU * liq * rho * M_W / RHO_L) ** (1 / n_kr)     # 액상 / 기체 실제 부피 유량 비
        s_e = t / (1 + t)
        wetness = np.clip(liq / (0.05 * c_sat), 0, 1)
        s_r = wetness * s_r0 / (1 + K0 * gradp / SIGMA / n_c)
        s_new = np.where(foam, np.minimum(s_r + (1 - s_r) * s_e, S_MAX), 0.0) if wet else s
        hist.append(np.abs(s_new - s)[foam].max())
        s = s + relax * (s_new - s)
        done = hist[-1] < tol and d_rho < 1e-5
        if v_cell is not None:
            a_new = np.clip(cw / c_sat[fl], 0.0, 1.0)
            a_w = a_new if a_w is None else a_w + 0.3 * (a_new - a_w)   # 활동도 ↔ 전류 되먹임 감쇠
            r_ohm = asr_c + r_mem(a_w, T_, t_mem)                     # Ω·cm², 기체가 있는 셀
            # 셀마다 국소 전류를 이분법으로: log ik = log i_ref + (E0 − V − R i(ik)) / b, i(ik) 는 현재 산소 농도에서의 전류
            cov = np.where(foam, (1 - s) ** cover, 1.0)[fl]
            rr = (R_OTHER + rho * r_fixed)[fl]
            cc = 4 * F_ * c_o2 * np.maximum(c, 0) * cov * rho[fl] / 1e4         # A/cm² per (1/k) 단위
            lo, hi = np.full(n, -8.0), np.full(n, np.log10(i_ref) + (e0 - v_cell) / b_t)
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                i_mid = cc / (4 * F_ * C_IN / (10 ** mid * 1e4) + rr)
                f = mid - np.log10(i_ref) - (e0 - v_cell - r_ohm * i_mid) / b_t
                hi, lo = np.where(f > 0, mid, hi), np.where(f > 0, lo, mid)
            lg_old = np.log10(ik_f[fl])
            ik_f[fl] = 10 ** (0.5 * (lo + hi))
            k_rxn = ik_f * 1e4 / (4 * F_ * C_IN)
            hist_v.append(np.abs(np.log10(ik_f[fl]) - lg_old)[act[fl]].max())
            done = done and hist_v[-1] < 1e-3 and np.abs(a_new - a_w)[act[fl]].max() < 1e-3
        if done:
            break

    i_loc = np.full(kind.shape, np.nan)
    i_loc[fl] = i_cell / 1e4
    i_loc[~act] = np.nan
    m = metrics(i_loc)
    extra = {}
    if v_cell is not None:
        ra = np.full(kind.shape, np.nan)
        ra[fl] = r_ohm
        ia_ = np.nan_to_num(i_loc)
        extra = dict(v_cell=v_cell, r_ohm=np.where(act, ra, np.nan), hfr=float(np.nanmean(ra[act])),
                     hfr_iw=float((ra[act] * ia_[act]).sum() / max(ia_[act].sum(), 1e-12)),
                     resid_v=hist_v[-1] if hist_v else 0.0, a_mean=float(a_w[act[fl]].mean()))
    return dict(**extra, dP_kPa=scale / 1e3, i=i_loc, s=np.where(foam, s, np.nan), i_mean=m["mean"], p05=m["p05"], i_min=m["min"],
                s_mean=float(s[foam].mean()), s_max=float(s[foam].max()), wet_frac=float((s[foam] > 0.02).mean()),
                resid=hist[-1] if hist else 0.0, iters=len(hist), speed=np.where(fl, u, np.nan), gradp=np.where(fl, gradp, np.nan), p_in_kPa=p_in / 1e3)


def opt_ports(name):
    with open(os.path.join(OUT, f"port_opt_{name}.json"), encoding="utf-8") as f:
        d = json.load(f)
    w = d["width_mm"] * 1e-3
    return [(v * 1e-3, w) for v in d["s_in_mm"]], [(v * 1e-3, w) for v in d["s_out_mm"]], d["header_mm"] * 1e-3


def designs():
    w = 6e-3
    u_in, u_out = LAYOUTS["U 옆변"]
    c_in, c_out = LAYOUTS["옆변 중앙 마주보기"]
    p2, p3, p2h = opt_ports("n2_h0"), opt_ports("n3_h0"), opt_ports("n2_h3")
    return [("기준 (대각 1 + 1)", BASE_IN, BASE_OUT, 0.0), ("U 옆변", [(u_in, w)], [(u_out, w)], 0.0),
            ("옆변 중앙 마주보기", [(c_in, w)], [(c_out, w)], 0.0), ("최적 2 + 2", *p2), ("최적 3 + 3", *p3),
            ("기준 + 헤더 3 mm", BASE_IN, BASE_OUT, 3e-3), ("최적 2 + 2 + 헤더 3 mm", *p2h)]


COLS = ["dP_kPa", "i_mean", "p05", "i_min", "s_mean", "s_max", "wet_frac", "resid", "iters"]


def row(tag, r):
    return dict(tag, **{k: r[k] for k in COLS})


def save(rows, name):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("→", os.path.normpath(path))


if __name__ == "__main__":
    mode = sys.argv[1]
    dx = 0.5e-3
    if mode == "check":
        from port_sweep import evaluate
        for hd in (0.0, 3e-3):
            m = evaluate(BASE_IN, BASE_OUT, dx, hd, n_forch=8)
            r = solve_wet(grid(BASE_IN, BASE_OUT, dx, hd), rh_in=0.0, wet=False)
            print(f"헤더 {hd * 1e3:g} mm | port_sweep: dP {m['dP_kPa']:.3f} mean {m['i076']:.4f} p05 {m['p05_076']:.4f} | "
                  f"물 끔: dP {r['dP_kPa']:.3f} mean {r['i_mean']:.4f} p05 {r['p05']:.4f}")
        for d in (1.0e-3, 0.5e-3, 0.25e-3):
            for it in (80, 300):
                r = solve_wet(grid(BASE_IN, BASE_OUT, d, 0.0), rh_in=1.0, n_it=it)
                print(f"RH 100 %, dx {d * 1e3:g} mm, 반복 {r['iters']:3d}: dP {r['dP_kPa']:.2f} mean {r['i_mean']:.3f} "
                      f"p05 {r['p05']:.3f} s {r['s_mean']:.3f} 잔차 {r['resid']:.1e}")
    elif mode == "designs":
        rows = []
        for name, pin, pout, hd in designs():
            g = grid(pin, pout, dx, hd)
            for rh in (0.0, 0.5, 1.0):
                d = solve_wet(g, rh_in=rh, wet=False)
                r = solve_wet(g, rh_in=rh, n_it=300, **DEFAULT)
                rows.append(row(dict(design=name, rh_in=rh, dP_dry_kPa=d["dP_kPa"], i_dry=d["i_mean"], p05_dry=d["p05"]), r))
                print(f"{name:22s} RH {rh:.1f} | 마름 dP {d['dP_kPa']:5.2f} i {d['i_mean']:.3f} p05 {d['p05']:.3f} | 젖음 dP {r['dP_kPa']:6.2f} "
                      f"i {r['i_mean']:.3f} p05 {r['p05']:.3f} s {r['s_mean']:.3f} (최대 {r['s_max']:.2f}) 잔차 {r['resid']:.0e}")
        save(rows, "water_designs.csv")
    elif mode == "plot":
        import matplotlib.pyplot as plt
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cfd"))
        from compare import SEQ, INK, INK2, MUTED, AXIS, SURF
        pick = [d for d in designs() if d[0] in ("기준 (대각 1 + 1)", "U 옆변", "최적 2 + 2", "기준 + 헤더 3 mm")]
        fig, axs = plt.subplots(4, len(pick), figsize=(3.3 * len(pick) + 1, 13.6), facecolor=SURF)
        for n, (name, pin, pout, hd) in enumerate(pick):
            g = grid(pin, pout, dx, hd)
            cut = (slice(None), g["act"].any(axis=0))
            for k, rh in enumerate((0.0, 1.0)):
                r = solve_wet(g, rh_in=rh, n_it=300, **DEFAULT)
                d = solve_wet(g, rh_in=rh, wet=False)
                im_s = axs[2 * k, n].imshow(r["s"][g["act"].any(axis=1)][cut], origin="lower", extent=[0, 50, 0, 50], cmap=SEQ, vmin=0, vmax=0.5)
                im_i = axs[2 * k + 1, n].imshow(r["i"][g["act"].any(axis=1)][cut], origin="lower", extent=[0, 50, 0, 50], cmap=SEQ, vmin=0, vmax=1.6)
                axs[2 * k, n].set_title(f"{name}\n입구 RH {rh * 100:.0f} % · 평균 포화도 {r['s_mean']:.2f}\n압력강하 {d['dP_kPa']:.1f} → {r['dP_kPa']:.1f} kPa",
                                        fontsize=9.5, color=INK)
                axs[2 * k + 1, n].set_title(f"평균 {d['i_mean']:.2f} → {r['i_mean']:.2f} A/cm²\n하위 5 % {d['p05']:.2f} → {r['p05']:.2f}",
                                            fontsize=9, color=INK2)
                for ims, a, lab in ((im_s, axs[2 * k], "액상 포화도"), (im_i, axs[2 * k + 1], "국소 전류밀도, 0.76 V (A/cm²)")):
                    if n == len(pick) - 1:
                        cb = fig.colorbar(ims, ax=a, fraction=0.012, pad=0.01)
                        cb.set_label(lab, color=INK2, fontsize=9); cb.ax.tick_params(colors=MUTED, labelsize=8); cb.outline.set_visible(False)
            for a in axs[:, n]:
                a.set_xticks([]); a.set_yticks([])
                for sp_ in a.spines.values():
                    sp_.set_color(AXIS)
        fig.subplots_adjust(hspace=0.42)
        fig.suptitle("물을 넣은 2D 모델 (가정값 s_r0 0.4, N_c 1e-4, n 3) — 화살표 왼쪽은 물을 끈 계산", fontsize=11, color=INK, y=0.995)
        path = os.path.join(OUT, f"water_maps_{sys.argv[2]}.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        print("→", os.path.normpath(path))
    elif mode == "sens":
        rows = []
        for name, pin, pout, hd in designs():
            g = grid(pin, pout, 1.0e-3, hd)
            for s_r0 in (0.0, 0.2, 0.4, 0.6):
                for n_c in (1e-5, 1e-4, 1e-3):
                    if s_r0 == 0.0 and n_c != 1e-4:
                        continue
                    r = solve_wet(g, rh_in=1.0, n_it=300, s_r0=s_r0, n_c=n_c)
                    rows.append(row(dict(design=name, s_r0=s_r0, n_c=n_c), r))
            print(name, "끝")
        save(rows, "water_sens.csv")
