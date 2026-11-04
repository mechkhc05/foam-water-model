"""25 cm² 포켓 셀: 폼의 구역별 기공률 최적화 (2D 깊이 평균 모델).

    python graded_porosity_opt.py check                 # 균일 폼 / 헤더 3 mm 를 3D CFD 값과 비교
    python graded_porosity_opt.py opt stock 3           # 재고 폼 범위(기공률 0.49~0.745), 3 × 3 구역
    python graded_porosity_opt.py opt free 3            # 주문 폼 범위(기공률 0.49~0.90), 3 × 3 구역
    python graded_porosity_opt.py plot                  # results/ 에 그림과 표

형상·물성은 cfd/make_case.py 의 pocket25 케이스와 같다 (대기압, 70 °C, 공기 0.532 L/min,
포켓 0.5 mm, GDL 0.3 mm). 유동은 Darcy–Forchheimer, 산소는 대류·확산·소모.
기공률 → 투과도는 Kozeny–Carman 식, 기준점은 학생 자료의 1 mm 폼 (K 8.81e-10 m², 기공률 0.8725).
접촉 저항과 물은 모델에 없다.
"""
import json, os, sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

F_, R_ = 96485.33, 8.314462
T, P, X_O2, MU = 343.15, 101325.0, 0.21, 2.05e-5
RHO = P * 0.02897 / (R_ * T)
NU = MU / RHO
D_GAS = 1.8e-5 * 1.5e5 / P
C_IN = X_O2 * P / (R_ * T)
W = L = 50e-3
PAD = 2e-3
H_F, H_GDL, K_GDL, F_GDL = 0.5e-3, 0.3e-3, 1.0e-11, 0.30
R_OTHER, C_F = 30.0, 0.1
Q = 0.532 / 6e4 * (T / 273.15) * (101325 / P)
K_REF, EPS_REF = 8.81e-10, 0.8725
EPS_BASE = 0.745                    # 1 mm 폼을 0.5 mm 로 눌렀을 때
SOLID, OPEN, FOAM = 0, 1, 2
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")


def k_of_eps(eps):
    f = lambda e: e ** 3 / (1 - e) ** 2
    return K_REF * f(np.asarray(eps, float)) / f(EPS_REF)


def grid(dx=0.5e-3, header=0.0):
    nx, ny = int(round((W + 2 * PAD) / dx)), int(round(L / dx))
    X, Y = np.meshgrid(-PAD + (np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dx)
    kind = np.full(X.shape, FOAM)
    if header:
        kind[(Y < header) | (Y > L - header)] = OPEN
    p_in = (X < 0) & (Y > 43e-3) & (Y < 49e-3)
    p_out = (X > W) & (Y > 1e-3) & (Y < 7e-3)
    kind[(X < 0) | (X > W)] = SOLID
    kind[p_in | p_out] = OPEN
    return dict(dx=dx, X=X, Y=Y, kind=kind, p_in=p_in, p_out=p_out, act=(X > 0) & (X < W))


def zones_to_field(g, zones):
    z = np.atleast_2d(np.asarray(zones, float))
    ix = np.clip((g["X"] / W * z.shape[1]).astype(int), 0, z.shape[1] - 1)
    jy = np.clip((g["Y"] / L * z.shape[0]).astype(int), 0, z.shape[0] - 1)
    return z[jy, ix]


def hmean(a, b):
    return 2 * a * b / (a + b + 1e-300)


def solve(g, eps, ik_list=(2.0,), n_forch=8):
    """eps: 셀별 기공률 (폼 셀에서만 쓴다). 반환: 압력강하와 ik 별 국소 전류밀도."""
    kind, dx, act = g["kind"], g["dx"], g["act"]
    fl = kind != SOLID
    idx = -np.ones(kind.shape, int)
    idx[fl] = np.arange(fl.sum())
    n = fl.sum()
    foam = kind == FOAM
    K0 = np.where(foam, k_of_eps(np.where(foam, eps, EPS_BASE)), H_F ** 2 / 12)
    Tg = np.where(act, H_GDL * K_GDL / MU, 0.0)
    ax = fl[:, :-1] & fl[:, 1:]
    ay = fl[:-1, :] & fl[1:, :]
    ia, ib = np.concatenate([idx[:, :-1][ax], idx[:-1, :][ay]]), np.concatenate([idx[:, 1:][ax], idx[1:, :][ay]])
    s = np.zeros(n)
    s[idx[g["p_in"]]] = Q / g["p_in"].sum()
    gbig = np.zeros(n)
    gbig[idx[g["p_out"]]] = 1.0

    def faces(field):
        return np.concatenate([hmean(field[:, :-1], field[:, 1:])[ax], hmean(field[:-1, :], field[1:, :])[ay]])

    Kapp = K0.copy()
    for _ in range(n_forch):
        Tc = H_F * Kapp / MU + Tg
        Tf = faces(Tc)
        A = sp.coo_matrix((np.concatenate([Tf, Tf, -Tf, -Tf, gbig]),
                           (np.concatenate([ia, ib, ia, ib, np.arange(n)]), np.concatenate([ia, ib, ib, ia, np.arange(n)]))),
                          shape=(n, n)).tocsc()
        p = spsolve(A, s)
        Ff = Tf * (p[ia] - p[ib])                                   # m³/s, a → b
        # 셀 중심 속도 (폼 안 겉보기 속도)
        qx, qy = np.zeros(kind.shape), np.zeros(kind.shape)
        nxf = ax.sum()
        fx, fy = np.zeros(ax.shape), np.zeros(ay.shape)
        fx[ax], fy[ay] = Ff[:nxf], Ff[nxf:]
        qx[:, :-1] += fx / 2; qx[:, 1:] += fx / 2
        qy[:-1, :] += fy / 2; qy[1:, :] += fy / 2
        u = np.hypot(qx, qy) / (dx * H_F) * (H_F * Kapp / MU) / (Tc + 1e-300)
        new = np.where(foam, 1 / (1 / K0 + C_F * u / (NU * np.sqrt(K0))), K0)
        Kapp = 0.5 * Kapp + 0.5 * new
    dP = p[idx[g["p_in"]]].mean()

    Dc = np.where(foam, D_GAS * np.where(foam, eps, 1.0) ** 1.5, D_GAS)
    Df = faces(H_F * Dc)
    r_fixed = H_GDL / (D_GAS * F_GDL) + H_F / (3 * Dc)
    out = dict(dP=dP, i=[], speed=np.where(fl, u, np.nan))
    Fp, Fm = np.maximum(Ff, 0), np.maximum(-Ff, 0)
    for ik in ik_list:
        k_rxn = ik * 1e4 / (4 * F_ * C_IN)
        k_tot = np.where(act, 1 / (1 / k_rxn + R_OTHER + r_fixed), 0.0)
        diag = np.bincount(ia, Fp + Df, n) + np.bincount(ib, Fm + Df, n) + gbig * p + k_tot[fl] * dx * dx
        M = sp.coo_matrix((np.concatenate([diag, -(Fm + Df), -(Fp + Df)]),
                           (np.concatenate([np.arange(n), ia, ib]), np.concatenate([np.arange(n), ib, ia]))),
                          shape=(n, n)).tocsc()
        c = spsolve(M, s)
        i_loc = np.full(kind.shape, np.nan)
        i_loc[fl] = 4 * F_ * C_IN * k_tot[fl] * c / 1e4
        i_loc[~act] = np.nan
        out["i"].append(i_loc)
    return out


def metrics(i_loc):
    v = i_loc[np.isfinite(i_loc)]
    return dict(mean=v.mean(), min=v.min(), p05=np.percentile(v, 5), cv=v.std() / v.mean())


def report(name, g, eps):
    r = solve(g, eps, (2.0, 1000.0))
    a, b = metrics(r["i"][0]), metrics(r["i"][1])
    print(f"{name:28s} dP {r['dP'] / 1e3:6.2f} kPa | 0.76 V: mean {a['mean']:.3f} min {a['min']:.3f} p05 {a['p05']:.3f} "
          f"cv {a['cv']:.3f} | limit: mean {b['mean']:.3f} min {b['min']:.3f}")
    return dict(name=name, dP_kPa=r["dP"] / 1e3, i076=a["mean"], min076=a["min"], p05_076=a["p05"], cv076=a["cv"],
                ilim=b["mean"], minlim=b["min"])


BOUNDS = {"stock": (0.49, 0.745), "free": (0.49, 0.90)}
G_OPT = None


def cost(x, nz, cap, header=0.0):
    global G_OPT
    if G_OPT is None:
        G_OPT = grid(1.0e-3, header)
    r = solve(G_OPT, zones_to_field(G_OPT, np.reshape(x, (nz, nz))), (2.0,), n_forch=5)
    m = metrics(r["i"][0])
    return -m["p05"] + 2.0 * max(0.0, r["dP"] / cap - 1.0)


def optimize(scen, nz, cap_factor, header=0.0):
    from scipy.optimize import differential_evolution, minimize
    g = grid(1.0e-3, header)
    base = solve(g, np.full(g["X"].shape, EPS_BASE), (2.0,), n_forch=5)
    cap = cap_factor * base["dP"]
    lo, hi = BOUNDS[scen]
    res = differential_evolution(cost, [(lo, hi)] * (nz * nz), args=(nz, cap, header), popsize=10 if nz < 5 else 5, maxiter=60, tol=1e-4,
                                 seed=1, polish=False, updating="deferred", workers=5,
                                 x0=np.full(nz * nz, min(EPS_BASE, hi - 1e-6)))
    res2 = minimize(cost, res.x, args=(nz, cap, header), method="Nelder-Mead",
                    options=dict(maxfev=600, xatol=1e-3, fatol=1e-5), bounds=[(lo, hi)] * (nz * nz))
    x = res2.x if res2.fun < res.fun else res.x
    zones = np.reshape(x, (nz, nz))
    tag = f"{scen}_{nz}x{nz}_cap{cap_factor:g}" + ("_header" if header else "")
    gf = grid(0.5e-3, header)
    row = report("opt " + tag, gf, zones_to_field(gf, zones))
    with open(os.path.join(OUT, f"graded_opt_{tag}.json"), "w") as f:
        json.dump(dict(scenario=scen, nz=nz, cap_factor=cap_factor, header=header, eps_zones=zones.tolist(),
                       note="eps_zones[jy][ix], y 0 = 배출 포트 쪽(아래), x 0 = 공급 포트 쪽(왼쪽)", **row), f, indent=1)
    print(np.round(zones[::-1], 3))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "check":
        for dx in (1.0e-3, 0.5e-3, 0.25e-3):
            g = grid(dx)
            report(f"uniform dx {dx * 1e3:g} mm", g, np.full(g["X"].shape, EPS_BASE))
        g = grid(0.5e-3, header=3e-3)
        report("header 3 mm", g, np.full(g["X"].shape, EPS_BASE))
    elif mode == "opt":
        optimize(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 1.0,
                 float(sys.argv[5]) if len(sys.argv) > 5 else 0.0)
    elif mode == "plot":
        import csv
        import matplotlib.pyplot as plt
        sys.path.insert(0, os.path.join(HERE, "..", "cfd"))
        from compare import SEQ, INK, INK2, MUTED, AXIS, SURF
        tag = sys.argv[2]

        def zones(name):
            with open(os.path.join(OUT, f"graded_opt_{name}.json")) as f:
                return np.array(json.load(f)["eps_zones"])

        cases = [("균일 폼", 0.0, None), ("3 × 3 구역 (재고 폼 범위)", 0.0, zones("stock_3x3_cap1.5")),
                 ("3 × 3 구역 (주문 폼 범위)", 0.0, zones("free_3x3_cap1")), ("5 × 5 구역 (주문 폼 범위)", 0.0, zones("free_5x5_cap1")),
                 ("균일 폼 + 헤더 3 mm", 3e-3, None)]
        fig, axs = plt.subplots(2, len(cases), figsize=(15.5, 6.6), facecolor=SURF)
        rows = []
        for n, (name, hd, z) in enumerate(cases):
            g = grid(0.5e-3, hd)
            eps = np.full(g["X"].shape, EPS_BASE) if z is None else zones_to_field(g, z)
            rows.append(report(name, g, eps))
            r = solve(g, eps, (2.0,))
            show = np.where(g["kind"] == FOAM, eps, np.nan)[:, g["act"][0]]
            ext = [0, 50, 0, 50]
            im0 = axs[0, n].imshow(show, origin="lower", extent=ext, cmap="Greys_r", vmin=0.45, vmax=0.92)
            if z is not None:
                nz = z.shape[0]
                for j in range(nz):
                    for i in range(nz):
                        axs[0, n].text((i + .5) * 50 / nz, (j + .5) * 50 / nz, f"{z[j, i]:.2f}", ha="center", va="center",
                                       fontsize=8 if nz == 3 else 6.5, color="#111" if z[j, i] > 0.68 else "#fff")
            m = rows[-1]
            axs[0, n].set_title(f"{name}\n압력강하 {m['dP_kPa']:.1f} kPa", fontsize=10, color=INK)
            im1 = axs[1, n].imshow(r["i"][0][:, g["act"][0]], origin="lower", extent=ext, cmap=SEQ, vmin=0, vmax=1.6)
            axs[1, n].set_title(f"평균 {m['i076']:.2f} A/cm²\n하위 5 % {m['p05_076']:.2f}, 최소 {m['min076']:.2f}", fontsize=9, color=INK2)
            for ax in axs[:, n]:
                ax.set_xticks([]); ax.set_yticks([])
                for s_ in ax.spines.values():
                    s_.set_color(AXIS)
        for im, row, lab in ((im0, 0, "폼 기공률 (흰색 = 빈 헤더)"), (im1, 1, "국소 전류밀도, 0.76 V (A/cm²)")):
            cb = fig.colorbar(im, ax=axs[row], fraction=0.012, pad=0.01)
            cb.set_label(lab, color=INK2, fontsize=9); cb.ax.tick_params(colors=MUTED, labelsize=8); cb.outline.set_visible(False)
        fig.savefig(os.path.join(OUT, f"graded_porosity_{tag}.png"), dpi=150, bbox_inches="tight")
        with open(os.path.join(OUT, f"graded_porosity_{tag}.csv"), "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
