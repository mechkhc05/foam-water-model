"""25 cm² 포켓 셀: 입출구(포트) 위치·폭·개수 전수 탐색 (2D 깊이 평균 모델).

    python port_sweep.py check              # 기준 배치를 graded_porosity_opt.py(3D CFD와 맞춘 판)와 비교, 격자 수렴
    python port_sweep.py pos [헤더 mm]      # 입구 1 + 출구 1, 폭 6 mm: 둘레 위 위치 전수 탐색
    python port_sweep.py width              # 대표 배치에서 포트 폭
    python port_sweep.py count              # 윗변 입구 N개 × 아랫변 출구 M개
    python port_sweep.py header             # 대표 배치 × 헤더 폭
    python port_sweep.py opt N [헤더 mm]    # 입구 N + 출구 N 의 위치 최적화 (차분 진화)

물성·조건·유동/산소 식은 graded_porosity_opt.py 와 같다 (대기압, 70 °C, 공기 0.532 L/min, 포켓 0.5 mm,
균일 폼 기공률 0.745). 다른 점은 포트 처리뿐이다: 포트를 둘레 좌표 s 로 지정하고, 입구들은 한 매니폴드에
물린 것으로 보아 같은 압력을 건다(유량 배분은 계산 결과). 출구들도 같은 압력이다.

둘레 좌표 s (m): 왼쪽 변 아래 → 위 (0 ~ 0.05), 윗변 왼 → 오른 (0.05 ~ 0.10),
오른쪽 변 위 → 아래 (0.10 ~ 0.15), 아랫변 오른 → 왼 (0.15 ~ 0.20). 포트 = (중심 s, 폭).
기준 배치(실험 판): 입구 (0.046, 0.006), 출구 (0.146, 0.006).
"""
import csv, json, os, sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from graded_porosity_opt import (F_, MU, NU, D_GAS, C_IN, W, L, PAD, H_F, H_GDL, K_GDL, F_GDL, R_OTHER, C_F, Q,
                                 EPS_BASE, SOLID, OPEN, FOAM, OUT, k_of_eps, hmean, metrics)

PERIM = 2 * (W + L)
G_PORT = 1e-5                       # 포트 셀 ↔ 매니폴드 컨덕턴스 (m³/s/Pa), 빈 셀 사이 값의 약 20배
BASE_IN, BASE_OUT = [(46e-3, 6e-3)], [(146e-3, 6e-3)]
NPROC = 6


def grid(ports_in, ports_out, dx=1.0e-3, header=0.0):
    nx, ny = int(round((W + 2 * PAD) / dx)), int(round((L + 2 * PAD) / dx))
    X, Y = np.meshgrid(-PAD + (np.arange(nx) + 0.5) * dx, -PAD + (np.arange(ny) + 0.5) * dx)
    act = (X > 0) & (X < W) & (Y > 0) & (Y < L)
    kind = np.where(act, FOAM, SOLID)
    if header:
        kind[act & ((Y < header) | (Y > L - header))] = OPEN
    s = np.full(X.shape, np.nan)
    inx, iny = (X > 0) & (X < W), (Y > 0) & (Y < L)
    m = (X < 0) & iny; s[m] = Y[m]
    m = (Y > L) & inx; s[m] = L + X[m]
    m = (X > W) & iny; s[m] = L + W + (L - Y[m])
    m = (Y < 0) & inx; s[m] = 2 * L + W + (W - X[m])

    def mask(ports):
        out = np.zeros(X.shape, bool)
        for s0, w in ports:
            out |= np.abs((s - s0 + PERIM / 2) % PERIM - PERIM / 2) < w / 2 - 1e-9
        return out

    p_in, p_out = mask(ports_in), mask(ports_out)
    kind[p_in | p_out] = OPEN
    return dict(dx=dx, X=X, Y=Y, kind=kind, p_in=p_in, p_out=p_out, act=act)


def solve(g, ik_list=(2.0, 1000.0), n_forch=8, eps=EPS_BASE):
    """반환: 압력강하(Pa), ik 별 국소 전류밀도, 입구 포트 셀별 유량."""
    kind, dx, act = g["kind"], g["dx"], g["act"]
    fl = kind != SOLID
    idx = -np.ones(kind.shape, int)
    idx[fl] = np.arange(fl.sum())
    n = fl.sum()
    foam = kind == FOAM
    K0 = np.where(foam, k_of_eps(eps), H_F ** 2 / 12)
    Tg = np.where(act, H_GDL * K_GDL / MU, 0.0)
    ax = fl[:, :-1] & fl[:, 1:]
    ay = fl[:-1, :] & fl[1:, :]
    ia, ib = np.concatenate([idx[:, :-1][ax], idx[:-1, :][ay]]), np.concatenate([idx[:, 1:][ax], idx[1:, :][ay]])
    k_in, k_out = idx[g["p_in"]], idx[g["p_out"]]
    gport = np.zeros(n)
    gport[k_in] = G_PORT
    gport[k_out] = G_PORT
    rhs = np.zeros(n)
    rhs[k_in] = G_PORT                                              # 매니폴드 압력 1 로 풀고 유량에 맞춰 늘린다
    all_n = np.arange(n)

    def faces(field):
        return np.concatenate([hmean(field[:, :-1], field[:, 1:])[ax], hmean(field[:-1, :], field[1:, :])[ay]])

    Kapp = K0.copy()
    for _ in range(n_forch):
        Tc = H_F * Kapp / MU + Tg
        Tf = faces(Tc)
        A = sp.coo_matrix((np.concatenate([Tf, Tf, -Tf, -Tf, gport]),
                           (np.concatenate([ia, ib, ia, ib, all_n]), np.concatenate([ia, ib, ib, ia, all_n]))),
                          shape=(n, n)).tocsc()
        p = spsolve(A, rhs)
        q_in = G_PORT * (1.0 - p[k_in])
        scale = Q / q_in.sum()
        p, q_in = p * scale, q_in * scale
        Ff = Tf * (p[ia] - p[ib])                                   # m³/s, a → b
        qx, qy = np.zeros(kind.shape), np.zeros(kind.shape)
        nxf = ax.sum()
        fx, fy = np.zeros(ax.shape), np.zeros(ay.shape)
        fx[ax], fy[ay] = Ff[:nxf], Ff[nxf:]
        qx[:, :-1] += fx / 2; qx[:, 1:] += fx / 2
        qy[:-1, :] += fy / 2; qy[1:, :] += fy / 2
        u = np.hypot(qx, qy) / (dx * H_F) * (H_F * Kapp / MU) / (Tc + 1e-300)
        new = np.where(foam, 1 / (1 / K0 + C_F * u / (NU * np.sqrt(K0))), K0)
        Kapp = 0.5 * Kapp + 0.5 * new
    s = np.zeros(n)
    s[k_in] = q_in

    Dc = np.where(foam, D_GAS * eps ** 1.5, D_GAS)
    Df = faces(H_F * Dc)
    r_fixed = H_GDL / (D_GAS * F_GDL) + H_F / (3 * Dc)
    out = dict(dP=scale, i=[], speed=np.where(fl, u, np.nan), q_in=q_in, p=np.where(fl, 0.0, np.nan))
    out["p"][fl] = p
    Fp, Fm = np.maximum(Ff, 0), np.maximum(-Ff, 0)
    for ik in ik_list:
        k_rxn = ik * 1e4 / (4 * F_ * C_IN)
        k_tot = np.where(act, 1 / (1 / k_rxn + R_OTHER + r_fixed), 0.0)
        diag = np.bincount(ia, Fp + Df, n) + np.bincount(ib, Fm + Df, n) + k_tot[fl] * dx * dx
        diag[k_out] += G_PORT * p[k_out]
        M = sp.coo_matrix((np.concatenate([diag, -(Fm + Df), -(Fp + Df)]),
                           (np.concatenate([all_n, ia, ib]), np.concatenate([all_n, ib, ia]))), shape=(n, n)).tocsc()
        c = spsolve(M, s)
        i_loc = np.full(kind.shape, np.nan)
        i_loc[fl] = 4 * F_ * C_IN * k_tot[fl] * c / 1e4
        i_loc[~act] = np.nan
        out["i"].append(i_loc)
    return out


KEYS = ["dP_kPa", "i076", "min076", "p05_076", "cv076", "ilim", "minlim"]


def evaluate(ports_in, ports_out, dx=1.0e-3, header=0.0, n_forch=6):
    """포트가 겹치거나 없으면 None."""
    g = grid(ports_in, ports_out, dx, header)
    if (g["p_in"] & g["p_out"]).any() or not g["p_in"].any() or not g["p_out"].any():
        return None
    r = solve(g, n_forch=n_forch)
    a, b = metrics(r["i"][0]), metrics(r["i"][1])
    return dict(dP_kPa=r["dP"] / 1e3, i076=a["mean"], min076=a["min"], p05_076=a["p05"], cv076=a["cv"],
                ilim=b["mean"], minlim=b["min"])


def _job(job):
    tag, ports_in, ports_out, dx, header = job
    m = evaluate(ports_in, ports_out, dx, header)
    return dict(tag, **(m or {k: float("nan") for k in KEYS}))


def run(jobs, name):
    from multiprocessing import Pool
    with Pool(NPROC) as pool:
        rows = pool.map(_job, jobs, chunksize=max(1, len(jobs) // (NPROC * 8)))
    path = os.path.join(OUT, name)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} 건 → {os.path.normpath(path)}")
    return rows


def gap_ok(s_a, s_b, w_a, w_b, wall=1e-3):
    d = abs((s_a - s_b + PERIM / 2) % PERIM - PERIM / 2)
    return d >= (w_a + w_b) / 2 + wall - 1e-9


def top(x):
    return L + x


def bottom(x):
    return 2 * L + W + (W - x)


def right(y):
    return L + W + (L - y)


LAYOUTS = {                           # 이름: (입구 s, 출구 s) — 폭 6 mm 기준
    "Z 옆변 (기준)": (46e-3, right(4e-3)),
    "U 옆변": (46e-3, 4e-3),
    "옆변 중앙 마주보기": (25e-3, right(25e-3)),
    "Z 윗변–아랫변": (top(4e-3), bottom(46e-3)),
    "U 윗변–아랫변": (top(4e-3), bottom(4e-3)),
    "윗변 중앙–아랫변 중앙": (top(25e-3), bottom(25e-3)),
}


def cost(x, n_in, w, header, cap):
    s = np.asarray(x) % PERIM
    for a in range(len(s)):
        for b in range(a):
            if not gap_ok(s[a], s[b], w, w):
                return 10.0
    m = evaluate([(v, w) for v in s[:n_in]], [(v, w) for v in s[n_in:]], 1.0e-3, header, n_forch=5)
    if m is None:
        return 10.0
    return -m["p05_076"] + 2.0 * max(0.0, m["dP_kPa"] / cap - 1.0)


if __name__ == "__main__":
    mode = sys.argv[1]
    mm = 1e-3
    if mode == "check":
        import graded_porosity_opt as old
        for hd in (0.0, 3e-3):
            for dx in (1.0e-3, 0.5e-3, 0.25e-3):
                go = old.grid(dx, hd)
                r = old.solve(go, np.full(go["X"].shape, EPS_BASE), (2.0, 1000.0))
                a = metrics(r["i"][0])
                m = evaluate(BASE_IN, BASE_OUT, dx, hd, n_forch=8)
                print(f"헤더 {hd * 1e3:g} mm, dx {dx * 1e3:g} mm | 이전 판: dP {r['dP'] / 1e3:6.2f} mean {a['mean']:.3f} "
                      f"p05 {a['p05']:.3f} min {a['min']:.3f} | 이 판: dP {m['dP_kPa']:6.2f} mean {m['i076']:.3f} "
                      f"p05 {m['p05_076']:.3f} min {m['min076']:.3f} ilim {m['ilim']:.3f}")
    elif mode == "pos":
        hd = float(sys.argv[2]) * mm if len(sys.argv) > 2 else 0.0
        w = 6 * mm
        s_in_max = 75 if hd else 50                       # 대칭: 헤더가 없으면 둘레의 1/8, 있으면 1/4
        jobs = [(dict(s_in_mm=a, s_out_mm=b, header_mm=hd / mm), [(a * mm, w)], [(b * mm, w)], 1.0e-3, hd)
                for a in range(25, s_in_max + 1) for b in range(200) if gap_ok(a * mm, b * mm, w, w)]
        run(jobs, f"port_pos_map_h{hd / mm:g}.csv")
    elif mode == "width":
        ws = [2, 4, 6, 8, 12, 16, 24, 32, 40, 48]
        jobs = []
        for wi in ws:                                     # 모서리에서 1 mm 띄운 쪽을 고정하고 넓힌다
            for wo in ws:
                a, b = wi * mm, wo * mm
                jobs.append((dict(layout="Z 옆변", w_in_mm=wi, w_out_mm=wo), [(49 * mm - a / 2, a)], [(right(1 * mm + b / 2), b)], 0.5e-3, 0.0))
                if wi + wo <= 46:
                    jobs.append((dict(layout="U 옆변", w_in_mm=wi, w_out_mm=wo), [(49 * mm - a / 2, a)], [(1 * mm + b / 2, b)], 0.5e-3, 0.0))
                jobs.append((dict(layout="옆변 중앙 마주보기", w_in_mm=wi, w_out_mm=wo), [(25 * mm, a)], [(right(25 * mm), b)], 0.5e-3, 0.0))
        run(jobs, "port_width.csv")
    elif mode == "count":
        jobs = []
        for total, label in ((12, "총 폭 12 mm"), (None, "개당 6 mm")):
            for n_i in (1, 2, 3, 4, 6):
                for n_o in (1, 2, 3, 4, 6):
                    wi = (total / n_i if total else 6) * mm
                    wo = (total / n_o if total else 6) * mm
                    pin = [(top((k + 0.5) / n_i * W), wi) for k in range(n_i)]
                    pout = [(bottom((k + 0.5) / n_o * W), wo) for k in range(n_o)]
                    jobs.append((dict(rule=label, n_in=n_i, n_out=n_o, w_in_mm=wi / mm, w_out_mm=wo / mm), pin, pout, 0.5e-3, 0.0))
        run(jobs, "port_count.csv")
    elif mode == "header":
        jobs = [(dict(layout=name, header_mm=h), [(a, 6 * mm)], [(b, 6 * mm)], 0.5e-3, h * mm)
                for name, (a, b) in LAYOUTS.items() for h in (0, 1, 2, 3, 4, 5, 6, 8)]
        run(jobs, "port_header.csv")
    elif mode == "opt":
        from scipy.optimize import differential_evolution
        n_p = int(sys.argv[2])
        hd = float(sys.argv[3]) * mm if len(sys.argv) > 3 else 0.0
        w = 6 * mm
        cap = evaluate(BASE_IN, BASE_OUT, 1.0e-3, 0.0)["dP_kPa"]           # 압력강하 상한 = 기준 배치(헤더 없음)
        res = differential_evolution(cost, [(0, PERIM)] * (2 * n_p), args=(n_p, w, hd, cap), popsize=12, maxiter=80,
                                     tol=1e-5, seed=1, polish=False, updating="deferred", workers=NPROC)
        s = np.sort(res.x[:n_p] % PERIM).tolist(), np.sort(res.x[n_p:] % PERIM).tolist()
        m = evaluate([(v, w) for v in s[0]], [(v, w) for v in s[1]], 0.5e-3, hd, n_forch=8)
        print(f"N {n_p}, 헤더 {hd / mm:g} mm, 계산 {res.nfev} 회 | 입구 s {np.round(np.array(s[0]) / mm, 1)} 출구 s {np.round(np.array(s[1]) / mm, 1)}")
        print({k: round(v, 4) for k, v in m.items()})
        with open(os.path.join(OUT, f"port_opt_n{n_p}_h{hd / mm:g}.json"), "w", encoding="utf-8") as f:
            json.dump(dict(n_ports=n_p, header_mm=hd / mm, width_mm=w / mm, dP_cap_kPa=cap, s_in_mm=(np.array(s[0]) / mm).tolist(),
                           s_out_mm=(np.array(s[1]) / mm).tolist(), nfev=int(res.nfev),
                           note="s: 왼쪽 변 아래→위 0~50, 윗변 왼→오른 50~100, 오른쪽 변 위→아래 100~150, 아랫변 오른→왼 150~200 (mm)", **m), f, indent=1, ensure_ascii=False)
