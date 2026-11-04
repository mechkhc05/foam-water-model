"""Energies 1차 심사(energies-4681468, 2026-11) 대응 추가 계산 → results/water_paper/revision/

    python water_revision.py grid        # 격자 수렴: 설계 6종 × dx 1.0 / 0.5 / 0.25 mm × RH 50 / 100 %  (R1-2)
    python water_revision.py fine        # dx 0.125 mm: Z, Z+H1.5 (RH 100 %), 가장 민감한 i5% 확인
    python water_revision.py metrics     # 약한 영역 지표 대안(min, p01, p05, p10, CoV, 저전류 면적, 젖은 면적), 설계 6종 × RH 0/50/100 (R2 i5%)
    python water_revision.py sens        # 90 조합 × Z, U, Z+H1.5, Z+H3, Z+H6 × RH 50/100, 대안 지표까지 (R1-5, R2-1, R2 i5%)
    python water_revision.py cap         # 모세관 압력 구배 / 기체 압력 구배 비, 중력 비, 트래핑 수 분포 (R1-3, R2-2)
    python water_revision.py summary     # 위 결과로 표(revision/summary.md)

식은 water_model.solve_wet 그대로이고 기본 조건은 water_paper.py 와 같다(0.6 V, 국소 절대압력).
"""
import csv, itertools, json, os, sys
from multiprocessing import Pool
import numpy as np
from water_model import solve_wet, DEFAULT
from water_paper import design, RES, V_REF, COMPRESS, SENS_GRID, DESIGNS
from port_sweep import grid
from graded_porosity_opt import H_F, EPS_BASE, k_of_eps

REV = os.path.join(RES, "revision")
os.makedirs(REV, exist_ok=True)
RHO_L = 977.8
SIGMA = 0.0644
NO_DRY = os.environ.get("NO_DRY") == "1"


def alt_metrics(r, g):
    """약한 영역 지표들. 활성 면적(헤더 밑 포함) 전체 셀 기준."""
    i = r["i"][g["act"]]
    i = i[np.isfinite(i)]
    s = np.nan_to_num(r["s"])
    foam = ~np.isnan(r["s"])
    m = i.mean()
    return dict(i_mean=m, i_min=i.min(), p01=np.percentile(i, 1), p05=np.percentile(i, 5), p10=np.percentile(i, 10),
                p25=np.percentile(i, 25), cov=i.std() / m,
                low_frac=float((i < 0.5 * m).mean()),            # 평균의 절반 미만인 활성 면적 비율
                flood_frac=float((s[foam] > 0.3).sum() / g["act"].sum()),   # s > 0.3 인 폼 면적 / 활성 면적
                s_mean=r["s_mean"], s_max=r["s_max"], dP_kPa=r["dP_kPa"],
                water_mg=float(s.sum() * g["dx"] ** 2 * H_F * EPS_BASE * RHO_L * 1e6),
                foam_cm2=float(foam.sum() * g["dx"] ** 2 * 1e4), iters=r["iters"], resid=r["resid"])


def job(a):
    name, dx, kw = a
    kw = dict(kw)
    g = grid(*design(name)[:2], dx, design(name)[2])
    r = solve_wet(g, n_it=400, v_cell=V_REF, compress=COMPRESS, **kw)
    out = dict(design=name, dx_mm=dx * 1e3, **kw)
    out.update(alt_metrics(r, g))
    if not NO_DRY:                                               # 물을 끈 계산은 상수와 무관하다 → sens 에서는 생략
        d = solve_wet(g, wet=False, n_it=400, v_cell=V_REF, compress=COMPRESS, rh_in=kw["rh_in"])
        dm = alt_metrics(d, g)
        out.update({f"{k}_dry": dm[k] for k in ("i_mean", "p05", "p10", "i_min", "cov", "low_frac", "dP_kPa")})
    return out


def run(jobs, fname, procs=int(os.environ.get("PROCS", 6))):
    rows = []
    with Pool(procs) as pool:
        for k, r in enumerate(pool.imap(job, jobs)):
            rows.append(r)
            if k % 20 == 0:
                print(f"{k + 1}/{len(jobs)}", flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path = os.path.join(REV, fname)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("→", os.path.normpath(path))


def load(fname):
    with open(os.path.join(REV, fname), encoding="utf-8-sig") as f:
        return [{k: (v if k == "design" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def leverett_dj(s):
    """|dJ/ds| (Udell 1985 형태, J = 1.417 s − 2.120 s² + 1.263 s³)."""
    return np.abs(1.417 - 4.240 * s + 3.789 * s ** 2)


def cap():
    """모세관 확산을 뺀 근사의 타당 범위. 기준 상수의 계산된 포화도 장에서
    Π = |∇p_c| / |∇p_g|, p_c = σ cosθ sqrt(ε/K) J(s) (|cosθ| = 1, 상한), 중력 G = ρ_l g / |∇p_g|."""
    out = {}
    K = k_of_eps(EPS_BASE)
    pc_scale = SIGMA * np.sqrt(EPS_BASE / K)
    out["pc_scale_Pa"] = float(pc_scale)
    out["sqrt_epsK_m"] = float(np.sqrt(EPS_BASE * K))
    for name in ("Z", "U", "Z+H3", "Z+H6"):
        for rh in (0.5, 1.0):
            g = grid(*design(name)[:2], 0.5e-3, design(name)[2])
            r = solve_wet(g, rh_in=rh, n_it=400, v_cell=V_REF, compress=COMPRESS, **DEFAULT)
            foam = ~np.isnan(r["s"])
            s = np.where(foam, r["s"], np.nan)
            sy, sx = np.gradient(s, g["dx"])                     # 폼 경계 셀은 NaN 이 된다
            grad_s = np.hypot(sx, sy)
            gp = r["gradp"]
            ok = foam & np.isfinite(grad_s) & np.isfinite(gp) & (gp > 0)
            pi = pc_scale * leverett_dj(s[ok]) * grad_s[ok] / gp[ok]
            wet = s[ok] > 0.02
            G = RHO_L * 9.81 / gp[ok]
            N = K * gp[ok] / SIGMA
            key = f"{name}|{rh}"
            out[key] = dict(
                pi_median=float(np.median(pi[wet])), pi_p90=float(np.percentile(pi[wet], 90)),
                frac_pi_gt1=float((pi[wet] > 1).mean()), frac_pi_gt01=float((pi[wet] > 0.1).mean()),
                # 물 질량 가중: 물이 많은 곳에서 Π 가 큰가
                frac_water_pi_gt1=float(s[ok][wet][pi[wet] > 1].sum() / s[ok][wet].sum()),
                G_median=float(np.median(G)), frac_G_gt01=float((G > 0.1).mean()),
                N_p05=float(np.percentile(N, 5)), N_median=float(np.median(N)), N_p95=float(np.percentile(N, 95)),
                gradp_median_kPa_m=float(np.median(gp[ok]) / 1e3))
            np.savez_compressed(os.path.join(REV, f"cap_{name.replace('+', '')}_{int(rh * 100)}.npz"),
                                s=s, pi=_scatter(ok, pi, s.shape), gp=gp, act=g["act"])
            print(key, out[key], flush=True)
    with open(os.path.join(REV, "cap.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("→ cap.json")


def _scatter(mask, vals, shape):
    a = np.full(shape, np.nan)
    a[mask] = vals
    return a


def gci(f1, f2, f3, r=2.0):
    """Roache GCI (fine 격자 f1). 관측 차수 p, 외삽값, GCI_fine (%)."""
    e21, e32 = f2 - f1, f3 - f2
    if abs(e21) < 1e-12 or e32 / e21 <= 0:
        return dict(p=np.nan, ext=f1, gci=abs(e21) / abs(f1) * 100 * 1.25 if f1 else np.nan, monotone=False)
    p = np.log(abs(e32 / e21)) / np.log(r)
    p_use = min(max(p, 0.5), 2.0)
    ext = f1 - e21 / (r ** p_use - 1)
    return dict(p=p, ext=ext, gci=1.25 * abs(e21) / abs(f1) / (r ** p_use - 1) * 100, monotone=True)


def summary():
    L = []
    # 1) 격자
    rows = load("grid.csv")
    fine = load("fine.csv") if os.path.exists(os.path.join(REV, "fine.csv")) else []
    rows += fine
    L += ["## 격자 수렴 (0.6 V)", "", "| 설계 | RH | 지표 | 1.0 | 0.5 | 0.25 | 0.125 | 관측 차수 p | 외삽값 | GCI(0.25→0.5 기준 0.5 mm 오차 %) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for d in DESIGNS:
        for rh in (0.5, 1.0):
            sel = {r["dx_mm"]: r for r in rows if r["design"] == d and r["rh_in"] == rh}
            for k in ("dP_kPa", "i_mean", "p05", "p10", "water_mg", "s_mean"):
                v = {dx: sel[dx][k] for dx in (1.0, 0.5, 0.25, 0.125) if dx in sel}
                txt = " | ".join(f"{v[dx]:.4g}" if dx in v else "–" for dx in (1.0, 0.5, 0.25, 0.125))
                if all(dx in v for dx in (0.5, 0.25, 0.125)):
                    gg = gci(v[0.125], v[0.25], v[0.5])
                elif all(dx in v for dx in (1.0, 0.5, 0.25)):
                    gg = gci(v[0.25], v[0.5], v[1.0])
                else:
                    gg = None
                if gg is not None:
                    err05 = abs(v[0.5] - gg["ext"]) / abs(gg["ext"]) * 100 if gg["ext"] else np.nan
                    L.append(f"| {d} | {rh:g} | {k} | {txt} | {gg['p']:.2f} | {gg['ext']:.4g} | {err05:.1f} |")
                else:
                    L.append(f"| {d} | {rh:g} | {k} | {txt} | – | – | – |")
    # 2) 대안 지표
    m = load("metrics.csv")
    L += ["", "## 약한 영역 지표 (0.5 mm, 0.6 V, 기준 상수)", "",
          "| 설계 | RH | i_avg | min | p01 | p05 | p10 | p25 | CoV | 저전류 면적(<0.5 i_avg) | 젖은 면적(s>0.3) | s_mean | 물 mg | 폼 cm² | 물 mg/cm² 활성 | 물 mg/cm² 폼 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in m:
        L.append(f"| {r['design']} | {r['rh_in']:g} | {r['i_mean']:.3f} | {r['i_min']:.3f} | {r['p01']:.3f} | {r['p05']:.3f} | {r['p10']:.3f} | "
                 f"{r['p25']:.3f} | {r['cov']:.3f} | {r['low_frac']:.3f} | {r['flood_frac']:.3f} | {r['s_mean']:.3f} | {r['water_mg']:.1f} | "
                 f"{r['foam_cm2']:.1f} | {r['water_mg'] / 25:.2f} | {r['water_mg'] / r['foam_cm2']:.2f} |")
    # 3) 90 조합 순위: 지표별
    s = load("sens.csv")
    by = {(r["design"], r["rh_in"], r["s_r0"], r["n_c"], r["n_kr"], r["cover"]): r for r in s}
    combos = sorted({k[2:] for k in by})
    L += ["", f"## 90 조합 성립 비율 (조합 수 {len(combos)})", "", "| 진술 | RH 50 % | RH 100 % |", "|---|---|---|"]
    hi = ("i_mean", "p01", "p05", "p10", "p25", "i_min")
    lo = ("cov", "low_frac", "flood_frac", "dP_kPa", "s_mean")
    for a, b in (("Z+H3", "Z"), ("Z+H1.5", "Z"), ("Z+H6", "Z"), ("Z", "U"), ("Z+H3", "Z+H6"), ("Z+H1.5", "Z+H6")):
        for k in hi + lo:
            cells = []
            for rh in (0.5, 1.0):
                ok = sum((by[(a, rh) + c][k] > by[(b, rh) + c][k]) if k in hi else (by[(a, rh) + c][k] < by[(b, rh) + c][k]) for c in combos)
                cells.append(f"{ok}/{len(combos)}")
            L.append(f"| {a} {'>' if k in hi else '<'} {b} : {k} | " + " | ".join(cells) + " |")
    # 물 없이(단상)도 순위가 같은가 → 설계 차이의 단상 기여
    L += ["", "## 물을 끈 계산과 비교 (기준 상수, 0.5 mm)", "", "| 설계 | RH | p05 젖음 | p05 마름 | i_avg 젖음 | i_avg 마름 |", "|---|---|---|---|---|---|"]
    for r in m:
        L.append(f"| {r['design']} | {r['rh_in']:g} | {r['p05']:.3f} | {r['p05_dry']:.3f} | {r['i_mean']:.3f} | {r['i_mean_dry']:.3f} |")
    # 4) 모세관
    if os.path.exists(os.path.join(REV, "cap.json")):
        c = json.load(open(os.path.join(REV, "cap.json"), encoding="utf-8"))
        L += ["", f"## 모세관·중력 비 (p_c 척도 σ√(ε/K) = {c['pc_scale_Pa']:.0f} Pa, √(εK) = {c['sqrt_epsK_m'] * 1e6:.1f} μm)", "",
              "| 설계 | RH | Π 중앙값 | Π p90 | Π>1 면적 | Π>0.1 면적 | Π>1 물 비율 | G 중앙값 | N p05 | N 중앙값 | N p95 | ∇p 중앙값 kPa/m |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, v in c.items():
            if "|" not in k:
                continue
            d, rh = k.split("|")
            L.append(f"| {d} | {rh} | {v['pi_median']:.3f} | {v['pi_p90']:.2f} | {v['frac_pi_gt1']:.3f} | {v['frac_pi_gt01']:.3f} | "
                     f"{v['frac_water_pi_gt1']:.3f} | {v['G_median']:.3f} | {v['N_p05']:.1e} | {v['N_median']:.1e} | {v['N_p95']:.1e} | {v['gradp_median_kPa_m']:.0f} |")
    txt = "\n".join(L)
    with open(os.path.join(REV, "summary.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    mode = sys.argv[1]
    base = dict(DEFAULT)
    if mode == "grid":
        jobs = [(d, dx, dict(rh_in=rh, **base)) for d in DESIGNS for rh in (0.5, 1.0) for dx in (1.0e-3, 0.5e-3, 0.25e-3)
                if not (d == "Z+H1.5" and dx == 1.0e-3)]
        jobs.sort(key=lambda j: j[1])                            # 작은 격자(오래 걸림)를 먼저
        run(jobs, "grid.csv")
    elif mode == "fine":
        run([(d, 0.125e-3, dict(rh_in=1.0, **base)) for d in ("Z", "Z+H1.5")], "fine.csv", procs=2)
    elif mode == "metrics":
        run([(d, 0.5e-3, dict(rh_in=rh, **base)) for d in DESIGNS for rh in (0.0, 0.5, 1.0)], "metrics.csv")
    elif mode == "sens":
        run([(d, 0.5e-3, dict(rh_in=rh, **c)) for d in ("Z", "U", "Z+H1.5", "Z+H3", "Z+H6") for rh in (0.5, 1.0) for c in SENS_GRID], "sens.csv")
    elif mode == "cap":
        cap()
    elif mode == "summary":
        summary()
