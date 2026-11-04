"""물 국내 논문(시뮬레이션만, 한국수소및신에너지학회논문집)용 정상 상태 계산.

    python water_paper.py pol         # 분극 곡선: 설계 6종 × RH 0 / 50 / 100 % × 0.85 ~ 0.35 V
    python water_paper.py rh          # 설계 6종 × 입구 RH 0 / 25 / 50 / 75 / 100 %, 물을 끈 계산과 함께
    python water_paper.py flow        # 공기 유량 0.3 ~ 1.6 L/min, 기준 대 헤더 3 mm, RH 50 / 100 %
    python water_paper.py sens        # 가정 상수 4개 전 조합(90) × 설계 3종 × RH 50 / 100 %
    python water_paper.py robust      # sens 결과로 결론별 성립 비율
    python water_paper.py maps        # 그림용 포화도·전류 지도(npz)

식은 water_model.solve_wet 그대로다. 1편(단상 포트 탐색)의 결과는 쓰지 않는다: 설계는 1편의 최적 배치가 아니라
단순 배치(대각, U, 중앙 마주보기)와 헤더 폭 0 / 1.5 / 3 / 6 mm(실험계획서 F1 / F2a / F2 / F2b와 같은 폭)다.
격자 0.5 mm (1 mm는 1.5 mm 헤더를 못 나타낸다).
모든 계산은 전압 고정(solve_wet v_cell): Tafel + 접촉 0.05 Ω·cm² + 막(Nafion 212, Springer, 애노드 포화 가정).
pol 밖의 계산은 V_REF = 0.6 V 에서 한다. 기체 밀도·산소·수증기는 국소 절대압력 기준(COMPRESS).
"""
import csv, itertools, os, sys
from multiprocessing import Pool
import numpy as np
from water_model import solve_wet, DEFAULT, OUT
from port_sweep import grid, BASE_IN, BASE_OUT, LAYOUTS

DX = 0.5e-3
W6 = 6e-3
RES = os.path.join(OUT, "water_paper")
os.makedirs(RES, exist_ok=True)


def design(name):
    if name.startswith("Z"):
        hd = {"Z": 0.0, "Z+H1.5": 1.5e-3, "Z+H3": 3e-3, "Z+H6": 6e-3}[name]
        return BASE_IN, BASE_OUT, hd
    a, b = LAYOUTS[{"U": "U 옆변", "C": "옆변 중앙 마주보기"}[name]]
    return [(a, W6)], [(b, W6)], 0.0


V_REF = 0.6
COMPRESS = True                     # 국소 절대압력 (2026-10-23, Energies 판). False 면 KHNES v2 결과 (results/water_paper_isobaric_KHNESv2)
VOLTS = (0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35)
DESIGNS = ["Z", "U", "C", "Z+H1.5", "Z+H3", "Z+H6"]
COLS = ["hfr", "hfr_iw", "a_mean", "resid_v", "dP_kPa", "i_mean", "p05", "i_min", "s_mean", "s_max", "wet_frac", "resid", "iters"]
GRIDS = {}


def g_of(name):
    if name not in GRIDS:
        GRIDS[name] = grid(*design(name)[:2], DX, design(name)[2])
    return GRIDS[name]


def water_mg(r, name):
    """폼 안 액상 물 (mg). 폼 두께 0.5 mm, 공극률 0.745."""
    from graded_porosity_opt import H_F, EPS_BASE
    s = np.nan_to_num(r["s"])
    return float(s.sum() * DX * DX * H_F * EPS_BASE * 977.8 * 1e6)


def job(a):
    name, kw = a
    kw = dict(kw)
    kw.setdefault("v_cell", V_REF)
    kw.setdefault("compress", COMPRESS)
    g = g_of(name)
    dry = solve_wet(g, wet=False, n_it=400, **{k: v for k, v in kw.items() if k in ("rh_in", "q_slpm", "v_cell", "compress")})
    r = solve_wet(g, n_it=400, **kw)
    out = dict(design=name, **kw, dP_dry=dry["dP_kPa"], i_dry=dry["i_mean"], p05_dry=dry["p05"], hfr_dry=dry["hfr"],
               water_mg=water_mg(r, name))
    out.update({k: r[k] for k in COLS})
    return out


def run(jobs, fname):
    with Pool(6) as pool:
        rows = []
        for k, r in enumerate(pool.imap(job, jobs)):
            rows.append(r)
            if k % 20 == 0:
                print(f"{k + 1}/{len(jobs)}", flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path = os.path.join(RES, fname)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("→", os.path.normpath(path))
    return rows


def load(fname):
    with open(os.path.join(RES, fname), encoding="utf-8-sig") as f:
        return [{k: (v if k == "design" else float(v == "True") if v in ("True", "False") else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]


SENS_GRID = [dict(s_r0=a, n_c=b, n_kr=c, cover=d)
             for a, b, c, d in itertools.product((0.0, 0.2, 0.4, 0.6), (1e-5, 1e-4, 1e-3), (2.0, 3.0, 4.0), (0.5, 1.0, 2.0))
             if not (a == 0.0 and b != 1e-4)]


def robust():
    rows = load("sens.csv")
    by = {}
    for r in rows:
        by[(r["design"], r["rh_in"], r["s_r0"], r["n_c"], r["n_kr"], r["cover"])] = r
    combos = sorted({k[1:] for k in by})
    claims = {
        "C1 헤더 3 mm가 기준보다 하위 5 % 전류가 높다": lambda Z, U, H: H["p05"] > Z["p05"],
        "C2 헤더 3 mm가 기준보다 젖은 ΔP가 낮다": lambda Z, U, H: H["dP_kPa"] < Z["dP_kPa"],
        "C3 순위 Z > U (하위 5 %)가 유지된다": lambda Z, U, H: Z["p05"] > U["p05"],
        "C4 물로 인한 하위 5 % 손실이 기준이 헤더보다 크다": lambda Z, U, H: (Z["p05_dry"] - Z["p05"]) > (H["p05_dry"] - H["p05"]),
        "C5 젖음/마름 ΔP 비가 헤더 쪽이 크다": lambda Z, U, H: H["dP_kPa"] / H["dP_dry"] > Z["dP_kPa"] / Z["dP_dry"],
        "C6 헤더가 기준보다 물을 더 머금는다": lambda Z, U, H: H["water_mg"] > Z["water_mg"],
        "C7 평균 전류 차이(헤더 − 기준) < 0.05 A/cm²": lambda Z, U, H: abs(H["i_mean"] - Z["i_mean"]) < 0.05,
        "C8 헤더 3 mm가 기준보다 평균 전류가 높다": lambda Z, U, H: H["i_mean"] > Z["i_mean"],
    }
    lines = ["| 결론 | RH 50 % | RH 100 % |", "|---|---|---|"]
    for name, f in claims.items():
        cells = []
        for rh in (0.5, 1.0):
            cs = [c for c in combos if c[0] == rh]
            ok = sum(f(by[("Z",) + c], by[("U",) + c], by[("Z+H3",) + c]) for c in cs)
            cells.append(f"{ok}/{len(cs)}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    # 상수 하나씩: 각 결과의 범위
    lines += ["", "| 지표 (RH 100 %) | 설계 | 최소 | 기본값 | 최대 |", "|---|---|---|---|---|"]
    for key in ("dP_kPa", "p05", "i_mean", "water_mg"):
        for d in ("Z", "Z+H3"):
            v = [by[(d,) + c][key] for c in combos if c[0] == 1.0]
            base = by[(d, 1.0, 0.4, 1e-4, 3.0, 1.0)][key]
            lines.append(f"| {key} | {d} | {min(v):.3g} | {base:.3g} | {max(v):.3g} |")
    # 상수별 기여: 한 상수만 움직일 때 범위 (나머지 기본값)
    lines += ["", "| 상수 하나만 바꿀 때 범위 (RH 100 %, 기준 Z) | ΔP kPa | 하위 5 % | 물 mg |", "|---|---|---|---|"]
    base = dict(s_r0=0.4, n_c=1e-4, n_kr=3.0, cover=1.0)
    for p, vals in (("s_r0", (0.0, 0.2, 0.4, 0.6)), ("n_c", (1e-5, 1e-4, 1e-3)), ("n_kr", (2.0, 3.0, 4.0)), ("cover", (0.5, 1.0, 2.0))):
        rs = [by[("Z", 1.0) + tuple(dict(base, **{p: v})[q] for q in ("s_r0", "n_c", "n_kr", "cover"))] for v in vals]
        fmt = lambda k, f: f"{min(r[k] for r in rs):{f}} ~ {max(r[k] for r in rs):{f}}"
        lines.append(f"| {p} {vals} | {fmt('dP_kPa', '.1f')} | {fmt('p05', '.3f')} | {fmt('water_mg', '.0f')} |")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(RES, "robust.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "pol":
        run([(d, dict(rh_in=rh, v_cell=v, **DEFAULT)) for d in DESIGNS for rh in (0.0, 0.5, 1.0) for v in VOLTS], "pol.csv")
    elif mode == "rh":
        run([(d, dict(rh_in=rh, **DEFAULT)) for d in DESIGNS for rh in (0.0, 0.25, 0.5, 0.75, 1.0)], "rh.csv")
    elif mode == "flow":
        run([(d, dict(rh_in=rh, q_slpm=q, **DEFAULT)) for d in ("Z", "Z+H3") for rh in (0.5, 1.0)
             for q in (0.3, 0.532, 0.8, 1.2, 1.6)], "flow.csv")
    elif mode == "sens":
        run([(d, dict(rh_in=rh, **c)) for d in ("Z", "U", "Z+H3") for rh in (0.5, 1.0) for c in SENS_GRID], "sens.csv")
    elif mode == "robust":
        robust()
    elif mode == "maps":
        data = {}
        for d in DESIGNS:
            g = g_of(d)
            for rh in (0.0, 0.5, 1.0):
                r = solve_wet(g, rh_in=rh, n_it=400, v_cell=V_REF, compress=COMPRESS, **DEFAULT)
                data[f"{d}|{rh}|s"], data[f"{d}|{rh}|i"] = r["s"], r["i"]
            data[f"{d}|act"] = g["act"]
        np.savez_compressed(os.path.join(RES, "maps.npz"), **data)
        print("→ maps.npz")
