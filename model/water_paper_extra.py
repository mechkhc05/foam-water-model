"""물 논문 보조 계산 → results/water_paper/extra.json (docs/paper_water_EN/paper_numbers.py 가 읽는다).

    python water_paper_extra.py

1) 격자 수렴: 격자 1.0 / 0.5 / 0.25 mm, Z 와 Z+H3, RH 100 %, 0.6 V.
2) ΔP 가중 포화도: 물을 끈 계산의 국소 압력 손실 밀도 |∇p|·|u| 로 가중한 폼 평균 포화도 (면적 평균과 비교).
"""
import json, os
import numpy as np
from water_model import solve_wet, DEFAULT
from water_paper import design, RES, V_REF, COMPRESS, water_mg
from port_sweep import grid

out = {}
for d in ("Z", "Z+H3"):
    for dx in (1.0e-3, 0.5e-3, 0.25e-3):
        g = grid(*design(d)[:2], dx, design(d)[2])
        r = solve_wet(g, rh_in=1.0, n_it=400, v_cell=V_REF, compress=COMPRESS, **DEFAULT)
        s = np.nan_to_num(r["s"])
        w = float(s.sum() * dx * dx * 0.5e-3 * 0.745 * 977.8 * 1e6)
        for k, v in (("dP", r["dP_kPa"]), ("i", r["i_mean"]), ("p05", r["p05"]), ("mg", w)):
            out[f"grid|{d}|{dx * 1e3:g}|{k}"] = float(v)
        print(d, dx, out[f"grid|{d}|{dx * 1e3:g}|dP"], out[f"grid|{d}|{dx * 1e3:g}|p05"], flush=True)
for d in ("Z", "Z+H3"):
    for h in (0.5, 1.0):
        g = grid(*design(d)[:2], 0.5e-3, design(d)[2])
        dry = solve_wet(g, rh_in=h, wet=False, n_it=400, v_cell=V_REF, compress=COMPRESS)
        wet = solve_wet(g, rh_in=h, n_it=400, v_cell=V_REF, compress=COMPRESS, **DEFAULT)
        foam = ~np.isnan(wet["s"])
        wgt = (dry["gradp"] * dry["speed"])[foam]
        out[f"sw|{d}|{h}"] = float((wet["s"][foam] * wgt).sum() / wgt.sum())
        out[f"sa|{d}|{h}"] = float(wet["s"][foam].mean())
        out[f"pin|{d}|{h}"] = float(wet["p_in_kPa"])
        print(d, h, out[f"sw|{d}|{h}"], out[f"sa|{d}|{h}"], flush=True)
with open(os.path.join(RES, "extra.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("→ extra.json")
