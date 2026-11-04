"""물 국내 논문 그림 (영문 표기, 1편 port_figures.py 와 같은 양식): results/water_paper/ → results/water_paper/figs/.

    python water_paper_figs.py            # 있는 자료로 만들 수 있는 그림 전부
    python water_paper_figs.py pol rh     # 골라서
"""
import csv, glob, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle
from water_paper import RES, DESIGNS, g_of, DX, V_REF
from graded_porosity_opt import SOLID, OPEN, FOAM, PAD, W, L

plt.rcParams.update({"font.family": "Arial", "font.size": 10.5, "axes.labelsize": 10.5, "legend.fontsize": 9, "axes.unicode_minus": False, "savefig.dpi": 300})
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
C_IN, C_OUT = "#1baf7a", "#eb6834"
COL = {"Z": "#2a78d6", "U": "#eb6834", "C": "#1baf7a", "Z+H1.5": "#8a5cd6", "Z+H3": "#c7902a", "Z+H6": "#52514e"}
NAME = {"Z": "Z (diagonal)", "U": "U (same side)", "C": "C (centre, facing)",
        "Z+H1.5": "Z + 1.5 mm headers", "Z+H3": "Z + 3 mm headers", "Z+H6": "Z + 6 mm headers"}
MK = {"Z": "o", "U": "s", "C": "^", "Z+H1.5": "D", "Z+H3": "v", "Z+H6": "P"}      # fixed marker per design (review R1-8)
RCOL = {0.0: "#9ecae1", 0.25: "#6baed6", 0.5: "#3182bd", 0.75: "#1d5f99", 1.0: "#08306b"}   # RH: one hue, increasing darkness
RMK = {0.0: "o", 0.5: "s", 1.0: "^"}
FOOT = 9
RHS = (0.0, 0.5, 1.0)
FIG = os.path.join(RES, "figs")
os.makedirs(FIG, exist_ok=True)
EXT = [-PAD * 1e3, (W + PAD) * 1e3, -PAD * 1e3, (L + PAD) * 1e3]


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8-sig") as f:
        return [{k: (v if k == "design" else float(v == "True") if v in ("True", "False") else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9.5)
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def label(ax, k):
    ax.text(-0.02, 1.04, f"({'abcdefghijkl'[k]})", transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK, ha="right")


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("→", os.path.normpath(path))


def cell_frame(ax, d):
    """셀 테두리, 헤더 경계, 포트 표시."""
    g = g_of(d)
    ax.add_patch(Rectangle((0, 0), W * 1e3, L * 1e3, fill=False, ec=INK2, lw=0.6))
    for mask, col in ((g["p_in"], C_IN), (g["p_out"], C_OUT)):
        ax.imshow(np.ma.masked_where(~mask, np.ones(mask.shape)), origin="lower", extent=EXT,
                  cmap=ListedColormap([col]), interpolation="nearest")
    hd = {"Z+H1.5": 1.5, "Z+H3": 3.0, "Z+H6": 6.0}.get(d, 0.0)
    for y in ((hd, L * 1e3 - hd) if hd else ()):
        ax.plot([0, W * 1e3], [y, y], color=INK2, lw=0.4, ls=(0, (3, 2)))
    ax.set_xlim(EXT[0] - 0.5, EXT[1] + 0.5); ax.set_ylim(EXT[2] - 0.5, EXT[3] + 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)


def ls_handles(color=None):
    from matplotlib.lines import Line2D
    c = color or INK2
    return [Line2D([], [], color=c, lw=1.4, label="solid: with liquid water"),
            Line2D([], [], color=c, lw=1.1, ls=(0, (3, 2)), label="dashed: liquid water switched off")]


def map_note(fig, y=0.0):
    fig.text(0.5, y, "Green bar: inlet slot; orange bar: outlet slot (6 mm wide); dashed line: foam / open-header boundary; "
             "each panel shows the 50 × 50 mm active area.", ha="center", fontsize=FOOT, color=INK2)


def cbar(fig, im, axs, text):
    cb = fig.colorbar(im, ax=axs, fraction=0.015, pad=0.012)
    cb.set_label(text, color=INK2, fontsize=10); cb.ax.tick_params(colors=INK2, labelsize=9); cb.outline.set_visible(False)


# ------------------------------------------------------------------ Fig. 1 형상
def fig_geometry():
    fig, axs = plt.subplots(1, 6, figsize=(11.6, 2.4))
    for ax, d in zip(axs, DESIGNS):
        g = g_of(d)
        k = np.where(g["kind"] == FOAM, 1.0, np.where(g["kind"] == OPEN, 0.0, np.nan))
        ax.imshow(np.ma.masked_invalid(k), origin="lower", extent=EXT, cmap=ListedColormap(["#ffffff", "#d9d7cf"]),
                  vmin=0, vmax=1, interpolation="nearest")
        cell_frame(ax, d)
        ax.set_title(NAME[d], fontsize=10, color=INK)
    fig.text(0.5, 0.06, "grey: Ni foam (50 × 50 × 0.5 mm, ε 0.745)   white: open header   green: inlet slot   orange: outlet slot (6 mm)",
             ha="center", fontsize=FOOT, color=INK2)
    save(fig, "fig1_geometry.png")


# ------------------------------------------------------------------ Fig. 2 분극 곡선 + HFR
def fig_pol():
    rows = load("pol.csv")
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    for c, rh in enumerate(RHS):
        a, b = axs[0, c], axs[1, c]
        for d in DESIGNS:
            rs = sorted([r for r in rows if r["design"] == d and r["rh_in"] == rh], key=lambda r: -r["v_cell"])
            v = [r["v_cell"] for r in rs]
            a.plot([r["i_mean"] for r in rs], v, "-", marker=MK[d], ms=3.5, lw=1.3, color=COL[d], label=NAME[d])
            if d in ("Z", "Z+H3"):
                a.plot([r["i_dry"] for r in rs], v, ls=(0, (3, 2)), lw=0.9, color=COL[d])
            b.plot([r["i_mean"] for r in rs], [r["hfr"] * 1e3 for r in rs], "-", marker=MK[d], ms=2.5, lw=1.1, color=COL[d])
        a.set_title(f"Inlet RH {rh * 100:.0f} %", fontsize=11, color=INK)
        a.set_ylim(0.3, 0.9); b.set_ylim(80, 260)
        b.set_xlabel("Mean current density (A/cm$^2$)", color=INK2)
        for ax in (a, b):
            style(ax)
        label(a, c); label(b, 3 + c)
    axs[0, 0].set_ylabel("Cell voltage (V)", color=INK2)
    axs[1, 0].set_ylabel("Mean HFR (mΩ cm$^2$)", color=INK2)
    axs[0, 0].plot([], [], ls=(0, (3, 2)), color=MUTED, label="dashed: liquid water off (Z, Z + 3 mm)")
    fig.tight_layout(rect=(0, 0.02, 1, 0.91))
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.0, "Air 0.532 L/min (dry basis), 70 °C, reference two-phase constants.", ha="center", fontsize=FOOT, color=INK2)
    save(fig, "fig2_polarization.png")


# ------------------------------------------------------------------ Fig. 3, 4 지도
def fig_maps():
    z = np.load(os.path.join(RES, "maps.npz"))
    for field, vmax, text, name in (("s", 0.4, "Liquid saturation s (–)", "fig3_saturation_maps.png"),
                                    ("i", 1.6, f"Local current density at {V_REF:.1f} V (A/cm$^2$)", "fig4_current_maps.png")):
        fig, axs = plt.subplots(3, 6, figsize=(11.6, 6.6))
        for c, d in enumerate(DESIGNS):
            for r_, rh in enumerate(RHS):
                ax = axs[r_, c]
                f = z[f"{d}|{rh}|{field}"]
                im = ax.imshow(np.ma.masked_invalid(f), origin="lower", extent=EXT, cmap=SEQ, vmin=0, vmax=vmax, interpolation="nearest")
                cell_frame(ax, d)
                v = f[np.isfinite(f)]
                stat = f"mean {v.mean():.2f}" + (f", max {v.max():.2f}" if field == "s" else f", 5th pct {np.percentile(v, 5):.2f}")
                ax.set_title((NAME[d] + "\n" if r_ == 0 else "") + stat, fontsize=9, color=INK)
                if c == 0:
                    ax.text(-0.12, 0.5, f"RH {rh * 100:.0f} %", transform=ax.transAxes, rotation=90, va="center", ha="center",
                            fontsize=10.5, color=INK)
        cbar(fig, im, axs, text)
        map_note(fig, 0.06)
        save(fig, name)


# ------------------------------------------------------------------ Fig. 5 RH
def fig_rh():
    rows = load("rh.csv")
    keys = [("i_mean", "Mean current density (A/cm$^2$)"), ("p05", "5th-percentile local current (A/cm$^2$)"),
            ("dP_kPa", "Pressure drop (kPa)"), ("water_mg", "Liquid water in foam (mg)")]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    for k, (key, yl) in enumerate(keys):
        ax = axs[k]
        for d in DESIGNS:
            rs = sorted([r for r in rows if r["design"] == d], key=lambda r: r["rh_in"])
            x = [r["rh_in"] * 100 for r in rs]
            ax.plot(x, [r[key] for r in rs], "-", marker=MK[d], ms=4, lw=1.3, color=COL[d], label=NAME[d])
            dry = {"i_mean": "i_dry", "p05": "p05_dry", "dP_kPa": "dP_dry"}.get(key)
            if dry:
                ax.plot(x, [r[dry] for r in rs], ls=(0, (3, 2)), lw=0.8, color=COL[d])
        ax.set_xlabel("Inlet relative humidity (%)", color=INK2); ax.set_ylabel(yl, color=INK2)
        ax.set_xticks([0, 25, 50, 75, 100])
        style(ax); label(ax, k)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    h, l = axs[0].get_legend_handles_labels()
    hs = ls_handles()
    fig.legend(h + hs, l + [x.get_label() for x in hs], loc="upper center", ncol=4, frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, -0.04, f"{V_REF:.1f} V, air 0.532 L/min (dry basis), 70 °C, reference two-phase constants.", ha="center", fontsize=FOOT, color=INK2)
    save(fig, "fig5_inlet_rh.png")


# ------------------------------------------------------------------ Fig. 6 헤더 폭
def fig_header():
    rows = load("rh.csv")
    hw = {"Z": 0.0, "Z+H1.5": 1.5, "Z+H3": 3.0, "Z+H6": 6.0}
    rcol = RCOL
    keys = [("dP_kPa", "Pressure drop (kPa)"), ("ratio", "Wet / dry pressure drop"), ("p05", "5th-percentile local current (A/cm$^2$)"),
            ("water_mg", "Liquid water in foam (mg)")]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    for k, (key, yl) in enumerate(keys):
        ax = axs[k]
        for rh in RHS:
            rs = sorted([r for r in rows if r["design"] in hw and r["rh_in"] == rh], key=lambda r: hw[r["design"]])
            x = [hw[r["design"]] for r in rs]
            y = [r["dP_kPa"] / r["dP_dry"] if key == "ratio" else r[key] for r in rs]
            ax.plot(x, y, "-", marker=RMK[rh], ms=4.5, lw=1.4, color=rcol[rh], label=f"Inlet RH {rh * 100:.0f} %")
            if key in ("dP_kPa", "p05"):
                ax.plot(x, [r["dP_dry" if key == "dP_kPa" else "p05_dry"] for r in rs], ls=(0, (3, 2)), lw=0.8, color=rcol[rh])
        ax.set_xlabel("Header width (mm)", color=INK2); ax.set_ylabel(yl, color=INK2)
        ax.set_xticks([0, 1.5, 3, 6])
        style(ax); label(ax, k)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    h, l = axs[0].get_legend_handles_labels()
    hs = ls_handles()
    fig.legend(h + hs, l + [x.get_label() for x in hs], loc="upper center", ncol=5, frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, -0.04, f"Z (diagonal) ports, {V_REF:.1f} V; header width 0 = layout Z (foam 50 mm long), 6 mm = foam 38 mm long.",
             ha="center", fontsize=FOOT, color=INK2)
    save(fig, "fig6_header_width.png")


# ------------------------------------------------------------------ Fig. 7 공기 유량
def fig_flow():
    rows = load("flow.csv")
    keys = [("i_mean", "Mean current density (A/cm$^2$)"), ("p05", "5th-percentile local current (A/cm$^2$)"),
            ("dP_kPa", "Pressure drop (kPa)"), ("water_mg", "Liquid water in foam (mg)")]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    for k, (key, yl) in enumerate(keys):
        ax = axs[k]
        for d in ("Z", "Z+H3"):
            for rh, ls in ((0.5, (0, (4, 2))), (1.0, "-")):
                rs = sorted([r for r in rows if r["design"] == d and r["rh_in"] == rh], key=lambda r: r["q_slpm"])
                ax.plot([r["q_slpm"] for r in rs], [r[key] for r in rs], marker=MK[d], ms=4, lw=1.3, ls=ls, color=COL[d],
                        label=f"{NAME[d]}, RH {rh * 100:.0f} %")
        ax.set_xlabel("Air flow rate (L/min, dry basis)", color=INK2); ax.set_ylabel(yl, color=INK2)
        style(ax); label(ax, k)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 1.01))
    fig.text(0.5, -0.04, f"{V_REF:.1f} V; solid: inlet RH 100 %, dashed: inlet RH 50 %. Air stoichiometry at 1 A/cm$^2$: "
             "0.532 L/min ≈ 1.3.", ha="center", fontsize=FOOT, color=INK2)
    save(fig, "fig7_air_flow.png")


# ------------------------------------------------------------------ Fig. 8 과도
def fig_transient():
    tdir = os.path.join(RES, "transient")
    cases = [("Z_rh100", "Z", "RH 100 %", "-"), ("ZH3_rh100", "Z+H3", "RH 100 %", "-"),
             ("Z_rh50", "Z", "RH 50 %", (0, (4, 2))), ("ZH3_rh50", "Z+H3", "RH 50 %", (0, (4, 2))), ("U_rh100", "U", "RH 100 %", "-")]
    cases = [c for c in cases if os.path.exists(os.path.join(tdir, f"ts_{c[0]}.csv"))]
    snaps = (10, 30, 60, 120, 600)
    fig = plt.figure(figsize=(11.6, 7.4))
    sf_top, sf_bot = fig.subfigures(2, 1, height_ratios=[1, 1.75], hspace=0.02)
    top = sf_top.subplots(1, 3)
    for tag, d, rh, ls in cases:
        with open(os.path.join(tdir, f"ts_{tag}.csv")) as f:
            r = list(csv.DictReader(f))
        t = np.array([float(x["t_s"]) for x in r])
        for ax, key in zip(top, ("stored_mg", "dP_kPa", "i_mean")):
            ax.plot(t, [float(x[key]) for x in r], ls=ls, lw=1.3, color=COL[d], label=f"{NAME[d]}, {rh}")
    for k, (ax, yl) in enumerate(zip(top, ("Liquid water in foam (mg)", "Pressure drop (kPa)", "Mean current density (A/cm$^2$)"))):
        ax.set_xlabel("Time from dry start (s)", color=INK2); ax.set_ylabel(yl, color=INK2); ax.set_xlim(0, 600)
        style(ax); label(ax, k)
    top[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    sf_top.subplots_adjust(left=0.07, right=0.98, wspace=0.32, bottom=0.2, top=0.9)
    axs = sf_bot.subplots(2, 5)
    for row, (tag, d) in enumerate((("Z_rh100", "Z"), ("ZH3_rh100", "Z+H3"))):
        for c, ts in enumerate(snaps):
            ax = axs[row, c]
            p = os.path.join(tdir, f"snap_{tag}_{ts}.csv")
            if os.path.exists(p):
                f = np.genfromtxt(p, delimiter=",")
                im = ax.imshow(np.ma.masked_invalid(f), origin="lower", extent=EXT, cmap=SEQ, vmin=0, vmax=0.4, interpolation="nearest")
            cell_frame(ax, d)
            ax.set_title(f"t = {ts} s" if row == 0 else "", fontsize=10, color=INK)
            if c == 0:
                ax.text(-0.1, 0.5, NAME[d] + ", RH 100 %", transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=9, color=INK)
    label(axs[0, 0], 3)
    sf_bot.subplots_adjust(left=0.05, right=0.9, wspace=0.06, hspace=0.12, top=0.92, bottom=0.02)
    cax = sf_bot.add_axes([0.92, 0.15, 0.012, 0.65])
    cb = sf_bot.colorbar(im, cax=cax)
    cb.set_label("Liquid saturation s (–)", color=INK2, fontsize=10); cb.ax.tick_params(colors=INK2, labelsize=9); cb.outline.set_visible(False)
    sf_bot.text(0.47, -0.02, "Green bar: inlet; orange bar: outlet; dashed line: header boundary.", ha="center", fontsize=FOOT, color=INK2)
    save(fig, "fig8_transient.png")


# ------------------------------------------------------------------ Fig. 9 민감도
def fig_sens():
    rows = load("sens.csv")
    base = dict(s_r0=0.4, n_c=1e-4, n_kr=3.0, cover=1.0)
    params = [("s_r0", (0.0, 0.2, 0.4, 0.6), "$s_{r0}$ 0 – 0.6"), ("n_c", (1e-5, 1e-4, 1e-3), "$N_c$ 10$^{-5}$ – 10$^{-3}$"),
              ("n_kr", (2.0, 3.0, 4.0), "$n$ 2 – 4"), ("cover", (0.5, 1.0, 2.0), "$m$ 0.5 – 2")]
    keys = [("dP_kPa", "Pressure drop (kPa)"), ("p05", "5th-percentile local current (A/cm$^2$)"), ("water_mg", "Liquid water in foam (mg)")]

    def pick(d, rh, **kw):
        q = dict(base, **kw)
        for r in rows:
            if r["design"] == d and r["rh_in"] == rh and all(abs(r[k] - v) <= 1e-12 + 1e-9 * abs(v) for k, v in q.items()):
                return r

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for k, (key, xl) in enumerate(keys):
        ax = axs[k]
        for j, (d, off) in enumerate((("Z", 0.18), ("Z+H3", -0.18))):
            ref = pick(d, 1.0)[key]
            for n, (p, vals, lab) in enumerate(params):
                v = [pick(d, 1.0, **{p: x})[key] for x in vals]
                y = len(params) - 1 - n + off
                ax.plot([min(v), max(v)], [y, y], lw=6, color=COL[d], alpha=0.85, solid_capstyle="butt",
                        label=NAME[d] if n == 0 else None)
                ax.plot([ref], [y], "|", ms=9, color=INK)
            allv = [r[key] for r in rows if r["design"] == d and r["rh_in"] == 1.0]
            ax.plot([min(allv), max(allv)], [-1.1 + off] * 2, lw=6, color=COL[d], alpha=0.45, solid_capstyle="butt")
        ax.set_yticks(list(range(len(params) - 1, -1, -1)) + [-1.1])
        ax.set_yticklabels([p[2] for p in params] + ["all 90 combinations"] if k == 0 else [])
        ax.set_xlabel(xl, color=INK2)
        style(ax); label(ax, k)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, fontsize=9.5, labelcolor=INK2,
               bbox_to_anchor=(0.5, 1.07))
    fig.text(0.5, -0.06, f"RH 100 %, {V_REF:.1f} V. Dark bars: range when one constant is varied (others at the reference value, "
             "black tick); pale bars: all 90 combinations.", ha="center", fontsize=FOOT, color=INK2)
    fig.tight_layout()
    save(fig, "fig9_sensitivity.png")


ALL = dict(geometry=fig_geometry, pol=fig_pol, maps=fig_maps, rh=fig_rh, header=fig_header, flow=fig_flow,
           transient=fig_transient, sens=fig_sens)
NEEDS = dict(pol="pol.csv", maps="maps.npz", rh="rh.csv", header="rh.csv", flow="flow.csv", sens="sens.csv", transient="transient")

if __name__ == "__main__":
    for k in (sys.argv[1:] or ALL):
        if k in NEEDS and not os.path.exists(os.path.join(RES, NEEDS[k])):
            print(k, "자료 없음, 건너뜀")
            continue
        ALL[k]()
