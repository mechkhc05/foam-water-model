# foam-water-model

Code and data for

> H. Kim, *Effect of Port Layout and Open Headers on Liquid Water and Performance in a Metal-Foam Cathode Flow Field:
> A Sensitivity Analysis of Unmeasured Two-Phase Constants*, Energies (under review, manuscript energies-4681468).

A two-dimensional, depth-averaged model of the cathode of a 25 cm² PEM fuel cell whose flow field is a compressed nickel
foam in a 0.5 mm pocket. It couples compressible Darcy–Forchheimer gas flow, oxygen transport, total-water transport with
equilibrium condensation, an algebraic liquid-saturation closure (liquid moves with the gas pressure gradient, residual
saturation decreasing with the trapping number), a Springer membrane-resistance model and a Tafel–ohmic local current at
fixed cell voltage. A transient (fill-then-flow) version is written in C#, which also contains the capillary-diffusion
model-form test of the revision.

Code comments are partly in Korean; variable names follow the manuscript (see the Nomenclature).

## Layout

| Path | Content |
|---|---|
| `model/graded_porosity_opt.py` | constants (geometry, foam/GDL properties, 70 °C), permeability, metrics |
| `model/port_sweep.py` | grid and port geometry (`grid`), layouts Z/U/C |
| `model/water_model.py` | steady two-phase solver `solve_wet` (Sections 2.2–2.4, Appendix A) |
| `model/water_paper.py` | all steady results of the paper: `pol`, `rh`, `flow`, `sens`, `maps` |
| `model/water_paper_extra.py` | grid check (original manuscript) and pressure-loss-weighted saturation |
| `model/water_revision.py` | revision calculations: grid convergence (Table A1), alternative weak-region metrics (Table 5), sensitivity with five designs (Table 3), capillary/gravity ratios (Table 6) |
| `model/water_paper_figs.py` | Figures 1–9 |
| `transient/src/` | C# solver: `PocketModel.cs` (gas/oxygen/current), `WaterSim.cs` (transient water, capillary option `capCos`) |
| `transient/fill/` | dry-start filling runs (Section 3.6, Figure 8) |
| `transient/capillary/` | capillary model-form test (Section 4.3, Table 7, Figure 10) |
| `results/water_paper/` | result files used for the paper (CSV/JSON) |

## Reproducing

Python ≥ 3.10 with `numpy`, `scipy`, `matplotlib` (`pip install -r requirements.txt`).

```bash
cd model
python water_paper.py pol        # polarization curves (Figure 2)
python water_paper.py rh         # inlet humidity (Figures 5, 6; Table 4)
python water_paper.py flow       # air flow rate (Figure 7)
python water_paper.py sens       # 90 constant combinations (Figure 9)
python water_paper.py maps       # saturation/current maps (Figures 3, 4)
python water_paper_extra.py
python water_revision.py metrics # Table 5 (reference constants), Table 4
python water_revision.py sens    # Table 3 / Table 5 (90 combinations, five designs)
python water_revision.py grid    # Table A1 (1.0/0.5/0.25 mm); `fine` adds 0.125 mm
python water_revision.py cap     # Table 6
python water_revision.py summary
python water_paper_figs.py       # Figures 1–9
```

Results are written to `results/water_paper/`. A single steady case takes about 5 s on the 0.5 mm grid
(about 1 min on 0.25 mm). Set `PROCS` to change the number of worker processes and `NO_DRY=1` to skip the
liquid-switched-off solutions in `sens`.

Transient runs (.NET 8 SDK):

```bash
cd transient/fill      && dotnet run -c Release -- ../../results/water_paper/transient
cd transient/capillary && dotnet run -c Release -- ../../results/water_paper/revision/capillary main
```

## Status of the model

The single-phase part was compared with 3D simulations in the author's previous study. The two-phase closure has
**not** been validated against measurements; its four constants (`s_r0`, `N_c`, `n`, `m`) are unmeasured for metal
foams and are varied in the paper. See Sections 2.7, 4.3–4.5 of the manuscript.

## License

MIT (see `LICENSE`).
