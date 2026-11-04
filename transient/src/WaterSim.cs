// Liquid water in the 25 cm² foam pocket cell, run forward in time so that water can be watched accumulating and draining.
// Physics as model/water_model.py; its steady state is what this transient converges to.
//
//  1) Water balance: inlet vapour (RH) + a water source carried by the gas flow (advection–diffusion). The amount above
//     the saturation concentration is liquid (isothermal, equilibrium condensation).
//  2) Liquid moves with the gas along the same face flows, at a volume ratio λ(s) = (μg/μl) (s_e / (1 − s_e))^n of the gas
//     flow, s_e = (s − s_r) / (1 − s_r). Below the residual saturation s_r = wetness · s_r0 / (1 + N/N_c), N = K|∇p|/σ,
//     liquid is held (λ = 0): pores fill first, then water starts to move. Steady state: λ = liquid/gas ratio of item 1,
//     which is exactly the algebraic saturation of water_model.py.
//  3) Time stepping: backward Euler, one scalar equation per cell, cells visited in order of decreasing gas pressure
//     (liquid only moves downstream, so every inflow is known when a cell is solved). Unconditionally stable.
//  4) Feedback on the gas (permeability, diffusion, GDL face) through PocketModel.sat, refreshed when s has changed.
// Open cells (headers, ports) hold no liquid (as in water_model.py): what enters passes straight through.
// Plain C# (no UnityEngine) so it can be tested outside Unity.

using System;
using System.Collections.Generic;

namespace PorousFlowField
{
    public enum WaterSource { FuelCell = 0, ExSituFace = 1, ExSituInlet = 2 }

    [Serializable]
    public class WaterSettings
    {
        public WaterSource source = WaterSource.FuelCell;
        public float rhIn = 1.0f;            // inlet relative humidity (fuel cell mode)
        public float injMlMin = 0.09f;       // injected liquid water, mL/min (ex-situ modes)
        public float tCell = 70f, tLab = 25f; // °C
        public double sr0 = 0.4, nc = 1e-4, nKr = 3.0, cover = 1.0, sMax = 0.9;
        // vCell > 0 (fuel cell mode): fixed cell voltage. Local reaction rate from Tafel (0.85 V @ 0.1 A/cm², 70 mV/dec)
        // minus the ohmic loss (contact asrC + membrane from the local water activity, Springer, anode saturated).
        // vCell = 0: uniform reaction rate d.ik as before.
        public double vCell = 0.0, asrC = 0.05, tMem = 50.8e-6;
        public double capCos = 0.0;          // >0: capillary diffusion between foam cells, p_c = capCos σ sqrt(ε/K) J(s) (model-form test)
        public bool compress = false;        // local absolute pressure (PocketModel.compress); inlet RH at the inlet pressure
        public float TempC { get { return source == WaterSource.FuelCell ? tCell : tLab; } }
    }

    public class WaterProps
    {
        public double tk, mu, pSat, cSat, muL, sigma, rhoL;
        public const double Mw = 0.018015;

        // Same correlations as water_model.props(); at 70 °C the fixed constants of water_model.py (so results match).
        public static WaterProps At(double tC, double P)
        {
            var w = new WaterProps();
            w.tk = tC + 273.15;
            if (Math.Abs(tC - 70.0) < 1e-9) { w.mu = 2.05e-5; w.pSat = 31.2e3; w.muL = 4.04e-4; w.sigma = 0.0644; w.rhoL = 977.8; }
            else
            {
                w.mu = 1.716e-5 * Math.Pow(w.tk / 273.15, 1.5) * (273.15 + 110.4) / (w.tk + 110.4);
                w.pSat = 611.21 * Math.Exp((18.678 - tC / 234.5) * (tC / (257.14 + tC)));
                w.muL = 2.414e-5 * Math.Pow(10, 247.8 / (w.tk - 140));
                w.sigma = 0.0756 - 1.48e-4 * tC;
                w.rhoL = 1000 * (1 - (tC + 288.94) * (tC - 3.99) * (tC - 3.99) / (508929.2 * (tC + 68.13)));
            }
            w.cSat = w.pSat / (PocketModel.Rgas * w.tk);
            return w;
        }
    }

    public class WaterSample { public double t, dP, storedG, iAvg, outMgS, sMean; }

    public class WaterSim
    {
        public readonly PocketModel m;
        public readonly PocketDesign d;
        public WaterSettings ws;
        public WaterProps pr;

        public double[] s;                 // liquid saturation (foam cells)
        public double[] lam;               // liquid / gas volume-flow ratio leaving each cell
        public double[] gen;               // liquid production per cell [m^3/s] (negative = evaporation)
        public double[] sr;                // residual saturation
        public double[] cw;                // total water per gas volume [mol/m^3]
        public double[] liqX, liqY;        // liquid face flows [m^3/s]
        public double[] aW, rOhm;          // water activity (relaxed) and ohmic resistance [Ω·cm²] in fixed-voltage mode
        public int ikIters;
        public double ikResid;
        public double time;                // simulated seconds since the dry start
        public double storedM3, outM3s, genM3s;
        public int gasSolves, cwSweeps;
        public bool valid;
        public List<WaterSample> history = new List<WaterSample>();

        double[] sGas, store, inflow, cOut, wSrc;
        double[] cwDef, evapMol;           // capillary test: unused vapour capacity per gas volume [mol/m^3], evaporation of stored liquid [mol/s]
        int[] order;
        int n;

        public WaterSim(PocketDesign d, WaterSettings ws)
        {
            this.d = d; this.ws = ws;
            m = new PocketModel(d);
        }

        public double SatMean()
        {
            double sum = 0; int c = 0;
            for (int k = 0; k < n; k++) if (m.kind[k] == PocketModel.Foam) { sum += s[k]; c++; }
            return c > 0 ? sum / c : 0;
        }

        // Dry start: geometry, properties, gas flow without water.
        public void Reset()
        {
            pr = WaterProps.At(ws.TempC, d.P);
            d.T = pr.tk; d.mu = pr.mu;
            m.nKr = ws.nKr; m.coverExp = ws.cover; m.compress = ws.compress; m.rhoRel = null;
            m.xV = ws.source == WaterSource.FuelCell ? ws.rhIn * pr.pSat / d.P : 0.0;
            m.sat = null;
            m.ikLoc = null; aW = null;
            time = 0; storedM3 = 0; outM3s = 0; gasSolves = 0;
            history.Clear();
            s = null;
            GasUpdate();
        }

        const double E0 = 0.85, Btafel = 0.070, Iref = 0.1;
        bool VMode { get { return ws.vCell > 0 && ws.source == WaterSource.FuelCell; } }

        public void GasUpdate()
        {
            float ik = d.ik;
            if (ws.source != WaterSource.FuelCell) d.ik = 0f;            // no reaction in the ex-situ tests
            else if (VMode && m.ikLoc == null) d.ik = (float)(Iref * Math.Pow(10, (E0 - ws.vCell) / Btafel));
            if (ws.compress && ws.source == WaterSource.FuelCell && m.valid) m.xV = ws.rhIn * pr.pSat / m.pIn;
            m.Solve();
            d.ik = ik;
            valid = m.valid;
            if (!valid) return;
            Prepare();
            if (VMode)
            {
                if (m.ikLoc == null || m.ikLoc.Length != n)
                {
                    m.ikLoc = new double[n];
                    double ik0 = Iref * Math.Pow(10, (E0 - ws.vCell) / Btafel);
                    for (int k = 0; k < n; k++) m.ikLoc[k] = ik0;
                    aW = null;
                }
                for (ikIters = 0; ikIters < 200; ikIters++)
                {
                    SolveWater();
                    bool done = UpdateIk();
                    m.Solve();
                    if (!m.valid) { valid = false; return; }
                    if (done) break;
                }
            }
            gasSolves++;
            SolveWater();
            BuildOrder();
            for (int k = 0; k < n; k++) sGas[k] = s[k];
        }

        static double RMem(double a, double tk, double tMem)
        {
            double lamC = Math.Max(0.043 + 17.18 * a - 39.85 * a * a + 36 * a * a * a, 1.0), sum = 0;
            double arr = Math.Exp(1268 * (1 / 303.0 - 1 / tk));
            for (int q = 0; q <= 10; q++)
            {
                double z = q / 10.0, lam = lamC * (1 - z) + 14.0 * z;
                double w = q == 0 || q == 10 ? 0.5 : 1.0;
                sum += w / ((0.005139 * lam - 0.00326) * arr);
            }
            return tMem * 100 * sum / 10.0;
        }

        // One update of the local reaction rate (as water_model.solve_wet): relaxed water activity → membrane resistance,
        // then per cell bisection of log ik = log Iref + (E0 − V − R i(ik)) / b at the current O2 field. Returns converged.
        bool UpdateIk()
        {
            bool first = aW == null || aW.Length != n;
            if (first) { aW = new double[n]; rOhm = new double[n]; }
            double maxDl = 0, maxDa = 0, hF = d.hF * 1e-3;
            double c0 = 4 * PocketModel.Faraday * m.cIn * (1 - m.xV) / 1e4;
            for (int k = 0; k < n; k++)
            {
                if (m.kind[k] == PocketModel.Solid) continue;
                double aNew = Math.Min(Math.Max(cw[k] * m.RhoAt(k) / pr.cSat, 0), 1);
                aW[k] = first ? aNew : aW[k] + 0.3 * (aNew - aW[k]);
                rOhm[k] = ws.asrC + RMem(aW[k], pr.tk, ws.tMem);
                if (!m.act[k]) continue;
                maxDa = Math.Max(maxDa, Math.Abs(aNew - aW[k]));
                double sk = m.kind[k] == PocketModel.Foam && m.sat != null ? m.sat[k] : 0.0;
                double cc = c0 * Math.Max(m.c[k], 0) * Math.Pow(1 - sk, ws.cover) * m.RhoAt(k), rr = m.TransportR(k);
                double lo = -8.0, hi = Math.Log10(Iref) + (E0 - ws.vCell) / Btafel;
                for (int it = 0; it < 40; it++)
                {
                    double mid = 0.5 * (lo + hi);
                    double iMid = cc / (4 * PocketModel.Faraday * m.cIn / (Math.Pow(10, mid) * 1e4) + rr);
                    double f = mid - Math.Log10(Iref) - (E0 - ws.vCell - rOhm[k] * iMid) / Btafel;
                    if (f > 0) hi = mid; else lo = mid;
                }
                double lg = 0.5 * (lo + hi);
                maxDl = Math.Max(maxDl, Math.Abs(lg - Math.Log10(m.ikLoc[k])));
                m.ikLoc[k] = Math.Pow(10, lg);
            }
            ikResid = maxDl;
            return !first && maxDl < 1e-3 && maxDa < 1e-3;
        }

        void Prepare()
        {
            int nn = m.nx * m.ny;
            if (s == null || s.Length != nn)
            {
                n = nn;
                s = new double[n]; sGas = new double[n]; lam = new double[n]; gen = new double[n]; sr = new double[n];
                cw = new double[n]; store = new double[n]; inflow = new double[n]; cOut = new double[n]; wSrc = new double[n];
                liqX = new double[(m.nx + 1) * m.ny]; liqY = new double[m.nx * (m.ny + 1)];
                order = new int[n];
            }
        }

        // Steady advection–diffusion of total water (as water_model.py), then liquid production and residual saturation.
        void SolveWater()
        {
            int nx = m.nx, ny = m.ny;
            double cell = m.dx * m.dx, hF = d.hF * 1e-3, eps = d.eps;
            double cV = m.xV * d.P / (PocketModel.Rgas * pr.tk);
            double activeCells = 0, qInSum = 0;
            for (int k = 0; k < n; k++) { if (m.act[k]) activeCells++; qInSum += m.qIn[k]; }
            double inj = ws.injMlMin * 1e-6 / 60.0 * pr.rhoL / WaterProps.Mw;       // mol/s
            for (int k = 0; k < n; k++)
            {
                double src = m.qIn[k] * cV;
                if (ws.source == WaterSource.FuelCell) { if (m.act[k]) src += m.iLoc[k] * 1e4 * cell / (2 * PocketModel.Faraday); }
                else if (ws.source == WaterSource.ExSituFace) { if (m.act[k]) src += inj / activeCells; }
                else src += inj * m.qIn[k] / qInSum;
                if (ws.capCos > 0 && evapMol != null && evapMol.Length == n) src += evapMol[k];   // evaporated stored liquid joins the vapour
                wSrc[k] = src;
            }
            double scale = 0;
            for (int k = 0; k < n; k++) scale = Math.Max(scale, Math.Abs(cw[k]));
            double tol = 1e-9 * Math.Max(scale, pr.cSat);
            cwSweeps = 0;
            for (int sweep = 0; sweep < 20000; sweep++)
            {
                double maxDelta = 0;
                bool fwdI = (sweep & 1) == 0, fwdJ = (sweep & 2) == 0;
                for (int jj = 0; jj < ny; jj++)
                {
                    int j = fwdJ ? jj : ny - 1 - jj;
                    for (int ii = 0; ii < nx; ii++)
                    {
                        int i = fwdI ? ii : nx - 1 - ii;
                        int k = i + j * nx;
                        if (m.kind[k] == PocketModel.Solid) continue;
                        int f = i + j * (nx + 1);
                        double aP = m.qOut[k], b = wSrc[k];
                        if (i > 0) { double fw = m.qx[f], g = m.dfx[f]; aP += g + Math.Max(-fw, 0); b += (g + Math.Max(fw, 0)) * cw[k - 1]; }
                        if (i < nx - 1) { double fe = m.qx[f + 1], g = m.dfx[f + 1]; aP += g + Math.Max(fe, 0); b += (g + Math.Max(-fe, 0)) * cw[k + 1]; }
                        if (j > 0) { double fs = m.qy[k], g = m.dfy[k]; aP += g + Math.Max(-fs, 0); b += (g + Math.Max(fs, 0)) * cw[k - nx]; }
                        if (j < ny - 1) { double fn = m.qy[k + nx], g = m.dfy[k + nx]; aP += g + Math.Max(fn, 0); b += (g + Math.Max(-fn, 0)) * cw[k + nx]; }
                        double cn = aP > 0 ? b / aP : cw[k];
                        double dl = Math.Abs(cn - cw[k]);
                        if (dl > maxDelta) maxDelta = dl;
                        cw[k] = cn;
                    }
                }
                cwSweeps = sweep + 1;
                if (maxDelta < tol) break;
            }
            // liquid carried per gas volume, and the production that makes the advective liquid flux balance
            double vol = WaterProps.Mw / pr.rhoL;
            for (int k = 0; k < n; k++) cOut[k] = m.kind[k] == PocketModel.Solid ? 0 : Math.Max(cw[k] - pr.cSat / m.RhoAt(k), 0) * vol;
            if (cwDef == null || cwDef.Length != n) { cwDef = new double[n]; evapMol = new double[n]; }
            for (int k = 0; k < n; k++) cwDef[k] = m.kind[k] == PocketModel.Foam ? Math.Max(pr.cSat / m.RhoAt(k) - cw[k], 0) : 0;
            genM3s = 0;
            for (int k = 0; k < n; k++) gen[k] = m.qOut[k] * cOut[k];
            for (int j = 0; j < ny; j++)
                for (int i = 1; i < nx; i++)
                {
                    int a = i - 1 + j * nx, b2 = i + j * nx;
                    double F = m.qx[i + j * (nx + 1)];
                    if (F > 0) { gen[a] += F * cOut[a]; gen[b2] -= F * cOut[a]; } else { gen[b2] -= F * cOut[b2]; gen[a] += F * cOut[b2]; }
                }
            for (int j = 1; j < ny; j++)
                for (int i = 0; i < nx; i++)
                {
                    int a = i + (j - 1) * nx, b2 = i + j * nx;
                    double F = m.qy[b2];
                    if (F > 0) { gen[a] += F * cOut[a]; gen[b2] -= F * cOut[a]; } else { gen[b2] -= F * cOut[b2]; gen[a] += F * cOut[b2]; }
                }
            for (int k = 0; k < n; k++)
            {
                genM3s += gen[k];
                store[k] = m.kind[k] == PocketModel.Foam ? eps * hF * cell : 0.0;
                if (m.kind[k] != PocketModel.Foam) { sr[k] = 0; continue; }
                double wet = Math.Min(Math.Max(cOut[k] / vol / (0.05 * pr.cSat / m.RhoAt(k)), 0), 1);
                sr[k] = wet * ws.sr0 / (1 + m.k0[k] * m.gradP[k] / pr.sigma / ws.nc);
            }
        }

        void BuildOrder()
        {
            var keys = new double[n];
            for (int k = 0; k < n; k++) { order[k] = k; keys[k] = m.kind[k] == PocketModel.Solid ? double.MaxValue : -m.p[k]; }
            Array.Sort(keys, order);
        }

        // liquid volume flow per gas flow (face flows are volume at P, so ÷ rhoRel for the actual gas volume)
        double Lambda(double sk, double srk, double rh)
        {
            double se = (sk - srk) / (1 - srk);
            if (se <= 0) return 0;
            if (se > 0.999) se = 0.999;
            return pr.mu / pr.muL * Math.Pow(se / (1 - se), ws.nKr) / rh;
        }

        // Advance by dt seconds (backward Euler). Returns the largest saturation change since the last gas solve.
        public double Step(double dt)
        {
            if (!valid) return 0;
            int nx = m.nx;
            Array.Clear(inflow, 0, n);
            double outflow = 0;
            for (int q = 0; q < n; q++)
            {
                int k = order[q];
                if (m.kind[k] == PocketModel.Solid) break;
                int i = k % nx, j = k / nx, f = i + j * (nx + 1);
                double fo = m.qOut[k];
                if (i > 0 && m.qx[f] < 0) fo -= m.qx[f];
                if (i < nx - 1 && m.qx[f + 1] > 0) fo += m.qx[f + 1];
                if (j > 0 && m.qy[k] < 0) fo -= m.qy[k];
                if (j < m.ny - 1 && m.qy[k + nx] > 0) fo += m.qy[k + nx];
                double supply = gen[k] + inflow[k];
                double lk;
                if (store[k] <= 0)
                {
                    lk = fo > 0 ? Math.Max(supply, 0) / fo : 0;          // open cell: passes straight through
                }
                else
                {
                    double S = store[k], rhs = S * s[k] / dt + supply;
                    if (rhs <= 0) { s[k] = 0; lk = 0; }                       // evaporation exceeds the liquid present
                    else if (fo <= 0) { s[k] = Math.Min(rhs * dt / S, ws.sMax); lk = 0; }
                    else
                    {
                        double smax = ws.sMax;
                        if (S * smax / dt + fo * Lambda(smax, sr[k], m.RhoAt(k)) <= rhs)
                        {
                            s[k] = smax; lk = (rhs - S * smax / dt) / fo;  // full: the excess passes through
                        }
                        else
                        {
                            double lo = 0, hi = smax;
                            for (int it = 0; it < 48; it++)
                            {
                                double mid = 0.5 * (lo + hi);
                                if (S * mid / dt + fo * Lambda(mid, sr[k], m.RhoAt(k)) > rhs) hi = mid; else lo = mid;
                            }
                            s[k] = 0.5 * (lo + hi); lk = Lambda(s[k], sr[k], m.RhoAt(k));
                        }
                    }
                }
                if (ws.capCos > 0 && store[k] > 0)
                {
                    // stored liquid in a cell whose gas is below saturation evaporates up to the remaining vapour capacity of the
                    // gas leaving the cell; the vapour is added to the total-water source at the next gas/water solve.
                    double capM3 = cwDef[k] * Math.Max(fo, 0) * WaterProps.Mw / pr.rhoL;      // m^3/s of liquid the gas can take up
                    double ev = Math.Min(store[k] * s[k] / dt, capM3);
                    if (ev > 0) s[k] -= ev * dt / store[k];
                    evapMol[k] = ev * pr.rhoL / WaterProps.Mw;
                }
                lam[k] = lk;
                if (i > 0 && m.qx[f] < 0) inflow[k - 1] -= m.qx[f] * lk;
                if (i < nx - 1 && m.qx[f + 1] > 0) inflow[k + 1] += m.qx[f + 1] * lk;
                if (j > 0 && m.qy[k] < 0) inflow[k - nx] -= m.qy[k] * lk;
                if (j < m.ny - 1 && m.qy[k + nx] > 0) inflow[k + nx] += m.qy[k + nx] * lk;
                outflow += m.qOut[k] * lk;
            }
            if (ws.capCos > 0) CapillaryStep(dt);
            time += dt;
            outM3s = outflow;
            storedM3 = 0;
            double maxChange = 0;
            for (int k = 0; k < n; k++)
            {
                storedM3 += store[k] * s[k];
                if (store[k] > 0) maxChange = Math.Max(maxChange, Math.Abs(s[k] - sGas[k]));
            }
            UpdateFaceLiquid();
            return maxChange;
        }

        // Capillary redistribution between foam cells (operator splitting after the advective step), backward Euler with
        // face conductances frozen at the start of the step: S ds/dt = Σ T (s_nb − s). T = h K k_rl(s_e,donor) |dp_c/ds| / μ_l,
        // k_rl from the wetter (donor) cell, Udell J(s). Headers/ports exchange nothing (they hold no liquid).
        double[] capTx, capTy;
        public double capMoved; public int capSweeps;            // liquid volume moved by capillarity in the last step [m^3]
        void CapillaryStep(double dt)
        {
            int nx = m.nx, ny = m.ny;
            double hF = d.hF * 1e-3, eps = d.eps;
            if (capTx == null || capTx.Length != n) { capTx = new double[n]; capTy = new double[n]; }
            for (int k = 0; k < n; k++) { capTx[k] = 0; capTy[k] = 0; }
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++)
                {
                    int k = i + j * nx;
                    if (m.kind[k] != PocketModel.Foam) continue;
                    if (i < nx - 1 && m.kind[k + 1] == PocketModel.Foam) capTx[k] = CapT(k, k + 1, hF, eps);
                    if (j < ny - 1 && m.kind[k + nx] == PocketModel.Foam) capTy[k] = CapT(k, k + nx, hF, eps);
                }
            var s0 = (double[])s.Clone();
            capSweeps = 0;
            for (int sweep = 0; sweep < 3000; sweep++)
            {
                double maxD = 0; capSweeps++;
                for (int k = 0; k < n; k++)
                {
                    if (store[k] <= 0) continue;
                    int i = k % nx, j = k / nx;
                    double a = store[k] / dt, b = store[k] * s0[k] / dt;
                    if (i > 0) { double T = capTx[k - 1]; a += T; b += T * s[k - 1]; }
                    if (i < nx - 1) { double T = capTx[k]; a += T; b += T * s[k + 1]; }
                    if (j > 0) { double T = capTy[k - nx]; a += T; b += T * s[k - nx]; }
                    if (j < ny - 1) { double T = capTy[k]; a += T; b += T * s[k + nx]; }
                    double sn = Math.Min(Math.Max(s[k] + 1.7 * (b / a - s[k]), 0), ws.sMax);   // SOR
                    maxD = Math.Max(maxD, Math.Abs(sn - s[k]));
                    s[k] = sn;
                }
                if (maxD < 1e-9) break;
            }
            capMoved = 0;
            for (int k = 0; k < n; k++) if (store[k] > 0) capMoved += 0.5 * store[k] * Math.Abs(s[k] - s0[k]);
        }

        double CapT(int a, int b, double hF, double eps)
        {
            int dn = s[a] >= s[b] ? a : b;
            double se = (s[dn] - sr[dn]) / Math.Max(1 - sr[dn], 1e-6);
            if (se <= 0) return 0;
            if (se > 1) se = 1;
            double sf = 0.5 * (s[a] + s[b]);
            double dJ = Math.Abs(1.417 - 4.240 * sf + 3.789 * sf * sf);
            double k0 = m.k0[dn];
            return hF * k0 * Math.Pow(se, ws.nKr) / pr.muL * ws.capCos * pr.sigma * Math.Sqrt(eps / k0) * dJ;
        }

        void UpdateFaceLiquid()
        {
            int nx = m.nx, ny = m.ny;
            for (int j = 0; j < ny; j++)
                for (int i = 1; i < nx; i++)
                {
                    int f = i + j * (nx + 1);
                    double F = m.qx[f];
                    liqX[f] = F * (F > 0 ? lam[i - 1 + j * nx] : lam[i + j * nx]);
                }
            for (int j = 1; j < ny; j++)
                for (int i = 0; i < nx; i++)
                {
                    int k = i + j * nx;
                    double F = m.qy[k];
                    liqY[k] = F * (F > 0 ? lam[k - nx] : lam[k]);
                }
        }

        // Advance by dt with the gas re-solved whenever the saturation has moved by more than tolS since the last solve.
        public void Advance(double dt, double maxSub, double tolS, int maxGasSolves)
        {
            int solves = 0;
            while (dt > 1e-12)
            {
                double h = Math.Min(dt, maxSub);
                double ch = Step(h);
                dt -= h;
                if (ch > tolS && solves < maxGasSolves) { PushSat(); GasUpdate(); solves++; }
            }
        }

        public void PushSat()
        {
            if (m.sat == null || m.sat.Length != n) m.sat = new double[n];
            Array.Copy(s, m.sat, n);
        }

        public void Record()
        {
            history.Add(new WaterSample
            {
                t = time, dP = m.metrics.pressureDrop, storedG = storedM3 * pr.rhoL * 1e3, iAvg = m.metrics.iAvg,
                outMgS = outM3s * pr.rhoL * 1e6, sMean = SatMean()
            });
            if (history.Count > 4000) history.RemoveRange(0, 1000);
        }

        // Liquid interstitial velocity [m/s] at a point: liquid face flows over the liquid-filled pore area.
        // Open cells (no storage) carry liquid with the gas.
        public void LiquidVelocity(double x, double y, out double u, out double v)
        {
            int k = m.CellAt(x, y);
            if (m.kind[k] == PocketModel.Solid) { u = 0; v = 0; return; }
            if (store[k] <= 0) { m.Velocity(x, y, out u, out v); return; }
            double area = s[k] * d.eps * d.hF * 1e-3 * m.dx;
            if (s[k] < 1e-4 || lam[k] <= 0) { u = 0; v = 0; return; }
            int nx = m.nx, i = k % nx, j = k / nx, f = i + j * (nx + 1);
            double fx = (x + m.pad) / m.dx - i, fy = (y + m.pad) / m.dx - j;
            fx = Math.Min(Math.Max(fx, 0), 1); fy = Math.Min(Math.Max(fy, 0), 1);
            u = ((1 - fx) * liqX[f] + fx * liqX[f + 1]) / area;
            v = ((1 - fy) * liqY[k] + fy * liqY[k + nx]) / area;
        }
    }
}
