// 25 cm² pocket cell: metal foam in a plain pocket, slot ports anywhere on the perimeter, optional open headers.
// Depth-averaged Darcy–Forchheimer flow, O2 advection–diffusion–consumption, local current density, tracer stepping.
// Same model as model/port_sweep.py (domestic paper). Plain C# (no UnityEngine) so it can be tested outside Unity.
//
// Perimeter coordinate s (mm): left wall bottom -> top 0..L, top wall left -> right, right wall top -> bottom,
// bottom wall right -> left. A port is (centre s, width w).

using System;
using System.Collections.Generic;

namespace PorousFlowField
{
    [Serializable]
    public class SlotPort
    {
        public float s, w;   // mm
        public SlotPort() { }
        public SlotPort(float s, float w) { this.s = s; this.w = w; }
    }

    [Serializable]
    public class PocketDesign
    {
        // geometry, mm
        public float W = 50f, L = 50f, pad = 2f;
        public float header = 0f;          // open strip along the top and bottom walls
        public List<SlotPort> portsIn = new List<SlotPort>(), portsOut = new List<SlotPort>();
        // foam (1 mm Ni foam pressed to 0.5 mm). Permeability from porosity by Kozeny–Carman
        public float hF = 0.5f;            // mm
        public float eps = 0.745f;
        public double KRef = 8.81e-10, epsRef = 0.8725, cF = 0.1;
        // GDL and catalyst layer
        public float hGdl = 0.3f;          // mm
        public double KGdl = 1.0e-11, fGdl = 0.30, rOther = 30.0;
        // operation
        public float qSlpm = 0.532f;       // air, standard L/min
        public float ik = 2.0f;            // reaction current density at inlet concentration, A/cm² (2 = 0.76 V)
        public double T = 343.15, P = 101325.0, mu = 2.05e-5;
        public float dxmm = 0.5f;

        public float Perimeter { get { return 2f * (W + L); } }
        public double K { get { return KOfEps(eps); } }
        public double KOfEps(double e)
        {
            return KRef * (e * e * e / ((1 - e) * (1 - e))) / (epsRef * epsRef * epsRef / ((1 - epsRef) * (1 - epsRef)));
        }
        // iR-free voltage for a given ik: Tafel 70 mV/dec, 0.1 A/cm² at 0.85 V (same as cfd/post.py)
        public double Voltage { get { return 0.85 - 0.070 * Math.Log10(ik / 0.1); } }
    }

    public class PocketMetrics
    {
        public double flowRate, tau;                   // m^3/s, s
        public double pressureDrop;                    // Pa
        public double iAvg, iP05, iMin, iCV;           // A/cm²
        public double utilization, stoich;
        public double meanSpeed;
        public double massBalanceError;
        public double[] inletSplit = new double[0];
    }

    public class PocketModel
    {
        public const int Solid = 0, Open = 1, Foam = 2;
        public const double Faraday = 96485.33, Rgas = 8.314462, O2InAir = 0.21, GPort = 1e-5;

        public PocketDesign d;
        public int nx, ny;
        public double dx, W, L, pad;       // m
        public int[] kind, inPort, outPort;            // port index or -1
        public bool[] act;
        public double[] p;                 // Pa
        public double[] qx, qy;            // face flows [m^3/s]
        public double[] qIn, qOut;         // per-cell supply / removal [m^3/s]
        public double[] speed;             // superficial speed in the flow layer [m/s]
        public double[] c;                 // O2 / inlet O2 (flow-layer mean)
        public double[] iLoc;              // A/cm² (NaN outside the active area)
        public double cIn, dGas;

        // Liquid water coupling (model/water_model.py). sat = liquid saturation per cell, null = dry (single-phase, as before).
        // Foam cells only: gas permeability × (1 − s)^nKr, foam diffusivity × (1 − s)^1.5, GDL face × (1 − s)^coverExp.
        // xV = water vapour mole fraction at the inlet: total flow = dry air / (1 − xV), O2 diluted by (1 − xV).
        public double[] sat;
        public double nKr = 3.0, coverExp = 1.0, xV = 0.0;
        public double[] gradP;             // |∇p| in the flow layer [Pa/m]
        // compress: local absolute pressure P + p (outlet = P). rhoRel = (P + p) / P per cell. Face flows are then mass flows
        // expressed as volume at P; O2 / water fields stay mole fractions; reaction and rOther act on concentration (× rhoRel),
        // molecular diffusion resistances do not change. pIn = absolute inlet manifold pressure. false = P everywhere (as before).
        public bool compress;
        public double[] rhoRel;
        public double pIn;
        public double RhoAt(int k) { return compress && rhoRel != null && rhoRel.Length == kind.Length ? rhoRel[k] : 1.0; }
        // Per-cell reaction current density at inlet concentration [A/cm²], null = uniform d.ik (as before).
        // Set by WaterSim in fixed-voltage mode (Tafel + ohmic + membrane, as water_model.solve_wet v_cell).
        public double[] ikLoc;

        public bool valid;
        public string message = "";
        public PocketMetrics metrics = new PocketMetrics();
        public int cgIterations, o2Sweeps;
        public bool o2Converged;

        public double[] k0, kApp, dfx, dfy;
        double[] tc, tg, tx, ty, dc, kTot, diag, rhs, cgR, cgZ, cgP, cgQ, pn, inCum;
        int[] inCells;

        public PocketModel(PocketDesign d) { this.d = d; }

        static double Harm(double a, double b) { double s = a + b; return s > 0 ? 2.0 * a * b / s : 0.0; }

        void Allocate()
        {
            W = d.W * 1e-3; L = d.L * 1e-3; pad = d.pad * 1e-3;
            int nxNew = (int)Math.Round((d.W + 2 * d.pad) / d.dxmm), nyNew = (int)Math.Round((d.L + 2 * d.pad) / d.dxmm);
            if (nxNew != nx || nyNew != ny || p == null)
            {
                nx = nxNew; ny = nyNew;
                int n = nx * ny;
                kind = new int[n]; inPort = new int[n]; outPort = new int[n]; act = new bool[n];
                p = new double[n]; pn = new double[n]; c = new double[n]; speed = new double[n]; iLoc = new double[n]; gradP = new double[n];
                qIn = new double[n]; qOut = new double[n];
                k0 = new double[n]; kApp = new double[n]; tc = new double[n]; tg = new double[n]; dc = new double[n]; kTot = new double[n];
                qx = new double[(nx + 1) * ny]; qy = new double[nx * (ny + 1)];
                tx = new double[(nx + 1) * ny]; ty = new double[nx * (ny + 1)];
                dfx = new double[(nx + 1) * ny]; dfy = new double[nx * (ny + 1)];
                diag = new double[n]; rhs = new double[n];
                cgR = new double[n]; cgZ = new double[n]; cgP = new double[n]; cgQ = new double[n];
            }
            dx = d.dxmm * 1e-3;
        }

        // Perimeter coordinate [m] of a point in the pad next to a wall, or NaN (active area, corners).
        public double PerimeterS(double x, double y)
        {
            bool inx = x > 0 && x < W, iny = y > 0 && y < L;
            if (x < 0 && iny) return y;
            if (y > L && inx) return L + x;
            if (x > W && iny) return L + W + (L - y);
            if (y < 0 && inx) return 2 * L + W + (W - x);
            return double.NaN;
        }

        static int PortAt(double s, List<SlotPort> ports, double per)
        {
            if (double.IsNaN(s)) return -1;
            for (int k = 0; k < ports.Count; k++)
            {
                double dd = Math.Abs(((s - ports[k].s * 1e-3 + 1.5 * per) % per) - 0.5 * per);
                if (dd < 0.5e-3 * ports[k].w - 1e-9) return k;
            }
            return -1;
        }

        bool BuildGeometry()
        {
            Allocate();
            double hd = d.header * 1e-3, hF = d.hF * 1e-3, hG = d.hGdl * 1e-3, per = 2 * (W + L);
            double kFoam = d.K, kOpen = hF * hF / 12.0;
            cIn = O2InAir * d.P / (Rgas * d.T);
            dGas = 1.8e-5 * 1.5e5 / d.P;
            double dFoam = dGas * Math.Pow(d.eps, 1.5);
            double kRxn = d.ik * 1e4 / (4 * Faraday * cIn);
            int nIn = 0, nOut = 0;
            for (int j = 0; j < ny; j++)
            {
                double y = -pad + (j + 0.5) * dx;
                for (int i = 0; i < nx; i++)
                {
                    double x = -pad + (i + 0.5) * dx;
                    int k = i + j * nx;
                    bool a = x > 0 && x < W && y > 0 && y < L;
                    int kd = a ? Foam : Solid;
                    if (a && hd > 0 && (y < hd || y > L - hd)) kd = Open;
                    double s = a ? double.NaN : PerimeterS(x, y);
                    inPort[k] = PortAt(s, d.portsIn, per);
                    outPort[k] = PortAt(s, d.portsOut, per);
                    if (inPort[k] >= 0 && outPort[k] >= 0) { message = "입구와 출구가 겹칩니다."; return false; }
                    if (inPort[k] >= 0 || outPort[k] >= 0) kd = Open;
                    if (inPort[k] >= 0) nIn++;
                    if (outPort[k] >= 0) nOut++;
                    kind[k] = kd; act[k] = a;
                    k0[k] = kd == Foam ? kFoam : kOpen;
                    tg[k] = a ? hG * d.KGdl / d.mu : 0.0;
                    double sk = SatAt(k, kd);
                    dc[k] = kd == Foam ? dFoam * Math.Pow(1 - sk, 1.5) : dGas;
                    double kR = ikLoc != null && ikLoc.Length == kind.Length ? ikLoc[k] * 1e4 / (4 * Faraday * cIn) : kRxn;
                    double rh = RhoAt(k);
                    kTot[k] = a ? Math.Pow(1 - sk, coverExp) * rh / (1.0 / kR + d.rOther + rh * (hG / (dGas * d.fGdl) + hF / (3 * dc[k]))) : 0.0;
                }
            }
            if (nIn == 0 || nOut == 0) { message = nIn == 0 ? "입구 포트가 없습니다." : "출구 포트가 없습니다."; return false; }
            inCells = new int[nIn]; inCum = new double[nIn];
            for (int k = 0, m = 0; k < kind.Length; k++) if (inPort[k] >= 0) inCells[m++] = k;
            return true;
        }

        // O2 transport resistance between the flow layer and the catalyst layer, without the reaction term [s/m].
        public double TransportR(int k) { return d.rOther + RhoAt(k) * (d.hGdl * 1e-3 / (dGas * d.fGdl) + d.hF * 1e-3 / (3 * dc[k])); }

        double SatAt(int k, int kd) { return sat != null && sat.Length == kind.Length && kd == Foam ? sat[k] : 0.0; }
        double Krg(int k) { double s = SatAt(k, kind[k]); return s > 0 ? Math.Pow(1 - s, nKr) : 1.0; }

        void BuildSystem()
        {
            int n = nx * ny;
            double hF = d.hF * 1e-3;
            for (int k = 0; k < n; k++)
            {
                tc[k] = kind[k] == Solid ? 0.0 : RhoAt(k) * (hF * kApp[k] * Krg(k) / d.mu + tg[k]);
                bool port = inPort[k] >= 0 || outPort[k] >= 0;
                diag[k] = port ? GPort : 0.0;
                rhs[k] = inPort[k] >= 0 ? GPort : 0.0;          // manifold pressure 1, scaled to the flow rate afterwards
            }
            for (int j = 0; j < ny; j++)
                for (int i = 1; i < nx; i++)
                {
                    int a = i - 1 + j * nx, b = i + j * nx, f = i + j * (nx + 1);
                    double tf = Harm(tc[a], tc[b]);
                    tx[f] = tf; diag[a] += tf; diag[b] += tf;
                    dfx[f] = kind[a] == Solid || kind[b] == Solid ? 0.0 : hF * Harm(dc[a], dc[b]);
                }
            for (int j = 1; j < ny; j++)
                for (int i = 0; i < nx; i++)
                {
                    int a = i + (j - 1) * nx, b = i + j * nx;
                    double tf = Harm(tc[a], tc[b]);
                    ty[b] = tf; diag[a] += tf; diag[b] += tf;
                    dfy[b] = kind[a] == Solid || kind[b] == Solid ? 0.0 : hF * Harm(dc[a], dc[b]);
                }
            for (int k = 0; k < n; k++) if (kind[k] == Solid) { diag[k] = 1.0; pn[k] = 0.0; }
        }

        void Apply(double[] x, double[] y)
        {
            for (int j = 0; j < ny; j++)
            {
                int row = j * nx, frow = j * (nx + 1);
                for (int i = 0; i < nx; i++)
                {
                    int k = row + i;
                    double v = diag[k] * x[k];
                    if (i > 0) v -= tx[frow + i] * x[k - 1];
                    if (i < nx - 1) v -= tx[frow + i + 1] * x[k + 1];
                    if (j > 0) v -= ty[k] * x[k - nx];
                    if (j < ny - 1) v -= ty[k + nx] * x[k + nx];
                    y[k] = v;
                }
            }
        }

        // Jacobi-preconditioned conjugate gradient on the normalized pressure pn, warm-started.
        void SolvePressure()
        {
            BuildSystem();
            int n = nx * ny;
            Apply(pn, cgQ);
            double bnorm = 0, rz = 0;
            for (int k = 0; k < n; k++)
            {
                cgR[k] = rhs[k] - cgQ[k];
                cgZ[k] = cgR[k] / diag[k];
                cgP[k] = cgZ[k];
                rz += cgR[k] * cgZ[k];
                bnorm += rhs[k] * rhs[k];
            }
            double tol2 = 1e-26 * bnorm;
            int maxIter = 60 * (nx + ny) + 500;
            for (int it = 0; it < maxIter; it++)
            {
                double rr = 0;
                for (int k = 0; k < n; k++) rr += cgR[k] * cgR[k];
                if (rr <= tol2) break;
                Apply(cgP, cgQ);
                double pq = 0;
                for (int k = 0; k < n; k++) pq += cgP[k] * cgQ[k];
                if (pq <= 0) break;
                double alpha = rz / pq, rzNew = 0;
                for (int k = 0; k < n; k++)
                {
                    pn[k] += alpha * cgP[k];
                    cgR[k] -= alpha * cgQ[k];
                    cgZ[k] = cgR[k] / diag[k];
                    rzNew += cgR[k] * cgZ[k];
                }
                double beta = rzNew / rz;
                rz = rzNew;
                for (int k = 0; k < n; k++) cgP[k] = cgZ[k] + beta * cgP[k];
                cgIterations++;
            }
        }

        // Scale the unit-manifold solution to the flow rate, face flows, superficial speed. Returns the manifold pressure.
        double ComputeFluxes(double Q)
        {
            int n = nx * ny;
            double hF = d.hF * 1e-3, sumOut = 0, sumIn = 0;
            for (int k = 0; k < n; k++)
            {
                if (outPort[k] >= 0) sumOut += GPort * pn[k];
                if (inPort[k] >= 0) sumIn += GPort * (1.0 - pn[k]);
            }
            metrics.massBalanceError = Math.Abs(sumIn - sumOut) / sumOut;
            double scale = Q / sumOut;
            for (int k = 0; k < n; k++)
            {
                p[k] = pn[k] * scale;
                qOut[k] = outPort[k] >= 0 ? GPort * p[k] : 0.0;
                qIn[k] = inPort[k] >= 0 ? GPort * (1.0 - pn[k]) * Q / sumIn : 0.0;
            }
            for (int j = 0; j < ny; j++)
            {
                int frow = j * (nx + 1), row = j * nx;
                qx[frow] = 0; qx[frow + nx] = 0;
                for (int i = 1; i < nx; i++) qx[frow + i] = tx[frow + i] * (p[row + i - 1] - p[row + i]);
            }
            for (int i = 0; i < nx; i++)
            {
                qy[i] = 0; qy[i + ny * nx] = 0;
                for (int j = 1; j < ny; j++) qy[i + j * nx] = ty[i + j * nx] * (p[i + (j - 1) * nx] - p[i + j * nx]);
            }
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++)
                {
                    int k = i + j * nx;
                    if (kind[k] == Solid) { speed[k] = 0; gradP[k] = 0; continue; }
                    double u = 0.5 * (qx[i + j * (nx + 1)] + qx[i + 1 + j * (nx + 1)]), v = 0.5 * (qy[k] + qy[k + nx]);
                    double kr = Krg(k);
                    speed[k] = Math.Sqrt(u * u + v * v) / (dx * hF) * (hF * kApp[k] * kr / d.mu) / tc[k];   // flow-layer share, actual (÷ rhoRel)
                    gradP[k] = d.mu * speed[k] / (kApp[k] * kr);
                }
            return scale;
        }

        // First-order upwind advection + diffusion + first-order consumption, Gauss–Seidel sweeps.
        void SolveOxygen()
        {
            int n = nx * ny;
            double cell = dx * dx;
            for (int k = 0; k < n; k++) if (!(c[k] >= 0.0 && c[k] <= 1.0) || kind[k] == Solid) c[k] = 0.0;
            o2Sweeps = 0; o2Converged = false;
            for (int sweep = 0; sweep < 8000; sweep++)
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
                        if (kind[k] == Solid) continue;
                        int f = i + j * (nx + 1);
                        double aP = kTot[k] * cell + qOut[k], b = qIn[k];
                        if (i > 0) { double fw = qx[f], g = dfx[f]; aP += g + Math.Max(-fw, 0); b += (g + Math.Max(fw, 0)) * c[k - 1]; }
                        if (i < nx - 1) { double fe = qx[f + 1], g = dfx[f + 1]; aP += g + Math.Max(fe, 0); b += (g + Math.Max(-fe, 0)) * c[k + 1]; }
                        if (j > 0) { double fs = qy[k], g = dfy[k]; aP += g + Math.Max(-fs, 0); b += (g + Math.Max(fs, 0)) * c[k - nx]; }
                        if (j < ny - 1) { double fn = qy[k + nx], g = dfy[k + nx]; aP += g + Math.Max(fn, 0); b += (g + Math.Max(-fn, 0)) * c[k + nx]; }
                        double cn = b / aP;
                        double dl = Math.Abs(cn - c[k]);
                        if (dl > maxDelta) maxDelta = dl;
                        c[k] = cn;
                    }
                }
                o2Sweeps = sweep + 1;
                if (maxDelta < 1e-8) { o2Converged = true; break; }
            }
        }

        void ComputeMetrics(double Q)
        {
            PocketMetrics m = metrics;
            int n = nx * ny, na = 0, nf = 0;
            double hF = d.hF * 1e-3, sum = 0, sum2 = 0, min = double.MaxValue, pore = 0, sp = 0;
            var vals = new List<double>();
            for (int k = 0; k < n; k++)
            {
                if (kind[k] != Solid) pore += (kind[k] == Foam ? d.eps : 1.0) * hF * dx * dx;
                if (kind[k] == Foam) { sp += speed[k]; nf++; }
                if (!act[k]) { iLoc[k] = double.NaN; continue; }
                double v = 4 * Faraday * cIn * (1 - xV) * kTot[k] * c[k] / 1e4;
                iLoc[k] = v; vals.Add(v);
                sum += v; sum2 += v * v; na++;
                if (v < min) min = v;
            }
            vals.Sort();
            double pos = 0.05 * (na - 1); int lo = (int)pos;                       // linear interpolation, as numpy.percentile
            m.iAvg = sum / na; m.iMin = min;
            m.iP05 = vals[lo] + (pos - lo) * (vals[Math.Min(lo + 1, na - 1)] - vals[lo]);
            m.iCV = Math.Sqrt(Math.Max(sum2 / na - m.iAvg * m.iAvg, 0)) / m.iAvg;
            m.flowRate = Q; m.tau = pore / Q; m.meanSpeed = nf > 0 ? sp / nf : 0;
            double o2Feed = Q * cIn * (1 - xV), o2Used = m.iAvg * 1e4 * W * L / (4 * Faraday);
            m.utilization = o2Used / o2Feed; m.stoich = o2Feed / o2Used;
            m.inletSplit = new double[d.portsIn.Count];
            double cum = 0;
            for (int q = 0; q < inCells.Length; q++)
            {
                m.inletSplit[inPort[inCells[q]]] += qIn[inCells[q]] / Q;
                cum += qIn[inCells[q]] / Q; inCum[q] = cum;
            }
        }

        public void Solve()
        {
            valid = false;
            if (!BuildGeometry()) return;
            int n = nx * ny;
            double Q = d.qSlpm / 6e4 * (d.T / 273.15) * (101325.0 / d.P) / (1 - xV);
            double rho = d.P * 0.02897 / (Rgas * d.T), nu = d.mu / rho;
            for (int k = 0; k < n; k++) kApp[k] = k0[k];
            cgIterations = 0;
            double dP = 0;
            for (int it = 0; it < 8; it++)                                          // Forchheimer: under-relaxed apparent permeability
            {
                SolvePressure();
                dP = ComputeFluxes(Q);
                if (compress)
                {
                    if (rhoRel == null || rhoRel.Length != n) { rhoRel = new double[n]; for (int k = 0; k < n; k++) rhoRel[k] = 1.0; }
                    for (int k = 0; k < n; k++) rhoRel[k] = kind[k] == Solid ? 1.0 : 0.5 * rhoRel[k] + 0.5 * (1 + p[k] / d.P);
                }
                pIn = d.P + dP;
                if (it == 7) break;
                for (int k = 0; k < n; k++)                                         // Forchheimer with the mass flux (speed × rhoRel)
                    if (kind[k] == Foam)
                        kApp[k] = 0.5 * kApp[k] + 0.5 / (1.0 / k0[k] + d.cF * speed[k] * RhoAt(k) / (nu * Math.Sqrt(k0[k])));
            }
            if (!(metrics.massBalanceError < 1e-3)) { message = "압력 풀이가 수렴하지 않았습니다."; return; }
            metrics.pressureDrop = dP;
            SolveOxygen();
            ComputeMetrics(Q);
            valid = true;
            message = "";
        }

        // ---------------------------------------------------------------- tracers

        public int CellAt(double x, double y)
        {
            int i = (int)((x + pad) / dx), j = (int)((y + pad) / dx);
            if (i < 0) i = 0; else if (i > nx - 1) i = nx - 1;
            if (j < 0) j = 0; else if (j > ny - 1) j = ny - 1;
            return i + j * nx;
        }

        // Interstitial velocity [m/s] from linearly interpolated face flows.
        public void Velocity(double x, double y, out double u, out double v)
        {
            double fx = (x + pad) / dx, fy = (y + pad) / dx;
            int i = (int)fx, j = (int)fy;
            if (i < 0) i = 0; else if (i > nx - 1) i = nx - 1;
            if (j < 0) j = 0; else if (j > ny - 1) j = ny - 1;
            fx -= i; fy -= j;
            if (fx < 0) fx = 0; else if (fx > 1) fx = 1;
            if (fy < 0) fy = 0; else if (fy > 1) fy = 1;
            int f = i + j * (nx + 1), k = i + j * nx;
            if (kind[k] == Solid) { u = 0; v = 0; return; }
            double s = 1.0 / (d.hF * 1e-3 * (kind[k] == Foam ? d.eps : 1.0) * dx * RhoAt(k));
            u = ((1 - fx) * qx[f] + fx * qx[f + 1]) * s;
            v = ((1 - fy) * qy[k] + fy * qy[k + nx]) * s;
        }

        // Advance a tracer by up to dtMax (midpoint rule). (ddx, ddy) is a random-walk displacement, may be zero.
        // Returns the time used; exited = reached an outlet port.
        public double Step(ref double x, ref double y, double dtMax, double ddx, double ddy, out bool exited)
        {
            exited = false;
            double u1, v1;
            Velocity(x, y, out u1, out v1);
            double sp = Math.Sqrt(u1 * u1 + v1 * v1);
            double dt = sp > 1e-30 ? Math.Min(dtMax, 0.3 * dx / sp) : dtMax;
            double xm = x + 0.5 * dt * u1, ym = y + 0.5 * dt * v1;
            double u2, v2;
            Velocity(xm, ym, out u2, out v2);
            double xa = x + dt * u2, ya = y + dt * v2;
            double xn = xa + ddx, yn = ya + ddy;
            if (kind[CellAt(xn, yn)] == Solid) { xn = xa; yn = ya; }       // reject the random kick
            if (kind[CellAt(xn, yn)] != Solid) { x = xn; y = yn; }
            if (outPort[CellAt(x, y)] >= 0) exited = true;
            return dt;
        }

        // Random point in the inlet ports, weighted by the flow of each port cell (u1, u2, u3 uniform in 0..1).
        public void SampleInlet(double u1, double u2, double u3, out double x, out double y)
        {
            int lo = 0, hi = inCells.Length - 1;
            while (lo < hi) { int mid = (lo + hi) / 2; if (inCum[mid] < u1) lo = mid + 1; else hi = mid; }
            int k = inCells[lo];
            x = -pad + (k % nx + u2) * dx;
            y = -pad + (k / nx + u3) * dx;
        }
    }
}
