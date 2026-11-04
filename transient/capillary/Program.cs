using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Globalization;
using PorousFlowField;
// Model-form test for the Energies revision: transient water model with capillary diffusion between foam cells,
// run to steady state. args: outDir [quick]
class P {
  const string HEADER = "design,rh_in,cap_cos,s_r0,n_c,n_kr,cover,t_s,dP_kPa,water_mg,i_mean,i_min,p01,p05,p10,p25,cov,low_frac,s_mean,s_max";
  static double CapDt = double.Parse(Environment.GetEnvironmentVariable("CAPDT") ?? "5", CultureInfo.InvariantCulture);
  static string Row(string tag, float rh, double cc, double[] c, WaterSim sim, double t) {
    var m = sim.m;
    var v = Enumerable.Range(0, m.nx * m.ny).Where(k => m.act[k] && !double.IsNaN(m.iLoc[k])).Select(k => m.iLoc[k]).OrderBy(x => x).ToArray();
    Func<double, double> pct = q => { double pos = q / 100.0 * (v.Length - 1); int lo = (int)Math.Floor(pos); int hi = Math.Min(lo + 1, v.Length - 1); return v[lo] + (pos - lo) * (v[hi] - v[lo]); };
    double mean = v.Average(), sd = Math.Sqrt(v.Select(x => (x - mean) * (x - mean)).Average());
    int low = v.Count(x => x < 0.5 * mean);
    double smax = 0; for (int k = 0; k < m.nx * m.ny; k++) if (m.kind[k] == PocketModel.Foam) smax = Math.Max(smax, sim.s[k]);
    return string.Format(CultureInfo.InvariantCulture, "{0},{1},{2},{3},{4},{5},{6},{7:F2},{8:F4},{9:F3},{10:F5},{11:F5},{12:F5},{13:F5},{14:F5},{15:F5},{16:F5},{17:F5},{18:F4},{19:F4}",
      tag, rh, cc, c[0], c[1], c[2], c[3], t, m.metrics.pressureDrop / 1e3, sim.storedM3 * sim.pr.rhoL * 1e6, mean, v[0], pct(1), pct(5), pct(10), pct(25), sd / mean,
      (double)low / v.Length, sim.SatMean(), smax);
  }
  static string Run(string outDir, string tag, float sIn, float sOut, float header, float rh, double cc, double[] c, bool snap) {
    var d = new PocketDesign(); d.dxmm = 0.5f; d.header = header;
    d.portsIn.Add(new SlotPort(sIn, 6f)); d.portsOut.Add(new SlotPort(sOut, 6f));
    var ws = new WaterSettings{ rhIn = rh, vCell = 0.6, compress = true, capCos = cc, sr0 = c[0], nc = c[1], nKr = c[2], cover = c[3] };
    var sim = new WaterSim(d, ws); sim.Reset();
    double last = -1; double t = 0;
    while (sim.time < 6000) {
      sim.Advance(10.0, ws.capCos > 0 ? CapDt : 1.0, 0.02, 1);
      if (Environment.GetEnvironmentVariable("CAPDBG") == "1" && Math.Abs(sim.time % 100) < 1e-6)
        Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "  [{0}] t {1:F0} stored {2:F2} mg  out {3:F4} mg/s  sweeps {4}  i {5:F4}", tag, sim.time, sim.storedM3 * sim.pr.rhoL * 1e6, sim.outM3s * sim.pr.rhoL * 1e6, sim.capSweeps, sim.m.metrics.iAvg));
      if (sim.time >= 600 && Math.Abs(sim.time % 100) < 1e-6) {
        double now = sim.storedM3;
        if (last > 0 && Math.Abs(now - last) / now < 2e-4) break;   // steady: stored water changes < 0.01 % per 100 s
        last = now;
      }
    }
    // final gas/current solve with the final saturation
    sim.PushSat(); sim.GasUpdate();
    t = sim.time;
    if (snap) {
      var sb = new StringBuilder();
      for (int j = 0; j < sim.m.ny; j++) { for (int i = 0; i < sim.m.nx; i++) { int k = i + j*sim.m.nx; sb.Append(sim.m.kind[k]==PocketModel.Foam ? sim.s[k].ToString("F4", CultureInfo.InvariantCulture) : "nan"); sb.Append(i < sim.m.nx-1 ? "," : "\n"); } }
      File.WriteAllText(Path.Combine(outDir, string.Format(CultureInfo.InvariantCulture, "s_{0}_rh{1:F0}_cap{2:F1}.csv", tag, rh * 100, cc)), sb.ToString());
    }
    return Row(tag, rh, cc, c, sim, t);
  }
  static void Main(string[] a) {
    string o = a[0]; Directory.CreateDirectory(o);
    string mode = a.Length > 1 ? a[1] : "main";
    var designs = new (string tag, float si, float so, float h)[] { ("Z", 46f, 146f, 0f), ("U", 46f, 4f, 0f), ("C", 25f, 125f, 0f), ("Z+H1.5", 46f, 146f, 1.5f), ("Z+H3", 46f, 146f, 3f), ("Z+H6", 46f, 146f, 6f) };
    var jobs = new System.Collections.Generic.List<(string, float, float, float, float, double, double[], bool)>();
    var refc = new double[] { 0.4, 1e-4, 3.0, 1.0 };
    if (mode == "quick") jobs.Add(("Z", 46f, 146f, 0f, 1f, 1.0, refc, false));
    else if (mode == "dtcheck") { jobs.Add(("Z+H3", 46f, 146f, 3f, 0.5f, 1.0, refc, false)); jobs.Add(("Z", 46f, 146f, 0f, 1f, 1.0, refc, false)); }
    else if (mode == "dbg") jobs.Add(("Z+H3", 46f, 146f, 3f, 0.5f, 1.0, refc, false));
    else if (mode == "q2") { foreach (double cc in new[] { 0.0, 1.0 }) { jobs.Add(("Z+H3", 46f, 146f, 3f, 0.5f, cc, refc, false)); jobs.Add(("Z", 46f, 146f, 0f, 0.5f, cc, refc, false)); } }
    else if (mode == "main") {
      foreach (var dz in designs) foreach (float rh in new[] { 0.5f, 1f }) foreach (double cc in new[] { 0.0, 0.5, 1.0, 3.0 })
        jobs.Add((dz.tag, dz.si, dz.so, dz.h, rh, cc, refc, true));
    } else if (mode == "combos") {   // corner cases of the constant ranges, capCos = 1
      foreach (var s0 in new[] { 0.2, 0.6 }) foreach (var nc in new[] { 1e-5, 1e-3 }) foreach (var nk in new[] { 3.0 })
        foreach (var dz in designs.Where(x => x.tag == "Z" || x.tag == "U" || x.tag == "Z+H3" || x.tag == "Z+H6")) foreach (float rh in new[] { 1f }) foreach (double cc in new[] { 0.0, 1.0 })
          jobs.Add((dz.tag, dz.si, dz.so, dz.h, rh, cc, new double[] { s0, nc, nk, 1.0 }, false));
    }
    // resume: skip jobs already listed in <outDir>/cap_<mode>_done.csv (rows of earlier, interrupted runs)
    string donePath = Path.Combine(o, $"cap_{mode}_done.csv");
    var done = File.Exists(donePath) ? File.ReadAllLines(donePath).Skip(1).Where(l => l.Trim().Length > 0).ToList() : new System.Collections.Generic.List<string>();
    Func<(string, float, float, float, float, double, double[], bool), string> key = j => string.Format(CultureInfo.InvariantCulture,
      "{0},{1},{2},{3},{4},{5},{6}", j.Item1, j.Item5, j.Item6, j.Item7[0], j.Item7[1], j.Item7[2], j.Item7[3]);
    var doneKeys = new System.Collections.Generic.HashSet<string>(done.Select(l => string.Join(",", l.Split(',').Take(7))));
    int before = jobs.Count;
    jobs = jobs.Where(j => !doneKeys.Contains(key(j))).ToList();
    Console.WriteLine($"{mode}: {before - jobs.Count} done earlier, {jobs.Count} to run");
    var lk = new object();
    var rows = new string[jobs.Count];
    Parallel.For(0, jobs.Count, new ParallelOptions { MaxDegreeOfParallelism = 14 }, q => {
      var j = jobs[q];
      var sw = System.Diagnostics.Stopwatch.StartNew();
      rows[q] = Run(o, j.Item1, j.Item2, j.Item3, j.Item4, j.Item5, j.Item6, j.Item7, j.Rest.Item1);
      Console.WriteLine(rows[q] + $"  ({sw.Elapsed.TotalSeconds:F0} s)");
      lock (lk) { File.AppendAllText(donePath, (File.Exists(donePath) ? "" : HEADER + "\n") + rows[q] + "\n"); }   // survives interruption
    });
    File.WriteAllText(Path.Combine(o, $"cap_{mode}.csv"), HEADER + "\n" + string.Join("\n", done.Concat(rows)) + "\n");
  }
}
