using System;
using System.IO;
using System.Text;
using PorousFlowField;
// Transient fill-then-flow runs for the water domestic paper. args: outDir
class P {
  static void Run(string outDir, string tag, float sIn, float sOut, float header, float rh, double tEnd, double[] snaps) {
    var d = new PocketDesign(); d.dxmm = 0.5f; d.header = header;
    d.portsIn.Add(new SlotPort(sIn, 6f)); d.portsOut.Add(new SlotPort(sOut, 6f));
    var sim = new WaterSim(d, new WaterSettings{ rhIn = rh, vCell = 0.6, compress = true }); sim.Reset();
    var ts = new StringBuilder("t_s,dP_kPa,stored_mg,i_mean,out_mg_s,s_mean\n");
    int si = 0; double next = 0;
    while (true) {
      if (sim.time >= next - 1e-9) {
        ts.AppendLine($"{sim.time:F1},{sim.m.metrics.pressureDrop/1e3:F4},{sim.storedM3*sim.pr.rhoL*1e6:F3},{sim.m.metrics.iAvg:F4},{sim.outM3s*sim.pr.rhoL*1e6:F5},{sim.SatMean():F4}");
        next += sim.time < 60 ? 1.0 : 5.0;
      }
      if (si < snaps.Length && sim.time >= snaps[si] - 1e-9) {
        var sb = new StringBuilder();
        for (int j = 0; j < sim.m.ny; j++) { for (int i = 0; i < sim.m.nx; i++) { int k = i + j*sim.m.nx; sb.Append(sim.m.kind[k]==PocketModel.Foam ? sim.s[k].ToString("F4") : "nan"); sb.Append(i < sim.m.nx-1 ? "," : "\n"); } }
        File.WriteAllText(Path.Combine(outDir, $"snap_{tag}_{snaps[si]:F0}.csv"), sb.ToString());
        si++;
      }
      if (sim.time >= tEnd) break;
      sim.Advance(1.0, 1.0, 0.02, 1);
    }
    File.WriteAllText(Path.Combine(outDir, $"ts_{tag}.csv"), ts.ToString());
    Console.WriteLine($"{tag}: t {sim.time:F0} s dP {sim.m.metrics.pressureDrop/1e3:F2} kPa stored {sim.storedM3*sim.pr.rhoL*1e6:F1} mg i {sim.m.metrics.iAvg:F3} s {sim.SatMean():F3}");
  }
  static void Main(string[] a) {
    string o = a[0]; Directory.CreateDirectory(o);
    var snaps = new double[]{10, 30, 60, 120, 600};
    Run(o, "Z_rh100", 46f, 146f, 0f, 1f, 900, snaps);
    Run(o, "ZH3_rh100", 46f, 146f, 3f, 1f, 900, snaps);
    Run(o, "Z_rh50", 46f, 146f, 0f, 0.5f, 900, snaps);
    Run(o, "ZH3_rh50", 46f, 146f, 3f, 0.5f, 900, snaps);
    Run(o, "U_rh100", 46f, 4f, 0f, 1f, 900, snaps);
  }
}
