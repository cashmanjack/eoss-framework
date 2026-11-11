"""Build the dataset from saved runs (notes/setup3/runs, private), fit, report. Run: /usr/bin/python3 model/run_model.py"""
import json, os, sys, re
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import rettu_model as M

R = os.path.join(os.path.dirname(__file__), "..", "notes", "setup3", "runs")


def memory4_rows():
    d = json.load(open(f"{R}/memory4/summary.json"))
    nc = {(x["scene"], x["order"], x["state"], x["w"]): x for x in d["ncu"]}
    rows = []
    for t in d["timing"]:
        k = (t["scene"], t["variant"], t["state"], t["warps"])
        if k not in nc:
            continue
        n = nc[k]
        rows.append(dict(scene="soup" + t["scene"], variant=t["variant"], state=t["state"], W=t["warps"],
                         rpc=t["rays_per_cycle"], q=t["queries_per_ray"] + t["idle_lane_queries"],
                         S1=n["l1_sectors_per_ray"], F=n["l1_fill_sectors_per_ray"],
                         D=n["dram_bytes_per_ray"]))
    return rows


def err(p, rows):
    return np.array([M.predict_sm(p, r) / r["rpc"] - 1 for r in rows])


if __name__ == "__main__":
    rows = memory4_rows()
    print(len(rows), "rows")
    p = M.fit(rows)
    print({k: round(v, 3) for k, v in p.items()}, "all-data rms rel err", np.sqrt(np.mean(err(p, rows) ** 2)))
    for held in ["soup3000000", "soup100000"]:
        tr = [r for r in rows if r["scene"] != held]; te = [r for r in rows if r["scene"] == held]
        pp = M.fit(tr); e = err(pp, te)
        print("held scene", held, "mean abs", np.mean(abs(e)), "bias", e.mean())
    for held in ["b_grouped_ray", "b_interleaved_ray", "b_grouped_warp"]:
        tr = [r for r in rows if r["variant"] != held]; te = [r for r in rows if r["variant"] == held]
        pp = M.fit(tr); e = err(pp, te)
        print("held rayset", held, "mean abs", np.mean(abs(e)), "bias", e.mean())


# ---- whole GPU (Throughput_6, first query only, 32 warps per SM; all four scenes held out of the fit)
SCENES6 = {  # per ray descriptors (Memory_3 and step 21 ncu; coherent scenes: one warp query shares its sectors over 32 rays);
    # Ftot = all L2 sectors per ray incl. the harness stores (RT share 59% / 69% from step 21); h1 = L1 hit share assumed
    "tri1": dict(F=12.0/32, Ftot=12.0/32, h1=0.0, D={1: 0, 142: 0}),
    "par64": dict(F=104.0/32, Ftot=104.0/32, h1=0.0, D={1: 0, 142: 0}),
    "soup1e5": dict(F=45.3, Ftot=45.3/0.59, h1=0.2, D={1: 0, 32: 0, 64: 0, 142: 0}),
    "soup3e6": dict(F=68.7, Ftot=68.7/0.69, h1=0.2, D={1: 0, 32: 0, 64: 53, 142: 470}),
}


def throughput6_rows():
    out = []
    for ln in open(f"{R}/throughput6/report.txt"):
        m = re.match(r"(tri1|par64|soup1e5|soup3e6)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s+[\d.]+\s*\|\s*([\d.]+)", ln)
        if m:
            out.append((m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))))
    return out


def gpu_predictions(p):
    res = []
    for sc, G, per_sm, agg in throughput6_rows():
        d = SCENES6[sc]
        gs = sorted(d["D"]); D = float(np.interp(G, gs, [d["D"][g] for g in gs]))
        r = dict(q=1.0, S1=d["F"] / (1 - d["h1"]), F=d["F"], D=D, W=32)
        x1 = M.predict_sm(p, r)
        res.append(dict(scene=sc, G=G, meas=agg, pred=M.predict_gpu(x1, d.get("Ftot", d["F"]), G), x1=x1, meas1=per_sm))
    return res


def report():
    rows = memory4_rows()
    p = M.fit(rows)
    lines = []
    lines.append("# Model v1 (fitted numbers and errors)\n")
    lines.append("Fit on Memory_4 one SM data (72 points: soup 1e5 and 3e6, six ray orders, warm/cold, 4/8/16 warps). Error = predicted / measured - 1.\n")
    lines.append("## Fitted parameters (cycles per ray on one SM; x32 for cycles per warp batch)")
    for k, v in p.items():
        lines.append(f"- {k} = {v:.3f}  (per warp batch: {32*v:.0f})")
    lines.append(f"\nAll data fit: mean |err| {np.mean(abs(err(p, rows))):.3f}, rms {np.sqrt(np.mean(err(p, rows)**2)):.3f}\n")
    lines.append("## One SM, held out (fit without the named scene or ray order, predict it)")
    lines.append("| Held out | n | mean abs err | bias |\n|---|---|---|---|")
    held = []
    for key, vals in (("scene", ["soup100000", "soup3000000"]),
                      ("variant", ["interleaved", "groupedbox", "grouped", "b_grouped_warp", "b_grouped_ray", "b_interleaved_ray"])):
        for v in vals:
            tr = [r for r in rows if r[key] != v]; te = [r for r in rows if r[key] == v]
            e = err(M.fit(tr), te); held.append((key, v, np.mean(abs(e)), e.mean()))
            lines.append(f"| {key} {v} | {len(te)} | {np.mean(abs(e)):.1%} | {e.mean():+.1%} |")
    lines.append("\nBy warps (all-data fit): " + ", ".join(
        f"W{W} {np.mean(abs(err(p,[r for r in rows if r['W']==W]))):.1%}" for W in (4, 8, 16)))
    g = gpu_predictions(p)
    lines.append("\n## Whole GPU (Throughput_6, 32 warps per SM, first query; never in the fit)")
    lines.append("| Scene | one SM pred / meas | G=32 | G=71 | G=100 | G=142 | mean abs err all G |\n|---|---|---|---|---|---|---|")
    summ = {}
    for sc in SCENES6:
        gg = [x for x in g if x["scene"] == sc]
        def cell(G):
            x = [y for y in gg if y["G"] == G][0]; return f"{x['pred']:.3f} / {x['meas']:.3f} ({x['pred']/x['meas']-1:+.0%})"
        mae = np.mean([abs(x["pred"] / x["meas"] - 1) for x in gg]); summ[sc] = mae
        x1 = gg[0]
        lines.append(f"| {sc} | {x1['pred']:.4f} / {x1['meas']:.4f} | {cell(32)} | {cell(71)} | {cell(100)} | {cell(142)} | {mae:.0%} |")
    lines.append("\nGPU cells are aggregate rays per cycle, predicted / measured.")
    return p, rows, held, g, summ, "\n".join(lines)


def plot(p, rows, g):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    a = ax[0]
    for v, c in (("interleaved", "#1b6ca8"), ("grouped", "#2e8b57"), ("b_grouped_warp", "#d98c1f"), ("b_grouped_ray", "#b03a2e"),
                 ("groupedbox", "#7f7f7f"), ("b_interleaved_ray", "#8e44ad")):
        # held out ray order: fit without it
        pp = M.fit([r for r in rows if r["variant"] != v]); rr = [r for r in rows if r["variant"] == v]
        a.scatter([r["rpc"] for r in rr], [M.predict_sm(pp, r) for r in rr], s=18, color=c, label=v)
    lo, hi = 3e-4, 3e-2
    a.plot([lo, hi], [lo, hi], "k-", lw=0.8); a.plot([lo, hi], [lo*1.25, hi*1.25], "k:", lw=0.6); a.plot([lo, hi], [lo/1.25, hi/1.25], "k:", lw=0.6)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("measured rays per cycle (one SM)"); a.set_ylabel("predicted (ray order held out)")
    a.set_title("One SM: each ray order predicted without being fitted"); a.legend(fontsize=7)
    b = ax[1]
    for sc, c in (("tri1", "#1b6ca8"), ("par64", "#2e8b57"), ("soup1e5", "#d98c1f"), ("soup3e6", "#b03a2e")):
        gg = [x for x in g if x["scene"] == sc]
        b.plot([x["G"] for x in gg], [x["meas"] for x in gg], "o", color=c, ms=4, label=sc + " measured")
        b.plot([x["G"] for x in gg], [x["pred"] for x in gg], "-", color=c, lw=1.2)
    b.axvline(71, color="gray", lw=0.6, ls=":")
    b.set_yscale("log"); b.set_xlabel("active SMs (blocks)"); b.set_ylabel("aggregate rays per cycle"); b.set_title("Whole GPU: lines = model (scenes never fitted)")
    b.legend(fontsize=7)
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.text(0.5, 0.01, "Right panel: the whole GPU fit used partly harness limited data (our kernel, not an RT core limit); tri1 (one triangle) is not modeled.", ha="center", fontsize=8, wrap=True); fig.savefig(os.path.join(os.path.dirname(__file__), "..", "notes", "figures", "model_v1_error.png"), dpi=140)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "report":
    p, rows, held, g, summ, text = report()
    open(os.path.join(os.path.dirname(__file__), "..", "notes", "model_v1.md"), "w").write(text + "\n")
    plot(p, rows, g)
    print(text)
