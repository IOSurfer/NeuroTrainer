"""
Cache QC report: one line per check, over every cached subject.

    python -m apps.cortical_surface_flow.cache_report --cache_dir input/cortical_surface_flow/cache_ixi_rigid

Reads <subject>/qc.json (written by preprocess.py) and the cached surfaces, and writes
<cache>/cache_report.json.  A subject is listed under "attention" when any check is outside
its limit:

    rigid_ncc            registered native T1 vs T1_Rigid                       (rigid_dataset layout, >= 0.999)
    white_boundary       centre / best of the white-surface gradient profile    (>= 0.99; 1.0 = peak at zero shift)
    in_brainmask_frac    surface vertices inside the brain mask                 (>= 0.99)
    outside_grid_frac    surface vertices outside the T1 grid                   (== 0)
    pial_outside_white   pial vertices outside white along the white normal     (0.88 .. 0.97; catches flipped normals)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from apps.cortical_surface_flow.geometry import GridFrame, read_gifti, utf8_stdout, vertex_normals


def main(argv=None) -> None:
    utf8_stdout()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cache_dir", required=True)
    a = p.parse_args(argv)
    subs = sorted(d for d in os.listdir(a.cache_dir) if os.path.isfile(os.path.join(a.cache_dir, d, "qc.json")))
    rows, attention = [], {}
    frames = {}
    for s in subs:
        q = json.load(open(os.path.join(a.cache_dir, s, "qc.json")))
        bad = []
        r = {"subject": s}
        if "rigid_ncc" in q:
            r["rigid_ncc"], r["rigid_start"] = q["rigid_ncc"], q.get("rigid_start", "centre of mass")
            r["rotation_deg"], r["translation_mm"] = q["rigid_rotation_deg"], q["rigid_translation_mm"]
            if q["rigid_ncc"] < 0.999:
                bad.append("rigid_ncc %.5f" % q["rigid_ncc"])
        if "white_boundary" in q:
            r["white_boundary"] = q["white_boundary"]["centre_over_best"]
            if r["white_boundary"] < 0.99:
                bad.append("white_boundary %.4f" % r["white_boundary"])
        for h, per in q.get("surfaces", {}).items():
            for name, e in per.items():
                if name != "sphere" and e["in_brainmask_frac"] < 0.99:
                    bad.append("%s.%s in mask %.4f" % (h, name, e["in_brainmask_frac"]))
                if e["outside_grid_frac"] > 0:
                    bad.append("%s.%s outside grid %.4f" % (h, name, e["outside_grid_frac"]))
            wp = [n for n in ("white", "pial.T1") if n in per]
            if len(wp) == 2:
                if h not in frames:
                    frames[h] = GridFrame(a.cache_dir, h)
                w = read_gifti(os.path.join(a.cache_dir, s, "%s.white.surf.gii" % h))[0]
                pl = read_gifti(os.path.join(a.cache_dir, s, "%s.pial.T1.surf.gii" % h))[0]
                out = float((((pl - w) * vertex_normals(w, frames[h].faces)).sum(1) > 0).mean())
                r["pial_outside_white_" + h] = out
                if not 0.88 <= out <= 0.97:
                    bad.append("%s pial outside white %.3f" % (h, out))
        rows.append(r)
        if bad:
            attention[s] = bad

    def stat(k):
        v = np.array([r[k] for r in rows if k in r], float)
        return {"n": int(len(v)), "min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())} if len(v) else None

    summary = {"subjects": len(rows), "attention": attention,
               "stats": {k: stat(k) for k in ("rigid_ncc", "rotation_deg", "translation_mm", "white_boundary",
                                              "pial_outside_white_lh", "pial_outside_white_rh")},
               "rigid_start": {k: sum(1 for r in rows if r.get("rigid_start") == k)
                               for k in sorted({r.get("rigid_start") for r in rows if "rigid_start" in r})}}
    with open(os.path.join(a.cache_dir, "cache_report.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    print("%d subjects checked, %d need attention" % (len(rows), len(attention)))
    for k, v in summary["stats"].items():
        if v:
            print("  %-24s min %.5g  median %.5g  max %.5g  (n=%d)" % (k, v["min"], v["median"], v["max"], v["n"]))
    print("  rigid start:", summary["rigid_start"])
    for s, b in attention.items():
        print("  ATTENTION", s, b)


if __name__ == "__main__":
    main()
