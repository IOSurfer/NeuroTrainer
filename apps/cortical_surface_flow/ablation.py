"""
Cortical Surface Flow -- ablation study runner.

Trains every variant of an ablation spec one after another (never two GPU jobs at
once), evaluates each best.pth on the test split, and rewrites the result table after
every variant (markdown, JSON, LaTeX).  Resumable: a finished variant is skipped, an
interrupted one continues from its latest.pth.

Usage
-----
    python -m apps.cortical_surface_flow.ablation --spec apps/cortical_surface_flow/ablation_default.json
    python -m apps.cortical_surface_flow.ablation --spec ... --only full plain     # subset
    python -m apps.cortical_surface_flow.ablation --spec ... --table_only          # rebuild tables

Spec (JSON)
-----------
    {"name": "abl_lh_white_pial",                  experiment prefix and output folder
     "base_args": ["--cache_dir", "...", ...],     train.py flags shared by every variant
     "test_args": ["--render", "1"],               test.py flags shared by every variant
     "variants": [
        {"key": "full", "label": "Full model"},
        {"key": "plain", "label": "- residual / decoder", "args": ["--encoder_type", "plain"]},
        {"key": "no_kalman", "label": "- Kalman", "from": "full", "test_args": ["--kalman_q_mm", "0"]},
        ...]}

A variant with ``from`` is inference-only: test.py runs on that variant's checkpoint.
The variant ``references_from`` (default: the first) also gets the no-T1 reference rows.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from apps.cortical_surface_flow.geometry import utf8_stdout

METRICS = [("sym", "Surface (mm)", -1), ("t2p95", "T→P p95 (mm)", -1), ("corr", "Corr. (mm)", -1),
           ("fold4", "Fold r4", 1), ("fold16", "Fold r16", 1), ("flip", "Flipped (%)", -1),
           ("contact", "Self-contact", 0)]


def log(msg, fh):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def run(cmd, logfile, fh) -> int:
    log("$ " + " ".join(cmd), fh)
    with open(logfile, "ab") as f:
        return subprocess.run([sys.executable, "-u"] + cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def variant_paths(spec, v, output_dir):
    src = v.get("from", v["key"])
    exp = Path(output_dir) / ("%s_%s" % (spec["name"], src))
    out = exp / ("test_eval" if "from" not in v else "test_eval_%s" % v["key"])
    return exp, out


def _per_subject(m):
    """subject -> metric dict of the model, from a test.py metrics.json."""
    grp = m["groups"].get("test") or next(iter(m["groups"].values()))
    out = {}
    for s, r in grp.items():
        r = r["model"]
        fd, sm = r.get("fold_detail", {}), r["smoothness"]
        out[s] = {"sym": r["d_surface_symmetric_mean"], "t2p95": r["d_surface_target_to_pred"]["p95"],
                  "corr": r["correspondence_error"]["mean"], "fold4": fd.get("fold_r4", np.nan),
                  "fold16": fd.get("fold_r16", np.nan), "flip": sm["flipped_faces_vs_target"],
                  "contact": sm["self_contact_frac_pred"]}
    return out


def paired_tests(full_m, m):
    """Paired Wilcoxon signed-rank test of every metric against the full model, on the
    subjects both were evaluated on: metric -> (p, n where the full model is better, n)."""
    from scipy.stats import wilcoxon
    a, b = _per_subject(full_m), _per_subject(m)
    subs = sorted(set(a) & set(b))
    out = {}
    for k, _, better in METRICS:
        if better == 0:
            continue
        x = np.array([a[s][k] for s in subs])
        y = np.array([b[s][k] for s in subs])
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 6 or np.allclose(x[ok], y[ok]):
            continue
        pv = float(wilcoxon(x[ok], y[ok]).pvalue)
        n_full_better = int(((x > y) if better > 0 else (x < y))[ok].sum())
        out[k] = (pv, n_full_better, int(ok.sum()))
    return out


def _mark(pv):
    return "**" if pv < 0.001 else ("*" if pv < 0.01 else "")


def build_tables(spec, output_dir, out_dir: Path) -> None:
    rows = []
    for v in spec["variants"]:
        _, ev = variant_paths(spec, v, output_dir)
        p = ev / "metrics.json"
        if not p.is_file():
            rows.append((v, None, None))
            continue
        m = json.load(open(p))
        summ = m["summary"].get("test") or next(iter(m["summary"].values()))
        rows.append((v, summ["model"], m))
    ref_rows = []
    for v, s, m in rows:
        if m is not None:
            ref_rows = [(k, val) for k, val in m["summary"].get("test", {}).items() if k.startswith("ref_")]
            if ref_rows:
                break
    full = rows[0][1]
    full_m = rows[0][2]
    tests = {v["key"]: paired_tests(full_m, m) for v, s, m in rows[1:] if m is not None and full_m is not None}

    def fmt(k, x):
        return ("%.1f" % (100 * x)) if k == "flip" else ("%.3f" % x)

    # markdown
    md = ["# Ablation: %s" % spec["name"], "",
          spec.get("description", ""), "",
          "| Variant | " + " | ".join(n for _, n, _ in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for v, s, _ in rows:
        if s is None:
            md.append("| %s | " % v["label"] + " | ".join("pending" for _ in METRICS) + " |")
            continue
        cells = []
        for k, _, better in METRICS:
            c = fmt(k, s[k])
            if full is not None and s is not full and better != 0:
                d = s[k] - full[k]
                c += " (%s%s)" % ("+" if d >= 0 else "-", fmt(k, abs(d)))
                t = tests.get(v["key"], {}).get(k)
                if t and _mark(t[0]):
                    c += _mark(t[0]).replace("*", r"\*")
            cells.append(c)
        md.append("| %s | " % v["label"] + " | ".join(cells) + " |")
    md += ["", "Significance against the full model: paired Wilcoxon signed-rank test over the test "
           "subjects, \\* p < 0.01, \\*\\* p < 0.001.  p and the number of subjects on which the full "
           "model is better:", ""]
    for v, s, _ in rows[1:]:
        t = tests.get(v["key"])
        if t:
            md.append("- %s: " % v["label"] + "; ".join("%s p=%.2g (%d/%d)" % ((k,) + t[k]) for k in t))
    if ref_rows:
        md += ["", "No-T1 references (same test subjects):", "",
               "| Reference | " + " | ".join(n for _, n, _ in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
        for k, s in ref_rows:
            md.append("| %s | " % k + " | ".join(fmt(m, s[m]) for m, _, _ in METRICS) + " |")
    (out_dir / "ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # json
    json.dump({"spec": spec, "rows": [{"key": v["key"], "label": v["label"], "test": s} for v, s, _ in rows],
               "references": dict(ref_rows)}, open(out_dir / "ablation.json", "w"), indent=1)

    # LaTeX (booktabs); best value per column in bold
    done = [(v, s) for v, s, _ in rows if s is not None]
    best = {}
    for k, _, better in METRICS:
        if better and done:
            vals = [s[k] for _, s in done]
            best[k] = min(vals) if better < 0 else max(vals)
    arrows = {-1: r"$\downarrow$", 1: r"$\uparrow$", 0: ""}
    tex = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{l" + "c" * len(METRICS) + "}", r"\toprule",
           "Variant & " + " & ".join(n.replace("%", r"\%").replace("→", r"$\to$") + " " + arrows[b]
                                     for _, n, b in METRICS) + r" \\", r"\midrule"]
    for v, s in done:
        cells = []
        for k, _, b in METRICS:
            c = fmt(k, s[k])
            c = r"\textbf{%s}" % c if k in best and abs(s[k] - best[k]) < 1e-12 else c
            t = tests.get(v["key"], {}).get(k)
            if t and _mark(t[0]):
                c += r"$^{\ast\ast}$" if t[0] < 0.001 else r"$^{\ast}$"
            cells.append(c)
        tex.append(v["label"].replace("−", "--").replace("&", r"\&") + " & " + " & ".join(cells) + r" \\")
        if v is rows[0][0]:
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Ablation on the test split (%d subjects, %s).}" % (
                len(next(iter(json.load(open(variant_paths(spec, rows[0][0], output_dir)[1] / "metrics.json"))
                              ["groups"].values()))) if full is not None else 0,
                spec.get("schedule_note", "")),
            r"\label{tab:ablation}", r"\end{table}"]
    (out_dir / "ablation.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    utf8_stdout()
    p = argparse.ArgumentParser(description="Cortical Surface Flow -- ablation runner")
    p.add_argument("--spec", required=True)
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--only", nargs="*", default=None, help="Run only these variant keys")
    p.add_argument("--table_only", action="store_true")
    a = p.parse_args(argv)
    spec = json.load(open(a.spec, encoding="utf-8"))
    out_dir = Path(a.output_dir) / ("%s_ablation" % spec["name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(spec, open(out_dir / "spec.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    fh = open(out_dir / "ablation.log", "a", encoding="utf-8")
    if a.table_only:
        build_tables(spec, a.output_dir, out_dir)
        return
    ref_key = spec.get("references_from", spec["variants"][0]["key"])
    for v in spec["variants"]:
        if a.only and v["key"] not in a.only:
            continue
        exp, ev = variant_paths(spec, v, a.output_dir)
        if (ev / "metrics.json").is_file():
            log("%s: done, skipped" % v["key"], fh)
            continue
        t0 = time.time()
        if "from" not in v:
            best = exp / "checkpoints" / "best.pth"
            latest = exp / "checkpoints" / "latest.pth"
            done_training = best.is_file() and latest.is_file() and _finished(exp)
            if not done_training:
                if latest.is_file():
                    cmd = ["-m", "apps.cortical_surface_flow.train", "--reproduce", str(exp), "--resume", str(latest)]
                else:
                    cmd = (["-m", "apps.cortical_surface_flow.train"] + spec["base_args"] + v.get("args", []) +
                           ["--output_dir", a.output_dir, "--experiment_name", exp.name, "--no_test"])
                log("%s: training (%s)" % (v["key"], v["label"]), fh)
                rc = run(cmd, out_dir / ("%s.train.log" % v["key"]), fh)
                if rc != 0:
                    log("%s: training FAILED (exit %d), see %s.train.log" % (v["key"], rc, v["key"]), fh)
                    continue
        tcmd = (["-m", "apps.cortical_surface_flow.test", "--config_dir", str(exp), "--output_dir", str(ev)]
                + spec.get("test_args", []) + v.get("test_args", []))
        if v["key"] != ref_key:
            tcmd.append("--no_references")
        log("%s: testing" % v["key"], fh)
        rc = run(tcmd, out_dir / ("%s.test.log" % v["key"]), fh)
        log("%s: %s in %.1f min" % (v["key"], "done" if rc == 0 else "test FAILED (exit %d)" % rc,
                                    (time.time() - t0) / 60), fh)
        build_tables(spec, a.output_dir, out_dir)
    build_tables(spec, a.output_dir, out_dir)
    log("ablation finished -> %s" % out_dir, fh)


def _finished(exp: Path) -> bool:
    """latest.pth has reached the configured number of steps."""
    import torch
    try:
        ck = torch.load(exp / "checkpoints" / "latest.pth", map_location="cpu", weights_only=False)
        steps = ck["configs"]["Training"]["default"]["steps"]
        return ck["step"] >= steps
    except Exception:
        return False


if __name__ == "__main__":
    main()
