"""
Cortical Surface Flow -- run several experiments one after another on the GPU.

Usage
-----
    python -m apps.cortical_surface_flow.run_queue --spec apps/cortical_surface_flow/queue_formal_rigid.json

Spec (JSON)
-----------
    {"name": "formal_rigid",                      queue folder: <output_dir>/<name>_queue/
     "output_dir": "./output",
     "base_args": ["--cache_dir", "...", ...],    train.py flags shared by every experiment
     "experiments": [{"name": "lh_white_pial", "args": ["--hemi", "lh", ...]}, ...]}

Per experiment (never two at once -- the GPU is shared):
    metrics in <exp>/test_eval/metrics.json   -> done, skipped
    <exp>/checkpoints/latest.pth              -> ``train --reproduce <exp> --resume latest.pth``
    otherwise                                 -> fresh ``train`` (which also runs the test split at the end)
A crashed training is resumed (at most --retries times) before the queue moves on.

TensorBoard follows the queue: when an experiment starts, the queue's own TensorBoard is replaced
by one whose logdir is that experiment's tensorboard/ folder (not all of output/).  It uses the
first free port from --tb_port upwards (default 6006) and never touches a TensorBoard that
somebody else runs (the port is written to <queue folder>/tensorboard.port).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from apps.cortical_surface_flow.geometry import utf8_stdout

TENSORBOARD = r"C:\tools\miniconda3\envs\nnuNet\Scripts\tensorboard.exe"


def log(msg, fh):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def listening_ports():
    """{local TCP port: pid} of every listener (Windows netstat)."""
    out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True).stdout
    ports = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[3] == "LISTENING":
            ports[int(parts[1].rsplit(":", 1)[1])] = int(parts[4])
    return ports


def restart_tensorboard(logdir: Path, port: int, qdir: Path, fh) -> None:
    """Replace the queue's own previous TensorBoard by one on `logdir`.  Only that process is
    ever killed: a TensorBoard (or anything else) that somebody else runs on a port is left
    alone, and the next free port from `port` upwards is used instead."""
    pidfile = qdir / "tensorboard.pid"
    if pidfile.is_file():
        subprocess.run(["taskkill", "/F", "/T", "/PID", pidfile.read_text().strip()], capture_output=True)
        time.sleep(2)
    busy = listening_ports()
    while port in busy:
        port += 1
    out = open(qdir / "tensorboard.out", "ab")
    flags = 0x00000008 | 0x00000200          # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    p = subprocess.Popen([TENSORBOARD, "--logdir", str(logdir.resolve()), "--port", str(port), "--bind_all",
                          "--reload_interval", "30"], stdout=out, stderr=out, creationflags=flags)
    pidfile.write_text(str(p.pid))
    (qdir / "tensorboard.port").write_text(str(port))
    log("TensorBoard -> %s : http://localhost:%d (pid %d)" % (logdir, port, p.pid), fh)


def finished(exp: Path) -> bool:
    return (exp / "test_eval" / "metrics.json").is_file()


def run_experiment(name, spec, args, qdir, fh) -> bool:
    out = spec.get("output_dir", "./output")
    exp = Path(out) / name
    py = [sys.executable, "-u"]
    for attempt in range(args.retries + 1):
        if finished(exp):
            return True
        latest = exp / "checkpoints" / "latest.pth"
        if latest.is_file():
            cmd = ["-m", "apps.cortical_surface_flow.train", "--reproduce", str(exp), "--resume", str(latest)]
            log("%s: resuming (attempt %d)" % (name, attempt + 1), fh)
        else:
            e = next(x for x in spec["experiments"] if x["name"] == name)
            cmd = (["-m", "apps.cortical_surface_flow.train"] + spec["base_args"] + e.get("args", []) +
                   ["--output_dir", out, "--experiment_name", name])
            log("%s: starting" % name, fh)
        exp.mkdir(parents=True, exist_ok=True)
        with open(exp / "stdout.log", "ab") as so:
            proc = subprocess.Popen(py + cmd, stdout=so, stderr=subprocess.STDOUT)
            tb = exp / "tensorboard"
            t0, shown = time.time(), False
            while proc.poll() is None:
                if not shown and tb.is_dir() and any(tb.glob("events.out.*")):
                    restart_tensorboard(tb, args.tb_port, qdir, fh)
                    shown = True
                if not shown and time.time() - t0 > 600:    # never wait forever for the first event file
                    restart_tensorboard(tb, args.tb_port, qdir, fh)
                    shown = True
                time.sleep(10)
        log("%s: train exited with code %d" % (name, proc.returncode), fh)
        if finished(exp):
            return True
    return finished(exp)


def main(argv=None) -> None:
    utf8_stdout()
    p = argparse.ArgumentParser(description="Cortical Surface Flow -- sequential experiment queue")
    p.add_argument("--spec", required=True)
    p.add_argument("--tb_port", type=int, default=6006, help="First port tried; a busy port is skipped, never taken over")
    p.add_argument("--retries", type=int, default=2, help="Resumes of a crashed experiment before moving on")
    p.add_argument("--only", nargs="*", default=None, help="Run only these experiment names")
    a = p.parse_args(argv)
    spec = json.load(open(a.spec, encoding="utf-8"))
    qdir = Path(spec.get("output_dir", "./output")) / ("%s_queue" % spec["name"])
    qdir.mkdir(parents=True, exist_ok=True)
    json.dump(spec, open(qdir / "spec.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    fh = open(qdir / "queue.log", "a", encoding="utf-8")
    log("queue %s: %s" % (spec["name"], [e["name"] for e in spec["experiments"]]), fh)
    results = {}
    for e in spec["experiments"]:
        if a.only and e["name"] not in a.only:
            continue
        t0 = time.time()
        ok = run_experiment(e["name"], spec, a, qdir, fh)
        results[e["name"]] = ok
        log("%s: %s in %.1f min" % (e["name"], "DONE" if ok else "FAILED", (time.time() - t0) / 60), fh)
    log("queue finished: %s" % results, fh)


if __name__ == "__main__":
    main()
