#!/usr/bin/env python3
"""Local status dashboard for the scorvec.com pipelines + ECMWF downloads.

A dependency-free (stdlib only) live monitor: launchd pipeline health, in-flight
ECMWF store downloads with progress bars, cache/disk usage, and recent commits —
colour-coded green/yellow/red. Read-only; safe to run alongside everything.

    python scripts/status_dash.py            # then open http://localhost:8787
    python scripts/status_dash.py --port 9000 --once   # print one JSON snapshot

Keep it open in a browser tab; the page refreshes itself every few seconds.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "scripts" / "ecmwf" / "cache"
REFRESH_S = 5
STALL_S = 180                      # an active download untouched this long → flagged

import sys
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
try:
    import store as _store          # registry() → full download queue; SOURCES list
except Exception:                   # noqa: BLE001
    _store = None

# pipeline name, launchd label, run-log, expected cadence (h), recent-error patterns
PIPELINES = [
    dict(name="MJO / RMM + AAM", label="com.scorvec.mjo",
         log=REPO / "scripts/mjo/run_local.log", cadence=12,
         markers=REPO / "scripts/mjo/data"),
    dict(name="SST / RONI", label="com.scorvec.sst",
         log=REPO / "scripts/sst/run_local_sst.log", cadence=24, markers=None),
    dict(name="Synoptic maps", label="com.scorvec.synoptic",
         log=REPO / "scripts/synoptic/run_local_synoptic.log", cadence=6, markers=None),
]
ERR_RE = re.compile(r"\b(error|failed|fatal|could not|traceback|exit 1|skipped)\b", re.I)
OK_RE = re.compile(r"\b(pushed|complete|nothing to do|saved)\b", re.I)


# ── helpers ───────────────────────────────────────────────────────────────────
def _run(cmd: list[str], cwd=None) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=8).stdout
    except Exception:
        return ""


def _launchd() -> dict:
    out = _run(["launchctl", "list"])
    jobs = {}
    for ln in out.splitlines():
        p = ln.split("\t")
        if len(p) >= 3 and p[2].startswith("com.scorvec"):
            pid = None if p[0] in ("-", "") else int(p[0])
            jobs[p[2]] = dict(pid=pid, exit=p[1])
    return jobs


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.0f} {u}" if u in ("B", "KB") else f"{n:.1f} {u}"
        n /= 1024


def _age(secs: float) -> str:
    if secs < 90:
        return f"{secs:.0f}s ago"
    if secs < 5400:
        return f"{secs/60:.0f}m ago"
    if secs < 172800:
        return f"{secs/3600:.1f}h ago"
    return f"{secs/86400:.1f}d ago"


# ── collectors ────────────────────────────────────────────────────────────────
def collect_pipelines(jobs: dict) -> list[dict]:
    rows = []
    for p in PIPELINES:
        j = jobs.get(p["label"])
        log: Path = p["log"]
        status, notes = "muted", []
        last_run, running = None, bool(j and j["pid"])
        if log.exists():
            last_run = log.stat().st_mtime
            tail = "\n".join(log.read_text(errors="ignore").splitlines()[-120:])
            errs = ERR_RE.findall(tail)
            if errs:
                notes.append(f"{len(errs)} warn/err in recent log")
        loaded = j is not None
        last_exit = j["exit"] if j else "—"
        if not loaded:
            status = "red"; notes.insert(0, "launchd NOT loaded")
        elif last_exit not in ("0", "—"):
            status = "red"; notes.insert(0, f"last exit {last_exit}")
        elif running:
            status = "blue"; notes.insert(0, f"running (pid {j['pid']})")
        elif last_run is not None:
            stale = (time.time() - last_run) > p["cadence"] * 3600 * 1.6
            if notes:
                status = "yellow"
            elif stale:
                status = "yellow"; notes.insert(0, "stale (no recent run)")
            else:
                status = "green"
        rows.append(dict(name=p["name"], label=p["label"], status=status,
                         running=running, loaded=loaded, last_exit=last_exit,
                         last_run=_age(time.time() - last_run) if last_run else "—",
                         cadence=p["cadence"], notes=notes))
    return rows


def _expected_sizes() -> dict:
    """Map spec-filename -> a typical finalized size, to estimate in-flight progress."""
    sizes = {}
    if not CACHE.exists():
        return sizes
    for cyc in CACHE.glob("*z"):
        for model in cyc.iterdir() if cyc.is_dir() else []:
            for f in (model.iterdir() if model.is_dir() else []):
                if f.suffix == ".grib2" and f.is_file():
                    sizes[f.name] = max(sizes.get(f.name, 0), f.stat().st_size)
    return sizes


def _read_src(st: Path) -> str:
    """Mirror breadcrumb left by the store (`*.src` inside the .parts dir, newest wins)."""
    cands = list(st.glob("*.src")) if st.is_dir() else (
        [Path(str(st) + ".src")] if Path(str(st) + ".src").exists() else [])
    if not cands:
        return ""
    try:
        return max(cands, key=lambda f: f.stat().st_mtime).read_text().strip()
    except OSError:
        return ""


def _spec_label(spec) -> str:
    lev = "/".join(str(x) for x in spec.levelist) if spec.levelist else spec.levtype
    return f"{spec.model} · {spec.type} {spec.param}@{lev} ×{len(spec.steps)}"


def collect_queue(cyc_name: str, exp: dict) -> list[dict]:
    """The full registry of specs for the active cycle, each done / downloading / pending."""
    if _store is None or not cyc_name:
        return []
    rows = []
    for spec in _store.registry():
        md = CACHE / cyc_name / spec.model
        p = md / spec.filename
        parts = md / (".stage_" + spec.filename + ".parts")
        stage = md / (".stage_" + spec.filename)
        if p.exists():
            rows.append(dict(label=_spec_label(spec), status="done", pct=100))
        elif parts.exists() or stage.exists():
            st = parts if parts.exists() else stage
            cur, want = _dir_size(st), exp.get(spec.filename, 0)
            mt = max((c.stat().st_mtime for c in (st.rglob("*") if st.is_dir() else [st]) if c.exists()),
                     default=st.stat().st_mtime if st.exists() else 0)
            stalled = (time.time() - mt) > STALL_S
            rows.append(dict(label=_spec_label(spec),
                             status="stalled" if stalled else "active",
                             pct=min(100, 100 * cur / want) if want else None, src=_read_src(st)))
        else:
            rows.append(dict(label=_spec_label(spec), status="pending", pct=0))
    return rows


def collect_downloads() -> dict:
    exp = _expected_sizes()
    cycles = []
    active = []
    if CACHE.exists():
        for cyc in sorted(CACHE.glob("*z"), reverse=True):
            models = {}
            for model in sorted(p for p in cyc.iterdir() if p.is_dir()):
                done = [f for f in model.glob("*.grib2") if not f.name.startswith(".stage")]
                stages = [s for s in model.glob(".stage_*")
                          if not s.name.endswith((".src", ".json"))]
                models[model.name] = dict(done=len(done), staging=len(stages))
                for st in stages:
                    spec = st.name.replace(".stage_", "").replace(".parts", "")
                    cur = _dir_size(st) if st.is_dir() else (st.stat().st_size if st.exists() else 0)
                    want = exp.get(spec, 0)
                    mtime = max((c.stat().st_mtime for c in (st.rglob("*") if st.is_dir() else [st]) if c.exists()),
                                default=st.stat().st_mtime if st.exists() else 0)
                    idle = time.time() - mtime
                    active.append(dict(cycle=cyc.name, model=model.name, spec=spec,
                                       cur=cur, want=want, src=_read_src(st),
                                       pct=min(100, 100 * cur / want) if want else None,
                                       idle=idle, stalled=idle > STALL_S))
            cycles.append(dict(name=cyc.name, models=models,
                               size=sum(_dir_size(m) for m in cyc.iterdir() if m.is_dir())))
    queue = collect_queue(cycles[0]["name"] if cycles else "", exp)
    return dict(cycles=cycles, active=active, queue=queue,
                active_cycle=cycles[0]["name"] if cycles else "")


def collect_jobs_misc() -> dict:
    # background bulk pulls (ARCO/WB2 analog extractions)
    ps = _run(["pgrep", "-fl", "build_u850_bandseries.py"])
    pulls = [ln for ln in ps.splitlines() if "build_u850_bandseries" in ln]
    du, _, free = shutil.disk_usage(REPO)
    git_log = _run(["git", "log", "--oneline", "-5"], cwd=REPO).splitlines()
    git_sb = _run(["git", "status", "-sb"], cwd=REPO).splitlines()
    ahead = next((l for l in git_sb[:1]), "")
    return dict(pulls=pulls, disk_free=free, disk_total=du,
                cache_size=_dir_size(CACHE) if CACHE.exists() else 0,
                git_log=git_log, git_branch=ahead)


def collect() -> dict:
    jobs = _launchd()
    return dict(ts=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
                pipelines=collect_pipelines(jobs),
                downloads=collect_downloads(),
                misc=collect_jobs_misc())


# ── render ────────────────────────────────────────────────────────────────────
CLR = dict(green="#3fb950", yellow="#d29922", red="#f85149", blue="#58a6ff", muted="#8b949e")


def _badge(status: str, text: str) -> str:
    c = CLR.get(status, CLR["muted"])
    return f'<span class="badge" style="background:{c}1f;color:{c};border:1px solid {c}55">{html.escape(text)}</span>'


def _bar(pct, color="#58a6ff", stalled=False):
    if pct is None:
        return '<span class="muted">size unknown</span>'
    c = CLR["red"] if stalled else color
    return (f'<div class="track"><div class="fill" style="width:{pct:.0f}%;background:{c}"></div>'
            f'<span class="pct">{pct:.0f}%</span></div>')


def render(s: dict) -> str:
    ptiles = []
    for p in s["pipelines"]:
        c = CLR.get(p["status"], CLR["muted"])
        notes = " · ".join(html.escape(n) for n in p["notes"]) or "all good"
        loaded = "loaded" if p["loaded"] else '<b style="color:#f85149">UNLOADED</b>'
        ptiles.append(f"""<div class="tile" style="border-top:5px solid {c}">
          <div class="thead">{_badge(p['status'],'●')}&nbsp;<b>{html.escape(p['name'])}</b></div>
          <div class="kv"><span>launchd</span><span>{loaded}</span></div>
          <div class="kv"><span>last exit</span><span>{html.escape(str(p['last_exit']))}</span></div>
          <div class="kv"><span>last run</span><span>{html.escape(p['last_run'])}</span></div>
          <div class="note">{notes}</div>
          <div class="sub mono" style="margin-top:.5em">{html.escape(p['label'])} · ~{p['cadence']}h</div>
        </div>""")

    A = []
    for a in sorted(s["downloads"]["active"], key=lambda x: -(x["pct"] or 0)):
        flag = _badge("red", "STALLED " + _age(a["idle"])) if a["stalled"] else _badge("blue", "downloading")
        sz = _human(a["cur"]) + (f' / {_human(a["want"])}' if a["want"] else "")
        src = f'<div class="sub">via {html.escape(a["src"])}</div>' if a.get("src") else \
              '<div class="sub muted">source —</div>'
        A.append(f"""<tr><td class="mono">{html.escape(a['cycle'])}<div class="sub">{html.escape(a['model'])}</div></td>
          <td class="mono sub">{html.escape(a['spec'])}{src}</td>
          <td class="barcell">{_bar(a['pct'], stalled=a['stalled'])}</td>
          <td class="mono">{sz}</td><td>{flag}</td></tr>""")
    if not A:
        A.append('<tr><td colspan="5" class="muted" style="text-align:center;padding:1.4em">no active downloads</td></tr>')

    # full registry queue for the active cycle
    Q = []
    qd = s["downloads"].get("queue", [])
    ICON = {"done": ("✓", CLR["green"]), "active": ("⏳", CLR["blue"]),
            "stalled": ("✕", CLR["red"]), "pending": ("○", CLR["muted"])}
    for q in qd:
        ic, c = ICON.get(q["status"], ("○", CLR["muted"]))
        if q["status"] == "done":
            right = '<span style="color:#3fb950">done</span>'
        elif q.get("pct") is not None:
            right = f'{q["pct"]:.0f}%' + (f' · {html.escape(q["src"])}' if q.get("src") else "")
        else:
            right = "queued" if q["status"] == "pending" else "…"
        Q.append(f'<tr><td style="color:{c};width:1.4em;font-size:1.1em">{ic}</td>'
                 f'<td class="mono">{html.escape(q["label"])}</td>'
                 f'<td class="mono sub" style="text-align:right;white-space:nowrap">{right}</td></tr>')
    ndone = sum(1 for q in qd if q["status"] == "done")

    C = []
    for c in s["downloads"]["cycles"]:
        ms = " &nbsp; ".join(f'{html.escape(m)} {v["done"]}✓'
                             + (f' <span style="color:#d29922">{v["staging"]}⏳</span>' if v["staging"] else "")
                             for m, v in c["models"].items())
        C.append(f'<tr><td class="mono"><b>{html.escape(c["name"])}</b></td>'
                 f'<td class="mono">{_human(c["size"])}</td><td class="sub">{ms}</td></tr>')
    if not C:
        C.append('<tr><td class="muted">empty</td></tr>')

    m = s["misc"]
    pulls = "<br>".join(html.escape(p) for p in m["pulls"]) or '<span class="muted">none running</span>'
    gitlog = "<br>".join(f'<span class="mono sub">{html.escape(l)}</span>' for l in m["git_log"])
    disk_pct = 100 * (1 - m["disk_free"] / m["disk_total"])
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="{REFRESH_S}"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>scorvec status</title><style>
:root{{color-scheme:dark}}
*{{box-sizing:border-box}}
body{{background:#0d1117;color:#c9d1d9;font:clamp(13px,0.8vw,18px)/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0 auto;padding:26px clamp(16px,2vw,44px);width:min(1900px,90vw)}}
h1{{font-size:1.85em;margin:0 0 4px}} h2{{font-size:.95em;text-transform:uppercase;letter-spacing:.1em;color:#8b949e;margin:1.4em 0 .55em}}
.ts{{color:#8b949e;font-size:.7em;margin-bottom:.4em}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:22px}}
.tile{{background:#161b22;border:1px solid #21262d;border-radius:18px;padding:20px 26px}}
.tile.full{{grid-column:1/-1}}
.thead{{font-size:1.15em;margin-bottom:.45em}}
.kv{{display:flex;justify-content:space-between;padding:.4em 0;border-bottom:1px solid #21262d}}
.kv span:first-child{{color:#8b949e}} .kv:last-of-type{{border-bottom:none}}
.note{{margin-top:.6em}}
table{{border-collapse:collapse;width:100%}} td{{padding:.7em 1em;border-bottom:1px solid #21262d;vertical-align:middle}}
.barcell{{width:44%}}
.badge{{padding:.12em .7em;border-radius:20px;font-size:.82em;font-weight:600;white-space:nowrap}}
.sub{{color:#8b949e;font-size:.82em}} .muted{{color:#6e7681}} .mono{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.85em}}
.track{{position:relative;background:#21262d;border-radius:8px;height:2.1em;overflow:hidden}}
.fill{{height:100%;border-radius:8px;transition:width .4s}} .pct{{position:absolute;right:.6em;top:0;font-size:.8em;line-height:2.6em;color:#c9d1d9;font-weight:600}}
b{{color:#e6edf3}}
</style></head><body>
<h1>scorvec.com — pipeline &amp; download status</h1>
<div class="ts">updated {s['ts']} · auto-refresh {REFRESH_S}s</div>

<h2>Pipelines</h2>
<div class="cards">{''.join(ptiles)}</div>

<h2>Active downloads</h2>
<div class="tile full"><table>{''.join(A)}</table></div>

<h2>Download queue — {html.escape(s['downloads'].get('active_cycle','')) or 'n/a'} · {ndone}/{len(qd)} done</h2>
<div class="tile full"><table>{''.join(Q) or '<tr><td class=muted>registry unavailable</td></tr>'}</table></div>

<h2>Store cache &amp; system</h2>
<div class="cards">
  <div class="tile"><div class="thead"><b>Store cache</b></div><table>{''.join(C)}</table>
    <div class="sub" style="margin-top:.6em">total {_human(m['cache_size'])} · disk {disk_pct:.0f}% used · {_human(m['disk_free'])} free</div></div>
  <div class="tile"><div class="thead"><b>Recent commits</b></div>{gitlog}
    <div class="sub" style="margin-top:.5em">{html.escape(m['git_branch'])}</div></div>
  <div class="tile"><div class="thead"><b>Background pulls</b></div><span class="mono sub">{pulls}</span></div>
</div>
</body></html>"""


# ── server ────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path.startswith("/api"):
            body = json.dumps(collect(), default=str).encode()
            ctype = "application/json"
        else:
            body = render(collect()).encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8799)   # 8787 is often taken by a site preview
    ap.add_argument("--once", action="store_true", help="print one JSON snapshot and exit")
    a = ap.parse_args()
    if a.once:
        print(json.dumps(collect(), indent=2, default=str)); return 0
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"status dashboard → http://localhost:{a.port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
