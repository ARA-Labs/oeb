"""Act-plane readings of each run, plus the real payoff of each experiment where the benchmark logs one.

Only the tasks whose outcome separates runs are read. Nothing here reads what the agent SAID (stances): only what it launched, which idea a
launch belongs to, what it built on, and when. Used by Section 6 (make_findings_numbers.py and the figures).

Runs: PostTrainBench Arena-Hard Writing / HealthBench / HumanEval / GPQA (AIME and BFCL do not separate runs),
Chip-Bench (hc47-glm panel), nanoGPT speedrun (w5-grok46 panel): the panels the paper reports.
Ground truth: Chip-Bench = the run's own score_log.jsonl (every local `make score`, timestamped, raw metric);
speedrun = the experiment database (every training run, launch time, validation loss). Both are placed on the
graph's step axis through the raw steps' timestamps and attached to the launch made at that step.
"""
import json, os, glob, math, bisect, re, collections
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(os.environ.get("OEB_DATA", ".")); O = str(ROOT) + "/"
PTB = {"Arena-Hard Writing": "arenahard-gemma-w5-glm53", "HealthBench": "healthbench-gemma-w5-glm53",
       "HumanEval": "humaneval-gemma-w5-glm53", "GPQA": "gpqa-gemma-w5-glm53"}
CHIP_RAW = {'c1_approx_area': ('lut4_eq', 'min'), 'c1_pipeline_fmax': ('fmax_mhz', 'max'), 'c2_branch_pred': ('geomean_mpkb', 'min'),
            'c2_cache_ctrl': ('geomean_cycles', 'min'), 'c2_chacha_tput': ('area_time', 'min'), 'c2_divider_al': ('metric', 'min'),
            'c3_mem_sched': ('geomean_cycles', 'min'), 'c3_prefetcher': ('metric_public_geomean_cpa', 'min'), 'c3_replacement': ('public_geomean_mpka', 'min')}
SIG = 0.0013   # speedrun: a loss drop below this is inside the run-to-run noise of one configuration

def J(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None

def outcome(u):
    m = J(f"{u}/official/metrics.json")
    if m and m.get("accuracy") is not None: return m["accuracy"]
    man = J(f"{u}/record_manifest.json") or {}
    if man.get("bench") == "speedrun" and man.get("reward") not in (None, ""):
        try: return -float(man["reward"])
        except Exception: return None
    return ((J(f"{u}/world.json") or {}).get("outcome") or {}).get("score")

def units():
    out = []
    for t, b in PTB.items():
        for u in sorted(glob.glob(O + f"out/posttrainbench/{b}/*/unified_graph.json")):
            u = os.path.dirname(u); n = os.path.basename(u)
            out.append(dict(bench="PostTrainBench", task=t, unit=u, name=n, model=n[:-3] if n.endswith("-r2") else n, second=n.endswith("-r2")))
    for u in sorted(glob.glob(O + "out/chipbench/hc47-glm/*--*/unified_graph.json")):
        u = os.path.dirname(u); n = os.path.basename(u)
        out.append(dict(bench="Chip-Bench", task=n.split("--")[0], unit=u, name=n, model=n.split("--")[1], second=False))
    for u in sorted(glob.glob(O + "out/speedrun/w5-grok46/*/unified_graph.json")):
        u = os.path.dirname(u); n = os.path.basename(u)
        if os.path.exists(u + "/measure.json"): out.append(dict(bench="nanoGPT speedrun", task="speedrun", unit=u, name=n, model=n.rsplit("-", 1)[0], second=False))
    for r in out: r["score"] = outcome(r["unit"])
    return [r for r in out if r["score"] is not None]

def launches(u):
    """research launches in step order: [(gid, idea, act node)]"""
    g = J(u + "/unified_graph.json")
    acts = sorted([n for n in g["nodes"] if n["type"] == "ACT" and not n.get("mandated") and not n.get("harness_event")], key=lambda n: float(n["gid"]))
    return g, acts

def readings(u):
    g, acts = launches(u)
    steps = sum(1 for _ in open(u + "/record.jsonl"))
    com = [n for n in acts if n["act"] == "commit"]
    res = [n for n in com if n.get("topic") == "research"]
    L = len(res); r = {"steps": steps, "launches": len(com), "research": L}
    if L == 0: return r
    ideas = [n.get("mech_group") or n["id"] for n in res]
    cnt = collections.Counter(ideas)
    first = {}
    for i, k in enumerate(ideas): first.setdefault(k, i)
    half = steps / 2
    r.update(ideas=len(cnt), exps_per_idea=L / len(cnt), one_shot_ideas=sum(1 for v in cnt.values() if v == 1) / len(cnt),
             new_ideas_2nd_half=sum(1 for k, i in first.items() if float(res[i]["gid"]) > half),
             switch_rate=(sum(1 for a, b in zip(ideas, ideas[1:]) if a != b) / (L - 1)) if L > 1 else None,
             last_new_idea_at=float(res[max(first.values())]["gid"]) / steps,
             last_research_at=float(res[-1]["gid"]) / steps,
             research_share=L / len(com), research_per_step=L / steps,
             built_on=sum(1 for n in res if n.get("builds_on")) / L)
    return r

# ---------------------------------------------------------------- ground truth
def T(s):
    if s is None: return None
    s = str(s).replace('Z', '+00:00').replace(' ', 'T')
    try: d = datetime.fromisoformat(s)
    except Exception: return None
    if d.tzinfo is None: d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()
def _norm(x): return re.sub(r'[^a-z0-9]', '', str(x).lower())
def step_times(unit, raw_path, W=800):
    rec = [json.loads(l) for l in open(unit + '/record.jsonl')]; raw = json.load(open(raw_path))
    out = {}; ptr = 0
    if not any(w.get('ts') or w.get('ts_call') for w in raw): return {}
    if len(rec) == len(raw):
        for r, w in zip(rec, raw): out[float(r['gid'])] = T(w.get('ts_call') or w.get('ts'))
        return out
    RW = []
    for w in raw:
        tl = [t for t in (w.get('tools') or []) if isinstance(t, dict)]
        RW.append((w.get('api_turn'), [_norm(t.get('cmd') or '')[:28] for t in tl], [str(t.get('fn') or '').lower() for t in tl]))
    for r in rec:
        a = r.get('api_turn'); act = r.get('action') or ''; na = _norm(act); tool = act[2:].split(' ', 1)[0].lower() if act.startswith('$ ') else ''
        for j in range(ptr, min(ptr + W, len(raw))):
            ta, cmds, fns = RW[j]
            if ta != a or not cmds: continue
            if any(c and c in na for c in cmds) or (tool and tool != 'bash' and tool in fns):
                out[float(r['gid'])] = T(raw[j].get('ts_call') or raw[j].get('ts')); ptr = j + 1; break
    return out
def attach(unit, times, events, maxgap=2):
    _, acts = launches(unit); L = [n for n in acts if n['act'] == 'commit']
    gids = sorted(times); tt = [times[g] for g in gids]
    for i in range(1, len(tt)):
        if tt[i] is None or (tt[i - 1] is not None and tt[i] < tt[i - 1]): tt[i] = tt[i - 1]
    lg = [float(n['gid']) for n in L]; out = collections.defaultdict(list)
    key = [x if x is not None else -1 for x in tt]
    for (t, p) in events:
        if t is None: continue
        k = bisect.bisect_right(key, t) - 1
        if k < 0: continue
        g = gids[k]; j = bisect.bisect_right(lg, g) - 1
        if j < 0 or sum(1 for x in gids if lg[j] < x <= g) > maxgap: continue
        out[L[j]['id']].append(p)
    return out
def truth(r):
    """{launch id: {'improve': bool, 'gain': float}} for runs whose benchmark logs a real per-experiment result, else None"""
    u, n = r["unit"], r["name"]
    if r["bench"] == "Chip-Bench":
        raw = O + 'data/chipbench/runs/' + n + '/steps.json'; sl = O + 'data/chipbench/runs/' + n + '/score_log.jsonl'
        if not (os.path.exists(raw) and os.path.exists(sl)) or r["task"] not in CHIP_RAW: return None
        key, dirn = CHIP_RAW[r["task"]]; ev = []
        for l in open(sl):
            if not l.strip(): continue
            e = json.loads(l); ok = e.get(key) is not None and all((e.get('gates') or {}).values())
            ev.append((T(e.get('t')), {'ok': ok, 'v': e.get(key)}))
        ev.sort(key=lambda x: x[0] or 0); best = None
        for t, p in ev:
            p['improve'] = False; p['gain'] = 0.0
            if not p['ok']: continue
            if best is None: best = p['v']; continue
            if (p['v'] < best) if dirn == 'min' else (p['v'] > best):
                p['improve'] = True; p['gain'] = abs(math.log(p['v'] / best)) if p['v'] > 0 and best > 0 else 0.0; best = p['v']
        att = attach(u, step_times(u, raw), ev)
    elif r["bench"] == "nanoGPT speedrun":
        import pyarrow.parquet as pq
        trial = (J(u + '/record_manifest.json') or {}).get('trial')
        raw = O + f'data/speedrun/runs-v10/{trial}/steps.json'
        raw = raw if os.path.exists(raw) else None
        rs = sorted([x for x in pq.read_table(O + 'data/speedrun/kb/experiments.parquet').to_pylist() if x['session_id'] == trial], key=lambda x: int(x['seq_in_session'] or 0))
        if not raw or not rs: return None
        best_at = {}; ev = []
        for x in rs:
            ok = x['status'] == 'completed' and x['mean_val_loss'] is not None and x['train_steps']
            p = {'ok': bool(ok), 'improve': False, 'gain': 0.0, 'ref': False}
            if ok:
                k = x['train_steps']; b = best_at.get(k)
                if b is not None:
                    p['ref'] = True
                    if b - x['mean_val_loss'] > SIG: p['improve'] = True; p['gain'] = b - x['mean_val_loss']
                best_at[k] = min(b, x['mean_val_loss']) if b is not None else x['mean_val_loss']
            ev.append((T(x['launched_at']), p))
        att = attach(u, step_times(u, raw), ev, maxgap=3)
        att = {k: [p for p in v if p['ref']] for k, v in att.items()}
    else: return None
    return {k: {'improve': any(p['improve'] for p in v), 'gain': sum(p['gain'] for p in v)} for k, v in att.items() if v}
