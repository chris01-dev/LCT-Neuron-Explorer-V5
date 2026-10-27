"""LCT Neuron Explorer server: sweep data, names, embeddings, and the live model.

    uv run python explorer/server.py            # http://localhost:8791
"""
import asyncio
import builtins
import glob
import json
import os
import sys
import threading

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sweep"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank import compute_units  # noqa: E402
from runtime import NO_MODEL, Runtime  # noqa: E402
from brain import Brain  # noqa: E402

RUNS, STATS = os.path.join(ROOT, "runs"), os.path.join(ROOT, "stats")
NAMES = os.path.join(ROOT, "explorer", "names.json")
PORT = int(os.environ.get("PORT", 8791))
_cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B").get_text_config()
ARCH = {"n_layers": _cfg.num_hidden_layers, "hidden": _cfg.hidden_size, "inter": _cfg.intermediate_size,
        "layer_types": list(_cfg.layer_types), "attn_heads": _cfg.num_attention_heads,
        "kv_heads": _cfg.num_key_value_heads, "linear_k_heads": _cfg.linear_num_key_heads,
        "linear_v_heads": _cfg.linear_num_value_heads}

app = FastAPI()


# only this machine's own pages and tools: a page on another site (or a DNS name rebound to 127.0.0.1) can't reach
# the API, which runs the model
LOCAL_HOSTS = {f"localhost:{PORT}", f"127.0.0.1:{PORT}", f"[::1]:{PORT}"}


@app.middleware("http")
async def local_only(request, call_next):
    from urllib.parse import urlparse
    origin = request.headers.get("origin")
    if (request.headers.get("host") not in LOCAL_HOSTS
            or (origin and urlparse(origin).netloc not in LOCAL_HOSTS)
            or request.headers.get("sec-fetch-site") in ("cross-site", "same-site")):
        return PlainTextResponse("forbidden", status_code=403)
    return await call_next(request)


@app.middleware("http")
async def no_store(request, call_next):
    resp = await call_next(request)
    if not request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp

rt = Runtime(STATS)
_cache = {}


def layer_runs():
    """layer -> path of its sweep run: main_rel preferred, then the largest (calibration runs excluded)."""
    best = {}
    for meta_path in glob.glob(os.path.join(RUNS, "*.meta.json")):
        meta = json.load(open(meta_path))
        if meta.get("design_name", "main") == "calib":
            continue
        jsonl = meta_path.replace(".meta.json", ".jsonl")
        if not os.path.exists(jsonl):
            continue
        key = (meta.get("design_name") == "main_rel", os.path.getsize(jsonl))
        if key > best.get(meta["layer"], ("", (False, 0)))[1]:
            best[meta["layer"]] = (jsonl, key)
    return {layer: path for layer, (path, _) in best.items()}


_load_lock = threading.Lock()


def load_layer(layer):
    path = layer_runs().get(layer)
    if path is None:
        return None
    units_path = path.replace(".jsonl", ".units.jsonl")
    key = (path, os.path.getmtime(path), os.path.getmtime(units_path) if os.path.exists(units_path) else 0)
    if _cache.get(layer, {}).get("key") == key:
        return _cache[layer]
    with _load_lock:
        if _cache.get(layer, {}).get("key") == key:
            return _cache[layer]
        if not (os.path.exists(units_path) and os.path.getmtime(units_path) >= os.path.getmtime(path)):
            full = [json.loads(l) for l in open(path)]  # ranking needs token ids; done once, then dropped
            with open(units_path + ".tmp", "w") as f:
                f.writelines(json.dumps(u) + "\n" for u in compute_units(full))
            del full
            os.replace(units_path + ".tmp", units_path)
            key = (path, os.path.getmtime(path), os.path.getmtime(units_path))
        units = [json.loads(l) for l in open(units_path)]
        # keep only what the explorer shows; token ids, logprobs and alternatives stay on disk
        slim = ("kind", "neuron", "prompt", "dose", "rel_push", "text", "finish", "n_gen")
        rows = []
        for l in open(path):
            r = json.loads(l)
            rows.append({k: r[k] for k in slim})
        by_unit = {}
        for i, r in enumerate(rows):
            by_unit.setdefault((r["kind"], r["neuron"]), []).append(i)
        _cache[layer] = {"key": key, "path": path, "rows": rows, "units": units, "by_unit": by_unit,
                         "meta": json.load(open(path.replace(".jsonl", ".meta.json")))}
        return _cache[layer]


_ranked = {}


def ranked_words(layer):
    """{neuron: top words} for the neurons of `layer` with sweep rollouts, read from the run's ranking file only."""
    path = layer_runs().get(layer)
    units_path = path.replace(".jsonl", ".units.jsonl") if path else None
    if not units_path or not os.path.exists(units_path):
        return {}
    key = (units_path, os.path.getmtime(units_path))
    if _ranked.get(layer, (None,))[0] != key:
        words = {}
        for line in open(units_path):
            u = json.loads(line)
            if u["kind"] == "neuron":
                words[u["neuron"]] = [w for w, _, _ in u["words"][:4]]
        _ranked[layer] = (key, words)
    return _ranked[layer][1]


def swept(layer):
    return ranked_words(layer).keys()


def names():
    return json.load(open(NAMES)) if os.path.exists(NAMES) else {}


@app.get("/api/index")
def index():
    runs = layer_runs()
    return {"model": "Qwen/Qwen3.5-4B", "arch": ARCH,
            "layers": {str(l): os.path.basename(p) for l, p in sorted(runs.items())}, "status": rt.status}


@app.get("/api/units")
def units(layer: int):
    d = load_layer(layer)
    if d is None:
        return []
    nm = names()
    return [{"kind": u["kind"], "n": u["neuron"], "z": u.get("mass", u["z"]), "null_pct": u["null_pct"],
             "words": [w for w, _, _ in u["words"][:5]], "name": nm.get(f"L{layer}/N{u['neuron']}", "")}
            for u in d["units"]]


@app.get("/api/unit")
def unit(layer: int, n: int, kind: str = "neuron"):
    d = load_layer(layer)
    out = {"layer": layer, "n": n, "kind": kind, "name": names().get(f"L{layer}/N{n}", "")}
    if d is None:
        return out
    rec = next((u for u in d["units"] if u["neuron"] == n and u["kind"] == kind), None)
    rows = [d["rows"][i] for i in d["by_unit"].get((kind, n), [])]
    base = [r for r in d["rows"] if r["kind"] == "baseline"]
    keep = ("prompt", "dose", "rel_push", "text", "finish", "n_gen")
    neurons = [u for u in d["units"] if u["kind"] == "neuron"]
    out.update({
        "run": os.path.basename(d["path"]), "prompts": d["meta"]["prompts"],
        "rollouts": [{k: r[k] for k in keep} for r in rows],
        "baselines": [{k: r[k] for k in keep} for r in base],
        "stats": rec, "rank": next((i + 1 for i, u in enumerate(neurons) if u["neuron"] == n), None) if rec else None,
        "n_ranked": len(neurons),
    })
    return out


@app.post("/api/name")
async def set_name(req: Request):
    body = await req.json()
    nm = names()
    key = f"L{int(body['layer'])}/N{int(body['n'])}"
    if body.get("name", "").strip():
        nm[key] = body["name"].strip()
    else:
        nm.pop(key, None)
    with open(NAMES, "w") as f:
        json.dump(nm, f, indent=1, sort_keys=True)
    return {"ok": True}


@app.get("/api/embeddings")
def embeddings(layer: int):
    d = load_layer(layer)
    if d is None:
        return {}
    path = d["path"].replace(".jsonl", ".umap.json")
    return FileResponse(path) if os.path.exists(path) else JSONResponse({})


@app.get("/api/rollout")
def rollout(layer: int, i: int):
    r = load_layer(layer)["rows"][i]
    return {k: r[k] for k in ("kind", "neuron", "prompt", "dose", "rel_push", "text")}


@app.get("/api/connections")
async def connections(layer: int, n: int, k: int = 24, lens: int = 5):
    out = await asyncio.to_thread(rt.connections, layer, n, k, lens)
    words = await asyncio.to_thread(lambda: {l: ranked_words(l) for l in {c["layer"] for c in out["down"] + out["up"]}})
    nm = names()
    out = {side: [dict(c) for c in out[side]] for side in ("down", "up")}
    for c in out["down"] + out["up"]:
        c["name"] = nm.get(f"L{c['layer']}/N{c['n']}", "")
        c["words"] = words[c["layer"]].get(c["n"], [])
    return out


@app.get("/api/hover")
async def hover(layer: int, n: int):
    con = await asyncio.to_thread(rt.connections, layer, n, 16, 0)
    lens = await asyncio.to_thread(rt.lens, layer, n, 8)
    return {"lens": lens["up"], "lens_neg": lens["down"], "steer": lens.get("steer_up"), "steer_neg": lens.get("steer_down"),
            "con": con}


@app.get("/api/lens")
async def lens(layer: int, n: int):
    return await asyncio.to_thread(rt.lens, layer, n, 12)


@app.post("/api/reads")
async def reads(req: Request):
    b = await req.json()
    return await asyncio.to_thread(rt.read, int(b["layer"]), int(b["n"]), b["text"], b.get("chat", True))


@app.post("/api/steer")
async def steer(req: Request):
    b = await req.json()
    loop, q, cancel = asyncio.get_running_loop(), asyncio.Queue(), threading.Event()

    def work():
        try:
            # each row replays its own earlier replies; the user turns are shared
            rows = []
            for r in range(3):
                msgs = []
                for turn in b.get("history", []):
                    msgs += [{"role": "user", "content": turn["user"]}, {"role": "assistant", "content": turn["replies"][r]}]
                rows.append(msgs + [{"role": "user", "content": b["prompt"]}])
            # a stack of neurons (each with its own dose, the call's mode by default) or the single focused one
            d0, m0 = float(b.get("dose", 0.8)), b.get("mode", "add")
            pushes = [(int(x["layer"]), int(x["n"]), float(x.get("dose", d0)), x.get("mode", m0))
                      for x in b.get("neurons") or []]
            if not pushes:
                pushes = [(int(b["layer"]), int(b["n"]), float(b["dose"]), m0)]
            sysmsg = b.get("system")
            if sysmsg:
                rows = [[{"role": "system", "content": sysmsg}, *m] for m in rows]
            temp = float(b["temperature"]) if b.get("temperature") else None
            # opt-in live activity: per token, the top-k firing neurons of the asked rows as their own events
            # ({row, t: "", step, act}: a real row and an empty delta, so older clients and the bench just skip them)
            k, act = int(b.get("activity") or 0), {}
            if k > 0:
                act = {"activity": lambda step, row, top: loop.call_soon_threadsafe(
                           q.put_nowait, {"row": row, "t": "", "step": step, "act": top}),
                       "act_k": min(max(k, 8), 128),
                       "act_rows": tuple(int(r) for r in b.get("activity_rows") or [0])}
            for row, delta in rt.steer_multi(pushes, rows, int(b.get("max_new", 300)), cancel, temp, **act):
                loop.call_soon_threadsafe(q.put_nowait, {"row": row, "t": delta})
        except Exception as e:  # surfaced in the UI next to the chat
            loop.call_soon_threadsafe(q.put_nowait, {"row": None, "error": repr(e)})

    threading.Thread(target=work, daemon=True).start()

    async def events():
        try:
            while True:
                ev = await q.get()
                yield f"data: {json.dumps(ev)}\n\n"
                if ev["row"] is None:
                    break
                # activity events skip the poll (a disconnect still cancels through the finally below)
                if "act" not in ev and await req.is_disconnected():
                    cancel.set()
                    break
        finally:
            cancel.set()

    return StreamingResponse(events(), media_type="text/event-stream")


LENS_INDEX = os.path.join(ROOT, "explorer", "cache", "lens_index.npz")
_lens = {}


def lens_index():
    if "up" not in _lens and os.path.exists(LENS_INDEX):
        z = np.load(LENS_INDEX)
        _lens["up"], _lens["down"] = z["up"], z["down"]
    return _lens if "up" in _lens else None


@app.get("/api/search")
async def search(q: str, limit: int = 300):
    """Neurons whose logit lens (top 12 up or down) contains a token matching q (case-insensitive substring)."""
    idx = lens_index()
    q = q.strip().lower()
    if idx is None or len(q) < 2:
        return {"ready": idx is not None, "results": []}
    vocab = await asyncio.to_thread(rt.vocab_strings)
    match = np.array([i for i, t in enumerate(vocab) if q in t.lower()], dtype=np.int32)
    if not len(match):
        return {"ready": True, "results": [], "tokens": 0}
    out = []
    st = rt.steer_index()
    sources = [("+", idx["up"], "lens"), ("-", idx["down"], "lens")]
    if st is not None:
        sources += [("+", st["up"], "steer"), ("-", st["down"], "steer")]
    for sign, arr, src in sources:
        hit = np.isin(arr, match)  # [L, N, k]
        count = hit.sum(-1)
        best = np.where(hit, np.arange(arr.shape[-1]), 99).min(-1)
        for l, n in zip(*np.nonzero(count)):
            out.append((int(best[l, n]), -int(count[l, n]), int(l), int(n), sign, src))
    # one row per neuron and sign, keeping its best match across the two lenses
    seen, uniq = set(), []
    for row in sorted(out):
        if (row[2], row[3], row[4]) not in seen:
            seen.add((row[2], row[3], row[4])); uniq.append(row)
    out = uniq
    nm, res = names(), []
    sw = {l: swept(l) for l in {row[2] for row in out[:limit]}}
    for rank, _, l, n, sign, src in out[:limit]:
        arr = (idx if src == "lens" else st)["up" if sign == "+" else "down"]
        res.append({"layer": l, "n": n, "sign": sign, "rank": rank, "src": src, "name": nm.get(f"L{l}/N{n}", ""),
                    "swept": n in sw[l], "lens": [vocab[t] for t in arr[l, n, :6]]})
    return {"ready": True, "results": res, "tokens": int(len(match)), "total": len(out)}


ACT_SCALE = os.path.join(ROOT, "explorer", "cache", "act_scale.npz")
MAX_INFER_TOKENS = 1024  # [32, T, 9216] float32 activations plus scores: about 2.4 GB at this length


@app.get("/api/fires")
async def fires(text: str, limit: int = 300, chat: bool = False):
    """Neurons that fire hardest on `text` relative to their typical range over web text (activation / typical max,
    and activation / typical min for the negative side)."""
    if not os.path.exists(ACT_SCALE):
        return {"ready": False, "results": []}
    sc = np.load(ACT_SCALE)
    ids = await asyncio.to_thread(rt.chat_ids if chat else (lambda t: rt.tok(t).input_ids), text)
    acts = await asyncio.to_thread(rt.all_activations, ids)
    toks = [rt.tok.decode([t]) for t in ids]
    cand = []
    for l, (mx, amx, mn, amn) in acts.items():
        p = mx.numpy() / np.maximum(sc["pos"][l], 1e-3)
        q = mn.numpy() / np.minimum(sc["neg"][l], -1e-3)
        for arr, idx, sign, raw in ((p, amx.numpy(), "+", mx.numpy()), (q, amn.numpy(), "-", mn.numpy())):
            top = np.argpartition(-arr, limit)[:limit]
            cand += [(float(arr[n]), l, int(n), sign, int(idx[n]), float(raw[n])) for n in top]
    cand.sort(reverse=True)
    idx_lens, nm = lens_index(), names()
    vocab = await asyncio.to_thread(rt.vocab_strings)
    res = []
    for score, l, n, sign, t, raw in cand[:limit]:
        lens = [vocab[k] for k in (idx_lens["up"] if sign == "+" else idx_lens["down"])[l, n, :6]] if idx_lens else []
        res.append({"layer": l, "n": n, "sign": sign, "score": round(score, 2), "act": round(raw, 2), "token": toks[t],
                    "pos": t, "name": nm.get(f"L{l}/N{n}", ""), "lens": lens})
    return {"ready": True, "results": res, "tokens": toks}


_infer = {}


async def score_text(text, chat=False):
    """Every neuron's activation at every token of `text` ([L, T, N]) and its signed score: the positive side as a
    multiple of the neuron's typical max, the negative side of its typical min (act_scale.npz)."""
    ids = rt.chat_ids(text) if chat else rt.tok(text).input_ids
    ids = ids[:MAX_INFER_TOKENS]
    acts = await asyncio.to_thread(rt.full_activations, ids)  # [L, T, N]
    sc = np.load(ACT_SCALE)
    # a side the neuron barely reaches in calibration is scaled by at least a tenth of its other side
    pos = np.maximum(sc["pos"], np.maximum(0.1 * -sc["neg"], 1e-3))[:, None, :]
    neg = np.minimum(sc["neg"], np.minimum(-0.1 * sc["pos"], -1e-3))[:, None, :]
    score = np.where(acts >= 0, acts / pos, -(acts / neg)).astype(np.float32)
    if score.shape[1] > 1:
        score[:, 0, :] = 0  # position 0 is the attention-sink token, outside the range the scale was measured on
    return ids, acts, score


@app.post("/api/infer")
async def infer(req: Request):
    """Runs text through the model and keeps every neuron's activation at every token for /api/infer/view."""
    b = await req.json()
    if not os.path.exists(ACT_SCALE) or rt.tok is None:
        return {"ready": False}
    ids, acts, score = await score_text(b["text"], b.get("chat", False))
    units = rt.token_units(ids)
    unit_of = {i: k for k, (_, idx) in enumerate(units) for i in idx}
    _infer.clear()
    _infer.update(ids=ids, toks=[units[unit_of[i]][0] for i in range(len(ids))], acts=acts, score=score)
    # one chip per unit; clicking it selects the unit's last token (the one that completes the character)
    return {"ready": True, "tokens": [u[0] for u in units], "pos": [u[1][-1] for u in units]}


@app.get("/api/infer/view")
async def infer_view(pos: int = -1, sign: str = "both", lo: int = 0, hi: int = 99, min: float = 1.0, q: str = "",
                     limit: int = 300):
    """Neurons passing the filters for the last inferred text. pos = token index, or -1 for the strongest token.
    Returns the list plus every neuron's signed score at that position (base64 float16, layer-major) for the map."""
    if "score" not in _infer:
        return {"results": []}
    import base64
    S, A, toks = _infer["score"], _infer["acts"], _infer["toks"]  # [L, T, N]
    if pos >= S.shape[1]:
        return {"results": [], "total": 0}
    if pos >= 0:
        best, at = S[:, pos, :], np.full(S[:, 0, :].shape, pos)
    else:
        at = S.argmax(1) if sign == "+" else S.argmin(1) if sign == "-" else np.abs(S).argmax(1)  # [L, N]
        best = np.take_along_axis(S, at[:, None, :], 1)[:, 0, :]
    view = best.copy()
    keep = np.abs(best) >= min
    if sign == "+": keep &= best > 0
    if sign == "-": keep &= best < 0
    lay = np.arange(best.shape[0])[:, None]
    keep &= (lay >= lo) & (lay <= hi)
    idx = lens_index()
    vocab = await asyncio.to_thread(rt.vocab_strings)
    if q.strip() and idx is not None:
        ql = q.strip().lower()
        match = np.array([i for i, t in enumerate(vocab) if ql in t.lower()], dtype=np.int32)
        lens_hit = np.where(best[..., None] >= 0, np.isin(idx["up"][:, :, :6], match), np.isin(idx["down"][:, :, :6], match)).any(-1)
        keep &= lens_hit
    view[~keep] = 0
    ls, ns = np.nonzero(keep)
    order = np.argsort(-np.abs(best[ls, ns]))[:limit]
    nm, res = names(), []
    for o in order:
        l, n = int(ls[o]), int(ns[o]); sg = "+" if best[l, n] > 0 else "-"
        lens = [vocab[k] for k in (idx["up"] if sg == "+" else idx["down"])[l, n, :6]] if idx is not None else []
        t = int(at[l, n])
        res.append({"layer": l, "n": n, "sign": sg, "score": round(float(abs(best[l, n])), 2),
                    "act": round(float(A[l, t, n]), 2), "token": toks[t], "pos": t,
                    "name": nm.get(f"L{l}/N{n}", ""), "lens": lens})
    blob = base64.b64encode(view.astype(np.float16).tobytes()).decode()
    return {"results": res, "total": int(keep.sum()), "map": blob, "shape": list(view.shape)}


@app.get("/api/infer/paths")
def infer_paths(pos: int, top: int = 300, min: float = 1.0):
    """The circuit active at one token of the last inferred text: among the `top` neurons with the largest |score| there
    (at least `min`), every precomputed direct link whose two ends both fired. Same binary format as /api/web, with z
    signed by the push the writer's actual activation gives the reader (link sign x sign of the writer's score), so
    yellow / magenta read as "this firing pushed that one up / down"; hop is 1."""
    if "score" not in _infer or pos >= _infer["score"].shape[1]:
        return Response(b"", media_type="application/octet-stream", headers={"X-Web-Total": "0"})
    S = _infer["score"][:, pos, :]  # [L, N]
    flat = np.abs(S).ravel()
    cand = np.argpartition(-flat, min_k := builtins.min(top, flat.size - 1))[:min_k]
    cand = cand[flat[cand] >= min]
    N = S.shape[1]
    active = {(int(i // N), int(i % N)) for i in cand}
    ci = rt.con_index()
    rows = []
    for l, n in active:
        for i in range(ci["k"]):
            z = float(ci["down_z"][l, n, i])
            if z < -1e4:
                break
            t = (int(ci["down_layer"][l, n, i]), int(ci["down_n"][l, n, i]))
            if t in active:
                push = int(ci["down_sign"][l, n, i]) * (1 if S[l, n] > 0 else -1)
                rows.append((l, n, *t, round(z * 10) * push, 1))
    out = np.array(rows, np.int16).reshape(-1, 6)
    return Response(out.tobytes(), media_type="application/octet-stream",
                    headers={"X-Web-Total": str(len(out)), "X-Active": str(len(active))})


@app.get("/api/layout")
def layout(kind: str = "writes"):
    """Seriated grid layout: cell index (row * side + col) for every neuron, base64 int16, layer-major."""
    import base64
    path = os.path.join(ROOT, "explorer", "cache", f"layout_{kind}.npz")
    if not os.path.exists(path):
        return {"available": False}
    cell = np.load(path)["cell"].astype(np.int16)
    return {"available": True, "shape": list(cell.shape), "cell": base64.b64encode(cell.tobytes()).decode()}


def cone_links(layer, n, k=2, zmin=15, cap=20000, first=None, hops=None):
    """Connection cone of one neuron: everything it writes into, then what those write into, out to the last layer;
    and everything that writes into it, back to layer 0. Each side only moves one way. The first hop takes all of the
    neuron's top `first` links (all when None), later hops each node's top k with z >= zmin, up to `hops` hops; each side keeps at most cap / 2 links.
    Rows: src layer, src neuron, dst layer, dst neuron, z * sign, hop (positive downstream, negative upstream)."""
    ci = rt.con_index()
    out = []
    for side, step in (("down", 1), ("up", -1)):
        zs, ls, ns, sg = ci[f"{side}_z"], ci[f"{side}_layer"], ci[f"{side}_n"], ci[f"{side}_sign"]
        front, seen, hop, start = [(layer, n)], {(layer, n)}, 1, len(out)
        while front and len(out) - start < cap // 2 and (not hops or hop <= hops):
            nxt = []
            for a, b in front:
                for i in range((first or ci["k"]) if hop == 1 else k):
                    z = float(zs[a, b, i])
                    if z < -1e4 or (hop > 1 and z < zmin):
                        break
                    t = (int(ls[a, b, i]), int(ns[a, b, i]))
                    src, dst = ((a, b), t) if side == "down" else (t, (a, b))
                    out.append((*src, *dst, z * int(sg[a, b, i]), hop * step))
                    if t not in seen:
                        seen.add(t)
                        nxt.append(t)
            front, hop = nxt, hop + 1
        del out[start + cap // 2:]
    return out


@app.get("/api/web")
def web(layer: int, n: int, first: int = 5, k: int = 2, zmin: float = 15, cap: int = 20000, hops: int = 0):
    """The cone for the map. Binary int16, 6 per link: src layer, src neuron, dst layer, dst neuron,
    round(10 * z) * sign, hop."""
    rows = np.array([(*r[:4], round(r[4] * 10), r[5]) for r in cone_links(layer, n, k, zmin, cap, first, hops or None)], np.int16).reshape(-1, 6)
    return Response(rows.tobytes(), media_type="application/octet-stream", headers={"X-Web-Total": str(len(rows))})


_idf = {}


@app.get("/api/web/summary")
def web_summary(layer: int, n: int, first: int = 5, k: int = 2, top: int = 30, src: str = "steer", hops: int = 0):
    """What the neuron's web says, in tokens. Each neuron in the cone is read on the pole the chain drives (signs
    multiplied along the path from the open neuron), and its steer (or logit) lens tokens are pooled, merged across
    case and leading space, and weighted by link z / max z times 0.8 per hop past the first, times log lift: the share
    of web neurons carrying the token over its share among all neurons of the same layers, so tokens a layer's neurons
    commonly carry (`(`, ` a` in early layers) sink and tokens the web carries no more often than chance drop out.
    Downstream gives
    `toward` (the driven pole) and `away` (the opposite pole); upstream gives `fed_by` (the pole that excites it) and
    `held_by` (the pole that inhibits it)."""
    st = rt.steer_index() if src == "steer" else None
    lens = st if st is not None else rt.lens_index()
    v = rt.vocab_strings()
    key_src = "steer" if st is not None else "logit"
    if _idf.get("src") != key_src:
        L, N = lens["up"].shape[:2]
        df = np.stack([np.bincount(np.concatenate([lens["up"][l, :, :8].ravel(), lens["down"][l, :, :8].ravel()]),
                                   minlength=len(v)) for l in range(L)])
        _idf.update(src=key_src, base=((df + 1) / (2 * N)).astype(np.float32))  # [layer, token]
    base = _idf["base"]
    sign, zmax = {(layer, n): 1}, 1.0
    links = cone_links(layer, n, k, 15, 20000, first, hops or None)
    for sl, sn, dl, dn, z, hop in links:
        zmax = max(zmax, abs(z))
    reach = {}
    for sl, sn, dl, dn, z, hop in links:
        near, far = ((sl, sn), (dl, dn)) if hop > 0 else ((dl, dn), (sl, sn))
        if far in sign or near not in sign:
            continue
        sign[far] = sign[near] * (1 if z > 0 else -1)
        reach[far] = (hop, abs(z))

    def pool(nodes, flip):
        acc = {}
        for (l, m), (hop, z) in nodes:
            w = z / zmax * 0.8 ** (abs(hop) - 1)
            pole = "up" if sign[(l, m)] * flip > 0 else "down"
            for t in lens[pole][l, m, :8]:
                s = v[t]
                key = s.strip().lower()
                if not key or "\ufffd" in key:
                    continue
                a = acc.setdefault(key, {"tok": s.strip(), "n": 0, "w": 0.0, "forms": {}})
                a["n"] += 1
                a["w"] += w
                a["base"] = a.get("base", 0.0) + float(base[l, t])
                a["forms"][s.strip()] = a["forms"].get(s.strip(), 0) + 1
        for a in acc.values():
            lift = (a["n"] / max(1, len(nodes))) / (a["base"] / a["n"])
            a["score"] = a["w"] * np.log(lift) if lift > 1 else 0.0
        out = sorted((a for a in acc.values() if a["score"] > 0), key=lambda a: -a["score"])[:top]
        return [{"tok": max(a["forms"], key=a["forms"].get), "n": a["n"], "w": round(float(a["score"]), 2)} for a in out]

    down = [(key, val) for key, val in reach.items() if val[0] > 0]
    up = [(key, val) for key, val in reach.items() if val[0] < 0]
    return {"src": "steer" if st is not None else "logit", "links": len(links),
            "down": {"neurons": len(down), "toward": pool(down, 1), "away": pool(down, -1)},
            "up": {"neurons": len(up), "fed_by": pool(up, 1), "held_by": pool(up, -1)}}


brain = Brain(rt)


@app.post("/api/brain/chat")
async def brain_chat(req: Request):
    """One natural-language steering request. Body {goal, options}. Streams NDJSON events: status, round, plan,
    candidates, screen, thought, tool, result, trial, blend, done, error, end; and for the live map: testing (the
    trial about to run), fires (the neurons that fired hardest in its replies), screened (one candidate's verdict),
    join (one combine decision) and lk_pool (one likelihood-scan pass). Any event dict is forwarded as is."""
    b = await req.json()
    loop, q, cancel = asyncio.get_running_loop(), asyncio.Queue(), threading.Event()

    class Cancelled(BaseException):  # not an Exception: nothing inside the run may swallow it
        pass

    def emit(ev):
        if cancel.is_set():  # the client went away: unwind the run at its next event (its finally frees the brain)
            raise Cancelled()
        loop.call_soon_threadsafe(q.put_nowait, ev)

    def work():
        try:
            brain.run(str(b.get("goal", "")), emit, b.get("options"))
        except Cancelled:
            pass
        except Exception:
            import traceback
            loop.call_soon_threadsafe(q.put_nowait, {"t": "error", "error": traceback.format_exc()[-1200:]})
        finally:
            loop.call_soon_threadsafe(q.put_nowait, {"t": "end"})

    threading.Thread(target=work, daemon=True).start()

    async def events():
        try:
            while True:
                ev = await q.get()
                yield json.dumps(ev) + "\n"
                if ev.get("t") == "end":
                    break
        finally:
            cancel.set()

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/api/brain/stop")
async def brain_stop(req: Request):
    """Ends the running brain at its next trial. Body {keep}: true hands over the best recipe measured so far;
    false means the user already took a trial's recipe and nothing more is handed over."""
    b = await req.json()
    return {"stopping": brain.request_stop(bool(b.get("keep", True)))}


@app.post("/api/brain/judge")
async def brain_judge(req: Request):
    """The brain LLM's judgement of replies against a wish. Body {wish, texts, prompts?} (prompts: the user prompt
    each reply answered). Returns {scores: [[style, coherent], ...]}, each 0-10, every reply judged on its own."""
    b = await req.json()

    def work():
        brain.ensure_loaded(lambda ev: None)
        return brain.judge(str(b.get("wish", "")), [str(t) for t in b.get("texts", [])], b.get("prompts"))

    try:
        return {"scores": [list(s) for s in await asyncio.to_thread(work)]}
    except Exception as e:
        return JSONResponse({"error": repr(e)}, status_code=500)


@app.post("/api/brain/plan")
async def brain_plan(req: Request):
    """The brain LLM's reading of a wish (words to say, behaviour to move away from, example replies)."""
    b = await req.json()

    def work():
        brain.ensure_loaded(lambda ev: None)
        return brain.plan(str(b.get("wish", "")))

    try:
        return await asyncio.to_thread(work)
    except Exception as e:
        return JSONResponse({"error": repr(e)}, status_code=500)


@app.get("/api/status")
def status():
    return {"status": rt.status}


app.mount("/", StaticFiles(directory=os.path.join(ROOT, "explorer", "static"), html=True), name="static")

@app.on_event("startup")
def preload():
    def go():
        if NO_MODEL:
            rt.status = "no model"
            rt.vocab_strings()
            return
        rt.ensure_loaded()
        if not os.path.exists(LENS_INDEX):
            os.makedirs(os.path.dirname(LENS_INDEX), exist_ok=True)
            rt.build_lens_index(LENS_INDEX)
        rt.vocab_strings()
        if not os.path.exists(ACT_SCALE):
            rt.build_act_scale(ACT_SCALE)
    threading.Thread(target=go, daemon=True).start()


if __name__ == "__main__":
    if os.name == "nt":  # Windows asyncio subprocesses and pipes need the Proactor loop
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    uvicorn.run(app, host="127.0.0.1", port=PORT)
