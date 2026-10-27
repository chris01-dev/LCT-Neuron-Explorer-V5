// live.js — the map's live layer: what the steering brain is doing right now (scanning the pool, testing a neuron,
// rating it, keeping it) and which neurons fire in the chat as each token is generated. Fed by the "lct:live" events
// chat.js dispatches (app.js forwards them to handle); draws through map.setLive at most once per frame, from a
// ~30 fps loop that runs only while something moves and stops itself when the picture is still.
// Colours keep the app's meanings: amber = pushed up / +dose, magenta = pushed down / -dose, white = the focus.

const C = {
  pool: [0.07, 0.33, 0.45],      // proposed by the brain: dim cyan
  scan: [0.3, 0.68, 1.5],        // likelihood scan: cyan-blue sparkle
  test: [1.45, 1.45, 1.5],       // the neuron under test, pulsing
  rest: [0.72, 0.76, 0.82],      // the rest of the tested recipe
  beam: [0.36, 0.39, 0.44],      // focus -> recipe
  fireUp: [0.55, 1.12, 1.22],    // fires positive: light cyan
  fireDown: [0.7, 0.3, 1.32],    // fires negative: violet
  good: [0.2, 1.08, 0.4], inert: [0.27, 0.29, 0.31], broke: [1.25, 0.13, 0.09],
  up: [1.2, 0.84, 0.32], down: [1.12, 0.26, 0.92],  // the recipe
  drop: [0.62, 0.07, 0.05],
};
const LEGEND = {
  brain: [[["#2b8db0"], "proposed"], [["#5ab4ff"], "scan"], [["#ffffff"], "testing"], [["#9eefff", "#b06cff"], "fires +/−"],
    [["#3fdc6e"], "useful"], [["#6d7378"], "inert"], [["#ff3b2a"], "broke"], [["#ffb547", "#ff47d6"], "recipe +/−"]],
  chat: [[["#ffb547", "#ff47d6"], "pushed +/−"], [["#9eefff"], "fires +"], [["#b06cff"], "fires −"]],
  blend: [[["#ffb547"], "pushed up"], [["#ff47d6"], "pushed down"]],
};
const STAGE = { plan: "PLAN", discover: "DISCOVER", screen: "SCREEN", combine: "COMBINE", "tune and verify": "TUNE" };
const PTS = 8000, LINES = 1024, FX = 6000;
const SCAN = 1, FIRE = 2, KEPT = 3, DROP = 4, PING = 5;  // transient effects

const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const key = (l, n) => l * 65536 + n;
const isNeuron = b => b && Number.isInteger(b.layer) && Number.isInteger(b.n);
const id = b => `L${b.layer} N${b.n}`;
const num = d => String(Math.round(Math.abs(d) * 100) / 100);
const signed = d => (d < 0 ? "−" : "+") + num(d);
const cut = (s, k = 110) => { s = String(s ?? "").replace(/\s+/g, " ").trim(); return s.length > k ? s.slice(0, k - 1) + "…" : s; };
const now = () => performance.now() / 1000;

export function createLive(map, hud) {
  if (!map || !map.setLive) return { handle() {}, setOn() {}, on: false };
  hud = hud || document.createElement("div");
  let on = true, raf = 0, last = -1e9, minDt = 33;
  // reused every frame: setLive draws the first np / nl of these
  const P = Array.from({ length: PTS }, () => ({ layer: 0, n: 0, r: 0, g: 0, b: 0, size: 1, w: 0, c: null }));
  const LN = Array.from({ length: LINES }, () => ({ a: [0, 0], b: [0, 0], c: [0, 0, 0] }));
  let np = 0, nl = 0, drawn = false;
  // links between lit neurons of neighbouring layers: per layer the lit dots of this frame (indices into P), the
  // LINK_K brightest of them joined to their nearest lit neighbour on the sheet of the layer below
  const LINK_K = 4, LB = Array.from({ length: 32 }, () => []), TOP = Array.from({ length: 32 }, () => []), mix = [0, 0, 0];
  const fx = new Map(), spare = [];  // transient effects by kind + neuron, and recycled effect objects

  const S = {
    brain: false, ended: null, faded: false,  // a run is streaming / how the last one ended / a chat came after it
    stage: "", what: "", tail: "", sawTest: false, iBase: {},
    pool: new Map(), verdict: new Map(), test: null, brainRecipe: null, picked: false,
    chat: false, chatEnded: false, chatBlend: null, tokens: 0, firing: 0, peak: null, tokT: -1e9,
  };
  let recipe = [], recipeMode = "add", recipeKeys = new Set();

  // ---------- state ----------
  function setRecipe(list, mode = "add") {
    recipe = (Array.isArray(list) ? list : []).filter(b => isNeuron(b) && Number.isFinite(+b.dose))
      .map(b => ({ layer: b.layer, n: b.n, dose: +b.dose })).sort((a, b) => a.layer - b.layer || a.n - b.n);
    recipeMode = mode === "mul" ? "mul" : "add";
    recipeKeys = new Set(recipe.map(b => key(b.layer, b.n)));
    recipe.forEach((b, i) => {
      b.c = b.dose < 0 ? C.down : C.up;
      b.size = recipeMode === "mul" ? 1.5 : 1.05 + 0.85 * Math.sqrt(Math.min(Math.abs(b.dose), 3));
      if (i) { const a = recipe[i - 1]; b.lc = [0, 1, 2].map(j => (a.c[j] + b.c[j]) / 2); }  // constellation line from the previous
    });
  }
  const sameBlend = (a, b) => !!a && !!b && a.length === b.length && a.every((x, i) => x.layer === b[i].layer && x.n === b[i].n && +x.dose === +b[i].dose);
  function spark(kind, l, n, t0, dur, amp, sgn = 1) {
    if (!on || !Number.isInteger(l) || !Number.isInteger(n)) return;
    const k = kind * 16777216 + key(l, n), t = now();
    let e = fx.get(k);
    if (e) {
      // already lit: straight back to the peak (restarting the attack would strobe a neuron that fires every
      // token), keeping the stronger amplitude while the old flash is still bright; not lit yet: the earlier start
      const x = (t - e.t0) / e.dur;
      if (x >= 0) { if (x < 0.3) amp = Math.max(amp, e.amp); t0 = t - 0.1 * dur; } else t0 = Math.min(t0, e.t0);
    } else { if (fx.size >= FX) return; e = spare.pop() || {}; fx.set(k, e); }
    e.kind = kind; e.layer = l; e.n = n; e.t0 = t0; e.dur = dur; e.amp = amp; e.sgn = sgn;
  }
  const fireAmp = s => Math.min(1.8, 0.3 + 0.8 * Math.sqrt(Math.abs(s) / 100));
  function fires(list, t, sweep, k = 1) {
    if (!Array.isArray(list)) return;
    for (const f of list) {
      if (!Array.isArray(f) || !Number.isFinite(f[2])) continue;
      // brain trials (one summary per reply) sweep through the layers; chat tokens only lag a little per layer
      const l = Math.min(Math.max(f[0], 0), 31) || 0, delay = sweep ? l / 31 * 0.45 + Math.random() * 0.08 : l * 0.004;
      spark(FIRE, f[0], f[1], t + delay, sweep ? 0.75 : 0.7, fireAmp(f[2]) * k, f[2] < 0 ? -1 : 1);
    }
  }
  function resetBrain() {
    S.pool.clear(); S.verdict.clear(); S.test = null; S.brainRecipe = null; S.picked = false;
    S.ended = null; S.faded = false; S.sawTest = false; S.iBase = {}; S.stage = "BRAIN"; S.what = ""; S.tail = "";
    for (const [k, e] of fx) if (e.kind !== FIRE || !S.chat) { fx.delete(k); spare.push(e); }
  }
  function startBrain(goal) {
    resetBrain();
    S.brain = true; S.chatEnded = false; S.what = goal ? `“${cut(goal, 70)}”` : "";
    if (!S.chat) setRecipe([]);  // the run builds its own recipe; the chat's blend comes back when it ends
  }
  function endBrain() {
    if (!S.brain && !S.ended) return;  // no run to end
    if (!S.ended) {  // cancelled mid-run: nothing finished, so the run's marks go and the chat's blend shows again
      resetBrain(); S.ended = "cancelled"; S.what = "cancelled";
    }
    S.brain = false; S.test = null;
    if (S.chatBlend) setRecipe(S.chatBlend.blend, S.chatBlend.mode);
  }
  const verdictOf = v => v.utility >= 1 || v.strong ? "good" : v.utility <= -1 ? "broke" : "inert";
  const VERDICT = { good: "useful", inert: "inert", broke: "broke the replies" };
  function stageOf(ev) {
    const ph = String(ev.phase || "trial");
    if (Number.isInteger(ev.i) && Number.isInteger(ev.of)) {
      // i may count from 0 or from 1: the first candidate of a stage tells which
      if (S.iBase[ph] === undefined) S.iBase[ph] = ev.i === 0 ? 1 : 0;
      if (ev.i === 0) S.iBase[ph] = 1;
      const i = Math.min(ev.i + S.iBase[ph], ev.of);
      if (ph === "screen" || ph === "screen_b") return `${ph === "screen" ? "SCREEN" : "RESCREEN"} ${i}/${ev.of}`;
    }
    return ph === "screen" ? "SCREEN" : ph === "screen_b" ? "RESCREEN" : ph === "tune" ? "TUNE" : ph === "verify" ? "VERIFY"
      : ["lead", "join", "join_retry", "join_nudge"].includes(ph) ? "COMBINE" : ph.toUpperCase();
  }
  function testing(ev) {
    const rec = (Array.isArray(ev.recipe) ? ev.recipe : []).filter(isNeuron), f = isNeuron(ev.focus) ? ev.focus : null;
    S.sawTest = true;
    S.test = { recipe: rec, focus: f, t0: now() };
    if (f && !S.verdict.has(key(f.layer, f.n)) && !S.pool.has(key(f.layer, f.n))) S.pool.set(key(f.layer, f.n), { layer: f.layer, n: f.n });
    if (f) spark(PING, f.layer, f.n, now(), 0.55, 1);
    S.stage = stageOf(ev);
    const ph = ev.phase, d = f ? (ev.both ? "±" + num(f.dose) : signed(f.dose)) : "";
    S.what = ph === "screen" || ph === "screen_b" ? `${f ? `${id(f)} ${d}` : ""}${ev.rung ? ` · rung ${ev.rung + 1}` : ""}`
      : ph === "lead" ? `lead ${f ? `${id(f)} ${d}` : ""}`
      : ph === "join" || ph === "join_retry" || ph === "join_nudge" ? `+ ${f ? `${id(f)} ${d}` : "?"}${ph === "join_retry" ? " · retry" : ph === "join_nudge" ? " · nudged" : ""}`
      : ph === "tune" ? `${rec.length} neurons · doses ×${Number.isFinite(ev.scale) ? num(ev.scale) : "?"}`
      : ph === "verify" ? `${rec.length} neurons · held-out prompts`
      : `${rec.length} neurons`;
    S.tail = ev.cached ? "cached" : "testing";
  }
  function brainEvent(ev) {
    if (!ev || typeof ev !== "object") return;
    const t = ev.t, at = now();
    if (!S.brain && !["done", "error", "end"].includes(t)) startBrain(null);  // a stream that began before we listened
    switch (t) {
      case "round":
        S.stage = STAGE[ev.label] || (Number.isInteger(ev.i) ? `ROUND ${ev.i}` : "BRAIN"); S.what = ""; S.tail = ""; S.test = null;
        break;
      case "status":
        // the structured events say it better once trials run; before that the status line is the action
        if (!S.sawTest || /^(PLAN|DISCOVER|BRAIN|ROUND)/.test(S.stage)) { S.what = cut(ev.text); S.tail = ""; }
        break;
      case "plan": {
        const say = (ev.say || []).slice(0, 5).join(", "), avoid = (ev.avoid || []).slice(0, 3).join(", ");
        S.stage = "PLAN"; S.what = `says ${say || "—"}${avoid ? ` · avoids ${avoid}` : ""}`; S.tail = "";
        break;
      }
      case "lk_pool": {
        const list = Array.isArray(ev.list) ? ev.list : [];
        if (!list.length) break;
        const top = Math.max(1, ...list.map(f => (Array.isArray(f) && f[2] > 0 ? f[2] : 0)));
        for (const f of list) {
          if (!Array.isArray(f)) continue;
          const a = Math.max(0, f[2]) / top;
          spark(SCAN, f[0], f[1], at + Math.random() * 0.55, 1.2, 0.2 + 1.1 * a);
        }
        S.stage = "DISCOVER"; S.what = `likelihood scan · ${list.length} neurons${Number.isFinite(ev.dose) ? ` at ±${num(ev.dose)}` : ""}`; S.tail = "";
        break;
      }
      case "candidates": {
        const list = (Array.isArray(ev.list) ? ev.list : []).filter(isNeuron);
        if (!list.length) break;
        for (const c of list) {
          S.pool.set(key(c.layer, c.n), { layer: c.layer, n: c.n });
          spark(SCAN, c.layer, c.n, at + Math.random() * 0.3, 0.9, 0.7);
        }
        S.stage = "DISCOVER"; S.what = `${S.pool.size} proposed`; S.tail = "";
        break;
      }
      case "testing":
        if (isNeuron(ev.focus) || Array.isArray(ev.recipe) && ev.recipe.some(isNeuron)) testing(ev);
        break;
      case "fires":
        fires(ev.up, at, true); fires(ev.down, at, true, 0.6);
        if (Array.isArray(ev.up) && (S.tail === "testing" || S.tail === "cached")) S.tail = `${ev.up.length} fired · judging`;
        break;
      case "trial":
        // an older brain sends no testing events: the judged trial is the best "testing now" there is
        if (!S.sawTest && Array.isArray(ev.recipe)) {
          const rec = ev.recipe.filter(isNeuron);
          S.test = { recipe: rec, focus: rec.length ? rec[rec.length - 1] : null, t0: at };
        }
        if (ev.score && Number.isFinite(ev.score.up)) S.tail = `▲ ${ev.score.up} vs 0 ${ev.score.zero}${Number.isFinite(ev.score.down) ? ` · ▼ ${ev.score.down}` : ""}`;
        break;
      case "screened": {
        if (!isNeuron(ev)) break;
        const kind = verdictOf(ev);
        S.verdict.set(key(ev.layer, ev.n), { layer: ev.layer, n: ev.n, kind, t0: at });
        S.what = `${id(ev)}${Number.isFinite(ev.dose) ? ` ${signed(ev.dose)}` : ""} → ${VERDICT[kind]}`;
        S.tail = Number.isFinite(ev.utility) ? `utility ${ev.utility}` : "";
        break;
      }
      case "screen":  // the ranked list at the end of the stage: rates whatever no screened event did
        for (const c of ev.list || []) if (isNeuron(c) && !S.verdict.has(key(c.layer, c.n)))
          S.verdict.set(key(c.layer, c.n), { layer: c.layer, n: c.n, kind: verdictOf(c), t0: -1e9 });
        break;
      case "join": {
        if (!isNeuron(ev)) break;
        if (ev.kept) spark(KEPT, ev.layer, ev.n, at, 1.0, 1, ev.dose < 0 ? -1 : 1);
        else spark(DROP, ev.layer, ev.n, at, 0.5, 1);
        S.stage = "COMBINE"; S.what = `+ ${id(ev)}${Number.isFinite(ev.dose) ? ` ${signed(ev.dose)}` : ""} → ${ev.kept ? "kept" : "dropped"}`;
        S.tail = Number.isFinite(ev.value) ? `${ev.value.toFixed(1)} vs ${Number.isFinite(ev.best) ? ev.best.toFixed(1) : "?"}${ev.retry ? ` · again ${ev.retry}` : ""}` : "";
        break;
      }
      case "blend":
        if (!S.picked && Array.isArray(ev.blend)) { S.brainRecipe = ev.blend; setRecipe(ev.blend, "add"); }
        break;
      case "done":
        S.brain = false; S.test = null;
        S.ended = ev.stopped ? "stopped" : ev.verified ? "verified" : ev.handed_over ? "unverified" : "nothing";
        S.stage = ev.stopped ? "STOPPED" : "DONE";
        S.what = ev.stopped ? (S.picked ? "your pick is in the blend" : ev.handed_over ? "best recipe so far kept" : "nothing measured yet")
          : ev.verified ? `✓ verified · ${recipe.length} neurons`
          : ev.handed_over ? `not verified · ${recipe.length} neurons` : "nothing found";
        S.tail = "";
        break;
      case "error":
        S.brain = false; S.test = null; S.ended = "error"; S.stage = "ERROR"; S.what = cut(String(ev.error || "").split("\n")[0], 90); S.tail = "";
        break;
      case "end":
        S.test = null;
        break;
      case "thought": S.what = cut(ev.text); S.tail = ""; break;
      case "tool": S.what = `${cut(ev.name, 40)}(…)`; S.tail = ""; break;
      default: break;  // anything new: ignored until it gets a look
    }
  }
  function handle(d) {
    if (!d || typeof d !== "object") return;
    try {
      const at = now();
      switch (d.kind) {
        case "blend":
          S.chatBlend = { blend: Array.isArray(d.blend) ? d.blend : [], mode: d.mode };
          if (S.brain) {
            // during a run every brain blend comes back here at once (chat takes it); one that differs is the user's
            // own pick, and later brain blends no longer reach the chat, so they no longer show either
            if (!sameBlend(S.chatBlend.blend, S.brainRecipe)) S.picked = true;
            setRecipe(S.chatBlend.blend, S.chatBlend.mode);
          } else {
            if (S.ended) S.faded = true;  // changed after a run: the run's marks step back
            if (!S.chat) { S.chatEnded = false; setRecipe(S.chatBlend.blend, S.chatBlend.mode); }  // while streaming, what was sent stays
          }
          break;
        case "chat-start":
          S.chat = true; S.chatEnded = false; S.tokens = 0; S.firing = 0; S.peak = null;
          if (S.ended) S.faded = true;  // the last run's marks step back behind the chat
          if (!S.brain) setRecipe(d.blend, d.mode);
          break;
        case "act":
          if (d.row !== 0 || !Array.isArray(d.act)) break;  // the steered row only
          S.tokens = Math.max(S.tokens, (Number.isInteger(d.step) ? d.step : S.tokens) + 1);
          S.firing = d.act.length; S.peak = d.act[0] || null; S.tokT = at;
          fires(d.act, at, false);
          break;
        case "chat-end":
          S.chat = false; S.chatEnded = true;
          if (!S.brain && S.chatBlend) setRecipe(S.chatBlend.blend, S.chatBlend.mode);
          break;
        case "brain-start": startBrain(d.goal); break;
        case "brain": brainEvent(d.ev); break;
        case "brain-end": endBrain(); break;
        default: break;
      }
    } catch (e) { console.warn("live:", e); }
    kick();
  }

  // ---------- drawing ----------
  // w > 0: the dot takes part in the layer-to-layer links with that brightness (halos and rings do not)
  function pt(l, n, c, k, size, w = 0) {
    if (np >= PTS) return;
    const p = P[np++];
    p.layer = l; p.n = n; p.r = c[0] * k; p.g = c[1] * k; p.b = c[2] * k; p.size = size; p.w = w; p.c = c;
    if (w > 0.06 && l >= 0 && l < LB.length) LB[l].push(np - 1);
  }
  // per layer the LINK_K brightest distinct neurons, then each joined to the nearest of the layer below's on the
  // sheet; the line takes both dots' colours and the brightness of the weaker one
  function links() {
    for (let l = 0; l < LB.length; l++) {
      const b = LB[l], top = TOP[l];
      top.length = 0;
      if (!b.length) continue;
      b.sort((i, j) => P[j].w - P[i].w);
      for (const i of b) {
        if (top.length >= LINK_K) break;
        if (!top.some(j => P[j].n === P[i].n)) top.push(i);
      }
      for (const i of top) P[i].yz = map.sheetYZ(P[i].layer, P[i].n);
    }
    for (let l = 1; l < TOP.length; l++) {
      const below = TOP[l - 1];
      if (!below.length) continue;
      for (const i of TOP[l]) {
        const a = P[i];
        let best = -1, bd = Infinity;
        for (const j of below) {
          const d = (P[j].yz[0] - a.yz[0]) ** 2 + (P[j].yz[1] - a.yz[1]) ** 2;
          if (d < bd) { bd = d; best = j; }
        }
        const z = P[best], wa = Math.min(a.w, 2), wz = Math.min(z.w, 2);
        for (let k = 0; k < 3; k++) mix[k] = (a.c[k] * wa + z.c[k] * wz) / (wa + wz);
        line(z, a, mix, Math.min(0.9, 0.42 * Math.pow(Math.min(wa, wz), 0.8)));
      }
    }
  }
  function line(a, b, c, k) {
    if (nl >= LINES) return;
    const q = LN[nl++];
    q.a[0] = a.layer; q.a[1] = a.n; q.b[0] = b.layer; q.b[1] = b.n; q.c[0] = c[0] * k; q.c[1] = c[1] * k; q.c[2] = c[2] * k;
  }
  // one frame of the picture; returns 2 while something moves fast, 1 when only the pool breathes, 0 when still
  function compose(t) {
    np = nl = 0;
    for (const b of LB) b.length = 0;
    let moving = 0;
    const dim = S.faded ? 0.35 : 1;
    // the recipe: solid dots sized by |dose|, a faint halo, a constellation line in layer order; a token in the
    // chat makes them beat once
    const beat = S.chat ? Math.exp(-(t - S.tokT) * 7) : 0;
    if (beat > 0.02) moving = 2;
    for (let i = 0; i < recipe.length; i++) {
      const b = recipe[i];
      pt(b.layer, b.n, b.c, 1 + 0.45 * beat, b.size, 1 + 0.45 * beat);
      pt(b.layer, b.n, b.c, 0.1 + 0.12 * beat, b.size * 2.4);
      if (i) line(recipe[i - 1], b, b.lc, 0.22 + 0.2 * beat);
    }
    // testing now: the focus pulses white at 2 Hz with a halo and thin beams to the rest of the recipe; with no
    // focus (tune, verify) the whole recipe pulses
    if (S.test) {
      moving = 2;
      const p = 0.5 + 0.5 * Math.sin((t - S.test.t0) * Math.PI * 4), f = S.test.focus;
      for (const r of S.test.recipe) {
        if (f && r.layer === f.layer && r.n === f.n) continue;
        const k = f ? 0.6 + 0.15 * p : 0.45 + 0.65 * p;
        pt(r.layer, r.n, C.rest, k, f ? 1.3 : 1.35 + 0.55 * p, k);
        if (f) line(f, r, C.beam, 0.55 + 0.45 * p);
      }
      if (f) { pt(f.layer, f.n, C.test, 0.85 + 0.85 * p, 1.6 + 0.9 * p, 1.2 + 0.6 * p); pt(f.layer, f.n, C.test, 0.14 + 0.2 * p, 4.4 + 1.6 * p); }
    }
    // verdicts (a short pop when they land), then the proposed pool, breathing while the run goes on
    for (const [k, v] of S.verdict) {
      if (recipeKeys.has(k)) continue;
      const x = (t - v.t0) / 0.6, pop = x < 1 ? 1 - x : 0;
      if (pop) moving = 2;
      pt(v.layer, v.n, C[v.kind], (0.85 + 1.4 * pop) * dim, 1.15 + 0.9 * pop, (0.35 + 1.2 * pop) * dim);
    }
    for (const [k, p] of S.pool) {
      if (recipeKeys.has(k) || S.verdict.has(k)) continue;
      pt(p.layer, p.n, C.pool, (S.brain ? 0.75 + 0.25 * Math.sin(t * 2.4 + (k % 97)) : 0.7) * dim, 1, 0.2 * dim);
    }
    if (S.brain && S.pool.size && !moving) moving = 1;
    // transients
    for (const [k, e] of fx) {
      const x = (t - e.t0) / e.dur;
      if (x >= 1) { fx.delete(k); spare.push(e); continue; }
      moving = 2;
      if (x < 0) continue;
      if (e.kind === FIRE) {
        const env = x < 0.1 ? x / 0.1 : Math.pow((1 - x) / 0.9, 1.5);
        pt(e.layer, e.n, e.sgn < 0 ? C.fireDown : C.fireUp, e.amp * env, 0.8 + 0.6 * e.amp * env, e.amp * env);
      } else if (e.kind === SCAN) {
        const env = (x < 0.12 ? x / 0.12 : Math.pow((1 - x) / 0.88, 2)) * (0.72 + 0.28 * Math.sin(x * 38 + (e.n % 7)));
        pt(e.layer, e.n, C.scan, e.amp * env * dim, 0.85 + 0.5 * e.amp * env, 0.6 * e.amp * env * dim);
      } else if (e.kind === KEPT) {
        const c = e.sgn < 0 ? C.down : C.up, y = 1 - x;
        pt(e.layer, e.n, c, 2.1 * Math.pow(y, 1.5), 1.3 + 1.4 * y, 2.1 * Math.pow(y, 1.5));
        pt(e.layer, e.n, c, 0.45 * y * y, 1.6 + 5.5 * x);
      } else if (e.kind === DROP) {
        pt(e.layer, e.n, C.drop, Math.sin(Math.PI * x), 1.6, 0.4 * Math.sin(Math.PI * x));
      } else if (e.kind === PING) {
        pt(e.layer, e.n, C.test, 0.5 * (1 - x) * (1 - x), 1.4 + 4.5 * x);
      }
    }
    links();
    if (np || nl || drawn) { map.setLive(P, LN, np, nl); drawn = np > 0 || nl > 0; }
    return moving;
  }
  function frame(ts) {
    raf = 0;
    if (!on) return;
    if (ts - last < minDt - 3) { raf = requestAnimationFrame(frame); return; }
    last = ts;
    const moving = compose(ts / 1000);
    paintHud(moving);
    minDt = moving === 2 ? 1000 / 30 : 1000 / 15;
    if (moving) raf = requestAnimationFrame(frame);
  }
  function kick() {
    if (!on) return;
    minDt = Math.min(minDt, 1000 / 30);
    if (!raf) { last = -1e9; raf = requestAnimationFrame(frame); }
  }

  // ---------- HUD ----------
  hud.innerHTML = `<div class="lh-main"></div><div class="lh-legend"></div>`;
  const hudMain = hud.firstElementChild, hudLeg = hud.lastElementChild;
  let hudHTML = "", legKind = "", hudCls = "";
  const lineHTML = (stage, ...parts) => `<div class="lh-line"><b>${esc(stage)}</b>${parts.filter(Boolean).map(p => `<span class="lh-sep"> · </span>${esc(p)}`).join("")}</div>`;
  function paintHud(moving) {
    const lines = [];
    let leg = "";
    if (S.brain || (S.ended && !S.faded)) {
      lines.push(lineHTML(S.stage || "BRAIN", S.what, S.tail));
      leg = S.ended === "cancelled" ? (recipe.length ? "blend" : "") : "brain";
    }
    if (S.chat || (S.chatEnded && !S.brain)) {
      const pk = Array.isArray(S.peak) && Number.isFinite(S.peak[2]) ? `peak L${S.peak[0]} N${S.peak[1]} ${signed(S.peak[2] / 100)}×` : "";
      lines.push(S.chat ? lineHTML("CHAT", `token ${S.tokens}`, `${S.firing} firing`, pk) : lineHTML("CHAT", `${S.tokens} tokens`, "done"));
      leg = leg || "chat";
    }
    if (!lines.length && recipe.length) {
      lines.push(lineHTML("BLEND", `${recipe.length} neuron${recipe.length > 1 ? "s" : ""}`, recipeMode === "mul" ? "scaled in ▲" : "pushed in ▲"));
      leg = "blend";
    }
    const html = lines.join(""), cls = `${lines.length ? "on" : ""}${moving || S.brain || S.chat ? "" : " idle"}${S.brain || S.chat ? " busy" : ""}`;
    if (html !== hudHTML) { hudMain.innerHTML = html; hudHTML = html; }
    if (leg !== legKind) {
      legKind = leg;
      hudLeg.innerHTML = (LEGEND[leg] || []).map(([cs, t]) => `<span>${cs.map(c => `<i style="background:${c};color:${c}"></i>`).join("")}${esc(t)}</span>`).join("");
    }
    if (cls !== hudCls) { hud.className = cls.trim(); hudCls = cls; }
  }

  function setOn(v) {
    v = !!v;
    if (v === on) { if (on) kick(); return; }
    on = v;
    if (!on) {
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
      for (const [k, e] of fx) { fx.delete(k); spare.push(e); }
      map.clearLive(); drawn = false;
      hud.className = ""; hudCls = "";
    } else kick();
  }
  return { handle, setOn, get on() { return on; } };
}
