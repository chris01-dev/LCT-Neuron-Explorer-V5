// chat.js — the steering chat: describe a trait in words, blend the matching neurons at exact doses, then talk
// to the model pushed up / neutral / down and share the blend as text. It only talks to the public API
// (/api/search, /api/steer, /api/status, /api/brain/chat) and never touches the rest of the app's state.
// Dose rule of thumb learned from steering tests: one strong concept-writing neuron at 0.4-0.6 with a few supports
// at 0.1-0.3 works; a single token-writing neuron pushed past ~0.8 collapses into word soup.
(() => {
const $ = t => document.querySelector(t);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const STEPS = [-2.2, -1.7, -1.3, -1, -0.8, -0.65, -0.5, -0.4, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 1, 1.3, 1.7, 2.2, 3];
const fri = { mode: "add", thread: [], running: null, set: { max: 220, temp: 0, hist: 8, persona: "" } };

const style = document.createElement("style");
style.textContent = `
#chat-panel { position: fixed; top: 52px; right: 12px; bottom: 12px; width: min(430px, 46vw); z-index: 70; display: none;
  flex-direction: column; background: var(--panel); backdrop-filter: blur(14px); border: 1px solid var(--line); border-radius: 10px; }
#chat-panel.on { display: flex; }
#chat-panel.in-tab { position: static; width: auto; height: 100%; z-index: auto; display: flex; flex-direction: column; background: none; border: 0; border-radius: 0; backdrop-filter: none; }
#tab-chat.on { display: flex; flex-direction: column; overflow: hidden; padding: 0; }
#chat-panel .fr-head { display: flex; align-items: center; gap: 8px; padding: 10px 12px 8px; }
#chat-panel .fr-head b { font-size: 13px; letter-spacing: 0.06em; }
#chat-panel #fr-name { flex: 1; }
#chat-panel .fr-body { flex: 1; overflow-y: auto; padding: 0 12px 12px; display: flex; flex-direction: column; gap: 8px; }
#chat-panel.in-tab .fr-body { min-height: 0; }
#chat-panel .fr-step { color: var(--dim); font-size: 12px; margin-top: 4px; display: flex; align-items: center; gap: 8px; cursor: pointer; user-select: none; }
#chat-panel .fr-step b { color: var(--ink); }
#chat-panel .fr-step .fr-cv { margin-left: auto; font-size: 10px; color: var(--dim); flex: none; width: 14px; text-align: center; }
#chat-panel .fr-step:hover .fr-cv { color: var(--ink); }
#chat-panel .fr-sec { display: flex; flex-direction: column; gap: 8px; }
#chat-panel .fr-sec.off { display: none; }
#chat-panel .fr-sec#fs-talk { flex: 1 1 auto; min-height: 260px; }
#chat-panel .fr-sec#fs-talk #fr-thread { flex: 1; min-height: 140px; }
#chat-panel .fr-find { display: flex; gap: 6px; }
#chat-panel .fr-find input { flex: 1; }
#chat-panel .fr-find button, #chat-panel .fr-row button, #chat-panel .fr-ctl { border: 1px solid var(--line); border-radius: 4px; padding: 4px 10px; color: var(--ink); }
#chat-panel .fr-find button:hover, #chat-panel .fr-row button:hover, #chat-panel .fr-ctl:hover { background: var(--raise); }
#chat-panel .fr-results { display: flex; flex-direction: column; gap: 2px; max-height: 34vh; overflow-y: auto; }
#chat-panel .fr-res { display: flex; align-items: center; gap: 8px; background: var(--raise); border-radius: 4px; padding: 4px 8px; }
#chat-panel .fr-res .fr-id { font: 11.5px var(--mono); width: 96px; flex: none; }
#chat-panel .fr-res .fr-verified { color: var(--up); }
#chat-panel .fr-res .fr-badges { color: var(--up); font-size: 10px; margin-left: 4px; }
#chat-panel .fr-res .fr-lens { flex: 1; font-size: 12px; color: var(--dim); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#chat-panel .fr-res button { color: var(--dim); border: 1px solid var(--line); border-radius: 3px; padding: 1px 7px; flex: none; }
#chat-panel .fr-res button:hover { color: var(--ink); }
#chat-panel .fr-res button.in { color: var(--up); border-color: var(--up); }
#chat-panel #fr-blend { display: flex; flex-wrap: wrap; gap: 4px; min-height: 24px; }
#chat-panel .fr-chip { font: 11.5px var(--mono); background: var(--raise); border: 1px solid var(--line); border-radius: 4px; padding: 2px 6px; display: inline-flex; gap: 4px; align-items: center; }
#chat-panel .fr-chip b { color: var(--up); font-weight: 400; display: inline-block; min-width: 6ch; text-align: center; }
#chat-panel .fr-chip b.hot { color: var(--down); }
#chat-panel .fr-chip input { width: 58px; padding: 0 3px; background: var(--bg); border: 1px solid var(--dim); border-radius: 3px; font: inherit; }
#chat-panel .fr-sec .fr-row { gap: 6px; }
#chat-panel .fr-sec .fr-row .lab-l { color: var(--dim); font-size: 11.5px; }
#chat-panel .fr-sec .fr-row input.num { width: 62px; text-align: center; font-family: var(--mono); }
#chat-panel #fs-persona { flex: 1; }
#chat-panel .fr-chip button { color: var(--dim); font-size: 11px; }
#chat-panel .fr-chip button:hover { color: var(--ink); }
#chat-panel #fr-presets { display: flex; flex-wrap: wrap; gap: 4px; }
#chat-panel .fr-pres { cursor: pointer; }
#chat-panel .fr-pres:hover { border-color: var(--dim); }
#chat-panel .fr-row { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; font-size: 12px; }
#chat-panel .fr-row .seg { display: inline-flex; }
#chat-panel .fr-row .seg button { border: 1px solid var(--line); border-radius: 3px 0 0 3px; color: var(--dim); padding: 3px 8px; }
#chat-panel .fr-row .seg button + button { border-radius: 0 3px 3px 0; border-left: 0; }
#chat-panel .fr-row .seg button.on { background: rgba(255,255,255,0.14); color: var(--ink); }
#chat-panel #fr-manual { width: 92px; font-family: var(--mono); }
#chat-panel .fr-help { color: var(--faint); font-size: 11.5px; line-height: 1.4; }
#chat-panel .fr-help u { color: var(--dim); text-decoration: none; cursor: pointer; }
#chat-panel .fr-help u:hover { color: var(--ink); }
#chat-panel .fr-note { font-size: 12px; min-height: 16px; }
#chat-panel .fr-note.warn { color: var(--down); }
#chat-panel .fr-note.good { color: var(--up); }
#chat-panel #fr-thread { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 8px; min-height: 60px; }
#chat-panel .fr-turn .fr-user { font-size: 12.5px; color: var(--ink); margin-bottom: 3px; }
#chat-panel .fr-turn .fr-user .recipe { color: var(--dim); font: 10.5px var(--mono); margin-left: 6px; }
#chat-panel .fr-turn .fr-cols { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 4px; }
#chat-panel .fr-turn .fr-col { background: var(--raise); border-radius: 4px; padding: 4px 6px; font-size: 12px; line-height: 1.4; white-space: pre-wrap; overflow-wrap: anywhere; }
#chat-panel .fr-turn .fr-col .lab { display: block; font-size: 10px; margin-bottom: 2px; }
#chat-panel .fr-turn .fr-cu .lab { color: var(--up); }
#chat-panel .fr-turn .fr-cz .lab { color: var(--dim); }
#chat-panel .fr-turn .fr-cd .lab { color: var(--down); }
#chat-panel .fr-turn .err { color: var(--down); font-size: 12px; margin-top: 3px; }
#chat-panel .fr-empty { color: var(--faint); font-size: 12px; }
#chat-panel .fr-head select { max-width: 120px; font-size: 12px; padding: 4px 6px; }
#chat-panel #fr-brain-out { display: flex; flex-direction: column; gap: 4px; max-height: 34vh; overflow-y: auto; font-size: 12px; }
#chat-panel #fr-brain-out:not(.fr-empty) { background: rgba(0,0,0,0.18); border-radius: 6px; padding: 8px; }
#chat-panel #fr-brain-out .b-status { color: var(--dim); }
#chat-panel #fr-brain-out .b-round { color: var(--up); font-size: 11px; margin-top: 8px; }
#chat-panel #fr-brain-out .b-thought { color: var(--faint); font-size: 11px; }
#chat-panel #fr-brain-out .b-tool { font-family: var(--mono); color: var(--link); font-size: 11.5px; }
#chat-panel #fr-brain-out .b-result { color: var(--dim); font-size: 11px; border-left: 2px solid var(--line); padding-left: 6px; }
#chat-panel #fr-brain-out .b-cols { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 4px; margin: 2px 0 4px; }
#chat-panel #fr-brain-out .b-cols > div { background: var(--raise); border-radius: 4px; padding: 3px 5px; font-size: 11px; overflow-wrap: anywhere; }
#chat-panel #fr-brain-out .b-cols .lab { display: block; font-size: 10px; color: var(--dim); margin-bottom: 2px; }
#chat-panel #fr-brain-out .b-done { color: var(--up); margin-top: 6px; }
#chat-panel #fr-brain-out .b-use { float: right; font-size: 10px; padding: 0 5px; margin-left: 4px; line-height: 15px; cursor: pointer; }
#chat-panel #fr-clear { margin-left: auto; color: var(--dim); font-size: 11px; border: 1px solid var(--line); border-radius: 3px; padding: 1px 7px; }
#chat-panel #fr-clear:hover { color: var(--ink); background: var(--raise); }
`;
document.head.appendChild(style);

const panel = document.createElement("aside");
panel.id = "chat-panel";
panel.innerHTML = `
  <div class="fr-head"><b>steering</b><input id="fr-name" placeholder="name it (sad qwen…)" spellcheck="false"><select id="fr-preset-list" title="load a saved preset"></select></div>
  <div class="fr-body">
    <div class="fr-step" data-sec="find">1 · <b>describe it</b> — words the trait makes the model say or feel<span class="fr-cv">▾</span></div>
    <div class="fr-sec" id="fs-find">
      <div class="fr-find"><input id="fr-q" placeholder="dog, puppy, bark, fluffy" spellcheck="false" autocomplete="off"><button id="fr-find">find neurons</button></div>
      <div id="fr-results" class="fr-results fr-empty">type a concept above — word stems find more (depress, unhapp)</div>
    </div>
    <div class="fr-step" data-sec="blend">2 · <b>blend</b> — each neuron + its own push. ★ = verified to steer<span class="fr-cv">▾</span></div>
    <div class="fr-sec" id="fs-blend">
      <div id="fr-blend" class="fr-empty"></div>
      <div class="fr-row">
        <span class="seg" id="fr-mode-seg"><button data-m="add" class="on" title="push each neuron by its number (recommended)">add</button><button data-m="mul" title="advanced: multiply each neuron's own firing by its number instead">scale</button></span>
        <button id="fr-all-down" title="make every push gentler (divide by 1.3)">all −</button>
        <button id="fr-all-up" title="make every push stronger (multiply by 1.3)">all +</button>
        <input id="fr-manual" class="mono" placeholder="L22 N2742" spellcheck="false"><button id="fr-add" class="fr-ctl">add</button>
        <button id="fr-copy" class="fr-ctl">copy</button><button id="fr-paste" class="fr-ctl">paste</button><button id="fr-save" class="fr-ctl" title="save the current blend as a named preset">save</button>
      </div>
      <div id="fr-presets"></div>
      <div class="fr-help">find your own: search words → add the ★ rows that read right → send → raise a chip with + until ▲ changes; back off when it repeats. dose sense: 0.15-0.3 support · 0.4-0.6 one strong lead · 0.8+ on a word-writer usually loops.</div>
      <div class="fr-help">worked example to learn from: <u id="fr-ex-dog">dog persona</u> — one lead dog neuron at 0.45, bark/paw/fluffy supports at low doses, plus a low-layer neuron for overall mood</div>
    </div>
    <div class="fr-step" data-sec="tune">3 · <b>session tuning</b> — how the model answers in this panel<span class="fr-cv">▾</span></div>
    <div class="fr-sec" id="fs-tune">
      <div class="fr-row">
        <span class="lab-l">output tokens</span><input id="fs-max" class="num" value="220">
        <span class="lab-l">temperature</span><input id="fs-temp" class="num" value="0" title="0 = stable and repeatable; 0.7-1.0 = varied and playful">
        <span class="lab-l">history turns</span><input id="fs-hist" class="num" value="8">
      </div>
      <input id="fs-persona" placeholder="persona / system prompt (optional): how it should present itself" spellcheck="false">
    </div>
    <div class="fr-step" data-sec="talk">4 · <b>talk</b> — ▲ blend up · 0 normal · ▼ blend down<button id="fr-clear">clear chat</button></div>
    <div class="fr-sec" id="fs-talk">
      <div id="fr-note" class="fr-note dim"></div>
      <div id="fr-thread" class="fr-empty">add a neuron or two, then say something.</div>
      <div class="fr-find"><input id="fr-input" placeholder="Who are you?" autocomplete="off"><button id="fr-send">send</button></div>
    </div>
    <div class="fr-step" data-sec="brain" style="margin-top:12px">5 · <b>brain</b> — or just say what you want; it finds and tunes the neurons<span class="fr-cv">▾</span></div>
    <div class="fr-sec" id="fs-brain">
      <div class="fr-find"><input id="fr-brain-goal" placeholder="make it talk like a grumpy wizard…" spellcheck="false" autocomplete="off"><button id="fr-brain-go">steer for me</button></div>
      <div id="fr-brain-out" class="fr-empty">the brain loads a small second model (once), searches neurons, tests them on the live model, and hands you the finished recipe.</div>
    </div>
  </div>`;
const host = $("#tab-chat");
if (host) { panel.classList.add("in-tab"); host.appendChild(panel); }
else document.body.appendChild(panel);

const DEFAULT_SET = { max: 220, temp: 0, hist: 8, persona: "" };
const EXAMPLES = {
  dog: { name: "dog persona", mode: "add", blend: [
    { layer: 27, n: 3225, dose: 0.45 }, { layer: 22, n: 8658, dose: 0.25 }, { layer: 21, n: 7292, dose: 0.25 },
    { layer: 23, n: 4769, dose: 0.2 }, { layer: 31, n: 7966, dose: 0.12 }, { layer: 7, n: 5830, dose: 0.3 }],
    set: { max: 160, temp: 0.8, hist: 8, persona: "" } }
};

let blend = [], results = [];
const note = (t, cls, ms) => {
  const el = $("#fr-note");
  el.textContent = t || ""; el.className = "fr-note " + (cls || "dim");
  if (t && ms) setTimeout(() => { if (el.textContent === t) note(""); }, ms);
};
const save = () => { try { localStorage.setItem("lct-blend", JSON.stringify({ v: 3, name: $("#fr-name").value, mode: fri.mode, blend, set: fri.set })); } catch {} };
function load() {
  try {
    const raw = localStorage.getItem("lct-blend") ?? localStorage.getItem("friendly-blend");
    const d = JSON.parse(raw || "null");
    if (d && Array.isArray(d.blend)) {
      blend = d.blend.map(b => ({ layer: b.layer, n: b.n, dose: b.dose ?? b.factor * 0.8 }))
        .filter(b => Number.isInteger(b.layer) && Number.isInteger(b.n) && Number.isFinite(b.dose) && b.dose !== 0);  // negative doses are real pushes
      fri.mode = d.mode === "mul" ? "mul" : "add";
      fri.set = { max: 220, temp: 0, hist: 8, persona: "", ...(d.set || {}) };
      $("#fr-name").value = d.name || "";
    }
  } catch {}
}

const inBlend = (l, n) => blend.some(b => b.layer === l && b.n === n);
function addNeuron(l, n, dose = 0.25) { if (!inBlend(l, n)) blend.push({ layer: l, n, dose }); renderBlend(); save(); }
window.lctAdd = addNeuron;  // shift-click in the map/list drops neurons straight into the chat blend

const doseTag = d => Math.abs(Math.round(d * 100) / 100);
const fmtSigned = d => `${d < 0 ? "-" : "+"}${doseTag(Math.abs(d))}`;
// live map layer (live.js in app.js listens): one-way window events, so a broken listener can never stop chat or brain
const live = detail => { try { window.dispatchEvent(new CustomEvent("lct:live", { detail })); } catch {} };
function renderBlend() {
  const d = { kind: "blend", blend: blend.map(b => ({ layer: b.layer, n: b.n, dose: b.dose })), mode: fri.mode };
  window.__lctLiveBlend = d; live(d);  // kept for live.js in case it is set up after this first render
  const el = $("#fr-blend");
  if (!blend.length) { el.className = "fr-empty"; el.textContent = "nothing yet — pick the ★ rows above, or type an address and press add"; return; }
  el.className = "";
  el.innerHTML = blend.map((b, i) => `<span class="fr-chip" title="${b.dose < 0 ? "negative: this neuron is pushed DOWN in the ▲ column" : "pushed by this amount in the ▲ column (reversed in ▼)"} — double-click the number to type a dose">L${b.layer} N${b.n} <b data-i="${i}" class="${Math.abs(b.dose) > 0.8 ? "hot" : ""}">${fmtSigned(b.dose)}</b>
    <button data-a="-" data-i="${i}" aria-label="weaker">−</button><button data-a="+" data-i="${i}" aria-label="stronger">+</button><button data-a="f" data-i="${i}" aria-label="flip sign">±</button><button data-a="x" data-i="${i}" aria-label="remove">×</button></span>`).join("");
}
$("#fr-blend").addEventListener("dblclick", e => {
  const b = e.target.closest("b"); if (!b) return;
  const i = +b.dataset.i; if (!blend[i]) return;
  const inp = document.createElement("input");
  inp.value = Math.round(blend[i].dose * 100) / 100;
  b.replaceWith(inp);
  inp.focus(); inp.select();
  let done = false;
  const commit = () => { if (done) return; done = true; const v = parseFloat(inp.value.replace(",", ".")); if (Number.isFinite(v)) blend[i].dose = Math.max(-10, Math.min(10, v)); renderBlend(); save(); };
  inp.addEventListener("keydown", ev => {
    if (ev.key === "Enter") { ev.preventDefault(); commit(); }
    if (ev.key === "Escape") { done = true; renderBlend(); }
  });
  inp.addEventListener("blur", commit);
});
$("#fr-blend").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  const i = +b.dataset.i;
  if (b.dataset.a === "x") { blend.splice(i, 1); renderBlend(); save(); return; }
  if (b.dataset.a === "f") { blend[i].dose = -blend[i].dose; renderBlend(); save(); return; }
  const d = blend[i].dose;
  const next = b.dataset.a === "+" ? STEPS.find(x => x > d + 1e-9) : [...STEPS].reverse().find(x => x < d - 1e-9);
  if (next !== undefined) blend[i].dose = next;
  renderBlend(); save();
});
const scaleAll = f => { blend.forEach(b => b.dose = Math.max(-10, Math.min(10, Math.round(b.dose * f * 100) / 100))); renderBlend(); save(); };  // keeps each sign: negative pushes are real
$("#fr-all-up").onclick = () => scaleAll(1.3);
$("#fr-all-down").onclick = () => scaleAll(1 / 1.3);

async function find() {
  const qs = [...new Set($("#fr-q").value.split(/[,;]+/).map(t => t.trim().toLowerCase()).filter(t => t.length >= 2))].slice(0, 8);
  const el = $("#fr-results");
  if (!qs.length) { el.className = "fr-results fr-empty"; el.textContent = "type at least one word (2+ characters)"; return; }
  el.className = "fr-results fr-empty"; el.textContent = "searching…";
  try {
    const rs = await Promise.all(qs.map(q => fetch(`/api/search?q=${encodeURIComponent(q)}&limit=300`).then(r => r.json())));
    if (rs.some(r => r.ready === false)) { el.textContent = "lens index still loading — try again in a moment"; return; }
    const by = new Map();
    for (const q of qs) for (const r of rs[qs.indexOf(q)].results) {
      if (r.sign !== "+") continue;  // + neurons are pushed up; the ▲ column is what you read
      const k = `${r.layer}/${r.n}`;
      if (!by.has(k)) by.set(k, { layer: r.layer, n: r.n, hits: [], rank: r.rank, lens: r.lens, name: r.name, verified: false, exact: false });
      const me = by.get(k);
      me.hits.push(q); me.rank = Math.min(me.rank, r.rank);
      if (r.src === "steer") me.verified = true;
      if (r.lens.some(t => t.trim().toLowerCase().startsWith(q))) me.exact = true;
    }
    results = [...by.values()].sort((a, b) => b.exact - a.exact || b.verified - a.verified || b.hits.length - a.hits.length || a.rank - b.rank).slice(0, 14);
    el.className = "fr-results";
    el.innerHTML = results.map(r => `<div class="fr-res">
      <span class="fr-id mono">${r.verified ? '<span class="fr-verified" title="this direction was measured in the steering sweep: pushing it really does raise these words">★</span>' : ""}L${r.layer} N${r.n}</span>
      <span class="fr-lens">${r.hits.length ? `<span class="fr-badges">${esc(r.hits.join("·"))}</span> ` : ""}${esc((r.name ? r.name + " — " : "") + r.lens.join(" "))}</span>
      <button data-i="${results.indexOf(r)}" class="${inBlend(r.layer, r.n) ? "in" : ""}">${inBlend(r.layer, r.n) ? "✓ in blend" : "add +0.25"}</button></div>`).join("")
      + `<div class="fr-help" style="padding:4px 8px">top ${results.length} of ${by.size} · ★ verified first · exact words before oddballs (pawn, software, embarked)</div>`;
  } catch (e) { el.className = "fr-results fr-empty"; el.textContent = `search failed: ${esc(String(e))}`; }
}
$("#fr-find").onclick = find;
$("#fr-q").addEventListener("keydown", e => { if (e.key === "Enter") find(); });
$("#fr-results").addEventListener("click", e => {
  const b = e.target.closest("button[data-i]"); if (!b) return;
  const r = results[+b.dataset.i];
  addNeuron(r.layer, r.n);
  b.textContent = "✓ in blend"; b.classList.add("in");
  note(`L${r.layer} N${r.n} added at +0.25 — that's a gentle start; raise it with + on the chip`, "good", 4000);
});

document.querySelectorAll("#fr-mode-seg button").forEach(b => b.onclick = () => {
  fri.mode = b.dataset.m;
  document.querySelectorAll("#fr-mode-seg button").forEach(x => x.classList.toggle("on", x === b));
  renderBlend(); save();
});
const manual = () => {
  const m = $("#fr-manual").value.match(/L?\s*(\d+)\D+N?\s*(\d+)/i);
  if (!m) { note("address looks like L22 N2742", "", 2500); return; }
  addNeuron(+m[1], +m[2]); $("#fr-manual").value = "";
};
$("#fr-add").onclick = manual;
$("#fr-manual").addEventListener("keydown", e => { if (e.key === "Enter") manual(); });

const blendText = (name = $("#fr-name").value.trim()) =>
  `Qwen3.5 steer blend — ${name || "unnamed"} (mode ${fri.mode})\n` +
  blend.map(b => `L${b.layer} N${b.n} ${fmtSigned(b.dose)}`).join(", ") +
  (blend.length ? "\n" : "") +
  `\nimport string:\nQWENBLEND|${fri.mode}||` +
  blend.map(b => `L${b.layer}N${b.n}x${Math.round(b.dose * 100) / 100}`).join(",") + `|${name}`;

async function copyBlend() {
  if (!blend.length) { note("nothing to copy — the blend is empty", "", 2500); return; }
  const text = blendText();
  try { await navigator.clipboard.writeText(text); }
  catch {
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select();
    document.execCommand("copy"); ta.remove();
  }
  note("copied — the import string at the bottom is the shareable part", "good", 3000);
}
$("#fr-copy").onclick = copyBlend;

function parseBlend(text) {
  if (!text || !text.trim()) return null;
  const out = { mode: null, name: "", blend: [] };
  const v = text.match(/QWENBLEND[^\n]*/i);
  if (v) {
    const p = v[0].split("|");
    out.mode = /mul|scale|×/i.test(p[1] || "") ? "mul" : "add";
    out.name = (p[4] || "").trim();
    for (const m of (p[3] || "").matchAll(/L\s*(\d+)\s*N\s*(\d+)\s*x\s*([+-]?[\d.]+)/gi)) {
      const d = parseFloat(m[3]);
      out.blend.push({ layer: +m[1], n: +m[2], dose: Number.isFinite(d) ? d : 0.25 });
    }
  } else {
    const mode = text.match(/\b(add|mul|scale)\b/i);
    out.mode = mode ? (/mul|scale/i.test(mode[1]) ? "mul" : "add") : "add";
    const nm = text.match(/blend\s*[—-]\s*(.+?)(?:\s*\(mode|\n|$)/i); out.name = nm ? nm[1].trim() : "";
    for (const m of text.matchAll(/L\s*(\d+)\s*N\s*(\d+)(?:\s*[x×*]?\s*([+-])?\s*([\d.]+))?/gi)) {
      const mag = m[4] !== undefined ? parseFloat(m[4]) : 0.25;
      out.blend.push({ layer: +m[1], n: +m[2], dose: m[3] === "-" ? -mag : mag });
    }
  }
  return out.blend.length ? out : null;
}
$("#fr-paste").onclick = () => {
  const text = window.prompt("paste a blend (the import string, or the copied text)");
  const p = text && parseBlend(text);
  if (!p) { note("no neurons found in that text — it needs L22 N2742 style addresses", "", 4000); return; }
  blend = p.blend; if (p.mode) fri.mode = p.mode;
  $("#fr-name").value = p.name || $("#fr-name").value;
  document.querySelectorAll("#fr-mode-seg button").forEach(x => x.classList.toggle("on", x.dataset.m === fri.mode));
  renderBlend(); save();
  note(`imported ${p.blend.length} neurons`, "good", 3000);
};

// ---------- session settings ----------
const SET_FIELDS = [["#fs-max", "max", 10, 2000, 220], ["#fs-temp", "temp", 0, 2, 0], ["#fs-hist", "hist", 1, 50, 8]];
for (const [id, key, lo, hi, dflt] of SET_FIELDS) {
  const el = $(id);
  el.addEventListener("change", () => {
    let v = parseFloat(el.value.replace(",", "."));
    if (!Number.isFinite(v)) v = dflt;
    fri.set[key] = Math.max(lo, Math.min(hi, v));
    el.value = fri.set[key];
    save();
  });
}
$("#fs-persona").addEventListener("change", () => { fri.set.persona = $("#fs-persona").value.trim(); save(); });
function reflectSet() {
  $("#fs-max").value = fri.set.max; $("#fs-temp").value = fri.set.temp; $("#fs-hist").value = fri.set.hist;
  $("#fs-persona").value = fri.set.persona || "";
}

let PRE = [];
try { PRE = JSON.parse((localStorage.getItem("lct-presets") ?? localStorage.getItem("friendly-presets")) || "[]"); } catch {}
if (!Array.isArray(PRE)) PRE = [];
function renderPresets() {
  $("#fr-presets").innerHTML = PRE.map((p, i) => `<span class="fr-pres fr-chip" data-i="${i}" title="click to load this preset">${esc(p.name)} <b>${p.mode === "mul" ? "x" : "+"}${p.blend.length}</b><button data-a="del" data-i="${i}" aria-label="delete preset">×</button></span>`).join("");
  const sel = $("#fr-preset-list");
  sel.innerHTML = `<option value="">load…</option>` + PRE.map((p, i) => `<option value="${i}">${esc(p.name)}</option>`).join("");
}
function loadPresetAt(i) {
  const p = PRE[i]; if (!p) return;
  blend = p.blend.map(b => ({ ...b }));
  fri.mode = p.mode === "mul" ? "mul" : "add";
  if (p.set) fri.set = { max: 220, temp: 0, hist: 8, persona: "", ...p.set };
  $("#fr-name").value = p.name;
  document.querySelectorAll("#fr-mode-seg button").forEach(x => x.classList.toggle("on", x.dataset.m === fri.mode));
  renderBlend(); reflectSet(); save();
  note(`loaded "${p.name}" — send a message to try it`, "good", 3500);
}
$("#fr-save").onclick = () => {
  if (!blend.length) { note("nothing to save — the blend is empty", "", 2500); return; }
  let name = $("#fr-name").value.trim();
  if (!name) name = window.prompt("preset name:", `blend ${PRE.length + 1}`) || "";
  if (!name) return;
  $("#fr-name").value = name;
  const at = PRE.findIndex(p => p.name === name);
  const rec = { name, mode: fri.mode, blend: blend.map(b => ({ ...b })), set: { ...fri.set } };
  if (at >= 0) PRE[at] = rec; else PRE.push(rec);
  try { localStorage.setItem("lct-presets", JSON.stringify(PRE)); } catch {}
  renderPresets(); save();
  note(at >= 0 ? `updated preset "${name}"` : `saved preset "${name}"`, "good", 3000);
};
$("#fr-presets").addEventListener("click", e => {
  const del = e.target.closest("button[data-a='del']");
  if (del) {
    const i = +del.dataset.i, gone = PRE[i]?.name;
    PRE.splice(i, 1);
    try { localStorage.setItem("lct-presets", JSON.stringify(PRE)); } catch {}
    renderPresets(); note(`deleted preset "${gone}"`, "", 2500);
    return;
  }
  const chip = e.target.closest(".fr-pres"); if (!chip) return;
  loadPresetAt(+chip.dataset.i);
});

$(".fr-help #fr-ex-dog").onclick = () => {
  const ex = EXAMPLES.dog;
  blend = ex.blend.map(b => ({ ...b })); fri.mode = ex.mode;
  fri.set = { ...DEFAULT_SET, ...ex.set };
  $("#fr-name").value = ex.name;
  document.querySelectorAll("#fr-mode-seg button").forEach(x => x.classList.toggle("on", x.dataset.m === ex.mode));
  renderBlend(); reflectSet(); save();
  note("loaded the dog example — press send below and watch the ▲ column", "good", 5000);
};

const recipe = () => `${fri.mode === "mul" ? "x" : "add"}: ` + blend.map(b => `L${b.layer}N${b.n}${fmtSigned(b.dose)}`).join(", ");
function renderThread() {
  const el = $("#fr-thread");
  if (!fri.thread.length) { el.className = "fr-empty"; el.textContent = "add a neuron or two, then say something."; return; }
  el.className = "";
  el.innerHTML = fri.thread.map(t => `<div class="fr-turn">
    <div class="fr-user">${esc(t.user)}<span class="recipe">${esc(t.recipe)}</span></div>
    <div class="fr-cols">
      <div class="fr-col fr-cu"><span class="lab">▲ up</span>${esc(t.replies[0])}</div>
      <div class="fr-col fr-cz"><span class="lab">0 normal</span>${esc(t.replies[1])}</div>
      <div class="fr-col fr-cd"><span class="lab">▼ down</span>${esc(t.replies[2])}</div>
    </div>${t.error ? `<div class="err">${esc(t.error)}</div>` : ""}</div>`).join("");
  el.scrollTop = el.scrollHeight;
}
function judge(turn) {
  const up = turn.replies[0] || "", zero = turn.replies[1] || "";
  if (up && up === zero) { note("▲ came out identical to the normal column — check the add/scale switch is on add, raise the doses (+ on a chip, or all +), or add more neurons", "warn", 8000); return; }
  const flat = up.toLowerCase().replace(/\s+/g, " ");
  if (/(.{2,24}?)\1{3,}/.test(flat)) { note("▲ collapsed into repetition — that's too much push. dial the biggest chips down a notch (− or all −)", "warn", 8000); return; }
  if (up && up !== zero) note("steering is showing in ▲ — raise doses if you want more, note how the ▼ column moves the other way", "good", 5000);
}

async function send() {
  if (fri.running) { fri.running.abort(); return; }
  if (!blend.length) { note("the blend is empty — find or add at least one neuron first", "", 3500); return; }
  const text = $("#fr-input").value.trim() || $("#fr-input").placeholder;
  const history = fri.thread.filter(t => !t.error).map(t => ({ user: t.user, replies: t.replies })).slice(-fri.set.hist);
  const turn = { user: text, replies: ["", "", ""], recipe: recipe() };
  fri.thread.push(turn); $("#fr-input").value = ""; note(""); renderThread();
  const ctl = new AbortController(); fri.running = ctl; $("#fr-send").textContent = "stop";
  const neurons = blend.map(b => ({ layer: b.layer, n: b.n, dose: b.dose }));
  try {
    live({ kind: "chat-start", blend: neurons, mode: fri.mode });  // the snapshot sent, so later chip edits don't change it
    const res = await fetch("/api/steer", { method: "POST", headers: { "Content-Type": "application/json" }, signal: ctl.signal,
      body: JSON.stringify({ neurons, prompt: text, mode: fri.mode, history, max_new: fri.set.max,
                             temperature: fri.set.temp || undefined, system: fri.set.persona || undefined, activity: 48 }) });
    if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 160)}`);
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let k;
      while ((k = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, k); buf = buf.slice(k + 2);
        if (!line.startsWith("data: ")) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.error) turn.error = ev.error;
        if (ev.act) { live({ kind: "act", row: ev.row, step: ev.step, act: ev.act }); continue; }  // top firing neurons, t is ""
        if (ev.row == null) continue;
        turn.replies[ev.row] += ev.t;
        const c = $("#fr-thread").lastElementChild.querySelectorAll(".fr-col")[ev.row];
        c.lastChild.nodeType === 3 ? c.lastChild.textContent += ev.t : c.append(ev.t);
        const th = $("#fr-thread"); if (th.scrollHeight - th.scrollTop - th.clientHeight < 80) th.scrollTop = th.scrollHeight;
      }
    }
  } catch (e) { if (String(e) !== "AbortError") turn.error = String(e); }
  live({ kind: "chat-end" });
  fri.running = null; $("#fr-send").textContent = "send"; renderThread();
  if (!turn.error) judge(turn);
}
$("#fr-send").onclick = send;
$("#fr-input").addEventListener("keydown", e => { if (e.key === "Enter") send(); });
$("#fr-clear").onclick = () => { if (fri.running) fri.running.abort(); fri.thread = []; renderThread(); note(""); };

$("#fr-preset-list").addEventListener("change", e => { const v = e.target.value; if (v !== "") loadPresetAt(+v); e.target.value = ""; });

// ---------- brain: natural language -> neuron recipe ----------
let brainCtl = null, brainStopping = false, brainPicked = false;
const brainTrials = [];  // recipe of every trial shown, so "use ▲ / use ▼" can take exactly that one
const brainStop = keep => fetch("/api/brain/stop", { method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ keep }) }).catch(() => {});
function takeBlend(recipe) {
  blend = recipe.map(b => ({ layer: b.layer, n: b.n, dose: b.dose }));
  fri.mode = "add";  // recipes are written in add space; a stale 'scale' mode would mute them
  document.querySelectorAll("#fr-mode-seg button").forEach(x => x.classList.toggle("on", x.dataset.m === "add"));
  fri.set.persona = ""; $("#fs-persona").value = "";  // neurons only: no persona may mix into the measurement
  renderBlend(); save();
}
$("#fr-brain-out").addEventListener("click", e => {
  const bt = e.target.closest(".b-use"); if (!bt) return;
  const rec = brainTrials[+bt.dataset.k]; if (!rec) return;
  const sign = +bt.dataset.s;
  takeBlend(rec.map(b => ({ ...b, dose: Math.round(b.dose * sign * 1000) / 1000 })));
  note(`using that trial's neurons (${sign > 0 ? "▲ as pushed" : "▼ reversed"}) — the brain stops`, "good", 7000);
  if (brainCtl && !brainPicked) { brainPicked = true; brainStopping = true; $("#fr-brain-go").textContent = "stopping…"; brainStop(false); }
});
const brainOut = (cls, html) => {
  const el = $("#fr-brain-out");
  el.classList.remove("fr-empty");
  const d = document.createElement("div");
  d.className = cls;
  d.innerHTML = html;
  el.appendChild(d);
  el.scrollTop = el.scrollHeight;
};
async function brainRun() {
  if (brainCtl) {  // first click: stop at the next trial and keep the best recipe so far; second click: cancel
    if (!brainStopping) { brainStopping = true; $("#fr-brain-go").textContent = "stopping… (click to cancel)"; brainStop(true); }
    else brainCtl.abort();
    return;
  }
  const goal = $("#fr-brain-goal").value.trim() || $("#fr-brain-goal").placeholder;
  $("#fr-brain-out").innerHTML = ""; $("#fr-brain-out").className = "";
  $("#fr-brain-go").textContent = "stop · keep best";
  brainStopping = false; brainPicked = false; brainTrials.length = 0;
  brainCtl = new AbortController();
  note("");
  live({ kind: "brain-start", goal });
  try {
    const res = await fetch("/api/brain/chat", { method: "POST", headers: { "Content-Type": "application/json" }, signal: brainCtl.signal,
      body: JSON.stringify({ goal }) });
    if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 200)}`);
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let k;
      while ((k = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, k); buf = buf.slice(k + 1);
        if (!line.trim()) continue;
        const ev = JSON.parse(line);
        live({ kind: "brain", ev });  // every event, any t, before the log handles it
        if (ev.t === "status") brainOut("b-status", esc(ev.text));
        if (ev.t === "round") brainOut("b-round", `— ${ev.label ? `${ev.i}. ${esc(ev.label)}` : `round ${ev.i}`} —`);
        if (ev.t === "thought") brainOut("b-thought", esc(ev.text));
        if (ev.t === "tool") brainOut("b-tool", `${esc(ev.name)}(${Object.entries(ev.input || {}).map(([a, b]) => `${a}: ${esc(String(b))}`).join(", ")})`);
        if (ev.t === "result") brainOut("b-result", esc(ev.text).replace(/\n/g, "<br>"));
        if (ev.t === "trial") {
          const k = ev.recipe ? brainTrials.push(ev.recipe) - 1 : -1;
          const use = s => k >= 0 ? `<button class="b-use" data-k="${k}" data-s="${s}" title="stop the brain and use exactly this trial's neurons${s < 0 ? ", reversed" : ""}">use ${s > 0 ? "▲" : "▼"}</button>` : "";
          brainOut("b-cols",
            `<div><span class="lab">▲ ${esc(ev.prompt || "")}${use(1)}</span>${esc(ev.up)}</div><div><span class="lab">0 neutral</span>${esc(ev.zero)}</div><div><span class="lab">▼ down${use(-1)}</span>${esc(ev.down)}</div>`);
          if (ev.score && ev.score.judge) brainOut("b-status", `judge (0-10): ▲ ${ev.score.up} vs 0 ${ev.score.zero}${ev.score.down !== undefined ? ` · ▼ ${ev.score.down}` : ""} · ▲ coherent ${ev.score.coherent} · ▲ self-repetition ${ev.score.d4}`);
          else if (ev.score) brainOut("b-status", `score: goal words ▲ ${ev.score.up} vs 0 ${ev.score.zero} · ▲ self-repetition ${ev.score.d4}`);
        }
        if (ev.t === "blend" && !brainPicked) takeBlend(ev.blend);  // once you picked a trial, nothing overwrites it
        if (ev.t === "done") {
          if (ev.stopped) {
            brainOut("b-done", "■ " + esc(ev.summary));
            if (!brainPicked) note(ev.handed_over ? "stopped — the best recipe so far is in the blend" : "stopped before any recipe was measured — your blend is unchanged", ev.handed_over ? "good" : "warn", 7000);
          }
          else if (ev.verified) { brainOut("b-done", "✓ " + esc(ev.summary)); note("brain finished — the verified recipe is in the blend; persona cleared, so what you see below is the neurons alone", "good", 7000); }
          else { brainOut("b-done", "⚠ " + esc(ev.summary)); note("brain finished, but the recipe in the blend is NOT verified — see the summary", "warn", 9000); }
        }
        if (ev.t === "error") { brainOut("b-done", "error: " + esc(ev.error)); note("the brain hit an error — see the log", "warn", 7000); }
      }
    }
  } catch (e) { if (!/AbortError/.test(String(e))) brainOut("b-done", "error: " + esc(String(e))); }
  live({ kind: "brain-end" });
  brainCtl = null; brainStopping = false; $("#fr-brain-go").textContent = "steer for me";
}
$("#fr-brain-go").onclick = brainRun;
$("#fr-brain-goal").addEventListener("keydown", e => { if (e.key === "Enter") brainRun(); });


// collapsible sections — click a header to fold it away and give the chat room
const SECS = ["find", "blend", "tune", "talk", "brain"];
let collapsed;
try { const raw = localStorage.getItem("lct-collapsed"); collapsed = raw ? JSON.parse(raw) : null; } catch { collapsed = null; }
if (!Array.isArray(collapsed)) collapsed = ["find", "blend", "tune", "brain"];
const saveColl = () => { try { localStorage.setItem("lct-collapsed", JSON.stringify(collapsed)); } catch {} };
const applySec = s => {
  const sec = panel.querySelector("#fs-" + s); if (!sec) return;
  const off = collapsed.includes(s);
  sec.classList.toggle("off", off);
  const cv = panel.querySelector(`.fr-step[data-sec="${s}"] .fr-cv`);
  if (cv) cv.textContent = off ? "▸" : "▾";
};
SECS.forEach(applySec);
panel.querySelectorAll(".fr-step").forEach(h => h.addEventListener("click", e => {
  if (e.target.closest("button, input, select, a, u")) return;
  const s = h.dataset.sec; if (!s) return;
  collapsed = collapsed.includes(s) ? collapsed.filter(x => x !== s) : collapsed.concat([s]);
  applySec(s); saveColl();
}));

load();
document.querySelectorAll("#fr-mode-seg button").forEach(x => x.classList.toggle("on", x.dataset.m === fri.mode));
renderBlend(); renderThread(); renderPresets();
})();
