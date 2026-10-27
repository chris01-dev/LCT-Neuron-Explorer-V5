# LCT Neuron Explorer

**Turn what you say into activations.** LCT Neuron Explorer (LCT = Loop Close Technology) is a local
research instrument that lets you steer a language model through its neurons — not its prompt. You
describe a behavior in plain words; the app finds the neurons behind those words, and every reply
arrives in three columns side by side: the neurons pushed up, the untouched model, and the neurons
pushed down. The difference between the columns is not a guess — it is a measured effect of adding
to specific activations inside the network.

A built-in steering brain closes the loop the other way: give it a sentence like
*"make it talk like a grumpy wizard"* and it finds candidate neurons, tests each one live against the
model, combines and tunes the ones that measurably move the replies toward the wish, and hands you a
recipe labelled verified only when its numbers hold on prompts it never tuned on. Natural language
in, neuron activations out — that is the core of this tool.

Everything runs on your machine. No accounts, no telemetry, no cloud calls.

---

## What it does

- **3D map of the model** — every dot is a real neuron in a Qwen3.5-4B network, laid out by
  architecture. Hover for its vocabulary, click to open its measurements, shift-click to drop it
  straight into your steering blend.
- **Lens search** — find neurons by the words they respond to. Rows marked ★ were measured in
  steering sweeps: pushing them genuinely changes what the model writes.
- **Three-column chat** — every prompt is answered with the blend pushed up, neutral, and inverted,
  so the effect of each neuron is visible immediately.
- **The steering brain** — a second, local model that turns plain-language wishes into measured
  activation recipes while you watch every trial live: the replies, their scores, and why each neuron
  was kept or dropped.
- **Presets and sharing** — blends save as named presets and export as a small text recipe you can
  paste into any other install.

## How it works

The steered model is Qwen3.5-4B, running locally. Steering means adding a chosen value to a chosen
neuron's activation at a chosen layer for the whole reply — doses are in natural activation units,
measured per neuron. The app ships with pre-computed surveys of the model: each neuron's lens
(the tokens it raises and lowers, measured by a forward pass, not introspected by another model),
its strongest sweep examples, and its direct write/read connections to other neurons.

The steering brain is Gemma-4 12B (an agentic tune, loaded in 4-bit). It never prompts the chat
model — steering is by neurons only. For each wish it:

1. **plans** — reads the wish into words such replies would say, the plain default behavior the
   wish moves away from, and short example replies written in the wished style;
2. **discovers** candidates in any layer and either sign: neurons whose measured steering lens writes
   the wish's words or its format markers (emoji, \*actions\*, ALL-CAPS, "!!"), writers of the default
   behavior (to be pushed negative), and neurons that fire differently on the example replies than on
   the model's own plain replies. A fast likelihood screen then pushes ~150 such neurons both ways and
   measures, without generating anything, how much more likely each push makes the example replies
   than plain rewrites of the same content — its best ones go to the front of the queue. It runs again
   at twice the dose, because some neurons only show a persona when pushed harder;
3. **screens** each candidate alone on the live model, both directions at once, gentler again when a
   strong one floods the reply;
4. **combines** neurons one at a time while the judged effect grows and the replies stay coherent; a
   join that fails is tried once more at another strength (gentler when it broke a reply, the full
   dose when it added too little);
5. **tunes** one dose scale for the whole recipe;
6. **verifies** on held-out prompts no stage selected on.

Every reply is scored by the brain on its own, never knowing which one was steered: how strongly it
shows the wish (0–10) and how coherent it is (0–10), read as expected values of its score digits.
On hand-rated replies these scores agree with three independent raters at r = 0.86 (style) and
r = 0.94 (coherence). When the wish explicitly forbids something ("never mention being an AI"), the
brain also checks each reply for that violation and rewards recipes that remove it.

You do not have to wait for the end: **stop · keep best** ends the run at the next trial and puts the
best recipe measured so far in your blend, and every trial shown has **use ▲ / use ▼** buttons that
take exactly that trial's neurons (as pushed, or reversed) and stop the brain.

Interpretation stays with the researcher: the app presents rollouts beside baselines and measured
numbers, and it does not overclaim what any single neuron "means."

## Requirements

| | chat only | with the steering brain |
|---|---|---|
| OS | Windows 10/11 64-bit | Windows 10/11 64-bit |
| GPU | NVIDIA, 12 GB VRAM | NVIDIA, 16 GB VRAM (24 GB comfortable) |
| RAM | 16 GB (32 GB recommended) | 32 GB |
| free disk | ~17 GB | ~41 GB |

The model weights are not included in the download — see below. The NVIDIA driver should be
current (the runtime uses a CUDA 13 stack). AMD/Intel GPUs are not supported.

## Install

1. Download **`LCTNeuronExplorer.exe`** from [Releases](../../releases) (~100 MB). It is the only
   thing you download — the same file installs the app and starts it every time after.
2. Run it. The first run unpacks the app to `%LOCALAPPDATA%\LCTNeuronExplorer`, installs the Python
   + CUDA environment once (5–15 minutes), then asks how to get the two models:

   | choice | what happens |
   |---|---|
   | `[1]` download both | the chat model (~9 GB) and the brain (~24 GB) download with a progress bar |
   | `[2]` chat model only | everything except the brain works; add it later (see below) |
   | `[3]` I already have them | you paste the folder paths — each one is **verified to be exactly the right model** (type, architecture, and weights are checked); a wrong or incomplete folder is refused with the reason |
   | `[4]` skip | the chat model downloads on first window launch; the brain stays off |

   Both models are free, public downloads. If you prefer to fetch them yourself, download
   [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) and the
   [Gemma-4 12B agentic tune](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2),
   then pick `[3]` and point the installer at the folders. Either way the app refuses anything
   that is not the exact model it needs.

   Interrupted downloads resume: run the installer again and pick the same choice — finished files
   are kept.
3. The window opens. First launch loads the model into VRAM (~30–60 s); later launches are faster.

To update, download the newer `LCTNeuronExplorer.exe` and run it: it updates the app in place and
keeps your models and settings. Change models any time with `LCTNeuronExplorer.exe --setup`. Uninstalling is deleting the exe and
`%LOCALAPPDATA%\LCTNeuronExplorer` (the models live there too).

### From source

```powershell
git clone https://github.com/chris01-dev/LCT-Neuron-Explorer-V5.git
cd LCT-Neuron-Explorer-V5
uv sync                 # Python + PyTorch CUDA
uv run python desktop.py
```

For research use, the server exposes a plain HTTP API on `127.0.0.1:8791`
(`/api/lens`, `/api/steer`, `/api/reads`, `/api/search`, `/api/status`, the brain's
`/api/brain/chat` event stream, `/api/brain/plan` and `/api/brain/judge`), so experiments can be
scripted without the UI.

### Measuring the brain

`tools/brain_bench.py` scores the steering brain on a fixed suite of wishes (development wishes plus
held-out ones never used while developing the method), against hand-made reference recipes where
they exist and the untouched model, over a battery of prompts. Every reply is scored by the judge and
by a fixed per-wish word lexicon that involves no model at all. Each run appends one line to
`bench/history.jsonl`, so methods and versions compare on the same numbers.

Groups: `dev` (nine wishes used while developing the method: three short, six detailed persona
prompts), `heldout` (ten it never saw: seven short, three detailed), `user` (two wishes written by a
user, word for word, with the replies they described as the target), and `baseline6`, the six-wish
set the version history compares on.

```powershell
uv run python tools/brain_bench.py --wishes baseline6 --label v5
uv run python tools/brain_bench.py --wishes heldout,user --label v5-heldout
uv run python tools/brain_bench.py --wishes heldout --options "{\"sources\": \"say,avoid,marker,contrast,default\"}" --label no-likely
```

## Using the app

The main view is a 3D map: the vertical spine is the residual stream through 32 layers, each slab a
real attention/MLP block, every dot one neuron. The left panel lists the focused neuron's
connections and ranks the layer; the right panel is where steering happens — open it with the
**chat ›** handle at the right edge.

### The Chat panel

Five numbered sections, each collapsible — click a header to fold it away and give the chat room.
Your fold choices persist between sessions. The **dog persona** link loads a finished blend to
learn from.

**1 · describe it** — type words for the trait you want (word stems find more: `depress, unhapp`).
★ rows are directions that were measured to steer their words. Click a row to add it at a starter
dose.

**2 · blend** — every neuron becomes a chip like `L27 N3225 +0.45`: layer, neuron, dose. − / + step
the dose, ± flips its sign, × removes it, double-click the number to type an exact value, and
`all −` / `all +` scale everything. `copy` exports the recipe as text (the `QWENBLEND|…` line
re-imports on any install); `save` stores it as a named preset, listed in the dropdown next to the
blend's name.

**3 · session tuning** — output length, temperature (0 = repeatable, 0.7–1.0 = varied), how much
history the model sees, and an optional persona line for how it presents itself.

**4 · talk** — ask anything. Each reply comes as ▲ pushed up / 0 neutral / ▼ inverted. Raise doses
until ▲ changes; back off when it repeats. The panel explains both cases when they happen.

**5 · brain** — needs the optional brain model; without it, the section explains how to enable it.
Say what you want and watch it work: it searches, adds neurons, tests them live in the same three
columns, critiques, adjusts (a dozen rounds at most), and writes the finished recipe into your
blend. Search is stochastic — so every recipe stays fully editable afterwards.

### Dose sense, from live steering tests

- 0.15–0.3: gentle support
- 0.3–0.5: one strong lead concept
- above ~0.8 on a single word-writer: replies usually collapse into a repeated word

### The map and the other tabs

- Click a neuron: **Details** shows its measurements, connections (what feeds it, what reads it),
  and both sides of its lens. **Examples** shows stored sweep rollouts (pushed vs. baseline), when
  sweep data exists.
- Shift-click neurons on the map (or in the list) to add them to the chat blend directly.
- Highlight text in the Run tab to see which neurons fire on each token.

## Troubleshooting

| symptom | fix |
|---|---|
| first window takes long | the environment or chat model is still installing (choice `[4]` downloads ~9 GB on first launch) |
| no window appears | the one-time WebView2 runtime install was blocked — let it run, or install WebView2 from Microsoft |
| `the server did not come up in time` | another copy is already running (port 8791); close it first |
| low VRAM hiccups | close other AI apps; skip the brain |
| ▲ column identical to 0 | raise doses (+ on the chips) or add more neurons |
| ▲ column repeats one word | dial the biggest chips down |
| the brain says it is switched off | run `LCTNeuronExplorer.exe --setup`, pick `[1]` or `[3]` |
| installer rejects a folder I pointed it at | the folder must be that exact model's folder — the one containing `config.json` and the `.safetensors` weights |

## Credit

This is an original product of LCT. If you use this code or build on it for your own projects,
please credit LCT and link back to this repository.
