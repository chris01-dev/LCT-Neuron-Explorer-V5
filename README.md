# LCT Neuron Explorer

Steer a language model by turning individual neurons up or down instead of messing with the prompt.

You describe the behavior you want in normal words, the app finds neurons tied to those words, and
each reply shows up in three columns: neurons pushed up, the untouched model, and neurons pushed
down. Same prompt in all three, so whatever's different is coming from the neurons.

If you don't want to hunt for neurons yourself, there's a steering "brain" that does it for you.
Give it something like *"make it talk like a grumpy wizard"* and it tests neurons against the model,
keeps the ones that actually work, and gives you back a recipe. It only calls a recipe verified if it
still holds up on prompts it wasn't tuned on.

## What's in it

- A 3D map of Qwen3.5-4B where every dot is a real neuron. Hover one to see its vocabulary, click it
  for its measurements, shift-click to throw it into your blend.
- Lens search to find neurons by the words they push. Anything marked ★ was tested in steering
  sweeps and really changes the output.
- The three-column chat (up / neutral / inverted) so you see the effect immediately.
- The steering brain, with every trial shown live: replies, scores, and why a neuron got kept or
  cut.
- Presets, plus a short text format for sharing blends with other people.

## How it works

The steered model is Qwen3.5-4B. Steering a neuron means adding a fixed amount to its activation at
its layer for the whole reply, and the dose is scaled to that neuron's normal activation range. The
app comes with data precomputed for every neuron: its lens (which tokens it makes more or less
likely, measured directly with a forward pass), its strongest sweep examples, and which neurons it
writes to and reads from.

The brain is Gemma-4 12B, an agentic fine-tune loaded in 4-bit. It never writes prompts for the chat
model. It only steers through neurons. Here's roughly what it does with a request:

1. Figures out what words that style would use, what the model's normal behavior looks like, and
   writes a few example replies in the target style.
2. Collects candidate neurons from every layer, in both directions. That covers neurons whose lens
   writes the right words or formatting (emoji, \*actions\*, ALL CAPS, "!!"), neurons behind the
   default behavior (those get pushed negative), and neurons that fire differently on the example
   replies than on the model's normal ones.
3. Runs a quick likelihood screen on about 150 of those. Each is pushed both ways, and it measures
   how much more likely the push makes the example replies compared to plain rewrites of the same
   content. No text gets generated here, so it's fast. It does this twice, the second time at double
   the dose, because some neurons don't show anything until you push them harder.
4. Tests the best candidates one at a time on the live model, and eases off if one is strong enough
   to drown out the reply.
5. Stacks neurons one by one while the effect keeps improving and the replies still make sense. If
   one doesn't fit, it gets a second shot at a different strength (lower if it broke things, full
   strength if it barely did anything).
6. Tunes one overall dose for the recipe, then checks it on held-out prompts that weren't used
   anywhere above.

Replies are scored blind, so the brain doesn't know which column was steered. It gives each one a
0-10 score for how much it shows the style and a 0-10 for coherence (taken as the expected value of
the score digits). Compared against three human raters, those scores correlate at r = 0.86 for style
and 0.94 for coherence. If the request includes a rule like "never mention being an AI," every reply
also gets checked for it, and recipes that fix it are preferred.

You can bail early. **stop · keep best** ends the run on the next trial and drops the best recipe so
far into your blend, and **use ▲ / use ▼** on any trial takes exactly those neurons (as-is or
flipped) and stops the brain.

The app shows you the outputs and the numbers next to the baseline. It doesn't try to tell you what a
neuron "means". That part's up to you.

## Requirements

| | chat only | with the steering brain |
|---|---|---|
| OS | Windows 10/11 64-bit | Windows 10/11 64-bit |
| GPU | NVIDIA, 12 GB VRAM | NVIDIA, 16 GB VRAM (24 GB is comfortable) |
| RAM | 16 GB (32 GB recommended) | 32 GB |
| Disk space | ~17 GB | ~41 GB |

Model weights aren't included in the download. Make sure your NVIDIA driver is current since it runs
on CUDA 13. AMD and Intel GPUs won't work.

## Install

1. Download `LCTNeuronExplorer.exe` from [Releases](../../releases) (about 100 MB). That one file is
   the installer and the app.
2. Run it. The first time, it unpacks to `%LOCALAPPDATA%\LCTNeuronExplorer` and sets up Python and
   CUDA, which takes 5-15 minutes and only happens once. Then it asks about the models:

   | option | what it does |
   |---|---|
   | `[1]` download both | chat model (~9 GB) and brain (~24 GB) |
   | `[2]` chat model only | everything but the brain works; you can add it later |
   | `[3]` I already have them | paste the folder paths. It checks that each one is exactly the right model (type, architecture, weights) and tells you what's wrong if not |
   | `[4]` skip | the chat model downloads when you first open the app; the brain stays off |

   Both models are free. If you'd rather grab them yourself, here's
   [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) and the
   [Gemma-4 12B agentic tune](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2).
   Then pick `[3]`.

   Downloads can be resumed. If one gets cut off, run the exe again and pick the same option.
3. The app opens. Loading the model into VRAM takes 30-60 seconds the first time and is quicker
   after that.

**Updating:** download the new exe and run it. Your models and settings are kept.

**Changing models:** `LCTNeuronExplorer.exe --setup`

**Uninstalling:** delete the exe and `%LOCALAPPDATA%\LCTNeuronExplorer` (models are stored there
too).

### From source

```powershell
git clone https://github.com/chris01-dev/LCT-Neuron-Explorer-V5.git
cd LCT-Neuron-Explorer-V5
uv sync                 # Python + PyTorch CUDA
uv run python desktop.py
```

There's also an HTTP API on `127.0.0.1:8791` if you want to script things: `/api/lens`,
`/api/steer`, `/api/reads`, `/api/search`, `/api/status`, and for the brain `/api/brain/chat`
(event stream), `/api/brain/plan` and `/api/brain/judge`.

### Benchmarks

`tools/brain_bench.py` runs the brain on a fixed set of wishes and prompts, and compares it to
hand-made reference recipes (where there are any) and the untouched model. Each reply gets scored by
the judge and also by a per-wish word list, which doesn't use a model at all. Results get appended to
`bench/history.jsonl` so versions can be compared on the same numbers.

Wish sets:

- `dev` - 9 wishes used during development (3 short, 6 detailed personas)
- `heldout` - 10 never seen during development (7 short, 3 detailed)
- `user` - 2 wishes written by a user, word for word, with the replies they wanted
- `baseline6` - the 6-wish set used to compare versions

```powershell
uv run python tools/brain_bench.py --wishes baseline6 --label v5
uv run python tools/brain_bench.py --wishes heldout,user --label v5-heldout
uv run python tools/brain_bench.py --wishes heldout --options "{\"sources\": \"say,avoid,marker,contrast,default\"}" --label no-likely
```

## Using it

The main screen is a 3D map. The spine going up the middle is the residual stream through all 32
layers, each slab is an attention/MLP block, and each dot is a neuron. The left panel shows the
selected neuron's connections and where it ranks in its layer. Steering happens on the right; open
it with the **chat ›** tab on the edge of the screen.

### Chat panel

Five sections, all collapsible (it remembers which ones you closed). If you want an example to poke
at, the **dog persona** link loads a finished blend.

**1 · describe it** - Type words for the trait you're after. Stems work better (`depress, unhapp`).
★ rows were measured to actually steer. Click a row to add it at a starting dose.

**2 · blend** - Each neuron is a chip like `L27 N3225 +0.45` (layer, neuron, dose). − and + adjust
the dose, ± flips it, × removes it, and double-clicking the number lets you type a value. `all −` and
`all +` scale everything. `copy` gives you the recipe as text (paste the `QWENBLEND|…` line into any
install), and `save` stores it as a preset.

**3 · session tuning** - Output length, temperature (0 is repeatable, 0.7-1.0 for more variety), how
much history the model sees, and an optional persona line.

**4 · talk** - Chat normally. Every reply comes back as ▲ up / 0 neutral / ▼ inverted. Turn the doses
up until ▲ starts looking different, and back off if it starts repeating itself.

**5 · brain** - Needs the brain model (it'll tell you how to turn it on if it's missing). Say what you
want and watch it go. It searches, adds neurons, tests them in the same three columns, critiques,
and adjusts for up to about a dozen rounds, then puts the result in your blend. The search is a bit
random, so expect to tweak the recipe after.

### Dose cheat sheet

From testing:

- 0.15-0.3 is a light nudge
- 0.3-0.5 works for one strong concept
- Over ~0.8 on a single word-writing neuron and replies usually turn into one word repeated forever

### Map and other tabs

- Clicking a neuron opens **Details**: measurements, what feeds into it, what reads from it, and both
  sides of its lens. **Examples** shows sweep rollouts (pushed vs. baseline) when that data exists.
- Shift-click neurons on the map or in the list to add them to your blend.
- Highlight text in the Run tab to see which neurons fire on each token.

## Troubleshooting

| problem | fix |
|---|---|
| first launch is taking forever | it's still installing the environment or the chat model (option `[4]` pulls ~9 GB the first time you open it) |
| no window opens | the one-time WebView2 install got blocked. Let it through, or install WebView2 from Microsoft |
| `the server did not come up in time` | another copy is already running on port 8791. Close that one |
| stuttering / low VRAM | close other AI apps, or run without the brain |
| ▲ looks the same as 0 | turn the doses up or add more neurons |
| ▲ just repeats one word | turn the biggest chips down |
| brain says it's switched off | run `LCTNeuronExplorer.exe --setup` and choose `[1]` or `[3]` |
| installer won't accept my folder | point it at the model's own folder, the one with `config.json` and the `.safetensors` files |

## Credit

Made by LCT. If you use this code or build on it, please credit LCT and link back here.
