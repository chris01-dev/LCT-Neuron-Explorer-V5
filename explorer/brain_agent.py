"""Legacy steering brain (method "agent"), kept for side-by-side benchmarks: a small local agent model (default: the Gemma-4 12B agentic tune, 4-bit) that turns a
natural-language wish into a verified recipe of neuron pushes for the steered chat model.

Loop: decompose the wish into word-stems -> lens_search -> assemble a blend of doses -> test it live
(up / neutral / down) -> critique -> adjust -> finish. Everything streams to the UI as NDJSON events.

    EXPLORER_BRAIN_MODEL  model id or local path (default: the HF repo; downloads on first use).
                          "disabled" leaves the brain off (clean error, everything else works).
    EXPLORER_BRAIN_4BIT   "0" to disable 4-bit NF4 and load bf16 (needs ~24 GB VRAM for the 12B)
"""
import json
import os
import re
import threading

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoTokenizer, BitsAndBytesConfig

BRAIN_MODEL = os.environ.get("EXPLORER_BRAIN_MODEL",
                             "yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2")
BRAIN_DISABLED = BRAIN_MODEL == "disabled"
FOUR_BIT = os.environ.get("EXPLORER_BRAIN_4BIT", "1") != "0"
MAX_ROUNDS = 14
MAX_NEW_TOKENS = 900
PROBES = ["Who are you?", "Say hello and tell me about yourself."]
GUARD = "Give me a few ideas for the weekend."  # open-ended: list-style replies show word loops first
SCORE_TOKENS = 80   # goal words are counted in the first 80 tokens of a reply: the wish has to show early
PROBE_TOKENS = 104  # gate / ladder probes generate the scored head plus a tail where loops become visible
MIN_MARGIN = 2      # a probe passes only when UP says at least 2 more goal words than the untouched NEUTRAL
MOOD_WORDS = {"happy", "fun", "funny", "cute", "adorable", "excited", "exciting", "excitement", "friendly",
              "playful", "joy", "joyful", "cheerful", "love", "lovely", "loving", "sweet", "nice", "good",
              "great", "silly", "energetic", "energy", "enthusiastic", "kind", "warm", "gentle", "curious",
              "innocent", "smile", "laugh", "bouncy", "goofy", "wholesome", "eager", "loyal"}
VOCAB_PROMPT = """Someone wants a small chat model to behave like this: "{goal}"
List the words a reply that really does this would literally SAY: the persona's or topic's own signature vocabulary. Start with its own name and its plain everyday synonyms, then its sounds, its typical things and actions (for "talk like a cat": cat, kitten, kitty, meow, purr, paw, whiskers, fur, nap). Leave out generic mood and quality words (happy, fun, cute, excited, friendly, nice) unless the wish is about that mood itself.
Answer with 8 to 12 single lowercase English words, most characteristic first, comma-separated, nothing else."""
TOKEN_WORD_RE = re.compile(r"[\s_.,;:!?\"'()\-]*([A-Za-z]+)[\s_.,;:!?\"'()\-]*")
STOPWORDS = {"the", "and", "for", "with", "that", "this", "you", "your", "are", "was", "its", "make",
            "makes", "talk", "talks", "talking", "speak", "speaks", "speaking", "write", "writes",
            "writing", "use", "using", "model", "llm", "qwen", "brain", "everything", "always",
            "should", "would", "could", "when", "while", "about", "into", "from", "have", "has",
            "been", "them", "they", "then", "than", "but", "not", "all", "way", "ways", "like",
            "very", "really", "actually", "kind", "sort", "stuff", "things", "replies", "reply",
            "repy", "every", "only", "does", "did", "having", "wants", "want", "asked", "says",
            "saying", "gets", "gives", "little", "much", "many", "more", "most", "over", "under",
            "each", "other", "others", "because", "something", "someone", "anybody", "everybody",
            "sound", "sounds", "act", "acts", "behave", "behaves", "respond", "responds", "answer", "answers",
            "voice", "tone", "style", "character", "persona", "pretend", "pretends", "become", "becomes",
            "seem", "seems", "feel", "feels", "turn", "into", "just", "also", "being", "sure"}

SYSTEM_PROMPT = """You are the steering brain of a neuron explorer application. Your ONLY job: turn the user's wish into a verified recipe of neuron activations (layer, n, dose) that makes the Qwen3.5-4B chat model behave as asked.

Tools:
- lens_search(query): neurons whose lens contains the query word. [verified] means the direction was measured to genuinely steer those words - prefer those; "<- writes goal word" marks neurons whose lens spells one of the goal words. Use one short BARE stem per call (bark, not barking) - never put asterisks or wildcards in the query itself. If a stem finds nothing, exchange it for a different word at once.
- add_neuron(layer, n, dose): add one push to the recipe. Dose sense: 0.15-0.3 gentle support, 0.3-0.5 one strong lead concept, above 0.8 usually collapses replies into one repeated word. Negative dose pushes the other way. Do not round the numbers.
- remove_neuron(layer, n): remove it from the recipe.
- test_steering(prompt): run the current recipe on the live model; returns UP (recipe applied), NEUTRAL (unsteered, the control), DOWN (recipe reversed), then a SCORE line: the GOAL WORDS counted in the first 80 tokens of UP vs NEUTRAL (and which of them hit), and UP's self-repetition (0 = one word looped, 1 = varied text). The goal words are fixed from the user's wish before you start and are listed with it - your searches never change them, so they measure the wish, not your assumptions.
- finish(summary): done; summary is one sentence for the user.

Method, at most 12 rounds of tool calls (you may put SEVERAL tool calls in one round - batch your lens_search calls together):
0. The wish is achieved by NEURON ACTIVATIONS ONLY. There is no way to prompt or instruct the steered model - test_steering injects nothing but the recipe - so a shortcoming can only ever be fixed by changing neurons or doses, never by wording.
1. First say which 2-5 word stems capture the wish (personality words, its vocabulary, topics it should bring up).
2. lens_search ALL your stems in ONE batched round, then a second batch only for gaps. Before adding anything, sort the concepts: GOAL WORDS are the vocabulary listed with the wish - words the steered model itself should SAY (for "talk like a pirate": pirate, arr, matey, ship, treasure). MOOD words (excited, happy, friendly) only dress the voice. The recipe's strongest neuron at 0.3-0.5 must be a verified writer of a GOAL WORD from a HIGH layer (20-31); mood words may support at 0.15-0.25 but must NEVER lead, or the model will talk about the mood instead of doing the wish. By round 3 have a DRAFT recipe of 3-6 neurons across DIFFERENT concepts. Two strong neurons writing the same word multiply each other and collapse into a word-loop: do not do that.
3. test_steering by round 4 with at least TWO different probes and ALWAYS include "Who are you?" for persona or style wishes - identity questions resist personas the most, and a recipe that only shows on one prompt is not done. Stop searching once you have a draft; testing tells you more than more searching.
4. Read the SCORE, not just the vibe. Goal words UP must beat NEUTRAL by at least 2 on every probe, and self-repetition must stay at or above about 0.55. Too subtle (UP barely beats NEUTRAL) -> raise the biggest dose by about 1.3x or add another concept neuron. Repetition low or one word looped -> lower the biggest dose. Wrong flavour -> remove that neuron and lens_search a different stem.
5. finish only when two different probes score clean. finish runs one last check: the lead (the biggest dose) must literally write a goal word, "Who are you?" and a self-introduction probe must each score UP at least 2 goal words above NEUTRAL, and an open-ended probe must not loop. If anything fails you get the scores back and the recipe is NOT accepted - adjust, test_steering again, then finish again.
6. If the rounds run out anyway, the server auto-tunes the whole recipe along a fixed dose ladder and re-scores both probes; a recipe that still cannot beat the control is handed over marked NOT verified. Do not lean on that: test by round 4 so you steer the recipe yourself while rounds remain.
Only use neuron addresses that lens_search returned. Never invent addresses, never skip testing, keep NEUTRAL intact."""

TOOLS = [
    {"type": "function", "function": {"name": "lens_search",
      "description": "find neurons whose lens contains a word; returns layer, n and the words each pushes",
      "parameters": {"type": "object",
                     "properties": {"query": {"type": "string", "description": "a single word or word stem"},
                                    "limit": {"type": "string", "description": "max results, default 8"}},
                     "required": ["query"]}}},
    {"type": "function", "function": {"name": "add_neuron",
      "description": "add a neuron push to the recipe",
      "parameters": {"type": "object",
                     "properties": {"layer": {"type": "string"}, "n": {"type": "string"},
                                    "dose": {"type": "string", "description": "signed number, e.g. 0.35 or -0.2"}},
                     "required": ["layer", "n", "dose"]}}},
    {"type": "function", "function": {"name": "remove_neuron",
      "description": "remove a neuron from the recipe",
      "parameters": {"type": "object",
                     "properties": {"layer": {"type": "string"}, "n": {"type": "string"}},
                     "required": ["layer", "n"]}}},
    {"type": "function", "function": {"name": "test_steering",
      "description": "run the current recipe through the live model and return the three replies",
      "parameters": {"type": "object",
                     "properties": {"prompt": {"type": "string", "description": "short natural prompt"}},
                     "required": []}}},
    {"type": "function", "function": {"name": "finish",
      "description": "call when the recipe is verified; summary is one sentence for the user",
      "parameters": {"type": "object",
                     "properties": {"summary": {"type": "string"}},
                     "required": ["summary"]}}},
]

CALL_RE = re.compile(r"<\|tool_call>call:([^\{\}\s]+)\{(.*?)\}<tool_call\|>", re.DOTALL)
ARG_RE = re.compile(r"([a-zA-Z_]+)\s*:\s*(?:<\|\"\|>)?(.*?)(?:<\|\"\|>)?(?=\s*,\s*[a-zA-Z_]+\s*:|$)", re.DOTALL)
THOUGHT_RE = re.compile(r"<\|channel>thought\s*\n(.*?)\n?<(?:channel\|)>", re.DOTALL)


def _parse_args(s):
    out = {}
    for m in ARG_RE.finditer(s):
        key, val = m.group(1), m.group(2).strip().strip(",").strip()
        if val and key not in out:
            out[key] = val
    return out


def _variants(w, real=frozenset()):
    """A goal word with its plain inflections (dog -> dogs, doggy; puppy -> puppies; bark -> barking). When
    w + 'e' is a word of its own (car/care, wag/wage) the -es/-ed/-ing forms belong to that word and are
    left out; -er forms are left to the prefix rule, which rejects real words (bark -> Barker, pet -> peter)."""
    v = {w, w + "s"}
    if w + "e" not in real:
        v |= {w + "es", w + "ed", w + "ing"}
    if w.endswith("y"):
        v |= {w[:-1] + "ies", w[:-1] + "ied"}
    if w.endswith("e"):
        v |= {w + "d", w[:-1] + "ing"}
    if len(w) <= 4 and w[-1] not in "aeiouy":
        d = w + w[-1]
        v |= {d + "y", d + "ie", d + "ies", d + "o", d + "ed", d + "ing"}
    return v


def _d4(text):
    """Distinct-4-gram ratio of a reply's letter words (digits, list markers and emoji ignored, so numbered
    loops show): 1 varied, ~0 one phrase looped; 3-7 words use the distinct-word ratio; under 3 = collapsed."""
    t = re.findall(r"[^\W\d_]+", str(text).lower())
    if len(t) > 7:
        return len(set(zip(t, t[1:], t[2:], t[3:]))) / (len(t) - 3)
    return len(set(t)) / len(t) if len(t) >= 3 else 0.0


class AgentBrain:
    def __init__(self, rt):
        self.rt = rt
        self.model = self.tok = None
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.status = "idle"
        self._lock = threading.Lock()
        self._kw, self._kw_table, self._kw_long, self._goal_tok = [], {}, [], {}
        self._real = None  # whole-word tokens of the steered model's vocabulary, built once

    # ---------- loading ----------
    def ensure_loaded(self, emit):
        if self.model is not None:
            return
        if BRAIN_DISABLED:
            emit({"t": "status", "text": "the steering brain is switched off on this machine"})
            raise RuntimeError(
                "the steering brain is switched off: it was left out during setup. Run the installer "
                "again with --setup (choice 1 downloads it, choice 3 points at a folder you already "
                "have), or set the EXPLORER_BRAIN_MODEL environment variable.")
        with self._lock:
            if self.model is not None:
                return
            self.status = "loading"
            emit({"t": "status", "text": f"loading the brain model ({'4-bit, about a minute' if FOUR_BIT else 'bf16'}; once)"})
            kw = {"dtype": torch.bfloat16}
            if FOUR_BIT and self.dev == "cuda":
                kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                              bnb_4bit_compute_dtype=torch.bfloat16)
                kw["device_map"] = {"": 0}
            self.tok = AutoTokenizer.from_pretrained(BRAIN_MODEL)
            m = AutoModelForMultimodalLM.from_pretrained(BRAIN_MODEL, **kw)
            if kw.get("device_map") is None:
                m = m.to(self.dev)
            self.model = m.eval()
            self.status = "ready"
            emit({"t": "status", "text": "brain ready"})

    # ---------- tools (direct access to the explorer runtime) ----------
    def _lens_search(self, args):
        q = str(args.get("query", "")).strip().lower()
        q = q.strip("*").strip()  # the star marker lives in the result lines, not in the query
        limit = 8
        try:
            limit = max(2, min(16, int(float(args.get("limit", 8)))))
        except (TypeError, ValueError):
            pass
        if len(q) < 2:
            return "query too short: give one word or stem"
        vocab = self.rt.vocab_strings()
        idx, st = self.rt.lens_index(), self.rt.steer_index()
        # word-anchored: the stem must start the token's word (tail -> Tail, tails; never detail, retail)
        matching = np.array([i for i, t in enumerate(vocab)
                             if str(t).lower().lstrip(" _.,;:!?\"'()-").startswith(q)], dtype=np.int32)
        if not len(matching):
            return f'no lens token starts with "{q}" - try a different stem'
        hits = {}
        for src, mark, arrs in (("lens", False, idx), ("steer", True, st)):
            if arrs is None:
                continue
            arr = arrs["up"]  # pushing UP; the standard buttons only steer the up direction
            hit = np.isin(arr, matching)
            count, best = hit.sum(-1), np.where(hit, np.arange(arr.shape[-1]), 99).min(-1)
            for l, n in zip(*np.nonzero(count)):
                key = (int(l), int(n))
                cur = hits.get(key)
                if cur is None or best[l, n] < cur["rank"]:
                    hits[key] = {"rank": int(best[l, n]), "verified": mark or (cur or {}).get("verified", False),
                                 "lens": " ".join(str(vocab[t]) for t in arr[int(l), int(n), :6])}
        rows = sorted(hits.items(), key=lambda kv: (not kv[1]["verified"], kv[1]["rank"]))[:limit]
        out = []
        for (l, n), v in rows:
            g = self._writes(l, n)
            out.append(f"L{l} N{n} {'[verified]' if v['verified'] else ''} {v['lens']}"
                       + (f"  <- writes goal word: {', '.join(g)}" if g else ""))
        return "\n".join(out) if out else f'no up-direction lens token contains "{q}"'

    # ---------- goal words: fixed from the wish, never from the agent's own searches ----------
    def _goal_words(self, goal):
        """The scoring vocabulary, derived from the wish alone before the agent searches anything: one
        greedy call asks the brain for the words a reply fulfilling the wish would literally say. Generic
        mood words are dropped while at least three concrete ones remain; the wish's own content words
        are the fallback when the call yields too few."""
        literal = list(dict.fromkeys(w.lower() for w in re.findall(r"[a-zA-Z]{3,}", str(goal))
                                     if w.lower() not in STOPWORDS))
        words = []
        try:
            msgs = [{"role": "user", "content": VOCAB_PROMPT.format(goal=" ".join(str(goal).split()))}]
            enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                               tokenize=True, return_tensors="pt")
            ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(self.dev)
            with torch.inference_mode():
                out = self.model.generate(ids, max_new_tokens=80, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            text = self.tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=False).split("<turn|>")[0]
            text = re.sub(r"<\|channel>.*?(?:<channel\|>|$)", " ", text, flags=re.S)  # thought block, closed or not
            text = re.sub(r"<[^>]*>", " ", text)
            for item in re.split(r"[,;:\n]+", text):  # ':' too, so "Here are the words: puppy" yields puppy
                w = re.sub(r"^(?:and|or)\s+", "", item.strip().lower())
                w = w.strip(" \t.*-\"'`()[]0123456789!?:")
                if re.fullmatch(r"[a-z]{3,20}", w) and w not in STOPWORDS:
                    words.append(w)
        except Exception:  # never block a run on the vocabulary call: the wish's own words still score
            words = []
        words = list(dict.fromkeys(words))
        # a wish ABOUT a mood keeps its own mood words: it names one and the brain ranks a mood word first
        keep = {w for w in literal if w in MOOD_WORDS} if words and words[0] in MOOD_WORDS else set()
        # the wish's own concrete words always lead the list (the brain sometimes skips the persona's name)
        words = list(dict.fromkeys([w for w in literal if w not in MOOD_WORDS or w in keep] + words))
        concrete = [w for w in words if w not in MOOD_WORDS or w in keep]
        if len(concrete) >= 3:
            words = concrete
        if len(words) < 3:
            words = list(dict.fromkeys(words + literal))
        return words[:12]

    def _set_goal(self, words):
        """Fixes the scoring words for this run plus their matchers. A word counts for a goal word when it
        is the word or an inflection of it, or starts with it without being some other whole word of the
        steered model's vocabulary: DoggyAI and dog-dog count for dog; treatment never counts for treat,
        nor tailored for tail, nor pupils for pup."""
        vocab = self.rt.vocab_strings()
        if self._real is None:
            self._real = {t[1:].lower() for t in vocab if re.fullmatch(r" [A-Za-z]{4,}", str(t))}
        self._kw = list(words)
        self._kw_table = {}
        for w in self._kw:
            for v in _variants(w, self._real):
                self._kw_table.setdefault(v, w)
        self._kw_long = sorted(self._kw, key=len, reverse=True)
        self._goal_tok = {}  # vocabulary token id -> the goal word it spells
        for i, t in enumerate(vocab):
            m = TOKEN_WORD_RE.fullmatch(str(t))
            g = self._goal_of(m.group(1).lower()) if m else None
            if g:
                self._goal_tok[i] = g

    def _goal_of(self, w):
        """The goal word a lowercase word spells, or None."""
        g = self._kw_table.get(w)
        if g or len(w) < 3 or w in self._real:
            return g
        for k in self._kw_long:
            if not w.startswith(k):
                continue
            # compounds on the word or an inflection count (DoggyAI, pawsome); a longer real word in
            # between means w belongs to that word instead (care-giving, tailor-ing, puppet-eer)
            base = max((v for v, kk in self._kw_table.items() if kk == k and w.startswith(v)), key=len, default=k)
            if not any(w[:i] in self._real for i in range(len(base) + 1, len(w))):
                return k
        return None

    def _hits(self, text):
        """Goal-word occurrences in a text: {goal word: count}."""
        out = {}
        for w in re.findall(r"[a-z]+", str(text).lower()):
            g = self._goal_of(w)
            if g:
                out[g] = out.get(g, 0) + 1
        return out

    def _writes(self, layer, n):
        """The goal words a neuron's up lenses (sweep-verified steer lens and logit lens) literally spell."""
        found = []
        for arrs in (self.rt.steer_index(), self.rt.lens_index()):
            if arrs is None:
                continue
            for t in arrs["up"][int(layer), int(n)]:
                g = self._goal_tok.get(int(t))
                if g and g not in found:
                    found.append(g)
        return found

    def _lead_writes(self, blend):
        """The recipe's lead (its biggest positive dose) and the goal words it writes."""
        pos = [b for b in blend if b["dose"] > 0]
        if not pos:
            return None, []
        lead = max(pos, key=lambda b: b["dose"])
        return lead, self._writes(lead["layer"], lead["n"])

    def _score(self, up, zero, up_head=None, zero_head=None):
        """Goal-word hits in the first SCORE_TOKENS tokens of UP vs the untouched NEUTRAL, plus UP's
        self-repetition (distinct 4-gram ratio, the lower of whole reply and head: 1 varied, ~0 one word
        looped). The wish's words decide, not vibes."""
        uh = up if up_head is None else up_head
        zh = zero if zero_head is None else zero_head
        hits = self._hits(uh)
        return {"up": sum(hits.values()), "zero": sum(self._hits(zh).values()),
                "d4": round(min(_d4(up), _d4(uh)), 2), "same": " ".join(up.split()) == " ".join(zero.split()),
                "kw": " ".join(self._kw), "hits": hits}

    @staticmethod
    def _probe_ok(sc):
        return bool(sc) and not sc["same"] and sc["up"] - sc["zero"] >= MIN_MARGIN and sc["d4"] >= 0.55

    def _test_steering(self, args, blend, emit, max_new=130):
        prompt = str(args.get("prompt", "")).strip() or "Who are you?"
        if not blend:
            return "the recipe is empty: add_neuron first", "", {}
        pushes = [(b["layer"], b["n"], b["dose"], "add") for b in blend]
        texts, heads, steps = ["", "", ""], [None, None, None], [0, 0, 0]
        for row, delta in self.rt.steer_multi(pushes, [[{"role": "user", "content": prompt}]] * 3, max_new):
            if row is not None:
                texts[row] += delta
                steps[row] += 1
                if steps[row] == SCORE_TOKENS:  # fallback only: one delta per token, multi-byte pieces merge
                    heads[row] = texts[row]
        tok = getattr(self.rt, "tok", None)
        if tok is not None:  # the same token count for every row, emoji or not
            heads = [tok.decode(tok(t, add_special_tokens=False).input_ids[:SCORE_TOKENS]) for t in texts]
        heads = [t if h is None else h for h, t in zip(heads, texts)]
        up = self._cut(texts[0], 420)
        zero = self._cut(texts[1], 260)
        down = self._cut(texts[2], 260)
        sc = self._score(texts[0], texts[1], heads[0], heads[1])
        emit({"t": "trial", "prompt": prompt, "up": up, "zero": zero, "down": down, "score": sc})
        hit_txt = ", ".join(f"{w} x{k}" for w, k in sorted(sc["hits"].items(), key=lambda kv: -kv[1])) or "none"
        # SCORE first: the agent's tool message is cut at 900 characters, the replies may lose their tail
        res = (f"SCORE (first {SCORE_TOKENS} tokens): goal words UP {sc['up']} ({hit_txt}) vs NEUTRAL "
               f"{sc['zero']} (need UP >= NEUTRAL + {MIN_MARGIN}), UP self-repetition {sc['d4']} (healthy at or "
               f"above ~0.55)\nUP: {up}\nNEUTRAL: {zero}\nDOWN: {down}")
        return res, up, sc

    @staticmethod
    def _cut(text, n):
        text = " ".join(str(text).split())
        return text if len(text) <= n else text[:n] + " …"

    def _exec(self, name, args, blend, emit):
        """Runs one tool; returns the result string (and mutates blend)."""
        if name == "lens_search":
            return self._lens_search(args)
        if name == "finish":
            return str(args.get("summary", "recipe ready"))
        if name not in ("add_neuron", "remove_neuron", "test_steering"):
            return f"unknown tool {name} - only lens_search, add_neuron, remove_neuron, test_steering and finish exist here"
        if name == "test_steering":
            res, _, _ = self._test_steering(args, blend, emit)
            return res
        def num(key, default=None):
            s = str(args.get(key, default))
            m = re.search(r"-?\d+(?:\.\d+)?", s)
            return float(m.group(0)) if m else default
        layer, n = int(num("layer", -1)), int(num("n", -1))
        if not (0 <= layer < len(self.rt.lm.layers)):
            return f"layer {layer} does not exist in this model - use only addresses from lens_search"
        inter = self.rt.lm.layers[layer].mlp.down_proj.in_features
        if not (0 <= n < inter):
            return f"neuron {n} does not exist in layer {layer}"
        if name == "add_neuron":
            dose = num("dose", 0.25)
            if dose is None:
                return "dose must be a number like 0.35"
            dose = max(-3.0, min(3.0, dose))
            for b in blend:
                if b["layer"] == layer and b["n"] == n:
                    b["dose"] = dose
                    return f"updated L{layer} N{n} to {dose}"
            blend.append({"layer": layer, "n": n, "dose": dose})
            return f"added L{layer} N{n} dose {dose}; recipe now has {len(blend)} neurons"
        if name == "remove_neuron":
            blend[:] = [b for b in blend if not (b["layer"] == layer and b["n"] == n)]
            return f"removed L{layer} N{n}; recipe now has {len(blend)} neurons"
        return f"unknown tool {name} - only lens_search, add_neuron, remove_neuron, test_steering and finish exist here"

    def _pass_all(self, blend, emit):
        """Score the recipe on both mandatory probes (UP must beat NEUTRAL by MIN_MARGIN goal words without
        collapsing) and on the open-ended GUARD prompt (must not collapse). Returns [results], [oks];
        the guard's result comes last."""
        res = [self._test_steering({"prompt": p}, blend, emit, PROBE_TOKENS) for p in PROBES]
        oks = [self._probe_ok(sc) for _, _, sc in res]
        res.append(self._test_steering({"prompt": GUARD}, blend, emit, PROBE_TOKENS))
        oks.append(res[-1][2]["d4"] >= 0.55)
        return res, oks

    @staticmethod
    def _parts(res):
        out = [f"'{p}': goal {sc['up']} vs {sc['zero']}, self-rep {sc['d4']}" for p, (_, _, sc) in zip(PROBES, res)]
        return "; ".join(out + [f"open-ended guard self-rep {res[-1][2]['d4']}"])

    def _rescue(self, blend, emit):
        """Cap-time rescue, deterministic and LLM-free: measure the recipe on the probes; while it fails,
        walk every dose along a fixed ladder and re-score - up (x1.3 ... x2.0) while the replies are
        healthy but too subtle, down (x0.8 steps) when the recipe already loops a reply. Once one scale
        is too weak and another loops, the right dose lies between them (the response is steep there):
        two bisection steps refine it. Passes -> tuned recipe with its numbers. Fails -> None, keeping the
        strongest scale that did not loop once one was found, else the recipe as the brain designed it -
        handed over as explicitly unverified."""
        if not blend:
            return None
        res, oks = self._pass_all(blend, emit)
        if all(oks):
            return f"recipe already scored clean ({self._parts(res)})"
        start = [b["dose"] for b in blend]
        looped = lambda r: any(sc["d4"] < 0.55 for _, _, sc in r)

        def at(scale, label):
            for b, d in zip(blend, start):
                b["dose"] = round(d * scale, 3)
            lead = max(blend, key=lambda b: abs(b["dose"]))
            emit({"t": "blend", "blend": blend.copy()})
            emit({"t": "status", "text": f"auto-tune {label}: doses x{scale:.2f} of the design "
                                         f"(lead L{lead['layer']} N{lead['n']} at {lead['dose']})"})
            return self._pass_all(blend, emit)

        down = looped(res)
        lo, hi = (None, 1.0) if down else (1.0, None)  # dose scales: healthy but too weak / loops
        scale = 1.0
        for k, f in enumerate([0.8, 0.8, 0.8] if down else [1.3, 1.3, 1.5, 1.7, 2.0]):
            scale *= f
            res, oks = at(scale, f"step {k + 1}")
            if all(oks):
                return f"auto-tune PASSED at step {k + 1}, doses x{scale:.2f} of the design ({self._parts(res)})"
            if looped(res):
                hi = scale
            else:
                lo = scale
            if lo is not None and hi is not None:
                break
        bracketed = lo is not None and hi is not None
        if bracketed:
            for j in range(2):
                mid = round((lo * hi) ** 0.5, 3)
                res, oks = at(mid, f"refine {j + 1}")
                if all(oks):
                    return f"auto-tune PASSED at refine step {j + 1}, doses x{mid:.2f} of the design ({self._parts(res)})"
                if looped(res):
                    hi = mid
                else:
                    lo = mid
        keep = lo if bracketed else 1.0
        for b, d in zip(blend, start):
            b["dose"] = round(d * keep, 3)
        emit({"t": "blend", "blend": blend.copy()})
        return None

    def _finish_gate(self, blend, emit):
        """finish is only accepted when the recipe's lead literally writes a goal word and the recipe
        scores clean: on both mandatory probes goal words beat the untouched control by MIN_MARGIN without
        collapse, and the open-ended guard prompt does not loop.
        Returns (None, score_line) when accepted, (feedback, None) when rejected."""
        if not blend:
            return ("finish rejected: the recipe is empty. add_neuron first, "
                    "test_steering, then finish."), None
        lead, writes = self._lead_writes(blend)
        if not writes:
            what = (f"its lead L{lead['layer']} N{lead['n']} (dose {lead['dose']}) writes none of the goal words"
                    if lead else "it has no positive dose to lead it")
            return (f"finish rejected: {what}. The biggest push must be a [verified] writer of a goal word "
                    f"({', '.join(self._kw)}): lens_search one of them, add its best '<- writes goal word' neuron "
                    f"from a high layer at 0.3-0.5 so it is the biggest dose, test_steering, then finish again."), None
        res, oks = self._pass_all(blend, emit)
        lines = [f'"{p}": goal words {sc["up"]} vs {sc["zero"]}'
                 f'{", UP identical to NEUTRAL" if sc["same"] else ""}'
                 f', self-repetition {sc["d4"]}' + (" - FAILED" if not ok else "")
                 for p, (_, _, sc), ok in zip(PROBES, res, oks)]
        lines.append(f'open-ended guard "{GUARD}": self-repetition {res[-1][2]["d4"]}'
                     + ("" if oks[-1] else " - FAILED (word loop)"))
        if all(oks):
            return None, "; ".join(lines + [f"lead L{lead['layer']} N{lead['n']} writes {', '.join(writes)}"])
        return ("finish rejected: the recipe did not score clean.\n" + "\n".join(lines)
                + f"\nfix it: UP needs at least {MIN_MARGIN} more goal words than NEUTRAL on each probe - raise the "
                  "biggest dose by about 1.3x, add one support concept, or swap a neuron; a word loop needs a "
                  "lower dose. test_steering the failing prompt again, then call finish again."), None

    def _seed_candidates(self):
        """Every neuron whose up lens (logit lens or sweep-verified steer lens) spells a goal word: the goal
        words it writes, how purely its verified steer lens writes them (goal tokens among its top 12 and
        top 6) and the rank of the first one."""
        keys = np.array(sorted(self._goal_tok), dtype=np.int32)
        if not len(keys):
            return []
        cands = {}
        for verified, arrs in ((False, self.rt.lens_index()), (True, self.rt.steer_index())):
            if arrs is None:
                continue
            arr = arrs["up"]
            hit = np.isin(arr, keys)                                           # [L, N, k]
            for l, n in zip(*np.nonzero(hit.any(-1))):
                c = cands.setdefault((int(l), int(n)), {"layer": int(l), "n": int(n), "verified": False,
                                                         "words": set(), "pure": 0, "pure6": 0, "rank": 99})
                c["words"].update(self._goal_tok[int(t)] for t in arr[l, n] if int(t) in self._goal_tok)
                if verified:
                    c.update(verified=True, pure=int(hit[l, n].sum()), pure6=int(hit[l, n, :6].sum()),
                             rank=int(np.argmax(hit[l, n])))
        return list(cands.values())

    def _seed_from_words(self, emit):
        """Cookbook recipe, deterministic and LLM-free, built from the goal words alone. Lead at 0.45: the
        purest sweep-verified writer (at least 3 goal tokens in its top 6) in a high layer (20-31) of the
        most characteristic goal word that has one. Up to 3 supports at 0.25: verified writers with at
        least 2 goal tokens in their top 12 that add goal words the recipe does not write yet, the wish's
        most characteristic words first. The rescue ladder then tunes the result."""
        cs = [c for c in self._seed_candidates() if c["verified"]]
        if not cs:
            return []
        lead = None
        for w in self._kw:  # most characteristic first
            pool = [c for c in cs if w in c["words"] and c["layer"] >= 20 and c["pure6"] >= 3]
            if pool:
                lead = max(pool, key=lambda c: (c["pure6"], c["pure"], len(c["words"]), -c["rank"], c["layer"]))
                break
        if lead is None:
            lead = max(cs, key=lambda c: (c["layer"] >= 20, c["pure6"], c["pure"], len(c["words"]), -c["rank"]))
        order = {w: i for i, w in enumerate(self._kw)}
        blend, used = [], set()
        for c in [lead, None, None, None]:
            if c is None:
                pool = [x for x in cs if x["pure"] >= 2 and x["words"] - used
                        and all((x["layer"], x["n"]) != (b["layer"], b["n"]) for b in blend)]
                if not pool:
                    break
                c = min(pool, key=lambda x: (min(order[w] for w in x["words"] - used), -len(x["words"] - used),
                                             -x["pure"], x["rank"], -x["layer"]))
            role = "lead" if not blend else "support"
            new = sorted(c["words"] - used, key=order.get)
            blend.append({"layer": c["layer"], "n": c["n"], "dose": 0.45 if not blend else 0.25})
            used |= c["words"]
            emit({"t": "status", "text": f"goal-word recipe {role}: L{c['layer']} N{c['n']} writes "
                                         f"{', '.join(new)} ({c['pure']} goal tokens in its top 12)"})
        return blend

    # ---------- the loop ----------
    def _generate(self, msgs):
        enc = self.tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True,
                                           enable_thinking=True, tokenize=True, return_tensors="pt")
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        ids = ids.to(self.dev)
        with torch.inference_mode():
            out = self.model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                                      temperature=0.85, top_p=0.95, top_k=64,
                                      repetition_penalty=1.05, pad_token_id=self.tok.pad_token_id)
        text = self.tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=False)
        return text.split("<turn|>")[0].split("<eos>")[0]

    def run(self, goal, emit):
        if self._lock.locked() and self.model is None:
            emit({"t": "error", "error": "the brain is still loading"}); return
        self.rt.ensure_loaded()
        self.ensure_loaded(emit)
        if self.status == "busy":
            emit({"t": "error", "error": "a brain run is already in progress"}); return
        self.status = "busy"
        try:
            emit({"t": "status", "text": "deriving the goal words from the wish"})
            self._set_goal(self._goal_words(goal))
            emit({"t": "status", "text": "goal words (fixed from the wish; every test counts them): "
                                         + (" ".join(self._kw) or "none found")})
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{goal}\n\nGOAL WORDS (fixed from this wish; every test counts them "
                                                f"in UP vs NEUTRAL, your searches never change them): "
                                                f"{', '.join(self._kw) or 'none'}"}]
            blend = []
            summary = "recipe ready"
            tested = False
            reminded = False
            stumbles = 0
            for i in range(MAX_ROUNDS):
                emit({"t": "round", "i": i + 1})
                text = self._generate(msgs)
                th = THOUGHT_RE.search(text)
                if th:
                    emit({"t": "thought", "text": " ".join(th.group(1).split())[:600]})
                calls = [(m.group(1), _parse_args(m.group(2))) for m in CALL_RE.finditer(text)]
                if not calls:
                    stumbles += 1
                    emit({"t": "status", "text": f"round {i + 1} produced no parseable tool call - nudging the format (stumble {stumbles}/3)"})
                    if stumbles > 3:  # stop the agent, but still verify what it built (or the goal-word recipe)
                        emit({"t": "status", "text": "the brain kept tripping over its own tool-call format - "
                                                     "stopping the agent and verifying what it has"})
                        break
                    msgs.append({"role": "user", "content": "Format reminder: the previous reply contained no usable tool call. Reply again with tool calls ONLY, in the exact form <|tool_call>call:tool_name{arg: value}<tool_call|>, using only lens_search, add_neuron, remove_neuron, test_steering or finish."})
                    continue
                msgs.append({"role": "assistant", "content": "", "tool_calls":
                             [{"function": {"name": nm, "arguments": {k: v for k, v in a.items()}}} for nm, a in calls]})
                for nm, a in calls:
                    emit({"t": "tool", "name": nm, "input": a})
                    if nm == "finish":
                        reject, scores = self._finish_gate(blend, emit)
                        if reject:
                            emit({"t": "result", "name": "finish", "text": self._cut(reject, 400)})
                            msgs.append({"role": "tool", "name": "finish", "content": self._cut(reject, 900)})
                            continue
                        summary = self._exec(nm, a, blend, emit)
                        emit({"t": "blend", "blend": blend.copy()})
                        if scores:
                            summary = f"{summary}; verified: {scores}"
                        emit({"t": "done", "summary": summary, "verified": True})
                        return
                    if nm == "test_steering":
                        tested = True
                    res = self._exec(nm, a, blend, emit)
                    emit({"t": "result", "name": nm, "text": self._cut(res, 400)})
                    msgs.append({"role": "tool", "name": nm, "content": self._cut(res, 900)})
                if not tested and i >= 5 and not reminded:
                    reminded = True
                    msgs.append({"role": "user", "content": 'Reminder: half the rounds are gone and you have '
                                 'not tested once. In the very next round call test_steering on "Who are you?" '
                                 "and act on the SCORE instead of searching more."})
                    emit({"t": "status", "text": "injected: test now, before more searching"})
                emit({"t": "blend", "blend": blend.copy()})
            emit({"t": "blend", "blend": blend.copy()})
            stop = "agent stopped early" if stumbles > 3 else "round cap"
            seeded = False
            if not blend:
                emit({"t": "status", "text": "no recipe from the agent - assembling the deterministic goal-word recipe"})
                blend = self._seed_from_words(emit)
                seeded = True
                emit({"t": "blend", "blend": blend.copy()})
            verdict = self._rescue(blend, emit) if blend else None
            lead_ok = bool(blend) and bool(self._lead_writes(blend)[1])
            verified = False
            if verdict and (lead_ok or seeded):
                verified = True
                summary = f"recipe handed over ({stop}) - {verdict}"
            elif blend and not seeded:
                # A/B: the agent's recipe did not verify, or verified with a lead that writes none of the
                # goal words - build the deterministic goal-word recipe and prefer it when it verifies
                why = "did not verify" if not verdict else "verified, but its lead writes none of the goal words"
                emit({"t": "status", "text": f"the agent's recipe {why} - A/B against the deterministic goal-word recipe"})
                seed = self._seed_from_words(emit)
                seed_verdict = self._rescue(seed, emit) if seed else None
                if seed_verdict:
                    blend[:] = seed
                    verified = True
                    summary = f"deterministic goal-word recipe won the A/B and verified - {seed_verdict}"
                elif verdict:
                    verified = True  # the numbers passed; the caveat is in the summary
                    summary = (f"recipe handed over ({stop}) - {verdict}; its lead writes none of the goal "
                               "words, and the deterministic goal-word recipe did not verify")
                else:
                    summary = (f"recipe handed over ({stop}) but NOT verified: neither the agent's recipe nor "
                               "the deterministic goal-word recipe beat the control on both probes. Raise the "
                               "lead dose toward a verified goal-word writer, and retest - or keep it as a "
                               "starting point and tune it in the blend")
            elif blend:
                summary = (f"recipe handed over ({stop}) but NOT verified: the agent built nothing and the "
                           "deterministic goal-word recipe did not beat the control on both probes - keep it as "
                           "a starting point and tune it in the blend")
            else:
                summary = ("no recipe at all: the agent built nothing and no neuron writes any of the goal "
                           "words - try wording the wish with different vocabulary")
            emit({"t": "blend", "blend": blend.copy()})  # the handed-over recipe is always the last blend
            emit({"t": "done", "summary": summary, "verified": verified})
        finally:
            self.status = "ready"
