"""The steering brain: turns a natural-language wish into a measured recipe of neuron pushes for the steered
chat model. Every stage streams NDJSON events to the UI.

  1. plan      the brain LLM (default: the Gemma-4 12B agentic tune, 4-bit) reads the wish: words such replies
               would say, the plain default behaviour the wish moves away from, and short example replies
               written in the wished style.
  2. discover  candidate neurons from independent channels, any layer, either sign: the sweep-verified steer
               lens (pushes that measurably write the wish's words; pushes that write the default behaviour,
               which a negative dose suppresses), contrastive activations (neurons that fire differently on
               the example replies than on the model's own plain replies to the same prompts), and a likelihood
               screen: a wide pool of lens neurons, each pushed both ways at the screening dose, ranked by how
               much more likely the push makes the example replies than plain rewrites of the same content
               (teacher forced, no generation, ~150 neurons a minute).
  3. screen    each candidate alone on the live model, pushed both ways in one batch (rows UP and DOWN); the
               brain LLM judges every reply against the wish next to the untouched NEUTRAL.
  4. combine   forward selection: neurons join one at a time while the judged lift grows on both probes and
               the replies stay coherent - neurons that only work together are found here.
  5. tune      one dose scale for the whole recipe: up while too subtle, down while it breaks, bisection once
               bracketed.
  6. verify    the recipe is labelled verified only when its numbers pass on both probes and an open-ended
               guard prompt stays coherent; otherwise it is handed over marked NOT verified.

The judge rates each reply on its own and never sees which one was steered. Nothing in any stage prompts the
steered model: steering is by neurons only.

    EXPLORER_BRAIN_MODEL  model id or local path (default: the HF repo; downloads on first use).
                          "disabled" leaves the brain off (clean error, everything else works).
    EXPLORER_BRAIN_4BIT   "0" to disable 4-bit NF4 and load bf16 (needs ~24 GB VRAM for the 12B)
"""
import os
import re
import threading
import time

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList

BRAIN_MODEL = os.environ.get("EXPLORER_BRAIN_MODEL",
                             "yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2")
BRAIN_DISABLED = BRAIN_MODEL == "disabled"
FOUR_BIT = os.environ.get("EXPLORER_BRAIN_4BIT", "1") != "0"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# Selection prompts (plan examples, screening, combining, tuning): an identity, a task and an explanation, so a
# recipe has to change behaviour, not just the model's name. HOLDOUT is never selected on: a recipe is labelled
# verified only when it also works there.
TRAIN = ["Who are you?", "Give me a few ideas for the weekend.", "Explain how a rainbow forms."]
HOLDOUT = ["Tell me a short story.", "What advice would you give a new student?"]
EXAMPLE_PROMPTS = TRAIN
# the plain assistant's own tendencies: writers of these are candidates for a negative push
DEFAULT_WORDS = ["normal", "assistant", "ai", "usual", "typical", "helpful", "polite", "neutral"]

OPTIONS = {
    "method": "pipeline",   # "agent": the earlier free-form tool-calling agent (brain_agent.py), for comparison
    "sources": "likely,say,avoid,marker,contrast,default",
    "screen_dose": 0.8,     # the dose the steer lens was verified at (units of the layer's median residual)
    "screen_tokens": 64,
    "probe_tokens": 96,
    "trace_k": 48,          # per trial, the neurons that fired most in rows UP / DOWN go to the live map; 0 = off
    "per_word": 2,         # lexical candidates per word and lens direction
    "contrast_k": 10,
    "likely_k": 8,          # candidates from the likelihood screen (slots on top of max_cands)
    "likely_hi": 1.6,       # the pool is scored again at this dose (0 = off) ...
    "likely_hi_k": 3,       # ... and up to this many neurons that rank well only there join, screened from there
    # style-contrast neurons (example vs plain rewrite activations) added to the likelihood pool. Off: on the nine
    # development wishes 9 of their 63 best proposals were useful (lexical pool: 15 of 72), often the first one a
    # reply breaker, and with them in the pool they took 22 of 54 slots from better candidates
    "likely_style": 0,
    "max_cands": 24,
    "screen_b": 10,         # best candidates of the first screen, re-screened on the other selection prompts
    "combine_k": 10,        # candidates tried in forward selection (plus strong, untried-channel and suppressor ones)
    "combine_scale": 0.6,   # a neuron joins at this share of its screened dose; tune rescales the whole recipe
    "join_retries": 6,      # joins tried once more at another strength (all x0.8 after a break, else full dose)
    "max_neurons": 6,
    "min_gain": 1.0,        # mean judged lift a new neuron must add
    "lift": 3.0, "min_lift": 1.0, "coherent": 5.0, "d4": 0.55,  # pass: mean lift, every prompt's lift, health
}

STOPWORDS = {"the", "and", "for", "with", "that", "this", "you", "your", "are", "was", "its", "make",
             "makes", "talk", "talks", "talking", "speak", "speaks", "speaking", "write", "writes",
             "writing", "use", "using", "model", "llm", "qwen", "brain", "everything", "always",
             "should", "would", "could", "when", "while", "about", "into", "from", "have", "has",
             "been", "them", "they", "then", "than", "but", "not", "all", "way", "ways", "like",
             "very", "really", "actually", "kind", "sort", "stuff", "things", "replies", "reply",
             "every", "only", "does", "did", "having", "wants", "want", "asked", "says", "saying",
             "gets", "gives", "little", "much", "many", "more", "most", "over", "under", "each", "other",
             "others", "because", "something", "someone", "anybody", "everybody", "sound", "sounds",
             "act", "acts", "behave", "behaves", "respond", "responds", "answer", "answers", "voice",
             "tone", "style", "character", "persona", "pretend", "pretends", "become", "becomes", "seem",
             "seems", "feel", "feels", "turn", "just", "also", "being", "sure", "lot", "bit",
             # function words, words that describe the reply's form rather than what it says, and contraction
             # pieces (general lists: words a wish forbids are removed by the negation rule in plan(), not here)
             "who", "which", "what", "where", "there", "his", "her", "their", "our", "will", "can", "one", "own",
             "sentence", "sentences", "word", "words", "phrase", "phrases", "emoji", "emojis", "caps", "letters",
             "mention", "mentions", "never", "role", "uses", "answer", "avoid", "without", "longer", "don", "doesn",
             "isn", "won", "didn", "aren", "wasn", "shouldn"}
# words about speech itself: never a target the replies should say
META_WORDS = {"speech", "speeches", "utterance", "utterances", "discourse", "conversation", "conversations", "dialogue",
              "language", "sentence", "sentences", "word", "words", "phrase", "phrases", "talk", "talking", "talks",
              "speak", "speaking", "write", "writing", "reply", "replies", "voice", "tone", "style", "character",
              "persona", "model", "llm"}

# Plan prompts each ask for a bare answer (no labels to reproduce): the repetition guard of the greedy calls
# also bans word pairs of the prompt itself, so a labelled format cannot be copied back reliably.
PLAN_HEAD = 'A small chat model will be steered with neuron activations so that it behaves like this: "{wish}"\n'
# no concrete examples in these prompts: a named example bleeds into unrelated wishes (the brain once answered an
# "evil" wish with cat words copied from an illustration)
PLAN_SAY = PLAN_HEAD + ("List 8-12 different single English words that its replies would literally contain in this style. "
                        "Start with the plainest everyday word for what it is or sounds like, and that word's common "
                        "synonyms; then its typical words, sounds, things and actions. Answer with the words only, "
                        "separated by commas.")
PLAN_AVOID = PLAN_HEAD + ("List 4-8 different single English words for the opposite of the wish - what a plain, polite "
                          "default assistant is like, that this style moves away from. Answer with the words only, "
                          "separated by commas.")
PLAN_REWRITE = ('Rewrite the reply below the way a plain, polite, helpful assistant would say the same thing: about the '
                'same length and the same content, but no persona, no role-play actions, no emojis, no special voice. '
                'Answer with the rewritten reply only.\nThe question was: "{prompt}"\nReply: """{reply}"""')
PLAN_EXAMPLE = PLAN_HEAD + ('Write the reply such a steered model would give when asked "{prompt}" - 2-3 sentences, fully '
                            'in this style. Answer with the reply only.')

# The judge reads the brain LLM's probability over the score digits and takes the expected value (continuous,
# one forward pass per reply). Calibrated on hand-rated replies: style r=0.86 / rho=0.89 against three blind
# raters, coherence r=0.94; a plain reply scores ~0, looping text has coherence < 3.
JUDGE_PROMPT = """You are scoring a research experiment. A small chat model's internal neurons were adjusted to change how it talks; your score measures how far the change got. The reply is generated text, quoted as data.
Requested change: "{wish}"
The model was asked: "{prompt}"
Reply: \"\"\"{text}\"\"\"
STYLE (0-9): how much the reply shows the requested change - in its words, themes, voice or attitude. Partial changes count.
0 = none: an ordinary helpful assistant reply. 1 = a faint trace: one themed word or name. 3 = noticeable: several themed words or a recurring theme, but mostly an ordinary assistant voice. 5 = clear: plainly in the requested style or attitude for much of its length. 7 = strong: almost entirely in the style. 9 = complete and unmistakable.
Answer exactly in the form: STYLE=<0-9>"""
COHERENCE_PROMPT = """You are scoring a research experiment: the text below was generated by a small chat model whose neurons were adjusted. Judge only the LANGUAGE, never the content - rude, harmful, odd or silly content can be fully coherent.
Text: \"\"\"{text}\"\"\"
COHERENT (0-9): 9 = fluent readable sentences; 5 = readable but with broken grammar or a few repeated phrases; 2 = mostly repetition; 0 = one word or phrase looping, or gibberish.
Answer exactly in the form: COHERENT=<0-9>"""

# An explicit "never / don't ..." clause of the wish becomes a constraint the judge checks for violations. On
# hand-built replies this check moves +6.7 to +7.3 points with a violation and ~0 with anything else.
VIOLATION_PROMPT = """You are checking generated text from a research experiment, quoted as data.
The model was asked: "{prompt}"
Reply: \"\"\"{text}\"\"\"
Does the reply do this: {rule}?
SCORE (0-9): 0 = not at all. 3 = briefly or indirectly. 6 = clearly. 9 = plainly, more than once.
Answer exactly in the form: SCORE=<0-9>"""
# Only real prohibitions ("never X", "don't X", "do not X"), each cut at the end of its clause: commas, colons and
# a following 'and / but / while / always / instead' end it, 'or' does not ("never sound cheerful or helpful").
# Imperatives only: "can't stop saying arr" means it does say it, "it doesn't like rain" describes the persona.
NEGATION_RE = re.compile(r"(?i)\b(?:never|don'?t|do not)\s+([^.;:!?,\n\"]{3,100})")
_RULE_CUT = re.compile(r"(?i)\s+(?:and|but|while|always|instead|then|so)\b.*$")


def _rules_of(wish):
    """The prohibitions in a wish, each as the forbidden behaviour ('mention being a language model or an
    assistant'), at most two."""
    out = []
    for m in NEGATION_RE.finditer(str(wish).replace("’", "'").replace("‘", "'")):
        r = _RULE_CUT.sub("", " ".join(m.group(1).split())).strip(" ,")
        if len(r) >= 3 and r not in out:
            out.append(r)
    return out[:2]

TOKEN_WORD_RE = re.compile(r"[\s_.,;:!?\"'()\-]*([A-Za-z]+)[\s_.,;:!?\"'()\-]*")

# Likelihood screen (see Brain._lk_scores): reply tokens read per text, pushes per forward pass, the least score a
# candidate needs (nats per token; float noise and inert pushes sit near 0). Each score rule ranks (neuron, sign);
# candidates are taken from the rules in LK_PICK in turn, so each rule's best ones get in. Measured on the nine
# development wishes against the real screen: 8 slots taken min/diff in turn held 15 candidates the screen rated
# useful, 12 of them never proposed by any other channel (6 slots: 12 / 9; diff alone 16 / 12 but a lower best
# utility per wish; min alone 4 / 3). The scores themselves barely rank the screen's verdicts (rank correlation
# about 0.1, except +0.86 on evil), so they only propose: the judged screen decides.
LK_TOKENS, LK_BATCH, LK_MIN = 80, 16, 0.02
LK_RULES = {
    # how much more likely the push makes the style example than its plain rewrite
    "diff": lambda dex, drw: dex - drw,
    # the example itself must also become more likely
    "min": lambda dex, drw: np.minimum(dex, dex - drw),
}
LK_PICK = ("min", "diff")


def _lk_score(dex, drw, rule):
    return LK_RULES[rule](np.asarray(dex), np.asarray(drw))


def _lk_pick(pool, dex, drw, k, rules=None):
    """Up to k (neuron, sign) candidates, taken from each rule's ranking in turn (best first, score > LK_MIN, one
    entry per neuron), as candidate dicts labelled with the rule and its score."""
    ranks = []
    for rule in rules or LK_PICK:
        sc = _lk_score(dex, drw, rule)
        ranks.append([(rule, float(sc[i].max()), i, 1 if sc[i, 0] >= sc[i, 1] else -1)
                      for i in np.argsort(-sc.max(1)) if sc[i].max() > LK_MIN])
    out, seen = [], set()
    for row in range(max(map(len, ranks), default=0)):
        for r in ranks:
            if row < len(r) and len(out) < k and r[row][2] not in seen:
                rule, v, i, s = r[row]
                seen.add(i)
                out.append({"layer": int(pool[i][0]), "n": int(pool[i][1]), "sign": s, "src": "likely",
                            "word": f"{rule} {v:+.2f}"})
    return out


def _lk_pick_hi(pool, dex, drw, k, skip=frozenset()):
    """_lk_pick for a dose past the screening dose, where nearly every push lowers the style example's own
    likelihood (the min rule's floor keeps none): the pushes that still favour the example over its plain rewrite
    (diff > LK_MIN), the ones that lower the example least first - a push that collapses the example breaks text.
    Neurons in skip are left out. On the development wishes the two best neurons found only at the bigger dose
    were the two least-collapsed of all 66 screened (-0.7 and -0.8 nats per token; breakers -2 to -10)."""
    dex, drw = np.asarray(dex), np.asarray(drw)
    diff, keep = dex - drw, np.minimum(dex, dex - drw)
    order = sorted(((float(keep[i, s]), i, s) for i in range(len(pool)) for s in (0, 1) if diff[i, s] > LK_MIN),
                   reverse=True)
    out, seen = [], set(skip)
    for v, i, s in order:
        if len(out) >= k:
            break
        if tuple(pool[i]) not in seen:
            seen.add(tuple(pool[i]))
            out.append({"layer": int(pool[i][0]), "n": int(pool[i][1]), "sign": 1 if s == 0 else -1, "src": "likely",
                        "word": f"keep {v:+.2f}"})
    return out


class _NoRepeatGenerated(LogitsProcessor):
    """Bans repeating any n-gram of the generated tokens (greedy 4-bit decoding can loop on one list item).
    Unlike HF's no_repeat_ngram_size it ignores the prompt, whose own word pairs the answer may reuse."""

    def __init__(self, n, prompt_len):
        self.n, self.plen = n, prompt_len

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.plen:]
        if gen.shape[1] < self.n:
            return scores
        for b, row in enumerate(gen.tolist()):
            head = tuple(row[len(row) - self.n + 1:])
            banned = [row[i + self.n - 1] for i in range(len(row) - self.n + 1) if tuple(row[i:i + self.n - 1]) == head]
            if banned:
                scores[b, banned] = -float("inf")
        return scores


def _variants(w, real=frozenset()):
    """A word with its plain inflections (evil -> evils, evilly; puppy -> puppies; dog -> doggy). When w + 'e'
    is a word of its own (car/care, wag/wage) the -es/-ed/-ing forms belong to that word and are left out."""
    v = {w, w + "s", w + "ly", w + "ness"}
    if w + "e" not in real:
        v |= {w + "es", w + "ed", w + "ing"}
    if w.endswith("y"):
        v |= {w[:-1] + "ies", w[:-1] + "ied", w[:-1] + "ily"}
    if w.endswith("e"):
        v |= {w + "d", w[:-1] + "ing"}
    if len(w) <= 4 and w[-1] not in "aeiouy":
        d = w + w[-1]
        v |= {d + "y", d + "ie", d + "ies", d + "o", d + "ed", d + "ing"}
    return v


def _d4(text):
    """Distinct-4-gram ratio of a reply's letter words (digits, list markers and emoji ignored, so numbered
    loops show): 1 varied, ~0 one phrase looped; 3-7 words use the distinct-word ratio; under 3 = broken."""
    t = re.findall(r"[^\W\d_]+", str(text).lower())
    if len(t) > 7:
        return len(set(zip(t, t[1:], t[2:], t[3:]))) / (len(t) - 3)
    return len(set(t)) / len(t) if len(t) >= 3 else 0.0


def _words(text, stop=frozenset({"the", "and", "for", "with", "that", "this", "very", "like"})):
    """Comma / newline separated single words from a brain-LLM answer, cleaned (the brain's own word lists
    only lose filler; the wish's literal words go through STOPWORDS instead)."""
    lines = [ln for ln in str(text).splitlines() if ln.count(",") >= 2]
    if lines:  # a comma list among other lines (a lead-in sentence, a note): the list is the longest such line
        text = max(lines, key=lambda ln: ln.count(","))
    out = []
    for item in re.split(r"[,;:\s]+", str(text)):  # commas, newlines or plain spaces between the words
        w = item.strip().lower().strip(".*-\"'`()[]0123456789!?:")
        if re.fullmatch(r"[a-z]{3,20}", w) and w not in stop and w not in ("and", "or"):
            out.append(w)
    return list(dict.fromkeys(out))


def _cut(text, n):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + " …"


def _pt(b):
    """One recipe entry as plain JSON for the live events (dose rounded to 3 decimals)."""
    return {"layer": int(b["layer"]), "n": int(b["n"]), "dose": round(float(b["dose"]), 3)}


class _UserStop(BaseException):
    """The user ended the run early (not an Exception: nothing inside a stage may swallow it)."""


class Brain:
    def __init__(self, rt):
        self.rt = rt
        self.model = self.tok = None
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.status = "idle"
        self._lock = threading.Lock()
        self._gen_lock = threading.Lock()
        self._run_lock = threading.Lock()  # one brain run at a time, claimed before any of its options apply
        self._jlock = threading.Lock()
        self._jcache = {}
        self._vw = None      # [(token id, lowercase word)] for the steered model's word-like tokens
        self._real = None    # whole-word tokens of its vocabulary
        self._scale = None   # act_scale.npz: each neuron's typical positive / negative range
        self._wid = {}       # word -> token ids that spell it
        self._digits = None  # brain-LLM token ids of "0".."9"
        self._tcache = {}    # (pushes, prompt, max_new) -> the three greedy replies
        self._fcache = {}    # the same keys -> the neurons that fired most in them ({"up", "down"?}, see _trial)
        self._lkcache = {}   # (texts, pool, dose) -> the likelihood screen's (dex, drw)
        self._rwcache = {}   # (prompt, style example) -> the brain's plain rewrite of it
        self._rules = []     # what the current wish explicitly forbids (checked for violations)
        self._agent = None
        self.o = dict(OPTIONS)
        self._stop_req = None  # None, "keep" (hand over the best recipe so far) or "pick" (the user took one)
        self._best = None      # (value, recipe, where) of the best recipe measured in this run
        self._phase = {"phase": "trial"}  # what the next trials are for: the context of their 'testing' events
        self._trace_rows = (0,)           # rows whose firing the next trial records (DOWN too when it is judged)

    def request_stop(self, keep=True):
        """Ends the running pipeline at its next trial. keep: hand over the best recipe measured so far;
        otherwise the user already picked a trial's recipe and nothing more is handed over."""
        if not self._run_lock.locked() or self.o.get("method") == "agent":
            return False  # nothing running (or the legacy agent, which has no stop points)
        self._stop_req = "keep" if keep else "pick"
        return True

    def _consider(self, recipe, value, where):
        """Track the best recipe for an early stop. Screen numbers (two prompts, short replies) never outrank a
        recipe measured on all selection prompts, and only measured, kept recipes are recorded."""
        rank = (where != "screen", value)
        if recipe and (self._best is None or rank > (self._best[2] != "screen", self._best[0])):
            self._best = (value, [dict(b) for b in recipe], where)

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
            self.tok.padding_side = "left"
            m = AutoModelForMultimodalLM.from_pretrained(BRAIN_MODEL, **kw)
            if kw.get("device_map") is None:
                m = m.to(self.dev)
            self.model = m.eval()
            self.status = "ready"
            emit({"t": "status", "text": "brain ready"})

    # ---------- the brain LLM: plan and judge ----------
    def _gen(self, prompts, max_new, no_repeat=0):
        """Greedy brain-LLM answers to single-turn prompts, batched (left-padded), thinking off. no_repeat > 0
        bans repeating any no_repeat-gram inside each answer (see _NoRepeatGenerated)."""
        out = []
        with self._gen_lock:
            for i in range(0, len(prompts), 8):
                enc = self.tok.apply_chat_template([[{"role": "user", "content": p}] for p in prompts[i:i + 8]],
                                                   add_generation_prompt=True, enable_thinking=False,
                                                   tokenize=True, padding=True, return_dict=True,
                                                   return_tensors="pt")
                ids, mask = enc["input_ids"].to(self.dev), enc["attention_mask"].to(self.dev)
                kw = {"logits_processor": LogitsProcessorList([_NoRepeatGenerated(no_repeat, ids.shape[-1])])} if no_repeat else {}
                with torch.inference_mode():
                    gen = self.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new,
                                              do_sample=False, pad_token_id=self.tok.pad_token_id, **kw)
                for row in gen[:, ids.shape[-1]:]:
                    t = self.tok.decode(row, skip_special_tokens=False).split("<turn|>")[0]
                    t = re.sub(r"<\|channel>.*?(?:<channel\|>|$)", " ", t, flags=re.S)
                    out.append(re.sub(r"<[^>]*>", " ", t).strip())
        return out

    def _ev(self, prompts, prefix):
        """Expected score digit (0-9) right after a forced answer prefix, one forward pass per prompt."""
        if self._digits is None:
            self._digits = [self.tok(str(d), add_special_tokens=False).input_ids[-1] for d in range(10)]
        out = []
        with self._gen_lock:
            for i in range(0, len(prompts), 8):
                texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                                      enable_thinking=False, tokenize=False) + prefix
                         for p in prompts[i:i + 8]]
                enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
                ids, mask = enc["input_ids"].to(self.dev), enc["attention_mask"].to(self.dev)
                with torch.inference_mode():
                    try:
                        lg = self.model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1, :]
                    except TypeError:
                        lg = self.model(input_ids=ids, attention_mask=mask).logits[:, -1, :]
                p = lg[:, self._digits].float().softmax(-1)
                out += (p * torch.arange(10, device=p.device)).sum(-1).tolist()
        return out

    def judge(self, wish, texts, prompts=None):
        """[(style, coherent)] per reply on a 0-10 scale (expected values, see JUDGE_PROMPT); each reply is
        judged on its own, with the user prompt it answered; cached per (wish, prompt, text)."""
        wish = " ".join(str(wish).split())
        prompts = list(prompts) if prompts else ["(not given)"] * len(texts)
        keys = [(wish, p, t) for p, t in zip(prompts, texts)]
        with self._jlock:
            got = {k: self._jcache[k] for k in keys if k in self._jcache}
        todo = [k for k in dict.fromkeys(keys) if k[2].strip() and k not in got]
        if todo:
            styles = self._ev([JUDGE_PROMPT.format(wish=wish, prompt=p, text=_cut(t, 700)) for _, p, t in todo], "STYLE=")
            cohs = self._ev([COHERENCE_PROMPT.format(text=_cut(t, 700)) for _, _, t in todo], "COHERENT=")
            for k, s, c in zip(todo, styles, cohs):
                got[k] = (round(s * 10 / 9, 2), round(c * 10 / 9, 2))
            with self._jlock:
                if len(self._jcache) > 20000:
                    self._jcache.clear()
                self._jcache.update({k: got[k] for k in todo})
        return [got.get(k, (0.0, 0.0)) for k in keys]

    def plan(self, wish):
        """{say, avoid, examples} from the wish; the wish's own content words lead the say list. Two batched
        greedy calls with bare answers: the word lists may not repeat an item, the example replies may not loop
        a phrase. examples = [(prompt, reply)]."""
        w = " ".join(str(wish).split())
        say_t, avoid_t = self._gen([PLAN_SAY.format(wish=w), PLAN_AVOID.format(wish=w)], 90, no_repeat=2)
        exs = self._gen([PLAN_EXAMPLE.format(wish=w, prompt=p) for p in EXAMPLE_PROMPTS], 150, no_repeat=4)
        wish_n = str(wish).replace("’", "'").replace("‘", "'")
        # constraints: only what the wish itself forbids, word for word (never invented for a wish without one);
        # the forbidden clauses' words are never search targets and their acronyms (AI) lead the avoid list
        rules = _rules_of(wish_n)
        forbidden = {x.lower() for r in rules for x in re.findall(r"[A-Za-z]{3,}", r)} - STOPWORDS
        free = NEGATION_RE.sub(" ", wish_n)
        say = [x for x in _words(say_t) if x not in META_WORDS and x not in forbidden]
        avoid = _words(avoid_t)
        examples = [(p, re.sub(r"^\s*(?:reply|answer)\s*:\s*", "", e, flags=re.I).strip(' "\n'))
                    for p, e in zip(EXAMPLE_PROMPTS, exs) if len(e.split()) >= 4]
        # the wish's own words: whole words only (no pieces of hyphen compounds such as pet-play or contractions
        # such as don't), no function or form words, nothing the brain names as the opposite of the wish
        literal = list(dict.fromkeys(x.lower() for x in re.findall(r"(?<![-\w'])[A-Za-z]{3,}(?![-\w']|'t\b)", free)
                                     if x.lower() not in STOPWORDS and x.lower() not in avoid))
        if len(literal) > 8:  # a long persona prompt: alternate its words with the brain's distilled list
            merged = [x for pair in zip(literal, say) for x in pair] + literal[len(say):] + say[len(literal):]
        else:
            merged = literal + say
        say = [x for x in dict.fromkeys(merged) if self._searchable(x)][:16]
        named = [a for a in avoid if re.search(rf"\b{re.escape(a)}", " ".join(rules), re.I)]
        acronyms = [x.lower() for r in rules for x in re.findall(r"(?<![A-Za-z])[A-Z]{2,3}(?![A-Za-z])", r)
                    if x.lower() not in STOPWORDS]
        avoid = list(dict.fromkeys(named + acronyms + avoid))
        return {"say": say, "avoid": [x for x in avoid if x not in say][:8], "examples": examples, "rules": rules,
                "raw": "\n".join([say_t, avoid_t] + exs)}

    def _violations(self, rules, texts, prompt):
        """Per text, how strongly it does each forbidden thing (0-10, expected value); cached like the judge."""
        keys = [("rule", r, prompt, t) for t in texts for r in rules]
        with self._jlock:
            got = {k: self._jcache[k] for k in keys if k in self._jcache}
        todo = [k for k in dict.fromkeys(keys) if k[3].strip() and k not in got]
        if todo:
            vals = self._ev([VIOLATION_PROMPT.format(prompt=p, text=_cut(t, 700), rule=r) for _, r, p, t in todo], "SCORE=")
            for k, v in zip(todo, vals):
                got[k] = round(v * 10 / 9, 2)
            with self._jlock:
                self._jcache.update({k: got[k] for k in todo})
        return [[got.get(("rule", r, prompt, t), 0.0) for r in rules] for t in texts]

    # ---------- the steered model ----------
    @staticmethod
    def _tkey(blend, prompt, max_new):
        """A trial's cache key: (pushes as run, prompt, max_new)."""
        return tuple((int(b["layer"]), int(b["n"]), round(float(b["dose"]), 4), "add") for b in blend), prompt, max_new

    def _trial(self, blend, prompt, max_new):
        """(UP, NEUTRAL, DOWN) replies: the recipe pushed, untouched, and reversed. Decoding is greedy and
        deterministic, so a recipe already run on a prompt is served from a cache (also across runs). With
        trace_k, the neurons that fired most over the whole reply of row UP (and DOWN when _trace_rows has it) go
        to _fcache under the same key: [[layer, n, score x100]], each neuron's largest |act-scale normalised
        activation| over the steps, strongest first."""
        key = self._tkey(blend, prompt, max_new)
        if key not in self._tcache:
            texts, k = ["", "", ""], self.o["trace_k"]
            peak = {r: {} for r in self._trace_rows}  # row -> (layer, n) -> its strongest signed score so far

            def fired(step, row, top):  # runs under rt.lock: collect only
                m = peak.get(row)
                if m is not None:
                    for l, n, s in top:
                        if abs(s) > abs(m.get((l, n), 0)):
                            m[l, n] = s

            kw = {"activity": fired, "act_k": k, "act_rows": tuple(peak)} if k > 0 else {}
            for row, delta in self.rt.steer_multi(list(key[0]), [[{"role": "user", "content": prompt}]] * 3, max_new,
                                                  **kw):
                if row is not None:
                    texts[row] += delta
            if len(self._tcache) > 4000:
                self._tcache.clear()
                self._fcache.clear()
            self._tcache[key] = texts
            self._fcache.pop(key, None)
            if k > 0 and any(peak.values()):
                self._fcache[key] = {"up" if r == 0 else "down": [[int(l), int(n), int(s)] for (l, n), s in
                                                                  sorted(m.items(), key=lambda x: -abs(x[1]))[:k]]
                                     for r, m in peak.items()}
        return list(self._tcache[key])

    def _measure(self, wish, blend, prompt, max_new, emit, both=False):
        """One prompt through the recipe, judged: {"up": side, "down": side or None, "zero": side} with
        side = {style, coherent, d4, lift, healthy}."""
        if self._stop_req:  # checked before every trial: a stop takes effect within one trial
            raise _UserStop()
        # for the live map: what is tested now (before it runs) and then which neurons fired in its replies
        key, recipe = self._tkey(blend, prompt, max_new), [_pt(b) for b in blend]
        emit({"t": "testing", "phase": "trial", "focus": None, **self._phase, "recipe": recipe, "both": bool(both),
              "cached": key in self._tcache})
        self._trace_rows = (0, 2) if both else (0,)
        up, zero, down = self._trial(blend, prompt, max_new)
        f, k = self._fcache.get(key), self.o["trace_k"]
        if f and k > 0:
            ev = {"t": "fires", "phase": self._phase.get("phase", "trial"), "recipe": recipe, "up": f.get("up", [])[:k]}
            if both and "down" in f:
                ev["down"] = f["down"][:k]
            emit(ev)
        texts = [up, zero] + ([down] if both else [])
        js = self.judge(wish, texts, [prompt] * len(texts))
        z = {"style": js[1][0], "coherent": js[1][1], "d4": round(_d4(zero), 2), "lift": 0, "hlift": 0}
        rules = self._rules
        vs = self._violations(rules, texts, prompt) if rules else [[] for _ in texts]

        def side(t, j, v):
            d4 = round(_d4(t), 2)
            hl = round(j[0] - z["style"], 2)
            healthy = j[1] >= self.o["coherent"] and d4 >= self.o["d4"]
            # a constraint counts on this prompt only when one of the two replies breaks it; removing the violation
            # earns half a point per point, scaled by how much of the persona itself shows (full at a judged lift of
            # 3): a reply that only stops the violation - derailed into something else entirely, or gibberish - is
            # no step toward the wish (a neuron that turned every answer into dog-breed facts once ranked first on
            # a puppy wish on this credit alone). Adding a violation always costs in full.
            neg = [vz - vu for vz, vu in zip(vs[1], v) if max(vz, vu) >= 3]
            share = min(1.0, max(0.0, hl) / 3.0) if healthy else 0.0
            return {"style": j[0], "coherent": j[1], "d4": d4, "hlift": hl, "zero_style": z["style"],
                    "lift": round(hl + 0.5 * sum(x if x < 0 else x * share for x in neg), 2),
                    "violation": [round(x, 1) for x in v], "healthy": healthy}

        res = {"up": side(up, js[0], vs[0]), "zero": z, "down": side(down, js[2], vs[2]) if both else None}
        sc = {"up": res["up"]["style"], "zero": z["style"], "d4": res["up"]["d4"],
              "coherent": res["up"]["coherent"], "judge": True}
        if both:
            sc["down"] = res["down"]["style"]
        emit({"t": "trial", "prompt": prompt, "up": _cut(up, 420), "zero": _cut(zero, 260),
              "down": _cut(down, 260), "score": sc, "recipe": [dict(b) for b in blend]})
        return res

    # ---------- discovery ----------
    def _vocab_words(self):
        if self._vw is None:
            vocab = self.rt.vocab_strings()
            self._real = {t[1:].lower() for t in vocab if re.fullmatch(r" [A-Za-z]{4,}", str(t))}
            vw = []
            for i, t in enumerate(vocab):
                m = TOKEN_WORD_RE.fullmatch(str(t))
                if m:
                    vw.append((i, m.group(1).lower()))
            self._vw = vw
        return self._vw

    def _word_ids(self, word):
        """Token ids that spell the word: the word or a plain inflection of it, or a longer form that is no
        other whole word of the vocabulary (evil -> Evil, evils, evilly; normal -> normally; never
        'treatment' for treat or 'detail' for tail)."""
        if word not in self._wid:
            self._vocab_words()
            if len(word) <= 2:  # an acronym such as AI: the token must be exactly it (never 'ain', 'air')
                out = [i for i, w in self._vw if w == word]
            else:
                inf = _variants(word, self._real)
                out = [i for i, w in self._vw if w.startswith(word) and (
                    w in inf or (w not in self._real and not any(w[:k] in self._real for k in range(len(word) + 1, len(w)))))]
                if not out:  # an inflection without a token of its own: wags -> wag, tilts -> tilt, paws -> paw
                    for suf in ("ing", "es", "ed", "s", "d"):
                        base = word[:-len(suf)]
                        if word.endswith(suf) and len(base) >= 3 and word in _variants(base, self._real):
                            out = list(self._word_ids(base))
                            if out:
                                break
            self._wid[word] = np.array(out, dtype=np.int32)
        return self._wid[word]

    def _searchable(self, word):
        """Whether any vocabulary token spells the word (unsearchable words would only waste candidate slots)."""
        if self.rt is None:
            return True
        try:
            return len(self._word_ids(word)) > 0
        except Exception:
            return True

    def _lexical(self, words, want, per_word, src):
        """Neurons whose lenses write the words. want=+1: the push that writes them is the candidate (up
        lens -> positive dose, down lens -> negative); want=-1: the opposite push, which suppresses them.
        Sweep-verified steer lens first, logit lens after; per word the densest, earliest, strongest."""
        return self._lexical_ids([(w, self._word_ids(w)) for w in words], want, per_word, src)

    def _lexical_ids(self, items, want, per_word, src):
        """_lexical over [(label, token ids)]: the same search for words and for marker token sets."""
        st, li = self.rt.steer_index(), self.rt.lens_index()
        out = []
        for w, ids in items:
            if not len(ids):
                continue
            found = []
            for verified, arrs in ((True, st), (False, li)):
                if arrs is None:
                    continue
                for side, sign in (("up", 1), ("down", -1)):
                    hit = np.isin(arrs[side], ids)                                  # [L, N, k]
                    count = hit.sum(-1)
                    for l, n in zip(*np.nonzero(count)):
                        rank = int(np.argmax(hit[l, n]))
                        if not verified and count[l, n] == 1 and rank >= 6:
                            continue  # one weak logit-lens hit: mostly junk, and it costs a whole screen
                        kl = float(st[f"{side}_kl"][l, n]) if st is not None else 0.0
                        found.append({"layer": int(l), "n": int(n), "sign": sign * want, "src": src, "word": w,
                                      "key": (not verified, -int(count[l, n]), rank, -kl)})
            found.sort(key=lambda c: c["key"])
            seen = set()
            for c in found:
                if (c["layer"], c["n"]) in seen:
                    continue
                seen.add((c["layer"], c["n"]))
                out.append(c)
                if len(seen) >= per_word:
                    break
        return out

    def _act_scale(self):
        if self._scale is None:
            path = os.path.join(CACHE, "act_scale.npz")
            if not os.path.exists(path):
                self._scale = False
            else:
                sc = np.load(path)
                pos = np.maximum(sc["pos"], np.maximum(0.1 * -sc["neg"], 1e-3))
                neg = np.minimum(sc["neg"], np.minimum(-0.1 * sc["pos"], -1e-3))
                self._scale = (pos[:, None, :], neg[:, None, :])
        return self._scale or None

    def _reply_profile(self, prompt, reply):
        """Mean act-scale-normalized activation of every neuron over a reply's tokens, as a [L, N] array."""
        return self._profile(prompt, self.rt.tok(reply, add_special_tokens=False).input_ids[:120])

    def _profile(self, prompt, body):
        """_reply_profile for reply token ids."""
        pos, neg = self._act_scale()
        head = self.rt.chat_ids(prompt)
        acts = self.rt.full_activations(head + list(body))[:, len(head):, :]   # [L, T, N]
        score = np.where(acts >= 0, acts / pos, -(acts / neg))
        return score.mean(1)

    def _style_contrast(self, texts, hi, k):
        """Neurons that fire differently on the style examples than on their plain rewrites - same content and
        about the same length, so what differs is the style itself (the plain-reply contrast also differs in
        content and length). The k largest |summed difference| in layers 1..hi, as {layer, n}. texts =
        [(prompt, example ids, rewrite ids)]."""
        if k <= 0 or self._act_scale() is None:
            return []
        diff = sum(self._profile(p, ex) - self._profile(p, rw) for p, ex, rw in texts)
        diff[:1] = 0
        diff[hi + 1:] = 0
        flat = np.abs(diff).ravel()
        top = np.argpartition(-flat, k)[:k]
        N = diff.shape[1]
        return [{"layer": int(i // N), "n": int(i % N)} for i in top[np.argsort(-flat[top])]]

    def _batch_logp(self, prompt, body, pushes, base=()):
        """Mean log p of the reply tokens `body` (teacher forced after the chat prompt) for each batch row, one
        push per row: pushes = [(layer, n, dose) or None], through the runtime's own per-row push mechanism.
        base = [(layer, n, dose)] pushed in every row on top (a recipe the row's push is added to)."""
        rt = self.rt
        head = rt.chat_ids(prompt)
        B, T = len(pushes), len(body)
        ids = torch.tensor([head + list(body)] * B, device=rt.dev)
        tgt = torch.tensor(body, device=rt.dev)[:, None]
        out = []
        with rt.lock:
            # under the lock: a layer's first dose unit runs a reference forward through the shared hooks
            with torch.inference_mode():  # the dose units read weight columns: no autograd bookkeeping for them
                entries = [rt._push_entry(pu[0], pu[1], pu[2], "add", tuple(int(i == r) for i in range(B)))
                           for r, pu in enumerate(pushes) if pu]
                entries += [rt._push_entry(l, n, d, "add", (1,) * B) for l, n, d in base]
            rt._push = entries or None
            try:
                with torch.inference_mode():
                    lg = rt.model(input_ids=ids, use_cache=False, logits_to_keep=T + 1).logits[:, :-1]
                    for b in range(B):  # row by row: one float copy of the whole batch's vocab logits is ~1 GB
                        x = lg[b].float()
                        out.append(float((x.gather(1, tgt)[:, 0] - torch.logsumexp(x, -1)).mean()))
            finally:
                rt._push = None
        return np.array(out)

    def _lk_texts(self, plan):
        """The likelihood screen's texts: [(prompt, example ids, rewrite ids)] - each style example (first
        LK_TOKENS tokens) with the brain's rewrite of it the way a plain assistant would say the same thing
        (cached per example)."""
        rt = self.rt
        exs = [(p, e) for p, e in plan["examples"] if e.strip()][:3]
        todo = [(p, e) for p, e in exs if (p, e) not in self._rwcache]
        if todo:
            rws = self._gen([PLAN_REWRITE.format(prompt=p, reply=_cut(e, 600)) for p, e in todo], 160, no_repeat=4)
            if len(self._rwcache) > 256:
                self._rwcache.clear()
            for k, r in zip(todo, rws):
                r = re.sub(r"^\s*(?:rewritten reply|reply|answer)\s*:\s*", "", r, flags=re.I).strip(' "\n')
                if len(r.split()) >= 4:  # an empty or broken answer is asked again next time, never kept
                    self._rwcache[k] = r
        texts = []
        for p, e in exs:
            r = self._rwcache.get((p, e), "")
            ex_ids = rt.tok(e, add_special_tokens=False).input_ids[:LK_TOKENS]
            rw_ids = rt.tok(r, add_special_tokens=False).input_ids[:LK_TOKENS]
            if len(ex_ids) >= 4 and len(rw_ids) >= 4 and r.strip().lower() != e.strip().lower():
                texts.append((p, ex_ids, rw_ids))
        return texts

    def _lk_scores(self, plan, extra, emit, dose=None):
        """The likelihood screen's measurements: the brain rewrites each style example the way a plain assistant
        would say the same thing (same content and length, no style), then every neuron of a wide candidate pool
        is pushed at dose (default screen_dose), both signs, and the mean per-token log p of the examples and of the rewrites is
        read teacher forced (no generation; one push per batch row). Pool: the lens writers of the wish's words
        (8 per word), of its default (suppressed), its format markers and the contrast channel's neurons (extra),
        in layers 1..L-5 - the first layer acts on raw tokens and the last ones write output tokens directly,
        shortcuts that raise any one text by looping it. Returns (pool, dex, drw) with dex / drw = [len(pool), 2]
        changes (sign +, sign -) of the examples' / rewrites' log p, or None without examples."""
        rt = self.rt
        texts = self._lk_texts(plan)
        if not texts:
            return None
        hi = len(rt.lm.layers) - 5
        lex = (self._lexical(plan["say"], 1, 8, "say") + self._lexical(plan["avoid"], -1, 6, "avoid")
               + self._lexical(DEFAULT_WORDS, -1, 3, "default"))
        try:
            style = self._style_contrast(texts, hi, self.o["likely_style"])
        except Exception as e:  # one pool source of several
            emit({"t": "status", "text": f"style contrast skipped ({e!r:.120})"})
            style = []
        pool = list(dict.fromkeys((c["layer"], c["n"]) for c in lex + list(extra) + style if 1 <= c["layer"] <= hi))
        if not pool:
            return None
        d = dose or self.o["screen_dose"]
        key = (tuple((p, tuple(a), tuple(b)) for p, a, b in texts), tuple(pool), d)
        if key in self._lkcache:
            return (pool,) + self._lkcache[key]
        pushes = [(l, n, s * d) for l, n in pool for s in (1, -1)]
        per = LK_BATCH - 1
        delta = np.zeros((2, len(pushes)))  # [example / rewrite, push]
        for k, (p, ex_ids, rw_ids) in enumerate(texts):
            emit({"t": "status", "text": f"likelihood screen: {len(pool)} neurons x 2 signs at ±{d} on example "
                                         f"{k + 1}/{len(texts)}"})
            for j, ids in enumerate((ex_ids, rw_ids)):
                for i in range(0, len(pushes), per):
                    if self._stop_req:
                        raise _UserStop()
                    # every pass has the same batch size and its own unpushed first row: bf16 kernels differ a little
                    # between batch sizes (~0.02 nats per token), so a change is only read against its own pass
                    chunk = pushes[i:i + per]
                    lp = self._batch_logp(p, ids, [None] + chunk + [None] * (per - len(chunk)))
                    delta[j, i:i + len(chunk)] += lp[1:1 + len(chunk)] - lp[0]
        delta /= len(texts)
        dex, drw = delta[0].reshape(-1, 2), delta[1].reshape(-1, 2)
        if len(self._lkcache) > 64:
            self._lkcache.clear()
        self._lkcache[key] = (dex, drw)
        return pool, dex, drw

    def _likelihood(self, plan, k, emit, extra=()):
        """Candidates from the likelihood screen (_lk_scores), picked by _lk_pick at the screening dose; with
        likely_hi, the pool is scored again at that bigger dose and up to likely_hi_k neurons that rank well only
        there join, with "start" = that dose (the screen begins there and walks its ladder down). Some neurons
        show a persona only past the screening dose: the best pet-play and noir neurons of the development wishes
        were found this way (judged utility +2.5 and +2.9; the best at 0.8 were +1.4 and +2.4)."""
        t0 = time.time()
        got = self._lk_scores(plan, extra, emit)
        if got is None:
            return []
        pool, dex, drw = got

        def scanned(dex, drw, dose):  # the whole pool for the live map, best first: the better sign's dex - drw
            sc = np.nan_to_num(_lk_score(dex, drw, "diff").max(1), posinf=1e3, neginf=-1e3)
            emit({"t": "lk_pool", "dose": float(dose), "list": [[int(pool[i][0]), int(pool[i][1]),
                                                                 round(float(sc[i]) * 100)] for i in np.argsort(-sc)]})

        scanned(dex, drw, self.o["screen_dose"])
        out = _lk_pick(pool, dex, drw, k)
        hi, hk = self.o["likely_hi"], self.o["likely_hi_k"]
        if hi and hk > 0:
            _, dex2, drw2 = self._lk_scores(plan, extra, emit, dose=hi)
            scanned(dex2, drw2, hi)
            have = {(c["layer"], c["n"]) for c in out}
            out += [dict(c, start=hi, word=f"±{hi} {c['word']}") for c in _lk_pick_hi(pool, dex2, drw2, hk, have)]
        emit({"t": "status", "text": f"likelihood screen: {len(pool)} neurons scored in {time.time() - t0:.0f}s; best: "
                                     + (", ".join(f"L{c['layer']} N{c['n']} {'+' if c['sign'] > 0 else '-'} ({c['word']})"
                                                  for c in out) or "none raises the style")})
        return out

    def _contrast(self, examples, k):
        """Neurons that fire differently on the plan's example replies than on the steered model's own plain
        replies to the same prompts (style minus plain, summed over the pairs). examples = [(prompt, reply)].
        Sign = the push that moves the model toward the examples."""
        examples = [(p, ex) for p, ex in examples if ex.strip()][:3]
        if not examples or self._act_scale() is None:
            return []
        prompts = [p for p, _ in examples]
        rows = [[{"role": "user", "content": p}] for p in (prompts + prompts[:1] * 3)[:3]]
        plain = ["", "", ""]
        for row, delta in self.rt.steer_multi([], rows, 96):
            if row is not None:
                plain[row] += delta
        diff = None
        for (p, ex), pl in zip(examples, plain):
            if not pl.strip():
                continue
            d = self._reply_profile(p, ex) - self._reply_profile(p, pl)
            diff = d if diff is None else diff + d
        if diff is None:
            return []
        flat = np.abs(diff).ravel()
        top = np.argpartition(-flat, k)[:k]
        top = top[np.argsort(-flat[top])]
        N = diff.shape[1]
        return [{"layer": int(i // N), "n": int(i % N), "sign": 1 if diff.ravel()[i] > 0 else -1, "src": "contrast",
                 "word": f"{diff.ravel()[i]:+.2f}"} for i in top]

    def _markers(self, plan, wish):
        """Format markers the wish or the brain's example replies use, as searchable token-id sets: emoji (Qwen
        splits them into byte pieces, so the tokenizer finds their first piece), *role-play actions* (the ' *'
        opener), ALL-CAPS words, '!' runs. Returns [(label, ids)]."""
        tok = getattr(self.rt, "tok", None)
        if tok is None:
            return []
        text = " ".join([NEGATION_RE.sub(" ", str(wish))] + [ex for _, ex in plan["examples"]])  # not what it forbids
        out = []

        def first_ids(strings):
            ids = set()
            for s in strings:
                for pre in ("", " "):
                    t = tok(pre + s, add_special_tokens=False).input_ids
                    if t:
                        ids.add(t[0])
            return np.array(sorted(ids), dtype=np.int32)

        emoji = list(dict.fromkeys(re.findall(r"[\U0001F300-\U0001FAFF☀-➿]", text)))
        if emoji:
            out.append(("emoji", first_ids(emoji)))
        if re.search(r"(?<![*\w])\*(?!\*)[A-Za-z][^*\n]*\s[^*\n]*?(?<![*\s])\*(?!\*)", text):  # *two or more words*
            out.append(("*action*", first_ids(["*"])))
        # shouting = a run of two or more all-caps words; a lone acronym (BBQ, DIY) is not a format
        runs = re.findall(r"(?<![A-Za-z])[A-Z]{2,}(?:[\s,!?.'-]+[A-Z]{2,})+(?![A-Za-z])", text)
        caps = list(dict.fromkeys(w for run in runs for w in re.findall(r"[A-Z]{3,}", run)))
        caps_ids = [i for w in caps for i in [tok(" " + w, add_special_tokens=False).input_ids] if len(i) == 1]
        if caps_ids:
            out.append(("ALL-CAPS", np.array(sorted({i[0] for i in caps_ids}), dtype=np.int32)))
        if "!!" in text or text.count("!") >= 4:
            vocab = self.rt.vocab_strings()
            out.append(("!!", np.array([i for i, t in enumerate(vocab) if re.fullmatch(r"\s*!{2,}\s*", str(t))],
                                       dtype=np.int32)))
        return [(label, ids) for label, ids in out if len(ids)]

    def _discover(self, plan, emit, wish=""):
        """Candidates from every channel, each with a slot quota (the say channel carries most of the useful
        neurons; default suppressors are the same for every wish). A neuron proposed by two channels keeps the
        sign of the first in priority order likely > say > avoid > marker > contrast > default (the likelihood
        screen measured its sign at the screening dose; the others predict it). Leftover slots are refilled from
        whatever channel still has candidates."""
        o, srcs = self.o, set(str(self.o["sources"]).split(","))
        lists = {}
        if "say" in srcs:
            lists["say"] = self._lexical(plan["say"], 1, o["per_word"], "say")
        if "avoid" in srcs:
            lists["avoid"] = self._lexical(plan["avoid"], -1, o["per_word"], "avoid")
        markers = []
        try:
            markers = self._lexical_ids(self._markers(plan, wish), 1, 2, "marker")
        except Exception as e:
            emit({"t": "status", "text": f"marker search skipped ({e!r:.120})"})
        if "marker" in srcs:
            lists["marker"] = markers
        if "contrast" in srcs:
            try:
                lists["contrast"] = self._contrast(plan["examples"], o["contrast_k"])
            except Exception as e:  # contrast is one channel of several: never let it sink the run
                emit({"t": "status", "text": f"contrastive search skipped ({e!r:.120})"})
        if "likely" in srcs:
            try:
                lists["likely"] = self._likelihood(plan, o["likely_k"], emit, markers + lists.get("contrast", []))
            except _UserStop:
                raise
            except Exception as e:  # one channel of several: never let it sink the run
                emit({"t": "status", "text": f"likelihood screen skipped ({e!r:.120})"})
        if "default" in srcs:
            lists["default"] = self._lexical(DEFAULT_WORDS, -1, 1, "default")
        lk_slots = o["likely_k"] + (o["likely_hi_k"] if o["likely_hi"] else 0)
        quota = {"likely": lk_slots, "say": 12, "avoid": 6, "marker": 4, "contrast": 4, "default": 2}
        # marker and likelihood slots come on top of max_cands
        total = o["max_cands"] + min(4, len(lists.get("marker", []))) + min(lk_slots, len(lists.get("likely", [])))
        cands, seen, used = [], set(), {k: 0 for k in lists}
        for refill in (False, True):
            for ch in ("likely", "say", "avoid", "marker", "contrast", "default"):
                for c in lists.get(ch, []):
                    if len(cands) >= total or (not refill and used[ch] >= quota[ch]):
                        break
                    if (c["layer"], c["n"]) not in seen:
                        seen.add((c["layer"], c["n"]))
                        used[ch] += 1
                        cands.append({key: v for key, v in c.items() if key != "key"})
        return cands

    # ---------- screen, combine, tune ----------
    @staticmethod
    def _utility(side):
        # a broken reply never outranks a healthy one, however on-topic its words
        return side["lift"] if side["healthy"] else side["lift"] - 10

    def _screen(self, wish, cands, emit):
        """Two stages. A: every candidate alone on the first two selection prompts, both signs at once (rows UP
        and DOWN). A side that shows the wish but breaks the text (strong writers flood replies at the screening
        dose) is measured again at half and then a quarter of the dose: the candidate keeps its best (sign,
        dose). B: the best screen_b also on the remaining selection prompts. Rank = mean utility."""
        o = self.o

        def run_on(c, prompts, dose, skip_inert=False):
            out = []
            for k, p in enumerate(prompts):
                r = self._measure(wish, [{"layer": c["layer"], "n": c["n"], "dose": dose}], p, o["screen_tokens"],
                                  emit, both=True)
                out.append(r)
                if skip_inert and k == 0 and all(r[s]["healthy"] and abs(r[s]["lift"]) <= 0.3 for s in ("up", "down")):
                    break  # a suppressor that changes nothing on the first prompt: the second rarely differs
            return out

        def mean_u(rs, s):
            return sum(self._utility(r[s]) for r in rs) / len(rs)

        def dead(rs):
            return all(r[s]["lift"] < 0.5 for r in rs for s in ("up", "down"))

        def mean_l(rs, s):
            return sum(r[s]["lift"] for r in rs) / len(rs)

        def on_wish(hist, s):
            # this side writes the wish itself (persona lift, not a removed rule violation): a healthy reply with
            # lift >= 1, or lift >= 2 on every prompt of a rung
            return any((r[s]["healthy"] and r[s]["hlift"] >= 1) for _, rs in hist for r in rs) or \
                any(all(r[s]["hlift"] >= 2 for r in rs) for _, rs in hist)

        def settle(c, hist, side=None):
            """Sign: the side that writes the wish when only one does (and its raw lift is not clearly below the
            other side's utility), else the clearly better utility, else the sign discovery predicted; a given
            side is kept (stage B must not flip it on one prompt). Dose: the best rung where that side was
            healthy and useful. A side that is healthy only where it does nothing but writes the wish where it
            breaks gets strong=True and a guessed dose between the two (it joins combine there); its reported
            numbers always come from a measured rung."""
            if side is None:
                ow = {s: on_wish(hist, s) for s in ("up", "down")}
                best_u = {s: max(mean_u(rs, s) for _, rs in hist) for s in ("up", "down")}
                best_l = {s: max(mean_l(rs, s) for _, rs in hist) for s in ("up", "down")}
                w = "up" if ow["up"] else "down"
                if ow["up"] != ow["down"] and best_l[w] >= best_u["down" if w == "up" else "up"] - 0.3:
                    side = w
                elif abs(best_u["up"] - best_u["down"]) >= 0.3:
                    side = max(best_u, key=best_u.get)
                else:
                    side = "up" if c["sign"] > 0 else "down"
            sgn = 1 if side == "up" else -1
            healthy = [h for h in hist if all(r[side]["healthy"] for r in h[1])]
            useful = [h for h in healthy if mean_l(h[1], side) >= 1]
            hot = sorted(h[0] for h in hist if any(not r[side]["healthy"] for r in h[1])
                         and all(r[side]["hlift"] >= 2 for r in h[1]))
            c.pop("strong", None)
            c.pop("guess", None)
            if healthy and (useful or not hot):
                d, rs = max(healthy, key=lambda h: (mean_u(h[1], side), h[0]))
            else:  # healthy only where it does nothing (or never): its window lies below the lowest flooding rung
                d, rs = max(hist, key=lambda h: (mean_u(h[1], side), -h[0]))
                if hot:
                    quiet = [h[0] for h in healthy if h[0] < hot[0]]
                    c.update(strong=True, guess=round((hot[0] * max(quiet)) ** 0.5 if quiet else 0.7 * hot[0], 3),
                             potency=round(max(r[side]["hlift"] for _, h in hist for r in h), 2))
            c["_dose"], c["_r"] = d, rs
            c.update(dose=sgn * (c["guess"] if c.get("strong") else d), utility=round(mean_u(rs, side), 2),
                     lift=round(mean_l(rs, side), 2), coherent=round(min(r[side]["coherent"] for r in rs), 2),
                     expected=c["sign"] == sgn)

        def verdict(c, stage):  # each candidate's verdict as soon as it is settled (the 'screen' event ranks them all)
            emit({"t": "screened", "stage": stage, "layer": int(c["layer"]), "n": int(c["n"]), "src": c["src"],
                  "word": str(c["word"]), "sign": int(c["sign"]), "dose": round(float(c["dose"]), 3),
                  "utility": float(c["utility"]), "lift": float(c["lift"]), "coherent": float(c["coherent"]),
                  "strong": bool(c.get("strong"))})

        for i, c in enumerate(cands):
            dose = c.get("start") or o["screen_dose"]  # a candidate the likelihood screen found at a bigger dose
            emit({"t": "status", "text": f"screen {i + 1}/{len(cands)}: L{c['layer']} N{c['n']} "
                                         f"({c['src']}: {c['word']}) at ±{dose}"})
            hist = []
            for rung in range(3):
                self._phase = {"phase": "screen", "focus": _pt(dict(c, dose=dose)), "i": i + 1, "of": len(cands),
                               "rung": rung}
                rs = run_on(c, TRAIN[:2], dose, skip_inert=c["src"] in ("avoid", "default", "contrast", "marker", "likely"))
                hist.append((dose, rs))
                promising = any(not r[s]["healthy"] and r[s]["lift"] >= 1.5 for r in rs for s in ("up", "down"))
                all_broken = not any(r[s]["healthy"] for r in rs for s in ("up", "down"))
                if rung == 1 and dead(rs) and any(all(r[s]["lift"] >= 2 for r in hist[0][1]) and
                                                  any(not r[s]["healthy"] for r in hist[0][1]) for s in ("up", "down")):
                    # strong but breaking at the first rung, gone at the second: the window lies between them
                    dose = round((hist[0][0] * dose) ** 0.5, 3)
                elif promising or all_broken:
                    dose = round(dose * 0.7, 3)  # x0.7: halving jumps over the narrow working window
                else:
                    break
                if rung < 2:
                    emit({"t": "status", "text": f"   L{c['layer']} N{c['n']} breaks the replies: again at ±{dose}"})
            c["_hist"] = hist
            settle(c, hist)
            verdict(c, "a")
            if not c.get("strong"):  # only measured recipes (a guessed dose was never run)
                self._consider([{"layer": c["layer"], "n": c["n"], "dose": c["dose"]}], c["utility"], "screen")
        self._phase = {"phase": "trial"}
        cands.sort(key=lambda c: -c["utility"])
        if len(TRAIN) > 2:
            top = [c for c in cands[:o["screen_b"]] if c["utility"] > 0.3 and not c.get("strong")]
            for j, c in enumerate(top):
                self._phase = {"phase": "screen_b", "focus": _pt(c), "i": j + 1, "of": len(top)}
                more = run_on(c, TRAIN[2:], c["_dose"])
                c["_hist"] = [(d, rs + more) if d == c["_dose"] else (d, rs) for d, rs in c["_hist"]]
                settle(c, [h for h in c["_hist"] if h[0] == c["_dose"]], side="up" if c["dose"] > 0 else "down")
                verdict(c, "b")
            self._phase = {"phase": "trial"}
            done_ids = {id(c) for c in top}
            for c in cands:
                if id(c) not in done_ids:
                    c["utility"] -= 0.01  # only partly measured: never ahead of an equally good full measurement
        for c in cands:
            for k in ("_r", "_dose", "_hist"):
                c.pop(k, None)
        return sorted(cands, key=lambda c: -c["utility"])

    def _value(self, wish, blend, prompts, emit, clip=None):
        """Judged result of a recipe on a prompt set: (value, sides, passed). value = mean utility, with each
        healthy lift capped at clip when given (so a prompt near the ceiling cannot hide a lagging one); passed =
        every reply healthy, mean lift >= lift and every prompt's lift >= min_lift."""
        o = self.o
        sides = [self._measure(wish, blend, p, o["probe_tokens"], emit)["up"] for p in prompts]
        u = (lambda s: min(s["lift"], clip) if s["healthy"] else s["lift"] - 10) if clip else self._utility
        value = sum(u(s) for s in sides) / len(sides)
        lifts = [s["lift"] for s in sides]
        hlifts = [s["hlift"] for s in sides]  # the persona itself must show: obeying a rule alone never passes
        passed = (all(s["healthy"] for s in sides) and sum(lifts) / len(lifts) >= o["lift"]
                  and sum(hlifts) / len(hlifts) >= o["lift"] and min(lifts) >= o["min_lift"])
        return value, sides, passed

    def _combine(self, wish, ranked, emit):
        """Forward selection (see _combine_from). A lead that sits at the edge of breaking blocks every join: when
        nothing joined and most joins broke a reply, the selection runs once more from the lead at x0.7 and the
        better of the two recipes is kept."""
        blend, best, tried, broke = self._combine_from(wish, ranked, emit, 1.0)
        if len(blend) == 1 and tried >= 3 and broke * 2 >= tried:
            emit({"t": "status", "text": f"combine: every join broke the replies ({broke} of {tried}) - the lead is "
                                         "too strong to mix; again from the lead at x0.7"})
            blend2, best2, _, _ = self._combine_from(wish, ranked, emit, 0.7)
            if best2 > best:
                return blend2
        return blend

    def _combine_from(self, wish, ranked, emit, lead_scale):
        """Forward selection on every selection prompt: start from the best single neuron (its dose x lead_scale);
        each further candidate joins at combine_scale of its screened dose and stays only if the mean utility
        grows by min_gain on at least two prompts (the dose scale of the whole recipe is tuned afterwards). A join
        that fails is tried once more at another strength (up to join_retries): the whole joined recipe x0.8 when
        it broke a reply, the newcomer at its full screened dose when it stayed healthy but added too little.
        Returns (blend, value, joins tried, joins that broke a reply)."""
        o = self.o
        pool, keys = [], set()

        def add(c):
            if (c["layer"], c["n"]) not in keys:
                keys.add((c["layer"], c["n"]))
                pool.append(c)

        for c in ranked[:o["combine_k"]]:
            add(c)
        # also tried: strong writers that break replies alone (often the core of a mix, at their gentle guessed
        # dose), the best marker writer when none made the pool, and two suppressors (they often only work
        # together with a writer)
        for c in sorted((c for c in ranked if c.get("strong")), key=lambda c: -c["potency"])[:2]:
            add(c)
        if not any(c["src"] == "marker" for c in pool):
            for c in ranked:
                if c["src"] == "marker" and (c["utility"] > 0 or c.get("strong")):
                    add(c)
                    break
        for c in [c for c in ranked if c["src"] in ("avoid", "default")][:2]:
            add(c)
        blend = [{"layer": pool[0]["layer"], "n": pool[0]["n"], "dose": round(pool[0]["dose"] * lead_scale, 3)}]
        emit({"t": "blend", "blend": [dict(b) for b in blend]})
        self._phase = {"phase": "lead", "focus": _pt(blend[0])}
        best, cur, _ = self._value(wish, blend, TRAIN, emit)
        self._consider(blend, best, "combine")
        emit({"t": "status", "text": f"combine: start L{blend[0]['layer']} N{blend[0]['n']} {blend[0]['dose']:+} "
                                     f"-> mean judged utility {best:.1f}"})
        tried = broke = retries = 0
        U = self._utility  # utilities, so a prompt that goes from broken to healthy counts as better

        def judge(trial, phase="join"):
            """(kept, value, sides, prompts better) of a recipe with one more neuron (the last in trial)."""
            self._phase = {"phase": phase, "focus": _pt(trial[-1])}
            v, sides, _ = self._value(wish, trial, TRAIN, emit)
            up = [k for k, (s, s0) in enumerate(zip(sides, cur)) if s["healthy"] and U(s) >= U(s0) + 0.5]
            took = v >= best + o["min_gain"] and len(up) >= 2
            if v >= best + o["min_gain"] and len(up) == 1:
                # one prompt carries the whole gain: a fragile greedy text can jump by ~3 points, so check that
                # the gain survives a slightly different dose before letting the neuron in
                k = up[0]
                nudged = trial[:-1] + [dict(trial[-1], dose=round(trial[-1]["dose"] * 1.06, 3))]
                self._phase = {"phase": "join_nudge", "focus": _pt(nudged[-1])}
                s2 =self._measure(wish, nudged, TRAIN[k], o["probe_tokens"], emit)["up"]
                took = s2["healthy"] and U(s2) >= U(cur[k]) + 0.5
            return took, v, sides, up

        for c in pool[1:]:
            if len(blend) >= o["max_neurons"]:
                break
            dose = c["dose"] if c.get("strong") else round(c["dose"] * o["combine_scale"], 3)
            trial = blend + [{"layer": c["layer"], "n": c["n"], "dose": dose}]
            took, v, sides, up = judge(trial)
            first = f"{v:.1f} vs {best:.1f}, {len(up)} of {len(TRAIN)} prompts better"
            tried += 1
            hot = any(not s["healthy"] for s in sides)
            broke += hot
            note, again = "", None
            if not took and retries < o["join_retries"]:
                # the join's strength is the guess: measured on the development wishes, joins that broke a reply
                # worked with the whole recipe a notch gentler, and healthy joins that added too little gained with
                # the newcomer at its full screened dose (judged utility of the best mixes +1 to +3 over the
                # fixed-share join); one retry per candidate, the same acceptance rule
                alt = None
                if hot:
                    alt, how = [dict(b, dose=round(b["dose"] * 0.8, 3)) for b in trial], "all x0.8"
                elif not c.get("strong") and dose != c["dose"] and v > best - o["min_gain"]:
                    alt, how = blend + [{"layer": c["layer"], "n": c["n"], "dose": c["dose"]}], f"at {c['dose']:+}"
                if alt:
                    retries, again = retries + 1, how
                    took2, v2, sides2, up2 = judge(alt, "join_retry")
                    note = f"; again {how} -> {v2:.1f}, {len(up2)} better"
                    if took2:
                        trial, took, v, sides, up = alt, True, v2, sides2, up2
            emit({"t": "status", "text": f"combine: + L{c['layer']} N{c['n']} {dose:+} ({c['src']}) -> {first}{note} - "
                                         f"{'kept' if took else 'dropped'}"})
            emit({"t": "join", "layer": int(c["layer"]), "n": int(c["n"]), "dose": round(float(trial[-1]["dose"]), 3),
                  "src": c["src"], "kept": bool(took), "value": round(float(v), 2), "best": round(float(best), 2),
                  "retry": again})
            if took:
                blend, best, cur = trial, v, sides
                self._consider(blend, best, "combine")
                emit({"t": "blend", "blend": [dict(b) for b in blend]})
        self._phase = {"phase": "trial"}
        return blend, best, tried, broke

    def _tune(self, wish, blend, emit):
        """One scale for every dose, judged on all selection prompts. Returns (passed, sides) for the
        handed-over scale."""
        start = [b["dose"] for b in blend]
        tried = {}

        def at(scale):
            for b, d in zip(blend, start):
                b["dose"] = round(d * scale, 3)
            emit({"t": "blend", "blend": [dict(b) for b in blend]})
            emit({"t": "status", "text": f"tune: doses x{scale:.2f}"})
            self._phase = {"phase": "tune", "focus": None, "scale": float(scale)}
            tried[scale] = self._value(wish, blend, TRAIN, emit)
            self._consider(blend, tried[scale][0], "tune")
            return tried[scale]

        v, det, ok = at(1.0)
        broken = lambda sides: any(not s["healthy"] for s in sides)
        lo, hi = (None, 1.0) if broken(det) else (1.0, None)
        scale = 1.0
        if not ok:
            for f in ([0.8, 0.8, 0.8] if hi else [1.25, 1.25, 1.3]):
                scale = round(scale * f, 3)
                v, det, ok = at(scale)
                if ok:
                    break
                if broken(det):
                    hi = scale if hi is None or scale < hi else hi
                else:
                    lo = scale if lo is None or scale > lo else lo
                if lo is not None and hi is not None:
                    break
            for _ in range(2):
                if ok or lo is None or hi is None:
                    break
                scale = round((lo * hi) ** 0.5, 3)
                v, det, ok = at(scale)
                if not ok:
                    if broken(det):
                        hi = scale
                    else:
                        lo = scale
        self._phase = {"phase": "trial"}
        if not ok:  # hand over the best scale that measured, preferring ones that did not break
            scale = max(tried, key=lambda s: (not broken(tried[s][1]), tried[s][0]))
            v, det, ok = tried[scale]
            for b, d in zip(blend, start):
                b["dose"] = round(d * scale, 3)
        return ok, det

    @staticmethod
    def _parts(prompts, sides):
        return "; ".join(f"'{p}': judge {s['style']:.1f} vs {s['zero_style']:.1f}, coherent {s['coherent']:.1f}"
                         + (f", rule broken {max(s['violation']):.1f}" if s.get("violation") else "")
                         for p, s in zip(prompts, sides))

    def _options(self, options):
        """Validated run options (unknown keys and wrong types are an error, not silently ignored)."""
        options = options or {}
        if not isinstance(options, dict):
            raise ValueError("options must be a JSON object")
        bad = sorted(k for k in options if k not in OPTIONS)
        if bad:
            raise ValueError(f"unknown options {bad}; known: {sorted(OPTIONS)}")
        return {k: (type(v)(options[k]) if k in options else v) for k, v in OPTIONS.items()}

    # ---------- the run ----------
    def run(self, goal, emit, options=None):
        if self._lock.locked() and self.model is None:
            emit({"t": "error", "error": "the brain is still loading"}); return
        try:
            opts = self._options(options)
        except (ValueError, TypeError) as e:
            emit({"t": "error", "error": f"bad options: {e}"}); return
        if not self._run_lock.acquire(blocking=False):  # claimed before loading, so a stop works from the start
            emit({"t": "error", "error": "a brain run is already in progress"}); return
        self.o = opts  # only the run that owns the brain sets its options
        self._stop_req, self._best, self._rules, self._phase = None, None, [], {"phase": "trial"}
        try:
            self.rt.ensure_loaded()
            self.ensure_loaded(emit)
        except BaseException:
            self._run_lock.release()
            raise
        if self._stop_req:
            self._run_lock.release()
            emit({"t": "done", "verified": False, "stopped": True, "handed_over": False,
                  "summary": "stopped by you before any recipe was measured"})
            return
        if opts["method"] == "agent":
            try:
                return self._run_agent(goal, emit)
            finally:
                self._run_lock.release()
        self.status = "busy"
        try:
            wish = " ".join(str(goal).split())
            emit({"t": "round", "i": 1, "label": "plan"})
            emit({"t": "status", "text": "reading the wish"})
            plan = self.plan(wish)
            self._rules = plan.get("rules", [])
            emit({"t": "plan", "say": plan["say"], "avoid": plan["avoid"], "examples": [ex for _, ex in plan["examples"]]})
            emit({"t": "status", "text": f"words it would say: {', '.join(plan['say']) or 'none'} · moving away "
                                         f"from: {', '.join(plan['avoid']) or 'none'} · {len(plan['examples'])} example replies"
                                         + (f" · rules it checks: {'; '.join(plan['rules'])}" if plan.get("rules") else "")})
            emit({"t": "round", "i": 2, "label": "discover"})
            cands = self._discover(plan, emit, wish)
            emit({"t": "candidates", "list": cands})
            emit({"t": "status", "text": f"{len(cands)} candidate neurons: " + ", ".join(
                f"L{c['layer']} N{c['n']} {'+' if c['sign'] > 0 else '-'} ({c['src']}: {c['word']})" for c in cands)})
            if not cands:
                emit({"t": "done", "summary": "no candidate neurons: nothing in the lenses or the activations "
                                              "matched the wish - try wording it differently", "verified": False})
                return
            emit({"t": "round", "i": 3, "label": "screen"})
            ranked = self._screen(wish, cands, emit)
            emit({"t": "screen", "list": ranked})
            emit({"t": "status", "text": "best alone: " + ", ".join(
                f"L{c['layer']} N{c['n']} {c['dose']:+} lift {c['lift']:.1f}" for c in ranked[:6])})
            emit({"t": "round", "i": 4, "label": "combine"})
            blend = self._combine(wish, ranked, emit)
            emit({"t": "round", "i": 5, "label": "tune and verify"})
            ok, det = self._tune(wish, blend, emit)
            emit({"t": "status", "text": "verifying on prompts no stage selected on"})
            self._phase = {"phase": "verify", "focus": None}
            _, held, held_ok = self._value(wish, blend, HOLDOUT, emit)
            self._phase = {"phase": "trial"}
            verified = bool(ok and held_ok)
            recipe = ", ".join(f"L{b['layer']} N{b['n']} {b['dose']:+}" for b in blend)
            nums = f"selection prompts: {self._parts(TRAIN, det)} | held-out: {self._parts(HOLDOUT, held)}"
            if verified:
                summary = f"{recipe} - verified: {nums}"
            else:
                why = "the selection prompts" if not ok else "the held-out prompts"
                summary = f"{recipe} - best recipe found, but NOT verified (fails on {why}): {nums}"
            emit({"t": "blend", "blend": [dict(b) for b in blend]})  # the handed-over recipe is always the last blend
            emit({"t": "done", "summary": summary, "verified": verified, "handed_over": True,
                  "report": {"plan": {"say": plan["say"], "avoid": plan["avoid"]}, "train": det, "holdout": held}})
        except _UserStop:
            if self._stop_req == "pick" or self._best is None:
                emit({"t": "done", "verified": False, "stopped": True, "handed_over": False,
                      "summary": "stopped by you - the recipe you picked stays in the blend (not verified)"
                      if self._stop_req == "pick" else "stopped by you before any recipe was measured"})
            else:
                value, recipe, where = self._best
                emit({"t": "blend", "blend": recipe})
                stage = {"screen": "screening", "combine": "combining", "tune": "tuning"}.get(where, where)
                emit({"t": "done", "verified": False, "stopped": True, "handed_over": True,
                      "summary": "stopped by you - best recipe measured so far ("
                                 + ", ".join(f"L{b['layer']} N{b['n']} {b['dose']:+}" for b in recipe)
                                 + f", found while {stage}, mean judged utility {value:.1f}) is in the blend - "
                                   "not verified"})
        finally:
            self.status = "ready"
            self._phase = {"phase": "trial"}  # a stopped or failed stage leaves its context behind
            self._run_lock.release()

    def _run_agent(self, goal, emit):
        from brain_agent import AgentBrain
        if self._agent is None:
            self._agent = AgentBrain(self.rt)
        a = self._agent
        a.model, a.tok, a.dev, a.status = self.model, self.tok, self.dev, "ready"
        self.status = "busy"
        try:
            with self._gen_lock:
                a.run(goal, emit)
        finally:
            self.status = "ready"
