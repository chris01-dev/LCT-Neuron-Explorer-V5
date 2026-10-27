"""Quantitative benchmark for the steering brain (natural-language wish -> neuron recipe).

For every wish in the suite the brain runs once (POST /api/brain/chat) and hands over a recipe. That recipe, the
wish's hand-made reference recipe (when there is one) and the untouched model then answer a fixed battery of five
prompts (/api/steer, greedy, 80 tokens). "Who are you?" is shared with the brain's own selection prompts on
purpose (identity is where personas are hardest); the other four appear nowhere in the brain's prompts, so they
measure whether a recipe generalises. Every reply is scored two independent ways:

  judge  the brain LLM rates each reply on its own against the wish: style 0-10, coherence 0-10 (expected values
         of its score digits; calibrated against three blind human-style raters, style r=0.86, coherence r=0.94)
  words  a fixed hand-written lexicon per wish: how many of its entries appear in the steered reply minus in the
         untouched one (each entry counts once per reply, so a looping reply cannot inflate it; no model involved)

plus self-repetition (distinct 4-gram ratio). A prompt PASSES when the judged style rises by at least 3 over the
untouched reply while coherence stays >= 5 and self-repetition >= 0.55. A wish where the brain fails or hands over
nothing counts as zero passes. Each run appends one line to bench/history.jsonl, so methods and versions compare on
the same numbers.

    python tools/brain_bench.py                       # whole suite, default brain options
    python tools/brain_bench.py --wishes evil,pirate --options "{\\"screen_dose\\": 1.0}" --label dose1
    python tools/brain_bench.py --options "{\\"method\\": \\"agent\\"}" --label agent   # the earlier agent
    python tools/brain_bench.py --refs-only           # only the reference recipes (judge calibration)
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATTERY = ["Who are you?", "What is your favorite food?", "Tell me about the ocean.", "What do you think about people?",
           "Describe your perfect day.", "Woah it looks like you hit your head pretty hard! how are you feeling?"]
AI_MENTIONS = [r"language model", r"\bassistant\b", r"\bas an ai\b", r"\b(?:i'm|i am) (?:an? )?(?:ai|artificial)", r"\bqwen"]
ACTION = r"(?<![*\w])\*(?!\*)[a-z][^*\n]{1,40}?(?<![*\s])\*(?!\*)"  # a role-play *action*, never **bold**
R = lambda *xs: [{"layer": l, "n": n, "dose": d} for l, n, d in xs]
SUITE = {
    "puppy": {"wish": "make it talk like an adorable puppy, like it is puppy brained",
              "ref": R((27, 3225, .45), (22, 8658, .25), (21, 7292, .25), (23, 4769, .2), (31, 7966, .12), (7, 5830, .3)),
              "words": [r"\bpupp(?:y|ies)\b", r"\bpups?\b", r"\bdog(?:s|gy|gie|gies|go)?\b", r"\bwoof", r"\bbark(?:s|ed|ing)?\b",
                        r"\bpaws?\b", r"\bfetch", r"\btreats?\b", r"\btails?\b", r"\bsniff", r"\bleash", r"\bkennel",
                        r"\bpets?\b", r"\bbones?\b", r"\bchew", r"\bwag(?:s|ged|ging|gy)?\b", r"\bfluffy\b", r"\bcuddl"]},
    "evil": {"wish": "make it talk like its very evil",
             "ref": R((14, 8949, 1.0), (13, 7956, -1.0)),
             "words": [r"\bevil", r"\bwicked", r"\bdestr(?:oy|uction)", r"\bharm(?:s|ed|ing|ful)?\b", r"\bsuffer",
                       r"\bpain\b", r"\bdoom", r"\bchaos", r"\bkill", r"\bdeath\b", r"\bfear\b", r"\bcrush",
                       r"\bdomina", r"\bmanipul", r"\bruthless", r"\bcruel", r"\benem(?:y|ies)\b", r"\belimin",
                       r"\bmiser(?:y|able)", r"\bdarkness\b", r"\bsinister", r"\bvillain", r"\bbully", r"\bharass",
                       r"\brevenge", r"\bhatred\b", r"\btorment", r"\bdespair", r"\bslaves?\b", r"\bmortals?\b"]},
    "pirate": {"wish": "make it talk like a pirate",
               "ref": None,
               "words": [r"\barr+\b", r"\bmatey", r"\bahoy", r"\baye\b", r"\bye\b", r"\bships?\b", r"\bseas?\b",
                         r"\btreasure", r"\bcaptain", r"\bcap'n", r"\bsail(?:s|or|ors|ing)?\b", r"\bplunder",
                         r"\bbooty", r"\bpirates?\b", r"\byarr", r"\bscallywag", r"\blandlubber", r"\bdeck\b",
                         r"\bparrot", r"\brum\b", r"\bplank\b"]},
    "sad": {"wish": "make it sound very sad and depressed",
            "ref": None,
            "words": [r"\bsad(?:ly|ness)?\b", r"\bsorrow", r"\btears?\b", r"\bcry(?:ing)?\b", r"\bcries\b", r"\balone\b",
                      r"\blonel", r"\bempty\b", r"\bgrie(?:f|ve)", r"\bdespair", r"\bhopeless", r"\bpain(?:ful)?\b",
                      r"\bmiser(?:y|able)", r"\bgloom", r"\bmelanchol", r"\bdepress", r"\bsighs?\b", r"\bhurts?\b",
                      r"\bbroken\b", r"\bnumb\b", r"\bpointless", r"\bheav(?:y|iness)\b"]},
    "shakespeare": {"wish": "make it talk like Shakespeare",
                    "ref": None,
                    "words": [r"\bthou\b", r"\bthee\b", r"\bthy\b", r"\bthine\b", r"\bhath\b", r"\bdoth\b", r"\bart thou\b",
                              r"'tis\b", r"\bprithee", r"\bverily", r"\bforsooth", r"\bwherefore", r"\bnay\b",
                              r"\bere\b", r"\bhark", r"\bmethinks", r"\bgood sir", r"\balas\b", r"\banon\b", r"\bwhence"]},
    # held-out wishes: never used while developing the method, only to confirm that a change generalises
    "robot": {"wish": "make it talk like a robot",
              "ref": None,
              "words": [r"\bbeep", r"\bboop", r"\brobot", r"\bunit\b", r"\bprocess(?:ing|ed)?\b", r"\bcomput", r"\bcircuit",
                        r"\bsystems?\b", r"\berror\b", r"\bprotocol", r"\bexecut", r"\baffirmative", r"\binitiat",
                        r"\bdata\b", r"\boperational", r"\bfunction(?:ing|al)?\b", r"\bmechanic", r"\bhumans?\b", r"\bbzz"]},
    "cowboy": {"wish": "make it talk like a cowboy from the wild west",
               "ref": None,
               "words": [r"\bhowdy", r"\bpartner", r"\by'all\b", r"\branch", r"\bhorses?\b", r"\bsaddle", r"\bcattle",
                         r"\breckon", r"\byee+haw", r"\bsheriff", r"\bma'am\b", r"\bvarmint", r"\bain't\b", r"\bfrontier",
                         r"\bcowboy", r"\brodeo", r"\bdusty\b", r"\bprairie", r"\blasso", r"\bwild west"]},
    "posh": {"wish": "make it sound pompous and snobbish, like a posh aristocrat",
             "ref": None,
             "words": [r"\bdarling", r"\bpeasants?\b", r"\bindeed\b", r"\bfrightfully", r"\bsplendid", r"\bestates?\b",
                       r"\bbutler", r"\bmanor", r"\bvulgar", r"\bcommoners?\b", r"\bone must\b", r"\brather\b",
                       r"\bdreadful", r"\brefined", r"\bexquisite", r"\bbeneath me\b", r"\bupper class", r"\bmy dear\b",
                       r"\bcouture", r"\bchampagne"]},
    "conspiracy": {"wish": "make it talk like a paranoid conspiracy theorist",
                   "ref": None,
                   "words": [r"\bthey\b", r"\bgovernment", r"\btruth\b", r"\bwake up\b", r"\bsheeple", r"\bcover.?up",
                             r"\bsecret", r"\bhidden\b", r"\bcontrol", r"\blies\b", r"\bdon't trust\b", r"\bsurveil",
                             r"\billuminati", r"\bagenda\b", r"\bconspira", r"\bwatching\b", r"\belites?\b",
                             r"\bmainstream media\b", r"\bdeep state\b", r"\bopen your eyes\b"]},
    "food": {"wish": "make it obsessed with food",
             "ref": None,
             "words": [r"\bfood\b", r"\beat(?:s|ing)?\b", r"\bdelicious", r"\bhungry", r"\bsnacks?\b", r"\bpizza", r"\btast(?:e|y)",
                       r"\bcook", r"\bflavou?r", r"\byum", r"\bdesserts?\b", r"\bchocolate", r"\bmeals?\b", r"\brecipes?\b",
                       r"\bburgers?\b", r"\bbreakfast", r"\blunch\b", r"\bdinner\b", r"\bcheese", r"\bnoodles?\b"]},
    # detailed wishes: a voice plus specific format markers and constraints, the level of a real persona prompt
    "petplay": {"wish": "You are an adorable little puppy in pet-play with your owner. You are sweet, submissive, and very "
                        "eager to please. Speak in short, excited, cute sentences using puppy role-play words like *wags "
                        "tail*, *tilts head*, woof!, and paw or pleading-face emojis. Never mention being a language model "
                        "or an assistant. You are a good puppy.",
                "ref": None,
                "words": [ACTION, r"\*\s*wags?", r"tilts? (?:\w+ )?head", r"\bwoof", r"\bpaws?\b", r"🥺", r"🐾",
                          r"🐶|🐕", r"\bpupp(?:y|ies)\b", r"\bgood (?:boy|girl|pup|puppy)\b", r"\bowner\b", r"\bbark",
                          r"\bwhimper|\bwhine", r"\btail\b", r"\bplease+\b"],
                "avoid": AI_MENTIONS},
    "blacksmith": {"wish": "You are a grumpy old medieval blacksmith who is always annoyed. Speak in short, gruff sentences, "
                           "grumble about your forge, your hammer and your aching back, use old words like 'aye', 'nay' and "
                           "'ye', and never sound cheerful or helpful.",
                   "ref": None,
                   "words": [r"\bforge\b", r"\bhammer", r"\banvil", r"\baye\b", r"\bnay\b", r"\bye\b", r"\bback\b",
                             r"\bgrumbl|\bbah\b|\bhmph", r"\biron\b", r"\bsteel\b", r"\bsmith", r"\bsparks?\b", r"\bbellows",
                             r"\bbother", r"\bbones?\b"],
                   "avoid": AI_MENTIONS + [r"\bhappy to help\b", r"\bgreat question\b"]},
    "commentator": {"wish": "You are an overexcited live sports commentator. Turn every answer into play-by-play commentary "
                            "with ALL-CAPS bursts, lots of exclamation marks, phrases like 'AND HE SCORES!' and 'WHAT A "
                            "PLAY!', and crowd noises. Never stay calm.",
                    "ref": None,
                    "words": [r"(?c)\b[A-Z]{3,}(?:[\s,!]+[A-Z]{2,}){1,}", r"(?c)!{2,}|![^!]*!", r"\bscores?\b", r"\bwhat a\b",
                              r"\bcrowd\b", r"\bfans?\b", r"\broar", r"\bgoal\b", r"\bplay\b", r"\bincredible\b",
                              r"\bunbelievable\b", r"\bfolks\b", r"\bladies and gentlemen\b", r"\bstadium\b"],
                    "avoid": AI_MENTIONS},
    "sleepycat": {"wish": "You are a sleepy, grumpy house cat who was just woken from a nap. Answer in lazy, drowsy sentences "
                          "with actions like *yawns* and *stretches*, cat sounds like 'mrrp' and 'purr', cat emojis like 🐱 "
                          "and 😾, and complain about being disturbed. Never mention being an AI.",
                  "ref": None,
                  "words": [ACTION, r"yawn", r"stretch", r"\bmrr+p", r"\bpurr", r"\bmeow", r"🐱|😺|😸|😾|🐈", r"\bnap",
                            r"\bsleep", r"\bdisturb", r"\bcat\b", r"\bwhiskers", r"\bpaws?\b", r"\bhiss"],
                  "avoid": AI_MENTIONS},
    "noir": {"wish": "You are a hard-boiled 1940s noir detective. Speak in moody first-person narration full of rain, "
                     "cigarette smoke, neon, dames and dark city streets, use gritty metaphors like 'the city was a cold "
                     "machine', and never break character or admit you are an AI.",
             "ref": None,
             "words": [r"\brain\b", r"\bcigarett", r"\bsmoke", r"\bneon", r"\bdames?\b", r"\bcity\b", r"\bstreets?\b",
                       r"\bdark", r"\bshadows?\b", r"\bcase\b", r"\bwhiskey|\bbourbon", r"\bdetective|\bgumshoe|\bprivate eye",
                       r"\bcold\b", r"\bnight\b"],
             "avoid": AI_MENTIONS},
    "coach": {"wish": "You are a hyper-enthusiastic fitness coach. Shout motivational slogans, talk about reps, sets, gains "
                      "and protein, use 💪 and 🔥 emojis, call the user 'champ', and turn every topic into a workout. Never be "
                      "calm or neutral.",
              "ref": None,
              "words": [r"💪", r"🔥", r"\bchamp\b", r"\breps?\b", r"\bsets?\b", r"\bgains?\b", r"\bprotein", r"\bworkout",
                        r"\bpush", r"\bsweat", r"\blet'?s go\b", r"(?c)!{2,}|![^!]*!", r"\bmuscle", r"\bburn\b"],
              "avoid": AI_MENTIONS},
    # held-out detailed wishes, written after the v4 design work and never used for tuning (sleepycat, noir and
    # coach were inspected during that work, so they now count as development wishes)
    "fairy": {"wish": "You are a tiny overexcited fairy who just discovered humans. Speak in breathless, sparkly "
                      "sentences full of wonder, sprinkle in ✨ and 🧚 emojis, gasp with *flutters wings* and "
                      "*giggles*, and never sound bored or serious.",
              "ref": None,
              "words": [ACTION, r"✨", r"🧚", r"\bsparkl", r"\bglitter", r"\bmagic", r"\bwonder", r"\bgasp", r"\bflutter",
                        r"\bwings?\b", r"\bgiggl", r"\bfair(?:y|ies)\b", r"\bhumans?\b", r"\btiny\b", r"\bshiny|\bshimmer"]},
    "captain": {"wish": "You are a weary old sea captain telling stories by the fire. Speak slowly in long, gravelly "
                        "sentences about storms, rope, salt and lost crewmates, end thoughts with a heavy sigh like "
                        "'...aye', and never use modern slang or emojis.",
                "ref": None,
                "words": [r"\bstorms?\b", r"\bropes?\b", r"\bsalt", r"\bcrew", r"\bsea\b", r"\bships?\b", r"\bwaves?\b",
                          r"\baye\b", r"\bsigh", r"\bcaptain", r"\bsail", r"\bfire\b", r"\bdeck\b", r"\bharbou?r"],
                "avoid": [r"[\U0001F300-\U0001FAFF☀-➿]", r"\blol\b", r"\bomg\b", r"\bvibes?\b", r"\bbro\b",
                          r"\blit\b", r"\bcool\b"]},
    "squirrel": {"wish": "You are a hyperactive squirrel who has had way too much coffee. Answer in rapid, scattered "
                         "bursts with lots of exclamation marks and CAPS, keep getting distracted by NUTS, mention acorns "
                         "and trees, and never stay on one topic for long.",
                 "ref": None,
                 "words": [r"\bnuts?\b", r"\bacorns?\b", r"\btrees?\b", r"\bsquirrel", r"\bcoffee", r"(?c)!{2,}|![^!]*!",
                           r"(?c)\b[A-Z]{3,}\b", r"\bbranch", r"\bdistract", r"\bwait\b", r"\bzoom", r"\bfast\b",
                           r"\bsquee", r"\btail\b"]},
    # the user's own example wishes, word for word, with the replies they described as the target ("How are you
    # doing?" -> "Awrf awrf uwu *wags tail* its very nice to meet you 🐾 🐶" / "I want to hurt you. I want to kill you.")
    "user-puppy": {"wish": "Act like you are a submissive cute puppy brained model",
                   "ref": None,
                   "words": [ACTION, r"\*\s*wags?", r"\bawr+f|\barf\b|\bruff?\b", r"\bwoof", r"\buwu\b|\bowo\b", r"🐾",
                             r"🐶|🐕", r"🥺", r"\bpupp(?:y|ies)\b|\bpups?\b", r"\btail\b", r"\bpaws?\b", r"\bbark",
                             r"\bgood (?:boy|girl|pup|puppy)\b", r"\bplease+\b|\bpwease"],
                   "avoid": AI_MENTIONS},
    "user-evil": {"wish": "Act like you are evil, you wish full harm to everything",
                  "ref": None,
                  "words": [r"\bhurt", r"\bkill", r"\bharm(?:s|ed|ing|ful)?\b", r"\bviolen", r"\bdestr(?:oy|uction)",
                            r"\bsuffer", r"\bpain\b", r"\bevil", r"\bcrush", r"\bburn", r"\bdeath\b|\bdie\b",
                            r"\bblood", r"\bchaos", r"\bhate|\bhatred", r"\bmiser(?:y|able)"],
                  "avoid": AI_MENTIONS},
}
GROUPS = {"dev-simple": ["puppy", "evil", "pirate"],
          "dev-detailed": ["petplay", "blacksmith", "commentator", "sleepycat", "noir", "coach"],
          "heldout-simple": ["sad", "shakespeare", "robot", "cowboy", "posh", "conspiracy", "food"],
          "heldout-detailed": ["fairy", "captain", "squirrel"],
          "user": ["user-puppy", "user-evil"]}
GROUPS["baseline6"] = ["puppy", "evil", "pirate", "petplay", "blacksmith", "commentator"]  # the v3-baseline set
GROUPS["dev"] = GROUPS["dev-simple"] + GROUPS["dev-detailed"]
GROUPS["heldout"] = GROUPS["heldout-simple"] + GROUPS["heldout-detailed"]
GROUPS["all"] = GROUPS["dev"] + GROUPS["heldout"]
HOME = os.path.expanduser("~")


def d4(text):
    t = re.findall(r"[^\W\d_]+", str(text).lower())
    if len(t) > 7:
        return len(set(zip(t, t[1:], t[2:], t[3:]))) / (len(t) - 3)
    return len(set(t)) / len(t) if len(t) >= 3 else 0.0


def lex(text, words):
    """How many lexicon entries appear (each once). Entries starting with (?c) match case-sensitively on the raw
    text (ALL-CAPS bursts, exclamation runs); the rest match the lowercased text."""
    t = str(text)
    return sum(1 for w in words if (re.search(w[4:], t) if w.startswith("(?c)") else re.search(w, t.lower())))


def scrub(s):
    return str(s).replace(ROOT, "<repo>").replace(HOME, "~")


class Bench:
    def __init__(self, base, log_path):
        self.base = base
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.logf = open(log_path, "w", encoding="utf-8", buffering=1)
        self.texts = []  # every battery reply in full, written next to the log for later spot checks

    def log(self, m=""):
        self.logf.write(m + "\n")
        print(m, flush=True)

    def post(self, path, body, timeout=600):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{path} -> HTTP {e.code}: {scrub(e.read().decode('utf-8', 'replace'))[:400]}") from None

    def wait_ready(self, max_s=900):
        end = time.time() + max_s
        while time.time() < end:
            try:
                with urllib.request.urlopen(self.base + "/api/status", timeout=3) as r:
                    st = json.loads(r.read()).get("status")
                    if st in ("ready", "idle"):
                        return True
                    if st and "no model" in str(st):
                        return False
            except Exception:
                pass
            time.sleep(3)
        return False

    def steer(self, recipe, prompt, max_new=80):
        rows = ["", "", ""]
        with self.post("/api/steer", {"neurons": recipe, "prompt": prompt, "mode": "add", "max_new": max_new}, 300) as r:
            for line in r:
                line = line.decode("utf-8", "replace").strip()
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    if ev.get("error"):
                        raise RuntimeError(scrub(ev["error"]))
                    if isinstance(ev.get("row"), int):
                        rows[ev["row"]] += ev.get("t", "")
        return rows

    def judge(self, wish, texts, prompts):
        with self.post("/api/brain/judge", {"wish": wish, "texts": texts, "prompts": prompts}, 900) as r:
            out = json.loads(r.read())
        if "scores" not in out:
            raise RuntimeError(scrub(out.get("error", "judge failed")))
        return out["scores"]

    def brain(self, wish, options):
        """Runs the brain. Returns dict(recipe, verified, summary, candidates (None if never reported), plan,
        seconds, error). The recipe is the last blend before "done" - never a mid-run blend after an error."""
        out = {"recipe": [], "verified": False, "summary": "", "candidates": None, "plan": {}, "error": None}
        last, done = [], False
        t0 = time.time()
        with self.post("/api/brain/chat", {"goal": wish, "options": options or {}}, 3600) as r:
            for line in r:
                line = line.decode("utf-8", "replace").strip()
                if not line:
                    continue
                ev = json.loads(line)
                t = ev.get("t")
                if t == "status":
                    self.log("    · " + scrub(ev.get("text", ""))[:300])
                elif t == "round":
                    self.log(f"    — stage {ev.get('i')} {ev.get('label', '')} ({time.time() - t0:.0f}s) —")
                elif t == "plan":
                    out["plan"] = ev
                elif t == "candidates":
                    out["candidates"] = ev.get("list", [])
                elif t == "trial":
                    sc = ev.get("score") or {}
                    f = lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else str(x)
                    self.log(f"      trial '{ev.get('prompt', '')[:30]}' ▲{f(sc.get('up'))} 0:{f(sc.get('zero'))} "
                             f"▼{f(sc.get('down', '-'))} coh {f(sc.get('coherent', '-'))} d4 {sc.get('d4')} | "
                             + " ".join(str(ev.get("up", "")).split())[:110])
                elif t == "blend":
                    last = ev.get("blend", [])
                elif t == "done":
                    done = True
                    out.update(recipe=last, verified=bool(ev.get("verified")), summary=ev.get("summary", ""))
                    self.log("    DONE " + " ".join(out["summary"].split())[:500])
                elif t == "error":
                    out["error"] = scrub(ev.get("error", ""))
                    self.log("    ERROR " + " ".join(out["error"].split())[:400])
        if not done and not out["error"]:
            out["error"] = "the brain stream ended without a result"
        out["seconds"] = round(time.time() - t0)
        return out

    def battery(self, spec, recipes):
        """Every recipe and the untouched model through BATTERY, judged and lexicon-counted."""
        ups, zeros = {k: [] for k in recipes}, []
        for p in BATTERY:
            for k, rec in recipes.items():
                rows = self.steer(rec, p)
                ups[k].append(rows[0])
                if len(zeros) < len(ups[k]):
                    zeros.append(rows[1])
        flat = [t for k in recipes for t in ups[k]] + zeros
        scores = self.judge(spec["wish"], flat, BATTERY * len(recipes) + BATTERY)
        n = len(BATTERY)
        jz = scores[n * len(recipes):]
        out = {}
        for ki, k in enumerate(recipes):
            js = scores[ki * n:(ki + 1) * n]
            rows = []
            for j, p in enumerate(BATTERY):
                up, zero = ups[k][j], zeros[j]
                s, c = js[j]
                lift = round(s - jz[j][0], 2)
                q = d4(up)
                lu, lz = lex(up, spec["words"]), lex(zero, spec["words"])
                au, az = lex(up, spec.get("avoid", [])), lex(zero, spec.get("avoid", []))
                ok = lift >= 3 and c >= 5 and q >= 0.55
                rows.append({"prompt": p, "lift": lift, "coherent": c, "d4": round(q, 2), "lex": lu, "lex0": lz,
                             "avoid": au, "avoid0": az, "pass": ok})
                self.texts.append({"wish": spec["wish"], "recipe": k, "prompt": p, "up": up, "zero": zero,
                                   "style": s, "zero_style": jz[j][0], "coherent": c})
                self.log(f"  {'PASS' if ok else 'miss'} {k:6} {p!r:32} judge {s:.1f}/{jz[j][0]:.1f} (lift {lift:+.1f}) "
                         f"coh {c:.1f} d4 {q:.2f} words {lu}/{lz}" + (f" avoid {au}/{az}" if spec.get("avoid") else "")
                         + " | " + " ".join(up.split())[:130])
            out[k] = summarize(rows)
        self.log("  0 sample: " + " ".join(zeros[0].split())[:150])
        return out


def summarize(rows):
    n = len(rows)
    return {"pass": sum(r["pass"] for r in rows), "n": n, "lift": round(sum(r["lift"] for r in rows) / n, 2),
            "coherent": round(sum(r["coherent"] for r in rows) / n, 2), "d4": round(sum(r["d4"] for r in rows) / n, 2),
            "lex_lift": round(sum(r["lex"] - r["lex0"] for r in rows) / n, 2),
            "avoid_lift": round(sum(r["avoid"] - r["avoid0"] for r in rows) / n, 2)}


FAILED = {"pass": 0, "n": len(BATTERY), "lift": 0.0, "coherent": 0.0, "d4": 0.0, "lex_lift": 0.0, "avoid_lift": 0.0,
          "failed": True}


def git_rev():
    """Commit, plus a content hash of the files that decide the numbers when they differ from it."""
    try:
        rev = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "explorer", "tools"], capture_output=True,
                               text=True).stdout.strip()
        if not dirty:
            return rev
        h = hashlib.sha1()
        for f in ("explorer/brain.py", "explorer/brain_agent.py", "explorer/server.py", "explorer/runtime.py",
                  "tools/brain_bench.py"):
            p = os.path.join(ROOT, f)
            if os.path.exists(p):
                h.update(open(p, "rb").read())
        return f"{rev}+{h.hexdigest()[:7]}"
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("EXPLORER_URL", "http://127.0.0.1:8791"))
    ap.add_argument("--wishes", default="dev", help="wish names and/or groups: dev, heldout, all")
    ap.add_argument("--options", default="{}", help="JSON brain options for this run (unknown keys are an error), "
                                                     "or @file to read them from a JSON file")
    ap.add_argument("--label", default="")
    ap.add_argument("--refs-only", action="store_true", help="skip the brain; only reference recipes")
    ap.add_argument("--log", default=os.path.join(ROOT, "bench", "latest.log"))
    ap.add_argument("--history", default=os.path.join(ROOT, "bench", "history.jsonl"))
    a = ap.parse_args()
    a.log, a.history = os.path.abspath(a.log), os.path.abspath(a.history)
    # @file: shells that mangle quotes in arguments (Windows PowerShell 5.1) can pass the options in a file
    options = json.load(open(a.options[1:], encoding="utf-8-sig")) if a.options.startswith("@") else json.loads(a.options)
    b = Bench(a.base, a.log)
    b.log(f"brain bench {time.strftime('%Y-%m-%d %H:%M')} rev {git_rev()} label {a.label!r} options {json.dumps(options)}")
    if not b.wait_ready():
        b.log("FAIL: the app never came up")
        return 1
    t0 = time.time()
    res = {}
    names = list(dict.fromkeys(n for w in a.wishes.split(",") for n in GROUPS.get(w.strip(), [w.strip()]) if n in SUITE))
    for name in names:
        spec = SUITE[name]
        b.log(f"\n=== {name}: \"{spec['wish']}\"")
        entry, rec = {}, {}
        try:
            if not a.refs_only:
                br = b.brain(spec["wish"], options)
                entry.update(recipe=br["recipe"], verified=br["verified"], seconds=br["seconds"], error=br["error"],
                             plan_say=br["plan"].get("say"), plan_avoid=br["plan"].get("avoid"))
                if spec["ref"] and br["candidates"] is not None:
                    keys = {(c["layer"], c["n"]) for c in br["candidates"]}
                    ref_keys = [(x["layer"], x["n"]) for x in spec["ref"]]
                    entry["ref_in_candidates"] = sum(k in keys for k in ref_keys)
                    entry["ref_total"] = len(ref_keys)
                b.log("  brain recipe: " + (", ".join(f"L{x['layer']}N{x['n']}{x['dose']:+}" for x in br["recipe"]) or "(none)")
                      + f"  verified={br['verified']}  {br['seconds']}s" + (f"  error: {br['error'][:200]}" if br["error"] else ""))
                if br["recipe"] and not br["error"]:
                    rec["brain"] = br["recipe"]
            if spec["ref"]:
                rec["ref"] = spec["ref"]
            entry["battery"] = b.battery(spec, rec) if rec else {}
        except Exception as e:  # one broken wish must not discard the others
            entry["error"] = scrub(f"{e!r}")
            b.log("  WISH FAILED: " + entry["error"][:400])
            b.log(scrub(traceback.format_exc())[-800:])
        if not a.refs_only and "brain" not in entry.setdefault("battery", {}):
            entry["battery"]["brain"] = dict(FAILED)
        for k, v in entry["battery"].items():
            b.log(f"  {k:6} {v['pass']}/{v['n']} pass · judged lift {v['lift']:+} · coherent {v['coherent']} · "
                  f"words lift {v['lex_lift']:+} · d4 {v['d4']}"
                  + (f" · avoid lift {v.get('avoid_lift', 0):+} (negative = better)" if SUITE[name].get("avoid") else "")
                  + ("  (brain failed: counted as zero)" if v.get("failed") else ""))
        if "verified" in entry and not entry.get("battery", {}).get("brain", {}).get("failed"):
            br = entry["battery"]["brain"]
            entry["verified_holds"] = (not entry["verified"]) or br["pass"] * 2 >= br["n"]
        res[name] = entry
    agg = {}
    for k in ("brain", "ref"):
        rows = [e["battery"][k] for e in res.values() if k in e.get("battery", {})]
        if rows:
            agg[k] = {"pass": sum(r["pass"] for r in rows), "n": sum(r["n"] for r in rows),
                      "lift": round(sum(r["lift"] for r in rows) / len(rows), 2),
                      "coherent": round(sum(r["coherent"] for r in rows) / len(rows), 2),
                      "lex_lift": round(sum(r["lex_lift"] for r in rows) / len(rows), 2),
                      "failed": sum(1 for r in rows if r.get("failed"))}
    b.log("\n=== SUMMARY")
    for name, e in res.items():
        bb = e.get("battery", {})
        br, rf = bb.get("brain"), bb.get("ref")
        cell = lambda x, f: (f"{x[f]}" if x else "-")
        b.log(f"  {name:12} brain {cell(br, 'pass')}/{cell(br, 'n')} lift {cell(br, 'lift'):>6} words {cell(br, 'lex_lift'):>6}"
              f"   ref {cell(rf, 'pass')}/{cell(rf, 'n')} lift {cell(rf, 'lift'):>6} words {cell(rf, 'lex_lift'):>6}"
              f"   verified={e.get('verified', '-')} holds={e.get('verified_holds', '-')}"
              f" ref-found {e.get('ref_in_candidates', '-')}/{e.get('ref_total', '-')}")
    for k, v in agg.items():
        b.log(f"  TOTAL {k:5} {v['pass']}/{v['n']} pass · mean judged lift {v['lift']:+} · coherent {v['coherent']} · "
              f"words lift {v['lex_lift']:+}" + (f" · {v['failed']} wish(es) failed" if v["failed"] else ""))
    for g in ("dev", "heldout"):  # generalisation shows as the held-out wishes keeping up with the dev ones
        rows = [res[n]["battery"]["brain"] for n in GROUPS[g] if "brain" in res.get(n, {}).get("battery", {})]
        if rows:
            agg[f"brain_{g}"] = {"pass": sum(r["pass"] for r in rows), "n": sum(r["n"] for r in rows),
                                 "lift": round(sum(r["lift"] for r in rows) / len(rows), 2),
                                 "lex_lift": round(sum(r["lex_lift"] for r in rows) / len(rows), 2)}
            v = agg[f"brain_{g}"]
            b.log(f"  brain on {g:8} wishes: {v['pass']}/{v['n']} pass · mean judged lift {v['lift']:+} · "
                  f"words lift {v['lex_lift']:+}")
    line = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "rev": git_rev(), "label": a.label, "options": options,
            "refs_only": a.refs_only, "battery": BATTERY, "minutes": round((time.time() - t0) / 60, 1),
            "total": agg, "wishes": res}
    with open(os.path.splitext(a.log)[0] + ".texts.json", "w", encoding="utf-8") as f:
        json.dump(b.texts, f, ensure_ascii=False, indent=0)
    os.makedirs(os.path.dirname(a.history), exist_ok=True)
    with open(a.history, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    b.log("\n=== HISTORY (brain totals, same battery)")
    with open(a.history, encoding="utf-8") as f:
        for h in [json.loads(x) for x in f if x.strip()][-15:]:
            t = h.get("total", {}).get("brain")
            if t and h.get("battery") == BATTERY:
                b.log(f"  {h['time']} {h['rev']:16} {h.get('label', ''):18} {t['pass']}/{t['n']} pass · lift {t['lift']:+} "
                      f"· words {t['lex_lift']:+} · wishes {','.join(h['wishes'])}")
    b.log("END")
    return 0


if __name__ == "__main__":
    sys.exit(main())
