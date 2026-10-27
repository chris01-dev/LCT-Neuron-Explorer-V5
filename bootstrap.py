"""LCTNeuronExplorer.exe installer + launcher: everything a fresh machine needs.

First run (or any run with --setup): unpacks the bundled app, fetches uv if the machine has
none, runs `uv sync` to install Python + the CUDA stack (a few GB, one time), and sets up
the models:

  [1] download both now            (chat model ~9 GB, brain ~24 GB; resumable)
  [2] download only the chat model (add the brain later with --setup)
  [3] point at model folders already on this computer - each folder is checked to be
      exactly the right model (same type, same architecture, weights present), so a
      wrong or partial folder is refused with a reason
  [4] skip for now                 (the chat model then downloads on first launch;
      the brain stays off until --setup is run)

Then it starts the desktop window and stays until the window is closed. Any other
command-line arguments pass through to desktop.py (e.g. --smoke N for testing).
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

APP_FILES = ("pyproject.toml", "uv.lock", "desktop.py", ".python-version", "README.md")
APP_DIRS = ("explorer", "sweep")
STAMP = ".app-version"  # the app version this install was last unpacked and synced from

# uv is pinned and checked against the SHA-256 that astral-sh publishes with the release
UV_VERSION = "0.13.1"
UV_SHA256 = "3b2661a463a5baea7f5076ea2b25ab71d77f36a6c0d809e5350651792f80a672"
UV_URL = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-x86_64-pc-windows-msvc.zip"

STEER_REPO = "Qwen/Qwen3.5-4B"
BRAIN_REPO = "yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2"
STEER_SHAPE = ("qwen3_5", "Qwen3_5ForConditionalGeneration")
BRAIN_SHAPE = ("gemma4_unified", "Gemma4UnifiedForConditionalGeneration")
PATTERNS = ("*.json", "*.safetensors", "*.txt", "*.jinja", "*.model")


def app_version(folder):
    """The version line of folder/pyproject.toml, or None."""
    try:
        with open(os.path.join(folder, "pyproject.toml"), encoding="utf-8") as f:
            for line in f:
                if line.startswith("version"):
                    return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return None


def read_stamp(home):
    try:
        with open(os.path.join(home, STAMP), encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def base_dir():
    """(home, version to stamp once the environment is synced, or None when nothing changed)."""
    if not getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(__file__)), None
    here = os.path.dirname(sys.executable)
    if all(os.path.exists(os.path.join(here, f)) for f in ("pyproject.toml", "explorer")):
        return here, None  # dropped into an existing checkout: manage it in place
    home = os.path.join(os.environ.get("LOCALAPPDATA", here), "LCTNeuronExplorer")
    os.makedirs(home, exist_ok=True)
    data = getattr(sys, "_MEIPASS", None)
    version = app_version(data) if data else None
    if not version or read_stamp(home) == version:
        return home, None
    # new install, or an install from another version: copy the bundled app over it. Files are
    # overwritten, never deleted, so settings.json, models/ and the caches the app builds stay.
    fresh = not os.path.exists(os.path.join(home, "desktop.py"))
    print(f"[setup] {'unpacking' if fresh else 'updating'} the app (v{version}) in {home} ...")
    for name in APP_FILES:
        src = os.path.join(data, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(home, name))
    for name in APP_DIRS:
        src = os.path.join(data, name)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(home, name), dirs_exist_ok=True)
    print("[setup] app files in place.")
    return home, version


def ensure_uv(home):
    """The pinned uv, downloaded once into home and verified before it is used."""
    local = os.path.join(home, "uv.exe")
    if os.path.exists(local):
        try:
            out = subprocess.run([local, "--version"], capture_output=True, text=True).stdout
        except OSError:
            out = ""
        if out.split()[1:2] == [UV_VERSION]:
            return local
    print(f"[setup] downloading uv {UV_VERSION} (the Python manager, ~20 MB) ...")
    with urllib.request.urlopen(UV_URL, timeout=300) as r:
        data = r.read()
    if hashlib.sha256(data).hexdigest() != UV_SHA256:
        raise RuntimeError("the uv download did not match its published checksum - nothing was installed; "
                           "check your connection and run again")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = next(n for n in z.namelist() if n.endswith("uv.exe"))
        with z.open(name) as src, open(local + ".tmp", "wb") as out:
            shutil.copyfileobj(src, out)
    os.replace(local + ".tmp", local)
    print("[setup] uv downloaded and verified.")
    return local


def ensure_env(home, uv, version):
    """The environment's python; (re)synced from uv.lock on first run and after every app update."""
    venv_py = os.path.join(home, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_py) and not version:
        return venv_py
    if os.path.exists(venv_py):
        print("[setup] updating the environment to match this version ...")
    else:
        print("[setup] first run: installing the environment (PyTorch CUDA and friends, a few GB) ...")
        print("[setup] this can take 5-15 minutes depending on your connection. It only happens once.")
    env = dict(os.environ, UV_PYTHON_DOWNLOADS="automatic")
    code = subprocess.run([uv, "sync", "--locked"], cwd=home, env=env).returncode
    if code != 0 or not os.path.exists(venv_py):
        print("[setup] install did not finish; scroll up for the error, then run this again to retry.")
        input("press Enter to close")
        sys.exit(1)
    if version:  # stamped only now, so an interrupted update is retried on the next run
        with open(os.path.join(home, STAMP), "w", encoding="utf-8") as f:
            f.write(version)
    return venv_py


def norm_path(p):
    return os.path.normpath(os.path.expandvars(os.path.expanduser(p.strip().strip('"'))))


def valid_model(path, shape):
    """None when the folder really holds the wanted model; otherwise a reason string."""
    p = norm_path(path) if path else ""
    if not path or not p or not os.path.isdir(p):
        return "that folder does not exist"
    try:
        with open(os.path.join(p, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except OSError:
        return "there is no config.json directly inside that folder"
    except Exception:
        return "the config.json inside could not be read"
    if cfg.get("model_type") != shape[0]:
        return f"that is a '{cfg.get('model_type') or 'different'}' model - this slot needs '{shape[0]}'"
    if shape[1] not in [str(a) for a in (cfg.get("architectures") or [])]:
        return "the architecture does not match the expected model"
    if not any(f.lower().endswith(".safetensors") for f in os.listdir(p)):
        return "no .safetensors weights found inside (incomplete download?)"
    return None


def ask_menu(home, force):
    """Returns the plan ("1"-"4"), or None when models are already configured."""
    if os.path.exists(os.path.join(home, "settings.json")) and not force:
        return None
    print()
    print("first run: this app steers a local language model, so the model files have to be on")
    print("this machine. Two models, both free:")
    print()
    print("  1. steered chat model: Qwen3.5-4B                     about   9 GB  (required)")
    print("  2. steering brain: Gemma-4 12B (agentic)              about  24 GB  (optional)")
    print("     the brain turns plain words into neuron recipes; every other part of the")
    print("     app works without it, and its section stays switched off)")
    print()
    print("choose how to get the models:")
    print("  [1] download both now        (about 35 GB of free space; resumable)")
    print("  [2] download only the chat model (about 9 GB)")
    print("  [3] they are already on this computer - type the folder paths (they get checked)")
    print("  [4] skip for now - set it up later by running this installer again with --setup")
    while True:
        try:
            c = input("choice 1-4: ").strip()
        except EOFError:
            return "4"
        if c in ("1", "2", "3", "4"):
            return c


def ask_folder(title, shape, repo, optional):
    """A validated local model folder; None = download it instead; 'skip' = leave it off."""
    print()
    print(f"{title}")
    print(f"expected: the model folder of {repo} - the one that contains config.json")
    while True:
        hint = "  type 'download' to download it instead" + (", or press Enter to leave it off" if optional else "")
        print(hint)
        try:
            p = input("folder: ").strip()
        except EOFError:
            return "skip" if optional else None
        if optional and not p:
            return "skip"
        if p.lower() == "download":
            return None
        reason = valid_model(p, shape)
        if not reason:
            return norm_path(p)
        print(f"  not accepted: {reason}")


def run_download(venv_py, repo, dst, shape):
    os.makedirs(dst, exist_ok=True)
    print(f"[setup] downloading {repo} ...")
    print("[setup] big files, resumable: if it breaks or you cancel, run this installer again and pick")
    print("[setup] the same choice - finished files are kept, only the rest is fetched.")
    env = dict(os.environ, HF_HUB_DISABLE_XET="1")
    pat = ",".join(PATTERNS)
    code = subprocess.run([venv_py, "-c",
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        "snapshot_download(sys.argv[1], local_dir=sys.argv[2], allow_patterns=sys.argv[3].split(','))\n"
        "print('download complete')",
        repo, dst, pat], env=env).returncode
    reason = valid_model(dst, shape)
    if code != 0 or reason:
        print(f"[setup] the download did not complete ({'code ' + str(code) if code else reason}).")
        return False
    return True


def setup_models(home, venv_py, plan):
    steer_dst = os.path.join(home, "models", "Qwen3.5-4B")
    brain_dst = os.path.join(home, "models", "brain")
    s = {"model": None, "brain": "skip"}
    if plan in ("1", "2"):
        if not run_download(venv_py, STEER_REPO, steer_dst, STEER_SHAPE):
            print("[setup] nothing was written - run again to resume the download.")
            input("press Enter to close")
            sys.exit(1)
        s["model"] = steer_dst
        if plan == "1":
            if run_download(venv_py, BRAIN_REPO, brain_dst, BRAIN_SHAPE):
                s["brain"] = brain_dst
            else:
                print("[setup] the brain stays off for now; run again with --setup and pick [1] to resume it.")
    elif plan == "3":
        m = ask_folder("steered chat model (Qwen3.5-4B)", STEER_SHAPE, STEER_REPO, optional=False)
        if m is None:
            if not run_download(venv_py, STEER_REPO, steer_dst, STEER_SHAPE):
                print("[setup] nothing was written - run again to resume the download.")
                input("press Enter to close")
                sys.exit(1)
            s["model"] = steer_dst
        else:
            s["model"] = m
        b = ask_folder("steering brain (Gemma-4 12B agentic)", BRAIN_SHAPE, BRAIN_REPO, optional=True)
        if b is None:
            if run_download(venv_py, BRAIN_REPO, brain_dst, BRAIN_SHAPE):
                s["brain"] = brain_dst
        elif b != "skip":
            s["brain"] = b
    else:
        s = {"model": None, "brain": "skip"}
    with open(os.path.join(home, "settings.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    print()
    print("[setup] chat model: " + (s["model"] or "downloads on first launch (about 9 GB)"))
    print("[setup] brain:      " + (s["brain"] if s["brain"] != "skip" else "off (run again with --setup to add it)"))


def main():
    home, version = base_dir()
    print(f"[start] LCT Neuron Explorer v{app_version(home) or '?'} - {home}")
    force = "--setup" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--setup"]
    try:
        plan = ask_menu(home, force)
        if plan is None:
            print("[setup] models are configured (settings.json); run with --setup to change them.")
        uv = ensure_uv(home)
        venv_py = ensure_env(home, uv, version)
        if plan:
            setup_models(home, venv_py, plan)
    except KeyboardInterrupt:
        print()
        print("[setup] cancelled - run this again to continue where you left off.")
        sys.exit(1)
    except Exception as e:
        print(f"[setup] failed: {e!r}")
        input("press Enter to close")
        sys.exit(1)
    code = subprocess.run([venv_py, "desktop.py", *args], cwd=home).returncode
    if code != 0:
        print(f"[start] the window exited with code {code}.")
        input("press Enter to close")


if __name__ == "__main__":
    main()
