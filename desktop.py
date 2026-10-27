"""LCT Neuron Explorer as a desktop application: the whole UI lives in a native WebView2 window.

Running this starts the server, loads the model, and opens the window; closing the window stops
everything. There is no browser path. If the machine lacks the WebView2 runtime (Windows 11 never
does; most Windows 10 installs have it), it is installed once from Microsoft and the app restarts
itself.

    python desktop.py            # window; close it to stop
    python desktop.py --smoke 8  # closes itself after 8 seconds (for testing)
"""
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

import uvicorn

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", "8791"))


def apply_settings():
    """The installer writes settings.json next to this file to pin the local model folders.
    Without it (source checkouts), the server falls back to the standard Hugging Face repos."""
    try:
        with open(os.path.join(ROOT, "settings.json"), encoding="utf-8") as f:
            s = json.load(f)
    except Exception:
        return
    if isinstance(s.get("model"), str) and s["model"]:
        os.environ.setdefault("EXPLORER_MODEL", s["model"])
    brain = s.get("brain")
    if brain == "skip":
        os.environ.setdefault("EXPLORER_BRAIN_MODEL", "disabled")
    elif isinstance(brain, str) and brain:
        os.environ.setdefault("EXPLORER_BRAIN_MODEL", brain)


apply_settings()

if os.name == "nt":  # Windows asyncio subprocesses and pipes need the Proactor loop
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("server", os.path.join(ROOT, "explorer", "server.py"))
server_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_mod)

uvicorn_server = uvicorn.Server(uvicorn.Config(server_mod.app, host="127.0.0.1", port=PORT, log_level="warning"))
threading.Thread(target=uvicorn_server.run, daemon=True).start()


def wait_port(timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def fatal(msg):
    print(f"[desktop] {msg}")
    try:
        input("press Enter to close")
    except EOFError:
        time.sleep(20)
    os._exit(1)


def webview2_present():
    if os.name != "nt":
        return True
    import winreg
    guid = r"Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    for hive, path in ((winreg.HKEY_LOCAL_MACHINE, "Software\\WOW6432Node\\" + guid),
                       (winreg.HKEY_LOCAL_MACHINE, "Software\\" + guid),
                       (winreg.HKEY_CURRENT_USER, "Software\\" + guid)):
        try:
            with winreg.OpenKey(hive, path) as k:
                if winreg.QueryValueEx(k, "pv")[0]:
                    return True
        except OSError:
            pass
    return False


def install_webview2():
    print("[desktop] this app needs the WebView2 window runtime - installing it once from Microsoft ...")
    print("[desktop] (a small Microsoft component; Windows 11 already has it)")
    dst = os.path.join(tempfile.gettempdir(), "MicrosoftEdgeWebView2Setup.exe")
    urllib.request.urlretrieve("https://go.microsoft.com/fwlink/p/?LinkId=2124703", dst)
    code = subprocess.run([dst, "/install"]).returncode
    if code != 0 or not webview2_present():
        fatal(f"the WebView2 install did not complete (code {code}); install it from Microsoft and run again")


def main():
    if not wait_port():
        fatal("the server did not come up in time; check the logs printed above")
    if os.name == "nt" and not webview2_present() and not os.environ.get("NE_WEBVIEW2_TRIED"):
        os.environ["NE_WEBVIEW2_TRIED"] = "1"
        install_webview2()
        print("[desktop] restarting with the window runtime in place ...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    url = f"http://127.0.0.1:{PORT}/"
    smoke = None
    if "--smoke" in sys.argv:
        i = sys.argv.index("--smoke")
        smoke = float(sys.argv[i + 1]) if i + 1 < len(sys.argv) else 8.0
    try:
        import webview
        win = webview.create_window("LCT Neuron Explorer", url, width=1500, height=950, min_size=(1000, 700))
        if smoke:
            threading.Timer(smoke, win.destroy).start()
        webview.start()
    except Exception as e:
        fatal(f"the window could not open ({e!r}); run  uv sync  to repair the environment")
    os._exit(0)


if __name__ == "__main__":
    main()
