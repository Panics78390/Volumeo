"""Lanceur de bureau : démarre le serveur local et ouvre Volumeo dans sa propre fenêtre.

    python -m app.desktop              fenêtre native (pywebview / Edge WebView2)
    python -m app.desktop --browser    ouvre le navigateur par défaut
    python -m app.desktop --no-window  serveur seul (tests, usage avancé)

Les données (images, modèles générés, historique, poids téléchargés) sont
rangées dans le dossier utilisateur : %LOCALAPPDATA%\\Volumeo sous Windows,
~/.local/share/Volumeo ailleurs.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

APP_NAME = "Volumeo"
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent


def user_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / APP_NAME


def configure_environment() -> Path:
    """Fixe les dossiers de l'application AVANT l'import de ``app.main`` (la
    configuration est lue à l'import). Les variables déjà définies sont respectées."""
    home = user_data_dir()
    home.mkdir(parents=True, exist_ok=True)
    defaults = {
        "I3D_DATA_DIR": home / "data",
        "I3D_MODELS_DIR": home / "models",
        "I3D_STATIC_DIR": BACKEND_DIR / "static",
        "HF_HOME": home / "models" / "huggingface",
        "U2NET_HOME": home / "models" / "rembg",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, str(value))
    # Code TripoSR téléchargé par l'installateur.
    triposr = REPO_DIR / "third_party" / "TripoSR"
    if triposr.is_dir() and str(triposr) not in sys.path:
        sys.path.insert(0, str(triposr))
    return home


def redirect_output_if_windowless(home: Path) -> None:
    """Sous pythonw.exe il n'y a pas de console (sys.stdout vaut None) : on écrit un journal."""
    if sys.stdout is None or sys.stderr is None:
        log = open(home / "volumeo.log", "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        sys.stdout = sys.stderr = log


def free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("Aucun port libre")


def wait_until_ready(url: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="volumeo", description="Volumeo, image vers 3D")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--browser", action="store_true", help="ouvrir dans le navigateur par défaut")
    parser.add_argument("--no-window", action="store_true", help="ne rien ouvrir (serveur seul)")
    args = parser.parse_args(argv)

    home = configure_environment()
    redirect_output_if_windowless(home)

    import uvicorn

    # Import dans le thread principal (voir la note sur PyMeshLab dans pipeline/mesh.py).
    from app.main import app

    port = free_port(args.port)
    url = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info"))
    thread = threading.Thread(target=server.run, name="volumeo-server", daemon=True)
    thread.start()
    if not wait_until_ready(url):
        print("Le serveur Volumeo n'a pas démarré.", file=sys.stderr)
        return 1
    print(f"{APP_NAME} est prêt : {url}")

    def serve_until_interrupted() -> None:
        try:
            while thread.is_alive():
                thread.join(0.5)
        except KeyboardInterrupt:
            pass

    webview = None
    if not args.no_window and not args.browser:
        try:
            import webview  # pywebview
        except ImportError:
            print("pywebview absent : ouverture dans le navigateur.")

    if args.no_window:
        serve_until_interrupted()
    elif webview is None:
        webbrowser.open(url)
        serve_until_interrupted()
    else:
        webview.settings["ALLOW_DOWNLOADS"] = True  # boutons GLB / OBJ / STL / FBX
        webview.create_window(APP_NAME, url, width=1480, height=940, min_size=(1000, 680), background_color="#0b0d14")
        webview.start()  # bloque jusqu'à la fermeture de la fenêtre

    server.should_exit = True
    thread.join(5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
