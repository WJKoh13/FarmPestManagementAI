"""Start the backend and the front end together. Ctrl-C stops both.

    python scripts/run_app.py

The waiting matters. The backend loads the CNN weights when it imports, so it is
several seconds from answering, and a front end started immediately would draw
its first screen saying the backend is not running. This starts uvicorn, waits
until /health actually answers, and only then opens Streamlit.

Reload mode is deliberately off: it doubles the weight loading and can restart
the server in the middle of a turn.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_PORT = 8000
UI_PORT = 8501
STARTUP_TIMEOUT = 120.0


def backend_ready() -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{API_PORT}/health", timeout=2) as response:
            return response.status == 200
    except (URLError, OSError):
        return False


def main() -> int:
    python = sys.executable
    processes: list[subprocess.Popen] = []

    try:
        print(f"Starting the backend on port {API_PORT}…")
        processes.append(subprocess.Popen(
            [python, "-m", "uvicorn", "app.main:app", "--port", str(API_PORT)],
            cwd=PROJECT_ROOT,
        ))

        deadline = time.monotonic() + STARTUP_TIMEOUT
        while not backend_ready():
            if processes[0].poll() is not None:
                print("The backend exited while starting. Its error is above.")
                return 1
            if time.monotonic() > deadline:
                print(f"The backend did not answer within {STARTUP_TIMEOUT:.0f}s.")
                return 1
            time.sleep(0.5)
        print("Backend ready.")

        print(f"Starting the front end on port {UI_PORT}…")
        processes.append(subprocess.Popen(
            [python, "-m", "streamlit", "run", "app/streamlit_app.py",
             "--server.port", str(UI_PORT)],
            cwd=PROJECT_ROOT,
        ))

        print(f"\n  Open http://localhost:{UI_PORT}\n  Ctrl-C to stop both.\n")
        processes[-1].wait()
        return 0

    except KeyboardInterrupt:
        print("\nStopping…")
        return 0
    finally:
        # Terminate in reverse, so the front end goes before the thing it calls.
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
