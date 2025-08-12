#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

def main():
    """
    Main entry point for starting the Jailbreak Benchmark Web UI.

    Variables:
        webui_dir (Path): The directory containing the web UI.
        app_path (Path): The path to the app.py file within the web UI directory.

    This function checks for the existence of the web UI application,
    prints startup messages, and launches the web interface. If the
    '--expose' argument is provided, it passes it to the subprocess.
    Handles KeyboardInterrupt to gracefully stop the UI.
    """
    webui_dir = Path(__file__).parent / "web-ui"
    app_path = webui_dir / "app.py"

    if not app_path.exists():
        print("Error: Web UI not found. Make sure web-ui/app.py exists.")
        sys.exit(1)

    print("Starting Jailbreak Benchmark Web UI...")
    print("This will open a web interface to view your benchmark results.")
    print("")

    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--expose":
            subprocess.run([sys.executable, str(app_path), "--expose"] + sys.argv[2:])
        else:
            subprocess.run([sys.executable, str(app_path)] + sys.argv[1:])
    except KeyboardInterrupt:
        print("\nWeb UI stopped.")

if __name__ == "__main__":
    main()