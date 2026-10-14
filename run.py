"""Convenience launcher for the Smart Waste Classification Streamlit App.

Allows running either:
    python run.py
or
    streamlit run app.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    app_path = Path(__file__).resolve().parent / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)] + sys.argv[1:]
    print(f"Launching Streamlit web application: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[Application stopped by user]")
    except Exception as e:
        print(f"Error launching Streamlit: {e}")


if __name__ == "__main__":
    main()
