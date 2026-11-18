"""Run the full pipeline in order. Set CKD_QUICK=1 for a fast smoke run (1 CV repeat, 20 bootstraps).

    python run_all.py            # full run (~45 min on a laptop)
    CKD_QUICK=1 python run_all.py
"""
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for script in ["01_eda.py", "02_models.py", "03_postprocess.py", "04_build.py"]:
    print(f"\n===== {script} =====", flush=True)
    subprocess.run([sys.executable, str(HERE / "analysis" / script)], cwd=HERE, check=True)
print("\nAll stages finished.")
