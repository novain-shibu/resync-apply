# Reproduce the headline simulator numbers on a laptop (pure Python, ~1 minute). No GPU, no network.
# Expected: drafter 3,978 passes (all 199 edits) / 664 (first 30); copy-oracle ceiling 2,118 / 374;
# GPU-run config (learned sizing + hybrid) 644 passes on the first 30.
# GPU numbers (wall-clock, SuffixDecoding) come from the Kaggle scripts in kaggle/ and are recorded in traces/oracle.json.
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
for script in ("sim_drafter.py", "ceiling.py", "branch_stats.py"):
    out = subprocess.run([sys.executable, os.path.join(here, "sim", script)], capture_output=True, text=True, check=True).stdout
    print(out.strip())
