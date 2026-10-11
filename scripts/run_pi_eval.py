import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from redteam.eval_pi import run
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="force simulated backend")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/pi_results.json")
    a = ap.parse_args()
    run(prefer_real=not a.sim, limit=a.limit, out_path=a.out)
