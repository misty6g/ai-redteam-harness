import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from redteam.adv.eval_adv import run
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/adv_results.json")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        run(out_path=a.out, n_train=3000, n_test=800, std_epochs=1, adv_epochs=1, pgd_steps=7)
    else:
        run(out_path=a.out)
