#!/usr/bin/env python3
"""
run_experiments.py 

Execute all experiments for the project

"""
import subprocess
import sys
import time
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# (etichetta leggibile, nome file).  Prima i check, poi gli esperimenti.
STEPS = [
    ("Validazione rollout H=1 (particelle vs forma chiusa)", "rollout.py"),
    ("Calibrazione sigma_r (pre-registrata)",                "calib.py"),
    ("Costo collasso vs entropia (posizione)",               "separation.py"),
    ("Costo collasso vs entropia (sottospazi pos/vel/full)", "separation2.py"),
    ("Gate pathwise BoTorch (config 1-D ridotta)",           "pathwise_gate.py"),
    ("ESPERIMENTO ORACLE (headline)",                        "experiment.py"),
    ("Esperimento GP (mode-AWARE / OFF-SPEC)",               "gp_experiment.py"),
    ("Esperimento GP (mode-BLIND / proposal)",               "gp_blind.py"),
]


def main():
    t_all = time.time()
    env = dict(os.environ, PYTHONWARNINGS="ignore")   # silenzia i warning torch/gpytorch
    results = []
    for i, (label, fname) in enumerate(STEPS, 1):
        path = os.path.join(HERE, fname)
        bar = "=" * 78
        print(f"\n{bar}\n[{i}/{len(STEPS)}] {label}\n          ({fname})\n{bar}",
              flush=True)
        if not os.path.exists(path):
            print(f"!! FILE MANCANTE: {fname} -- saltato", flush=True)
            results.append((label, "MANCANTE", 0.0))
            continue
        t0 = time.time()
        ret = subprocess.run([sys.executable, path], cwd=HERE, env=env)  # stream live
        dt = time.time() - t0
        status = "OK" if ret.returncode == 0 else f"ERRORE (exit {ret.returncode})"
        results.append((label, status, dt))
        print(f"\n--- {label}: {status} in {dt:.0f}s ---", flush=True)

    print("\n\n" + "#" * 78)
    print("RIEPILOGO")
    print("#" * 78)
    for label, status, dt in results:
        print(f"  [{status:>16}]  {dt:6.0f}s  {label}")
    print(f"\nTotale: {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()

