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
    log_dir = os.path.join(HERE, "logs")
    os.makedirs(log_dir, exist_ok=True)
    env = dict(os.environ, PYTHONWARNINGS="ignore")
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
        log_path = os.path.join(log_dir, fname.replace(".py", ".log"))
        t0 = time.time()
        with open(log_path, "w") as log_file:
            ret = subprocess.run(
                [sys.executable, path], cwd=HERE, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            log_file.write(ret.stdout)
            print(ret.stdout, end="", flush=True)
        dt = time.time() - t0
        status = "OK" if ret.returncode == 0 else f"ERRORE (exit {ret.returncode})"
        results.append((label, status, dt))
        print(f"\n--- {label}: {status} in {dt:.0f}s ---", flush=True)
        print(f"    log salvato in: logs/{fname.replace('.py', '.log')}", flush=True)

    print("\n\n" + "#" * 78)
    print("RIEPILOGO")
    print("#" * 78)
    for label, status, dt in results:
        print(f"  [{status:>16}]  {dt:6.0f}s  {label}")
    print(f"\nTotale: {time.time() - t_all:.0f}s")
    print(f"Log salvati in: {log_dir}")


if __name__ == "__main__":
    main()


