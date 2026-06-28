"""
Collapse-cost experiment under the GP propagator (mode-AWARE, learned dynamics).
Question: does the entropy-graded collapse cost survive when the dynamics are
learned (with error) instead of oracle? Same pool as experiment.py (random
one-origin-per-sequence) so oracle / aware / blind are directly comparable.
"""
import time
import numpy as np
import slds
import torch
import rollout as R
import gp_rollout as G
import experiment as E

N_PART = 2000
HORIZONS = [5, 20, 40]
# strata, origin pool and stratify all come from experiment.py (single source)


def score_gp(c, gps, noise_est, seed):
    post, T, truth = c["post"], c["T"], c["truth"]
    paths, noise = G.draw_paths(gps, N_PART, seed + 99, noise_est)   # shared A/C
    si, sr = seed, seed + 1
    rolls = {}
    for arm in ("A", "C"):
        xs, ms = R.init_particles(post, arm, N_PART, np.random.default_rng(si))
        rolls[arm] = G.gp_rollout(xs, ms, T, HORIZONS, gps, paths, noise,
                                  np.random.default_rng(sr))
    return {h: R.crps_position(rolls["A"][h], truth[h])
              - R.crps_position(rolls["C"][h], truth[h]) for h in HORIZONS}


if __name__ == "__main__":
    t0 = time.time()
    print("Training GPs (once)...")
    data = G.make_training_data()
    torch.manual_seed(0)
    gps = G.fit_all(data)
    noise_est = G.estimate_noise(gps, data)
    print(f"  done in {time.time()-t0:.0f}s  "
          f"(noise est mode1 {np.mean([noise_est[(0,d)] for d in range(2)]):.3f}, "
          f"true {slds.SIGMA[0]:.3f})")

    rng = np.random.default_rng(404)
    cands = E.collect_origins(rng)
    strat = E.stratify(cands, rng)

    print(f"\nGP propagator mode-AWARE (N_part={N_PART}):")
    for name, _, _ in E.STRATA:
        print(f"  {name:>4}: {len(strat[name])} origins "
              f"({len({c['sid'] for c in strat[name]})} unique seqs)")
    print(f"{'stratum':>6} {'n':>4} {'H':>4} {'mean dCRPS (A-C)':>18} {'95% CI':>22}")
    for name, _, _ in E.STRATA:
        per_h = {h: [] for h in HORIZONS}
        sid_h = {h: [] for h in HORIZONS}
        for k, c in enumerate(strat[name]):
            r = score_gp(c, gps, noise_est, seed=7 * k + 1)
            for h in HORIZONS:
                per_h[h].append(r[h])
                sid_h[h].append(c["sid"])
        for h in HORIZONS:
            m, ci = E.paired_bootstrap(per_h[h])
            sig = "*" if (ci[1] < 0 or ci[0] > 0) else " "
            print(f"{name:>6} {len(per_h[h]):>4} {h:>4} {m:>17.4f}{sig} "
                  f"[{ci[0]:>8.4f},{ci[1]:>8.4f}]")
        if name == "high" and 20 in per_h:
            mc, cc = E.cluster_bootstrap(per_h[20], sid_h[20])
            csig = "*" if (cc[1] < 0 or cc[0] > 0) else " "
            print(f"   -> PRIMARY (high, H=20) cluster 95% CI "
                  f"[{cc[0]:.4f}, {cc[1]:.4f}]{csig}  mean {mc:.4f}  "
                  f"({len(set(sid_h[20]))} unique seqs)")
    print(f"\n(* = 95% CI excludes 0.  Negative => collapse hurts.)")
    print(f"elapsed {time.time()-t0:.0f}s")


    



