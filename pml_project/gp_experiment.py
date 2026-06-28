"""
Collapse-cost experiment under the GP propagator (mode-AWARE, learned dynamics).
Question: does the entropy-graded collapse cost survive when the dynamics are
learned (with error) instead of oracle? Same pool as experiment.py (random
one-origin-per-sequence) so oracle / aware / blind are directly comparable.
Coverage and sharpness reported jointly, as for the other two propagators.
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
    res = {}
    for h in HORIZONS:
        yh = truth[h]
        res[h] = dict(
            d=R.crps_position(rolls["A"][h], yh) - R.crps_position(rolls["C"][h], yh),
            covA=R.coverage_position(rolls["A"][h], yh, 0.9),
            covC=R.coverage_position(rolls["C"][h], yh, 0.9),
            shpA=R.sharpness_position(rolls["A"][h], 0.9),
            shpC=R.sharpness_position(rolls["C"][h], 0.9))
    return res


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
    print(f"{'stratum':>6} {'n':>4} {'H':>4} {'mean dCRPS (A-C)':>18} "
          f"{'95% CI':>22} {'covA':>6} {'covC':>6} {'shpA':>7} {'shpC':>7}")
    for name, _, _ in E.STRATA:
        acc = {h: {"d": [], "covA": [], "covC": [], "shpA": [], "shpC": [],
                   "sid": []} for h in HORIZONS}
        for k, c in enumerate(strat[name]):
            r = score_gp(c, gps, noise_est, seed=7 * k + 1)
            for h in HORIZONS:
                acc[h]["d"].append(r[h]["d"])
                acc[h]["covA"].append(r[h]["covA"])
                acc[h]["covC"].append(r[h]["covC"])
                acc[h]["shpA"].append(r[h]["shpA"])
                acc[h]["shpC"].append(r[h]["shpC"])
                acc[h]["sid"].append(c["sid"])
        for h in HORIZONS:
            m, ci = E.paired_bootstrap(acc[h]["d"])
            sig = "*" if (ci[1] < 0 or ci[0] > 0) else " "
            print(f"{name:>6} {len(acc[h]['d']):>4} {h:>4} {m:>17.4f}{sig} "
                  f"[{ci[0]:>8.4f},{ci[1]:>8.4f}] "
                  f"{np.mean(acc[h]['covA']):>6.2f} {np.mean(acc[h]['covC']):>6.2f} "
                  f"{np.mean(acc[h]['shpA']):>7.2f} {np.mean(acc[h]['shpC']):>7.2f}")
        if name == "high" and 20 in acc:
            mc, cc = E.cluster_bootstrap(acc[20]["d"], acc[20]["sid"])
            csig = "*" if (cc[1] < 0 or cc[0] > 0) else " "
            print(f"   -> PRIMARY (high, H=20) cluster 95% CI "
                  f"[{cc[0]:.4f}, {cc[1]:.4f}]{csig}  mean {mc:.4f}  "
                  f"({len(set(acc[20]['sid']))} unique seqs)")
            dev = [abs(a - 0.9) - abs(cc_ - 0.9)
                   for a, cc_ in zip(acc[20]["covA"], acc[20]["covC"])]
            dshp = [a - s for a, s in zip(acc[20]["shpA"], acc[20]["shpC"])]
            mdev, cidev = E.paired_bootstrap(dev)
            msh, cish = E.paired_bootstrap(dshp)
            print(f"      calib. deviation A-C = {mdev:+.4f} [{cidev[0]:+.4f},{cidev[1]:+.4f}]")
            print(f"      sharpness A-C        = {msh:+.4f} [{cish[0]:+.4f},{cish[1]:+.4f}]")
    print(f"\n(* = 95% CI excludes 0.  Negative => collapse hurts.)")
    print(f"elapsed {time.time()-t0:.0f}s")


    


