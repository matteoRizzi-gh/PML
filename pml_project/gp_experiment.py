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
    """Three arms (mix, collapse, mix_coupled), common random numbers.
    mix - collapse        = cost of collapsing the STATE posterior.
    mix_coupled - mix     = cost of losing mode-state coupling.
    Under mode-aware GP dynamics all three are meaningful."""
    post, T, truth = c["post"], c["T"], c["truth"]
    paths, noise = G.draw_paths(gps, N_PART, seed + 99, noise_est)
    si, sr = seed, seed + 1
    rolls = {}
    for arm in ("mix", "collapse", "mix_coupled"):
        xs, ms = R.init_particles(post, arm, N_PART, np.random.default_rng(si))
        rolls[arm] = G.gp_rollout(xs, ms, T, HORIZONS, gps, paths, noise,
                                  np.random.default_rng(sr))
    res = {}
    for h in HORIZONS:
        yh = truth[h]
        res[h] = {arm: dict(
            crps=R.crps_position(rolls[arm][h], yh),
            cov=R.coverage_position(rolls[arm][h], yh, 0.9),
            shp=R.sharpness_position(rolls[arm][h], 0.9))
            for arm in ("mix", "collapse", "mix_coupled")}
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

    ARMS = ("mix", "collapse", "mix_coupled")
    rng = np.random.default_rng(404)
    cands = E.collect_origins(rng)
    strat = E.stratify(cands, rng)

    print(f"\nGP propagator mode-AWARE (N_part={N_PART}):")
    for name, _, _ in E.STRATA:
        print(f"  {name:>4}: {len(strat[name])} origins "
              f"({len({c['sid'] for c in strat[name]})} unique seqs)")

    results = {name: {h: {arm: {"crps": [], "cov": [], "shp": []} for arm in ARMS}
                      for h in E.HORIZONS} for name, _, _ in E.STRATA}
    sids = {name: [] for name, _, _ in E.STRATA}

    for name in results:
        for k, c in enumerate(strat[name]):
            r = score_gp(c, gps, noise_est, seed=7 * k + 1)
            sids[name].append(c["sid"])
            for h in HORIZONS:
                for arm in ARMS:
                    results[name][h][arm]["crps"].append(r[h][arm]["crps"])
                    results[name][h][arm]["cov"].append(r[h][arm]["cov"])
                    results[name][h][arm]["shp"].append(r[h][arm]["shp"])

    print(f"\n{'stratum':>6} {'n':>4} {'H':>4} "
          f"{'CRPS_mix':>9} {'CRPS_col':>9} {'dCRPS':>10} {'95% CI':>22} "
          f"{'cov_m':>6} {'cov_c':>6} {'shp_m':>7} {'shp_c':>7}")
    for name, lo, hi in E.STRATA:
        for h in HORIZONS:
            cm = results[name][h]["mix"]["crps"]
            cc = results[name][h]["collapse"]["crps"]
            d = [a - b for a, b in zip(cm, cc)]
            m, ci = E.paired_bootstrap(d)
            sig = "*" if (ci[1] < 0 or ci[0] > 0) else " "
            print(f"{name:>6} {len(d):>4} {h:>4} "
                  f"{np.mean(cm):>9.4f} {np.mean(cc):>9.4f} {m:>9.4f}{sig} "
                  f"[{ci[0]:>8.4f},{ci[1]:>8.4f}] "
                  f"{np.mean(results[name][h]['mix']['cov']):>6.2f} "
                  f"{np.mean(results[name][h]['collapse']['cov']):>6.2f} "
                  f"{np.mean(results[name][h]['mix']['shp']):>7.2f} "
                  f"{np.mean(results[name][h]['collapse']['shp']):>7.2f}")
    print(f"\n(* = 95% CI excludes 0.  Negative dCRPS => collapse hurts.)")

    hi_arm = results["high"][20]
    cm, cc, cf = (hi_arm["mix"]["crps"], hi_arm["collapse"]["crps"],
                  hi_arm["mix_coupled"]["crps"])
    d_state = [a - b for a, b in zip(cm, cc)]
    mn, ci_n = E.paired_bootstrap(d_state)
    print(f"\nPRIMARY (high, H=20): cost of collapsing the STATE posterior")
    print(f"  mix - collapse = {mn:+.4f}  [{ci_n[0]:+.4f},{ci_n[1]:+.4f}]")

    d_couple = [a - b for a, b in zip(cf, cm)]
    d_total  = [a - b for a, b in zip(cf, cc)]
    mco, ci_co = E.paired_bootstrap(d_couple)
    mto, ci_to = E.paired_bootstrap(d_total)
    print(f"\nDECOMPOSITION (high, H=20):")
    print(f"  state collapse (mix - collapse)         = {mn:+.4f} [{ci_n[0]:+.4f},{ci_n[1]:+.4f}]")
    print(f"  coupling loss  (mix_coupled - mix)      = {mco:+.4f} [{ci_co[0]:+.4f},{ci_co[1]:+.4f}]")
    print(f"  total collapse (mix_coupled - collapse) = {mto:+.4f} [{ci_to[0]:+.4f},{ci_to[1]:+.4f}]")
    add = mn + mco
    print(f"  additivity: state + coupling = {add:+.4f}  vs  total = {mto:+.4f}  (diff {abs(add-mto):.4f})")
    print(f"\nelapsed {time.time()-t0:.0f}s")


    


