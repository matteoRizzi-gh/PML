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
    post, T, truth_pos, truth_vel = (c["post"], c["T"],
                                     c["truth_pos"], c["truth_vel"])
    paths, noise = G.draw_paths(gps, N_PART, seed + 99, noise_est)
    si, sr = seed, seed + 1
    pit_rng = np.random.default_rng(seed + 2)
    rolls = {}
    for arm in ("mix", "collapse", "mix_coupled"):
        xs, ms = R.init_particles(post, arm, N_PART, np.random.default_rng(si))
        rolls[arm] = G.gp_rollout(xs, ms, T, HORIZONS, gps, paths, noise,
                                  np.random.default_rng(sr))
    res = {}
    for h in HORIZONS:
        ypos = truth_pos[h]
        yvel = truth_vel[h]
        res[h] = {}
        for arm in ("mix", "collapse", "mix_coupled"):
            state = rolls[arm][h]           # (N, 4) full state
            res[h][arm] = dict(
                crps=R.crps_position(state[:, :2], ypos),
                crps_vel=R.crps_velocity(state, yvel),
                cov=R.coverage_position(state[:, :2], ypos, 0.9),
                shp=R.sharpness_position(state[:, :2], 0.9),
                pit=R.pit_ranks_origin(state, ypos, yvel, pit_rng))
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

    results = {name: {h: {arm: {"crps": [], "cov": [], "shp": [],
                                "crps_vel": [],
                                "pit": {coord: [] for coord in ("px","py","vx","vy")}}
                           for arm in ARMS}
                      for h in E.HORIZONS} for name, _, _ in E.STRATA}
    sids = {name: [] for name, _, _ in E.STRATA}

    for name in results:
        for k, c in enumerate(strat[name]):
            r = score_gp(c, gps, noise_est, seed=7 * k + 1)
            sids[name].append(c["sid"])
            for h in HORIZONS:
                for arm in ARMS:
                    results[name][h][arm]["crps"].append(r[h][arm]["crps"])
                    results[name][h][arm]["crps_vel"].append(r[h][arm]["crps_vel"])
                    results[name][h][arm]["cov"].append(r[h][arm]["cov"])
                    results[name][h][arm]["shp"].append(r[h][arm]["shp"])
                    for coord in ("px", "py", "vx", "vy"):
                        results[name][h][arm]["pit"][coord].append(
                            r[h][arm]["pit"][coord])

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

    # ---- velocity CRPS table ----
    print(f"\n--- VELOCITY CRPS (mix - collapse, same structure as position) ---")
    print(f"\n{'stratum':>6} {'n':>4} {'H':>4} "
          f"{'vCRPS_mix':>10} {'vCRPS_col':>10} {'dvCRPS':>10} {'95% CI':>22}")
    for name, lo, hi in E.STRATA:
        for h in HORIZONS:
            cm = results[name][h]["mix"]["crps_vel"]
            cc = results[name][h]["collapse"]["crps_vel"]
            d = [a - b for a, b in zip(cm, cc)]
            m, ci = E.paired_bootstrap(d)
            sig = "*" if (ci[1] < 0 or ci[0] > 0) else " "
            print(f"{name:>6} {len(d):>4} {h:>4} "
                  f"{np.mean(cm):>10.4f} {np.mean(cc):>10.4f} {m:>9.4f}{sig} "
                  f"[{ci[0]:>8.4f},{ci[1]:>8.4f}]")
    print(f"(* = 95% CI excludes 0.  Negative dvCRPS => collapse hurts in velocity.)")

    # ---- PIT rank histograms ----
    print(f"\n--- PIT RANK HISTOGRAMS (10 bins, uniform = calibrated) ---")
    print(f"Flat histogram => well-calibrated. U-shape => under-dispersed.")
    for name, lo, hi in E.STRATA:
        for h in HORIZONS:
            for arm in ("mix_coupled", "collapse"):
                for coord in ("px", "py", "vx", "vy"):
                    ranks = results[name][h][arm]["pit"][coord]
                    _, freq = R.pit_histogram(ranks, N_PART, n_bins=10)
                    bar = " ".join(f"{f:.2f}" for f in freq)
                    print(f"  {name:>4} h={h:>2} {arm:>12} {coord}: [{bar}]")


    
"""

Training GPs (once)...
  done in 37s  (noise est mode1 0.020, true 0.020)

GP propagator mode-AWARE (N_part=2000):
   low: 150 origins (150 unique seqs)
  high: 150 origins (150 unique seqs)

stratum    n    H  CRPS_mix  CRPS_col      dCRPS                 95% CI  cov_m  cov_c   shp_m   shp_c
   low  150    5    0.3315    0.3324   -0.0009* [ -0.0015, -0.0003]   0.91   0.91    1.89    1.89
   low  150   20    3.6120    3.6119    0.0001  [ -0.0013,  0.0016]   0.92   0.91   24.33   24.34
   low  150   40   10.0073   10.0097   -0.0024  [ -0.0061,  0.0011]   0.94   0.94   78.32   78.23
  high  150    5    0.4035    0.4049   -0.0013* [ -0.0025, -0.0001]   0.87   0.87    1.95    1.96
  high  150   20    3.7695    3.7684    0.0012  [ -0.0016,  0.0040]   0.89   0.88   20.71   20.72
  high  150   40    9.2822    9.2819    0.0004  [ -0.0049,  0.0057]   0.90   0.90   59.73   59.72

(* = 95% CI excludes 0.  Negative dCRPS => collapse hurts.)

PRIMARY (high, H=20): cost of collapsing the STATE posterior
  mix - collapse = +0.0012  [-0.0016,+0.0040]

DECOMPOSITION (high, H=20):
  state collapse (mix - collapse)         = +0.0012 [-0.0016,+0.0040]
  coupling loss  (mix_coupled - mix)      = -0.0137 [-0.0386,+0.0101]
  total collapse (mix_coupled - collapse) = -0.0126 [-0.0383,+0.0118]
  additivity: state + coupling = -0.0126  vs  total = -0.0126  (diff 0.0000)

elapsed 6115s

--- VELOCITY CRPS (mix - collapse, same structure as position) ---

stratum    n    H  vCRPS_mix  vCRPS_col     dvCRPS                 95% CI
   low  150    5     0.1241     0.1241   -0.0000  [ -0.0001,  0.0000]
   low  150   20     0.3421     0.3420    0.0000  [ -0.0001,  0.0001]
   low  150   40     0.4207     0.4207   -0.0001  [ -0.0002,  0.0001]
  high  150    5     0.1433     0.1433   -0.0000  [ -0.0002,  0.0001]
  high  150   20     0.3153     0.3153    0.0001  [ -0.0001,  0.0002]
  high  150   40     0.3608     0.3609   -0.0001  [ -0.0002,  0.0001]
(* = 95% CI excludes 0.  Negative dvCRPS => collapse hurts in velocity.)

--- PIT RANK HISTOGRAMS (10 bins, uniform = calibrated) ---
Flat histogram => well-calibrated. U-shape => under-dispersed.
   low h= 5  mix_coupled px: [0.07 0.11 0.07 0.10 0.13 0.11 0.10 0.12 0.09 0.11]
   low h= 5  mix_coupled py: [0.07 0.10 0.11 0.11 0.13 0.07 0.15 0.10 0.10 0.06]
   low h= 5  mix_coupled vx: [0.05 0.05 0.12 0.09 0.13 0.16 0.14 0.09 0.11 0.06]
   low h= 5  mix_coupled vy: [0.08 0.12 0.09 0.07 0.11 0.15 0.18 0.07 0.04 0.08]
   low h= 5     collapse px: [0.07 0.11 0.07 0.12 0.11 0.10 0.11 0.11 0.11 0.10]
   low h= 5     collapse py: [0.07 0.10 0.12 0.08 0.11 0.09 0.17 0.11 0.07 0.06]
   low h= 5     collapse vx: [0.05 0.05 0.13 0.08 0.13 0.17 0.14 0.09 0.11 0.06]
   low h= 5     collapse vy: [0.09 0.11 0.09 0.09 0.10 0.17 0.17 0.07 0.04 0.08]
   low h=20  mix_coupled px: [0.05 0.08 0.11 0.11 0.10 0.11 0.12 0.13 0.09 0.11]
   low h=20  mix_coupled py: [0.09 0.12 0.11 0.08 0.10 0.08 0.16 0.15 0.09 0.03]
   low h=20  mix_coupled vx: [0.03 0.09 0.11 0.11 0.09 0.13 0.10 0.11 0.11 0.12]
   low h=20  mix_coupled vy: [0.06 0.11 0.14 0.11 0.11 0.11 0.10 0.11 0.10 0.05]
   low h=20     collapse px: [0.05 0.08 0.10 0.12 0.10 0.11 0.11 0.13 0.11 0.09]
   low h=20     collapse py: [0.09 0.13 0.11 0.08 0.09 0.10 0.15 0.14 0.08 0.04]
   low h=20     collapse vx: [0.04 0.09 0.11 0.10 0.10 0.13 0.09 0.11 0.12 0.11]
   low h=20     collapse vy: [0.07 0.09 0.14 0.11 0.11 0.13 0.09 0.09 0.11 0.05]
   low h=40  mix_coupled px: [0.03 0.11 0.11 0.12 0.11 0.10 0.09 0.10 0.12 0.11]
   low h=40  mix_coupled py: [0.09 0.11 0.08 0.15 0.15 0.11 0.10 0.09 0.08 0.06]
   low h=40  mix_coupled vx: [0.06 0.08 0.07 0.16 0.12 0.07 0.12 0.11 0.12 0.09]
   low h=40  mix_coupled vy: [0.06 0.10 0.11 0.14 0.15 0.08 0.12 0.07 0.14 0.03]
   low h=40     collapse px: [0.03 0.11 0.11 0.11 0.11 0.10 0.10 0.11 0.11 0.11]
   low h=40     collapse py: [0.09 0.10 0.11 0.13 0.13 0.11 0.11 0.08 0.08 0.06]
   low h=40     collapse vx: [0.06 0.07 0.07 0.16 0.11 0.09 0.12 0.12 0.11 0.09]
   low h=40     collapse vy: [0.06 0.09 0.12 0.15 0.14 0.08 0.11 0.08 0.14 0.03]
  high h= 5  mix_coupled px: [0.09 0.05 0.09 0.15 0.10 0.07 0.11 0.09 0.12 0.12]
  high h= 5  mix_coupled py: [0.13 0.11 0.12 0.09 0.11 0.11 0.08 0.09 0.08 0.07]
  high h= 5  mix_coupled vx: [0.07 0.07 0.11 0.10 0.05 0.11 0.14 0.11 0.13 0.11]
  high h= 5  mix_coupled vy: [0.13 0.10 0.13 0.17 0.13 0.07 0.05 0.06 0.09 0.07]
  high h= 5     collapse px: [0.07 0.08 0.11 0.13 0.09 0.06 0.11 0.10 0.13 0.12]
  high h= 5     collapse py: [0.16 0.11 0.12 0.09 0.08 0.09 0.09 0.06 0.08 0.11]
  high h= 5     collapse vx: [0.07 0.08 0.09 0.11 0.04 0.09 0.17 0.08 0.15 0.11]
  high h= 5     collapse vy: [0.15 0.10 0.15 0.11 0.13 0.09 0.05 0.04 0.07 0.12]
  high h=20  mix_coupled px: [0.07 0.07 0.09 0.08 0.07 0.09 0.14 0.11 0.13 0.14]
  high h=20  mix_coupled py: [0.09 0.11 0.15 0.18 0.13 0.07 0.06 0.07 0.07 0.06]
  high h=20  mix_coupled vx: [0.05 0.11 0.07 0.08 0.08 0.09 0.11 0.10 0.14 0.17]
  high h=20  mix_coupled vy: [0.11 0.10 0.15 0.17 0.15 0.09 0.05 0.06 0.06 0.06]
  high h=20     collapse px: [0.07 0.09 0.09 0.08 0.08 0.08 0.13 0.12 0.12 0.15]
  high h=20     collapse py: [0.11 0.17 0.11 0.15 0.11 0.07 0.06 0.05 0.08 0.08]
  high h=20     collapse vx: [0.05 0.10 0.09 0.06 0.09 0.09 0.11 0.08 0.15 0.18]
  high h=20     collapse vy: [0.11 0.11 0.16 0.17 0.11 0.09 0.07 0.06 0.05 0.07]
  high h=40  mix_coupled px: [0.07 0.02 0.11 0.08 0.11 0.11 0.09 0.13 0.15 0.12]
  high h=40  mix_coupled py: [0.11 0.09 0.16 0.15 0.11 0.10 0.09 0.03 0.10 0.07]
  high h=40  mix_coupled vx: [0.03 0.05 0.07 0.07 0.11 0.10 0.13 0.16 0.15 0.13]
  high h=40  mix_coupled vy: [0.11 0.07 0.13 0.23 0.14 0.09 0.05 0.05 0.08 0.07]
  high h=40     collapse px: [0.06 0.03 0.10 0.10 0.11 0.11 0.09 0.13 0.15 0.12]
  high h=40     collapse py: [0.11 0.11 0.13 0.16 0.11 0.10 0.06 0.05 0.10 0.07]
  high h=40     collapse vx: [0.03 0.06 0.06 0.07 0.11 0.09 0.13 0.15 0.17 0.12]
  high h=40     collapse vy: [0.11 0.07 0.13 0.23 0.13 0.09 0.05 0.04 0.09 0.06]

"""
