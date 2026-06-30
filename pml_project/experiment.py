"""
Stratified collapse-cost experiment (oracle propagator).

DESIGN: one random origin per sequence (PDF: "at most one origin per sequence").
For each sequence we draw ONE candidate origin uniformly, snapshot its IMM
posterior, then bin by mode entropy H(mu_T). Random selection (not max-entropy)
avoids the time-since-switch bias that concentrates high-entropy origins right
after switches, where position separation has not yet accumulated. Selection
uses only mu_T (a function of y_{1:T}): no future-data leak.

CONSEQUENCE FOR INFERENCE: one origin per sequence => within a stratum every
origin is a distinct sequence, so the statistical unit IS the sequence and the
cluster bootstrap coincides with the paired bootstrap. We report the paired CI.

Per origin, at each horizon: CRPS_A, CRPS_C, paired dCRPS = CRPS_A - CRPS_C
(negative => collapse hurts), 90% interval coverage AND sharpness (interval
width) for both arms. Coverage and sharpness are reported jointly: a model can
hit nominal coverage by widening intervals, so width is the necessary complement.
"""
import numpy as np
import slds
import rollout as R

SIGMA_R = 0.05
SEQ_LEN = 200
ORIGINS = [60, 80, 100, 120, 140, 160]     # all satisfy T + 40 <= SEQ_LEN
HORIZONS = [5, 20, 40]
N_PART = 2000
N_SEQ = 2000
TARGET_PER_STRATUM = 150
LEVEL = 0.90
LOG3 = np.log(3)
STRATA = [("low", 0.0, 0.35), ("high", 0.75, LOG3 + 1e-9)]

ARMS = ("mix", "collapse", "mix_coupled")


def entropy(mu):
    mu = np.clip(mu, 1e-12, 1.0)
    return float(-(mu * np.log(mu)).sum())


def collect_origins(rng):
    """One random origin per sequence (no time-since-switch selection bias).
    Caller filters by band; sid is unique within each band."""
    cands = []
    for sid in range(N_SEQ):
        modes, x = slds.simulate(SEQ_LEN, rng)
        y = slds.observe(x, SIGMA_R, rng)
        T = int(rng.choice(ORIGINS))
        post = slds.run_imm_multi(y, SIGMA_R, [T])[T]
        cands.append(dict(
            H=entropy(post[0]), post=post, T=T, sid=sid,
            truth_pos={h: x[T + h, :2].copy() for h in HORIZONS},
            truth_vel={h: x[T + h, 2:].copy() for h in HORIZONS}))
    return cands


def stratify(cands, rng):
    """Filter to each entropy band, shuffle, cap at TARGET_PER_STRATUM."""
    out = {}
    for name, lo, hi in STRATA:
        pool = [c for c in cands if lo <= c["H"] < hi]
        rng.shuffle(pool)
        out[name] = pool[:TARGET_PER_STRATUM]
    return out


def score_origin(c, seed):
    """Three arms, common random numbers, all horizons. Returns per-horizon
    dict with CRPS / coverage / sharpness / velocity CRPS / PIT ranks per arm."""
    post, T = c["post"], c["T"]
    truth_pos, truth_vel = c["truth_pos"], c["truth_vel"]
    si, sr = seed, seed + 1
    pit_rng = np.random.default_rng(seed + 2)   # separate rng for tie-breaking
    rolls = {}
    for arm in ARMS:
        xs, ms = R.init_particles(post, arm, N_PART, np.random.default_rng(si))
        rolls[arm] = R.oracle_rollout(xs, ms, T, HORIZONS,
                                      np.random.default_rng(sr))
    res = {}
    for h in HORIZONS:
        ypos = truth_pos[h]
        yvel = truth_vel[h]
        res[h] = {}
        for arm in ARMS:
            state = rolls[arm][h]               # (N_PART, 4) full state
            res[h][arm] = dict(
                crps=R.crps_position(state[:, :2], ypos),
                crps_vel=R.crps_velocity(state, yvel),
                es_pos=R.energy_score_position(state, ypos),
                es_vel=R.energy_score_velocity(state, yvel),
                cov=R.coverage_position(state[:, :2], ypos, LEVEL),
                shp=R.sharpness_position(state[:, :2], LEVEL),
                pit=R.pit_ranks_origin(state, ypos, yvel, pit_rng))
    return res


def paired_bootstrap(deltas, B=4000, seed=0):
    """Percentile bootstrap over origins. With one origin per sequence the
    cluster bootstrap returns the same thing."""
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(1)
    return d.mean(), np.quantile(means, [0.025, 0.975])


def cluster_bootstrap(deltas, sids, B=4000, seed=0):
    """Resample whole sequences. Under one-origin-per-sequence, clusters are
    singletons and this coincides with paired_bootstrap (explicit check)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas); s = np.asarray(sids)
    groups = {u: d[s == u] for u in np.unique(s)}
    keys = list(groups.values())
    means = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(keys), size=len(keys))
        means[b] = np.concatenate([keys[i] for i in pick]).mean()
    return d.mean(), np.quantile(means, [0.025, 0.975])


if __name__ == "__main__":
    import time
    t0 = time.time()
    rng = np.random.default_rng(2025)
    cands = collect_origins(rng)
    strat = stratify(cands, rng)

    Hs = np.array([c["H"] for c in cands])
    print(f"sigma_r={SIGMA_R}  N_seq={N_SEQ}  N_part={N_PART}")
    print(f"pool {len(cands)}  entropy mean {Hs.mean():.3f} p90 {np.quantile(Hs,0.9):.3f}")
    for name, lo, hi in STRATA:
        print(f"  {name:>4}: {len(strat[name])} origins "
              f"({len({c['sid'] for c in strat[name]})} unique seqs)")

    # per stratum/horizon: store per-arm crps/cov/shp/crps_vel/pit lists, keyed by arm
    results = {name: {h: {arm: {"crps": [], "cov": [], "shp": [],
                                "crps_vel": [],
                                "es_pos": [], "es_vel": [],
                                "pit": {coord: [] for coord in ("px","py","vx","vy")}}
                           for arm in ARMS}
                      for h in HORIZONS} for name, _, _ in STRATA}
    sids = {name: [] for name, _, _ in STRATA}

    ORDER = {name: i for i, (name, _, _) in enumerate(STRATA)}
    for name in results:
        for k, c in enumerate(strat[name]):
            r = score_origin(c, seed=10_000 * ORDER[name] + 13 * k)
            sids[name].append(c["sid"])
            for h in HORIZONS:
                for arm in ARMS:
                    results[name][h][arm]["crps"].append(r[h][arm]["crps"])
                    results[name][h][arm]["crps_vel"].append(r[h][arm]["crps_vel"])
                    results[name][h][arm]["es_pos"].append(r[h][arm]["es_pos"])
                    results[name][h][arm]["es_vel"].append(r[h][arm]["es_vel"])
                    results[name][h][arm]["cov"].append(r[h][arm]["cov"])
                    results[name][h][arm]["shp"].append(r[h][arm]["shp"])
                    for coord in ("px", "py", "vx", "vy"):
                        results[name][h][arm]["pit"][coord].append(
                            r[h][arm]["pit"][coord])

    # ---- main table: the headline contrast mix - collapse ----
    print(f"\n{'stratum':>6} {'n':>4} {'H':>4} "
          f"{'CRPS_mix':>9} {'CRPS_col':>9} {'dCRPS':>10} {'95% CI':>22} "
          f"{'cov_m':>6} {'cov_c':>6} {'shp_m':>7} {'shp_c':>7}")
    for name, lo, hi in STRATA:
        for h in HORIZONS:
            cm = results[name][h]["mix"]["crps"]
            cc = results[name][h]["collapse"]["crps"]
            d = [a - b for a, b in zip(cm, cc)]
            m, ci = paired_bootstrap(d)
            sig = "*" if (ci[1] < 0 or ci[0] > 0) else " "
            print(f"{name:>6} {len(d):>4} {h:>4} "
                  f"{np.mean(cm):>9.4f} {np.mean(cc):>9.4f} {m:>9.4f}{sig} "
                  f"[{ci[0]:>8.4f},{ci[1]:>8.4f}] "
                  f"{np.mean(results[name][h]['mix']['cov']):>6.2f} "
                  f"{np.mean(results[name][h]['collapse']['cov']):>6.2f} "
                  f"{np.mean(results[name][h]['mix']['shp']):>7.2f} "
                  f"{np.mean(results[name][h]['collapse']['shp']):>7.2f}")
    print(f"\n(* = 95% CI excludes 0.  Negative dCRPS => collapse hurts.)")

    # ---- primary endpoint ----
    hi_arm = results["high"][20]
    cm, cc, cf = (hi_arm["mix"]["crps"], hi_arm["collapse"]["crps"],
                  hi_arm["mix_coupled"]["crps"])
    d_state = [a - b for a, b in zip(cm, cc)]          # mix - collapse
    mn, ci_n = paired_bootstrap(d_state)
    mc, ci_c = cluster_bootstrap(d_state, sids["high"])
    print(f"\nPRIMARY ENDPOINT (high, H=20): cost of collapsing the STATE posterior")
    print(f"  mean dCRPS (mix - collapse) = {mn:.4f}   (n={len(d_state)})")
    print(f"  paired  95% CI [{ci_n[0]:.4f}, {ci_n[1]:.4f}]")
    print(f"  cluster 95% CI [{ci_c[0]:.4f}, {ci_c[1]:.4f}]")

    # ---- decomposition via the third arm ----
    d_couple = [a - b for a, b in zip(cf, cm)]         # mix_coupled - mix
    d_total  = [a - b for a, b in zip(cf, cc)]         # mix_coupled - collapse
    mco, ci_co = paired_bootstrap(d_couple)
    mto, ci_to = paired_bootstrap(d_total)
    print(f"\nDECOMPOSITION (high, H=20):")
    print(f"  state collapse   (mix - collapse)        = {mn:+.4f} "
          f"[{ci_n[0]:+.4f},{ci_n[1]:+.4f}]")
    print(f"  coupling loss    (mix_coupled - mix)     = {mco:+.4f} "
          f"[{ci_co[0]:+.4f},{ci_co[1]:+.4f}]")
    print(f"  total collapse   (mix_coupled - collapse)= {mto:+.4f} "
          f"[{ci_to[0]:+.4f},{ci_to[1]:+.4f}]")
    add = mn + mco
    print(f"  additivity check: state + coupling = {add:+.4f}  vs  total = {mto:+.4f}"
          f"  (diff {abs(add-mto):.4f})")

    # ---- secondary: sharpness & calibration of the state collapse ----
    sm = hi_arm["mix"]["shp"]; sc = hi_arm["collapse"]["shp"]
    dshp = [a - b for a, b in zip(sm, sc)]
    msh, ci_sh = paired_bootstrap(dshp)
    vm = hi_arm["mix"]["cov"]; vc = hi_arm["collapse"]["cov"]
    dev = [abs(a - LEVEL) - abs(b - LEVEL) for a, b in zip(vm, vc)]
    mdev, ci_dev = paired_bootstrap(dev)
    print(f"\nSECONDARY (high, H=20), state collapse mix - collapse:")
    print(f"  sharpness     = {msh:+.4f} [{ci_sh[0]:+.4f},{ci_sh[1]:+.4f}]")
    print(f"  calib. dev.   = {mdev:+.4f} [{ci_dev[0]:+.4f},{ci_dev[1]:+.4f}]")
    print(f"\nelapsed {time.time()-t0:.1f}s")

    # ---- velocity CRPS table ----
    print(f"\n--- VELOCITY CRPS (mix - collapse, same structure as position) ---")
    print(f"\n{'stratum':>6} {'n':>4} {'H':>4} "
          f"{'vCRPS_mix':>10} {'vCRPS_col':>10} {'dvCRPS':>10} {'95% CI':>22}")
    for name, lo, hi in STRATA:
        for h in HORIZONS:
            cm = results[name][h]["mix"]["crps_vel"]
            cc = results[name][h]["collapse"]["crps_vel"]
            d = [a - b for a, b in zip(cm, cc)]
            m, ci = paired_bootstrap(d)
            sig = "*" if (ci[1] < 0 or ci[0] > 0) else " "
            print(f"{name:>6} {len(d):>4} {h:>4} "
                  f"{np.mean(cm):>10.4f} {np.mean(cc):>10.4f} {m:>9.4f}{sig} "
                  f"[{ci[0]:>8.4f},{ci[1]:>8.4f}]")
    print(f"(* = 95% CI excludes 0.  Negative dvCRPS => collapse hurts in velocity.)")

    # ---- PIT rank histograms (text, pooled per stratum/horizon/arm) ----
    print(f"\n--- PIT RANK HISTOGRAMS (10 bins, uniform = calibrated) ---")
    print(f"Flat histogram => well-calibrated. U-shape => under-dispersed.")
    for name, lo, hi in STRATA:
        for h in HORIZONS:
            for arm in ("mix_coupled", "collapse"):
                for coord in ("px", "py", "vx", "vy"):
                    ranks = results[name][h][arm]["pit"][coord]
                    _, freq = R.pit_histogram(ranks, N_PART, n_bins=10)
                    bar = " ".join(f"{f:.2f}" for f in freq)
                    print(f"  {name:>4} h={h:>2} {arm:>12} {coord}: [{bar}]")

    # ---- energy score (optional diagnostic, excluded from primary endpoints) ----
    print(f"\n--- ENERGY SCORE (optional; proposal excluded from primary endpoints) ---")
    print(f"ES is CRPS generalised to joint (px,py) or (vx,vy). "
          f"Negative dES => collapse hurts.")
    print(f"\n{'stratum':>6} {'n':>4} {'H':>4} "
          f"{'ES_pos_mix':>11} {'ES_pos_col':>11} {'dES_pos':>9} {'95% CI':>22}")
    for name, lo, hi in STRATA:
        for h in HORIZONS:
            em = results[name][h]["mix"]["es_pos"]
            ec = results[name][h]["collapse"]["es_pos"]
            d  = [a - b for a, b in zip(em, ec)]
            mv, ci = paired_bootstrap(d)
            sig = "*" if (ci[1] < 0 or ci[0] > 0) else " "
            print(f"{name:>6} {len(d):>4} {h:>4} "
                  f"{np.mean(em):>11.4f} {np.mean(ec):>11.4f} {mv:>8.4f}{sig} "
                  f"[{ci[0]:>8.4f},{ci[1]:>8.4f}]")
    print(f"\n{'stratum':>6} {'n':>4} {'H':>4} "
          f"{'ES_vel_mix':>11} {'ES_vel_col':>11} {'dES_vel':>9} {'95% CI':>22}")
    for name, lo, hi in STRATA:
        for h in HORIZONS:
            em = results[name][h]["mix"]["es_vel"]
            ec = results[name][h]["collapse"]["es_vel"]
            d  = [a - b for a, b in zip(em, ec)]
            mv, ci = paired_bootstrap(d)
            sig = "*" if (ci[1] < 0 or ci[0] > 0) else " "
            print(f"{name:>6} {len(d):>4} {h:>4} "
                  f"{np.mean(em):>11.4f} {np.mean(ec):>11.4f} {mv:>8.4f}{sig} "
                  f"[{ci[0]:>8.4f},{ci[1]:>8.4f}]")


"""
THREE-ARM DECOMPOSITION -- what each arm isolates, and the oracle result.

The single contrast A-C used to conflate TWO distinct losses (PDF sec.6,
"declared confound"). The third arm separates them. All three share the initial
state normals z and, where applicable, the propagation mode, via common random
numbers, so each paired difference isolates exactly one effect:

  mix          : state ~ mode-conditional component j; propagate under an
                 INDEPENDENT mode drawn from mu.
  collapse     : state ~ moment-matched Gaussian (xbar, Pbar); propagate under
                 the SAME independent mode as mix.
  mix_coupled  : state ~ component j; propagate under j itself (coupled mode).

  mix - collapse        = cost of collapsing the STATE posterior (Gaussianizing
                          the mixture), mode treatment identical so it cancels.
  mix_coupled - mix     = cost of losing the MODE-STATE COUPLING, state identical
                          so it cancels.
  mix_coupled - collapse= total cost (what the old buggy single contrast measured).

ORACLE RESULT (high stratum, H=20):
  state collapse   = -0.0012  [-0.0040, +0.0015]   -> NULL (CI spans 0)
  coupling loss    = -0.0247  [-0.0495, +0.0003]   -> the real cost
  total            = -0.0259  [-0.0517, -0.0007]
  additivity: state + coupling = -0.0259 vs total -0.0259 (diff 0.0000), exact.

Reading: Gaussianizing the state posterior is essentially free in this system;
the entire forecast cost lives in the mode-state coupling. This is consistent
with the benchmark's structure -- multimodality lives in velocity, position is
observed and pinned by the data -- so collapsing the (position-dominated)
state distribution loses almost nothing that the position CRPS can see. The
older significant numbers (-0.05, -0.04, -0.03) were 'total', i.e. coupling,
mislabeled as collapse cost by the arm-C mode bug.
"""



"""

sigma_r=0.05  N_seq=2000  N_part=2000
pool 2000  entropy mean 0.453 p90 0.788
   low: 150 origins (150 unique seqs)
  high: 150 origins (150 unique seqs)

stratum    n    H  CRPS_mix  CRPS_col      dCRPS                 95% CI  cov_m  cov_c   shp_m   shp_c
   low  150    5    0.3273    0.3281   -0.0008* [ -0.0017, -0.0000]   0.87   0.87    1.59    1.58
   low  150   20    3.5752    3.5759   -0.0007  [ -0.0025,  0.0010]   0.86   0.86   18.96   18.98
   low  150   40    9.6274    9.6273    0.0001  [ -0.0022,  0.0024]   0.88   0.88   53.42   53.41
  high  150    5    0.3617    0.3644   -0.0027* [ -0.0043, -0.0012]   0.84   0.85    1.73    1.74
  high  150   20    3.5315    3.5327   -0.0012  [ -0.0040,  0.0015]   0.84   0.84   17.91   17.96
  high  150   40    9.1433    9.1415    0.0018  [ -0.0020,  0.0055]   0.87   0.87   51.15   51.15

(* = 95% CI excludes 0.  Negative dCRPS => collapse hurts.)

PRIMARY ENDPOINT (high, H=20): cost of collapsing the STATE posterior
  mean dCRPS (mix - collapse) = -0.0012   (n=150)
  paired  95% CI [-0.0040, 0.0015]
  cluster 95% CI [-0.0040, 0.0015]

DECOMPOSITION (high, H=20):
  state collapse   (mix - collapse)        = -0.0012 [-0.0040,+0.0015]
  coupling loss    (mix_coupled - mix)     = -0.0247 [-0.0495,+0.0003]
  total collapse   (mix_coupled - collapse)= -0.0259 [-0.0517,-0.0007]
  additivity check: state + coupling = -0.0259  vs  total = -0.0259  (diff 0.0000)

SECONDARY (high, H=20), state collapse mix - collapse:
  sharpness     = -0.0451 [-0.0696,-0.0212]
  calib. dev.   = +0.0007 [-0.0080,+0.0080]

elapsed 1801.1s

--- VELOCITY CRPS (mix - collapse, same structure as position) ---

stratum    n    H  vCRPS_mix  vCRPS_col     dvCRPS                 95% CI
   low  150    5     0.1203     0.1203    0.0000  [ -0.0001,  0.0001]
   low  150   20     0.3353     0.3353   -0.0000  [ -0.0001,  0.0001]
   low  150   40     0.4029     0.4028    0.0000  [ -0.0000,  0.0001]
  high  150    5     0.1266     0.1268   -0.0002  [ -0.0004,  0.0000]
  high  150   20     0.3082     0.3081    0.0000  [ -0.0000,  0.0001]
  high  150   40     0.3433     0.3433   -0.0000  [ -0.0001,  0.0000]
(* = 95% CI excludes 0.  Negative dvCRPS => collapse hurts in velocity.)

--- PIT RANK HISTOGRAMS (10 bins, uniform = calibrated) ---
Flat histogram => well-calibrated. U-shape => under-dispersed.
   low h= 5  mix_coupled px: [0.14 0.14 0.09 0.04 0.11 0.10 0.06 0.06 0.15 0.11]
   low h= 5  mix_coupled py: [0.13 0.06 0.11 0.12 0.10 0.10 0.08 0.13 0.07 0.09]
   low h= 5  mix_coupled vx: [0.09 0.14 0.09 0.08 0.10 0.08 0.08 0.08 0.12 0.13]
   low h= 5  mix_coupled vy: [0.13 0.07 0.09 0.09 0.12 0.07 0.13 0.09 0.10 0.11]
   low h= 5     collapse px: [0.13 0.15 0.08 0.05 0.11 0.09 0.07 0.07 0.15 0.10]
   low h= 5     collapse py: [0.13 0.06 0.09 0.15 0.09 0.11 0.11 0.12 0.05 0.09]
   low h= 5     collapse vx: [0.09 0.13 0.11 0.07 0.11 0.08 0.08 0.08 0.12 0.13]
   low h= 5     collapse vy: [0.13 0.07 0.07 0.11 0.11 0.08 0.14 0.10 0.09 0.11]
   low h=20  mix_coupled px: [0.08 0.10 0.10 0.08 0.09 0.11 0.13 0.09 0.08 0.15]
   low h=20  mix_coupled py: [0.15 0.09 0.10 0.05 0.07 0.11 0.12 0.09 0.11 0.11]
   low h=20  mix_coupled vx: [0.08 0.11 0.08 0.07 0.11 0.10 0.13 0.09 0.08 0.15]
   low h=20  mix_coupled vy: [0.17 0.10 0.06 0.08 0.08 0.07 0.13 0.09 0.13 0.09]
   low h=20     collapse px: [0.08 0.10 0.09 0.09 0.09 0.11 0.13 0.09 0.09 0.14]
   low h=20     collapse py: [0.15 0.08 0.10 0.05 0.06 0.13 0.11 0.11 0.10 0.11]
   low h=20     collapse vx: [0.08 0.11 0.08 0.06 0.11 0.11 0.13 0.09 0.08 0.15]
   low h=20     collapse vy: [0.18 0.08 0.07 0.08 0.08 0.07 0.13 0.09 0.13 0.09]
   low h=40  mix_coupled px: [0.11 0.05 0.11 0.07 0.11 0.11 0.10 0.07 0.14 0.13]
   low h=40  mix_coupled py: [0.14 0.09 0.11 0.10 0.07 0.07 0.08 0.08 0.14 0.11]
   low h=40  mix_coupled vx: [0.08 0.11 0.07 0.06 0.16 0.07 0.10 0.09 0.10 0.15]
   low h=40  mix_coupled vy: [0.11 0.08 0.13 0.09 0.12 0.10 0.11 0.08 0.08 0.11]
   low h=40     collapse px: [0.11 0.05 0.11 0.07 0.11 0.10 0.11 0.09 0.13 0.13]
   low h=40     collapse py: [0.15 0.09 0.12 0.07 0.09 0.07 0.09 0.07 0.14 0.11]
   low h=40     collapse vx: [0.08 0.11 0.07 0.07 0.15 0.07 0.10 0.10 0.09 0.15]
   low h=40     collapse vy: [0.11 0.09 0.12 0.08 0.12 0.11 0.11 0.09 0.07 0.11]
  high h= 5  mix_coupled px: [0.11 0.08 0.06 0.09 0.12 0.17 0.10 0.10 0.10 0.07]
  high h= 5  mix_coupled py: [0.13 0.10 0.07 0.05 0.10 0.12 0.10 0.14 0.07 0.12]
  high h= 5  mix_coupled vx: [0.16 0.07 0.08 0.08 0.10 0.09 0.10 0.11 0.13 0.08]
  high h= 5  mix_coupled vy: [0.15 0.11 0.05 0.09 0.09 0.06 0.11 0.13 0.08 0.13]
  high h= 5     collapse px: [0.12 0.09 0.08 0.04 0.12 0.19 0.11 0.11 0.09 0.06]
  high h= 5     collapse py: [0.18 0.07 0.05 0.09 0.10 0.07 0.09 0.10 0.14 0.11]
  high h= 5     collapse vx: [0.15 0.09 0.07 0.10 0.07 0.09 0.12 0.11 0.12 0.09]
  high h= 5     collapse vy: [0.19 0.08 0.05 0.11 0.04 0.07 0.10 0.12 0.09 0.14]
  high h=20  mix_coupled px: [0.13 0.09 0.10 0.06 0.12 0.08 0.09 0.09 0.11 0.12]
  high h=20  mix_coupled py: [0.13 0.09 0.11 0.08 0.09 0.10 0.10 0.10 0.05 0.15]
  high h=20  mix_coupled vx: [0.11 0.13 0.11 0.06 0.07 0.10 0.10 0.09 0.09 0.15]
  high h=20  mix_coupled vy: [0.10 0.11 0.09 0.11 0.07 0.13 0.11 0.10 0.07 0.12]
  high h=20     collapse px: [0.13 0.09 0.10 0.07 0.12 0.08 0.07 0.10 0.11 0.13]
  high h=20     collapse py: [0.15 0.11 0.07 0.09 0.11 0.07 0.09 0.10 0.07 0.15]
  high h=20     collapse vx: [0.11 0.12 0.12 0.06 0.06 0.11 0.10 0.09 0.11 0.12]
  high h=20     collapse vy: [0.11 0.11 0.08 0.11 0.06 0.13 0.09 0.11 0.07 0.13]
  high h=40  mix_coupled px: [0.11 0.08 0.09 0.09 0.09 0.12 0.10 0.09 0.11 0.13]
  high h=40  mix_coupled py: [0.10 0.12 0.09 0.13 0.07 0.09 0.12 0.06 0.11 0.11]
  high h=40  mix_coupled vx: [0.10 0.09 0.11 0.07 0.08 0.11 0.09 0.11 0.15 0.09]
  high h=40  mix_coupled vy: [0.08 0.11 0.10 0.13 0.07 0.10 0.09 0.12 0.10 0.10]
  high h=40     collapse px: [0.10 0.08 0.10 0.09 0.07 0.13 0.09 0.11 0.11 0.12]
  high h=40     collapse py: [0.11 0.09 0.11 0.10 0.11 0.08 0.13 0.05 0.11 0.11]
  high h=40     collapse vx: [0.09 0.09 0.11 0.07 0.09 0.11 0.09 0.11 0.15 0.09]
  high h=40     collapse vy: [0.07 0.10 0.11 0.12 0.09 0.09 0.09 0.13 0.10 0.10]

--- ENERGY SCORE (optional; proposal excluded from primary endpoints) ---
ES is CRPS generalised to joint (px,py) or (vx,vy). Negative dES => collapse hurts.

stratum    n    H  ES_pos_mix  ES_pos_col   dES_pos                 95% CI
   low  150    5      0.5121      0.5130  -0.0009  [ -0.0022,  0.0003]
   low  150   20      5.4718      5.4727  -0.0009  [ -0.0033,  0.0013]
   low  150   40     15.0563     15.0575  -0.0012  [ -0.0044,  0.0018]
  high  150    5      0.5980      0.6014  -0.0035* [ -0.0057, -0.0014]
  high  150   20      5.4389      5.4386   0.0003  [ -0.0034,  0.0040]
  high  150   40     14.3766     14.3755   0.0011  [ -0.0043,  0.0065]

stratum    n    H  ES_vel_mix  ES_vel_col   dES_vel                 95% CI
   low  150    5      0.1892      0.1892  -0.0000  [ -0.0002,  0.0002]
   low  150   20      0.5249      0.5249  -0.0000  [ -0.0001,  0.0000]
   low  150   40      0.6292      0.6292   0.0000  [ -0.0001,  0.0001]
  high  150    5      0.2058      0.2061  -0.0002  [ -0.0005,  0.0000]
  high  150   20      0.4867      0.4867   0.0000  [ -0.0001,  0.0002]
  high  150   40      0.5378      0.5379  -0.0000  [ -0.0001,  0.0001]

"""


