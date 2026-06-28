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
            truth={h: x[T + h, :2].copy() for h in HORIZONS}))
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
    dict with CRPS / coverage / sharpness for each arm."""
    post, T, truth = c["post"], c["T"], c["truth"]
    si, sr = seed, seed + 1
    rolls = {}
    for arm in ARMS:
        xs, ms = R.init_particles(post, arm, N_PART, np.random.default_rng(si))
        rolls[arm] = R.oracle_rollout(xs, ms, T, HORIZONS,
                                      np.random.default_rng(sr))
    res = {}
    for h in HORIZONS:
        yh = truth[h]
        res[h] = {arm: dict(
            crps=R.crps_position(rolls[arm][h], yh),
            cov=R.coverage_position(rolls[arm][h], yh, LEVEL),
            shp=R.sharpness_position(rolls[arm][h], LEVEL))
            for arm in ARMS}
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

    # per stratum/horizon: store per-arm crps/cov/shp lists, keyed by arm
    results = {name: {h: {arm: {"crps": [], "cov": [], "shp": []} for arm in ARMS}
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
                    results[name][h][arm]["cov"].append(r[h][arm]["cov"])
                    results[name][h][arm]["shp"].append(r[h][arm]["shp"])

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






