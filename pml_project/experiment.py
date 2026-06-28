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
    """Both arms, common random numbers, all horizons. Returns per-horizon dict
    with CRPS, coverage and sharpness for A and C."""
    post, T, truth = c["post"], c["T"], c["truth"]
    res = {}
    si, sr = seed, seed + 1
    rolls = {}
    for arm in ("A", "C"):
        xs, ms = R.init_particles(post, arm, N_PART, np.random.default_rng(si))
        rolls[arm] = R.oracle_rollout(xs, ms, T, HORIZONS,
                                      np.random.default_rng(sr))
    for h in HORIZONS:
        yh = truth[h]
        res[h] = dict(
            crpsA=R.crps_position(rolls["A"][h], yh),
            crpsC=R.crps_position(rolls["C"][h], yh),
            covA=R.coverage_position(rolls["A"][h], yh, LEVEL),
            covC=R.coverage_position(rolls["C"][h], yh, LEVEL),
            shpA=R.sharpness_position(rolls["A"][h], LEVEL),
            shpC=R.sharpness_position(rolls["C"][h], LEVEL))
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
        used = len(strat[name])
        uniq = len({c["sid"] for c in strat[name]})
        print(f"  {name:>4}: {used} origins ({uniq} unique seqs)")

    results = {name: {h: {"d": [], "crpsA": [], "crpsC": [],
                          "covA": [], "covC": [], "shpA": [], "shpC": [],
                          "sid": []}
                      for h in HORIZONS} for name, _, _ in STRATA}

    ORDER = {name: i for i, (name, _, _) in enumerate(STRATA)}
    for name in results:
        for k, c in enumerate(strat[name]):
            r = score_origin(c, seed=10_000 * ORDER[name] + 13 * k)
            for h in HORIZONS:
                results[name][h]["d"].append(r[h]["crpsA"] - r[h]["crpsC"])
                results[name][h]["covA"].append(r[h]["covA"])
                results[name][h]["covC"].append(r[h]["covC"])
                results[name][h]["shpA"].append(r[h]["shpA"])
                results[name][h]["shpC"].append(r[h]["shpC"])
                results[name][h]["sid"].append(c["sid"])
                results[name][h]["crpsA"].append(r[h]["crpsA"])
                results[name][h]["crpsC"].append(r[h]["crpsC"])

    print(f"\n{'stratum':>6} {'n':>4} {'H':>4} "
          f"{'CRPS_A':>8} {'CRPS_C':>8} {'dCRPS':>10} {'cost%':>7} "
          f"{'95% CI':>22} {'covA':>6} {'covC':>6} {'shpA':>7} {'shpC':>7}")
    for name, lo, hi in STRATA:
        for h in HORIZONS:
            d = results[name][h]["d"]
            if not d:
                continue
            m, ci = paired_bootstrap(d)
            cA = np.mean(results[name][h]["crpsA"])
            cC = np.mean(results[name][h]["crpsC"])
            cost = 100.0 * (cC - cA) / cA if cA else float("nan")
            covA = np.mean(results[name][h]["covA"])
            covC = np.mean(results[name][h]["covC"])
            shpA = np.mean(results[name][h]["shpA"])
            shpC = np.mean(results[name][h]["shpC"])
            sig = "*" if (ci[1] < 0 or ci[0] > 0) else " "
            print(f"{name:>6} {len(d):>4} {h:>4} "
                  f"{cA:>8.4f} {cC:>8.4f} {m:>9.4f}{sig} {cost:>6.1f}% "
                  f"[{ci[0]:>8.4f},{ci[1]:>8.4f}] {covA:>6.2f} {covC:>6.2f} "
                  f"{shpA:>7.2f} {shpC:>7.2f}")
    print(f"\n(* = 95% CI excludes 0.  Negative dCRPS => collapse hurts.)")

    pe = results["high"][20]
    if pe["d"]:
        mn, ci_n = paired_bootstrap(pe["d"])
        mc, ci_c = cluster_bootstrap(pe["d"], pe["sid"])
        n_uniq = len(set(pe["sid"]))
        print(f"\nPRIMARY ENDPOINT (high stratum, H=20):")
        print(f"  mean dCRPS = {mn:.4f}   (n={len(pe['d'])}, {n_uniq} unique seqs)")
        print(f"  paired  95% CI [{ci_n[0]:.4f}, {ci_n[1]:.4f}]")
        print(f"  cluster 95% CI [{ci_c[0]:.4f}, {ci_c[1]:.4f}]")

        # Secondary (ii): paired calibration deviation and sharpness, A-C.
        dev = [abs(a - LEVEL) - abs(c - LEVEL)
               for a, c in zip(pe["covA"], pe["covC"])]
        dshp = [a - c for a, c in zip(pe["shpA"], pe["shpC"])]
        mdev, ci_dev = paired_bootstrap(dev)
        msh, ci_sh = paired_bootstrap(dshp)
        print(f"\nSECONDARY ENDPOINT (high, H=20):")
        print(f"  calib. deviation A-C = {mdev:+.4f}  "
              f"95% CI [{ci_dev[0]:+.4f}, {ci_dev[1]:+.4f}]  "
              f"(~0 => collapse does not hurt calibration)")
        print(f"  sharpness A-C        = {msh:+.4f}  "
              f"95% CI [{ci_sh[0]:+.4f}, {ci_sh[1]:+.4f}]  "
              f"(~0 => collapse does not change interval width)")
    print(f"\nelapsed {time.time()-t0:.1f}s")


