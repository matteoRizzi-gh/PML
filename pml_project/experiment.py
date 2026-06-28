"""
Stratified collapse-cost experiment (oracle propagator).

DESIGN: one origin per sequence, per stratum. From each sequence's candidate
origins we keep the MAX-entropy one for the HIGH pool and the MIN-entropy one
for the LOW pool, then filter each pool to its entropy band. Selection uses
only mu_T (a deterministic function of y_{1:T}): no future-data leak.

CONSEQUENCE FOR INFERENCE: within a stratum every retained origin comes from a
distinct sequence, so the statistical unit IS the sequence. The cluster
bootstrap therefore coincides with the paired bootstrap (one cluster == one
origin); we report the paired CI and note the coincidence. This is the whole
point of the one-origin-per-sequence design: it removes the within-sequence
correlation that the old 6-origins-per-sequence pool had to correct for.

Per origin we compute: entropy H(mu_T), CRPS_A, CRPS_C, the paired difference
dCRPS = CRPS_A - CRPS_C (negative => collapse hurts), and 90% interval coverage
for both arms, at each horizon.
"""
import numpy as np
import slds
import rollout as R

SIGMA_R = 0.05
SEQ_LEN = 200
ORIGINS = [60, 80, 100, 120, 140, 160]     # all satisfy T + 40 <= SEQ_LEN
HORIZONS = [5, 20, 40]
N_PART = 2000
N_SEQ = 2000                                  # one origin/seq/stratum -> need many
TARGET_PER_STRATUM = 100
LEVEL = 0.90
LOG3 = np.log(3)
# Two-stratum design: the high-vs-low contrast carries the claim. Each band is
# the selection target for its pool (max-entropy -> high, min-entropy -> low).
STRATA = [("low", 0.0, 0.35), ("high", 0.75, LOG3 + 1e-9)]
SELECT = {"low": min, "high": max}           # which origin to keep per sequence


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
    """Filter each pool to its entropy band, shuffle, cap at TARGET_PER_STRATUM.
    The band filter matters: a sequence's max-entropy origin can still be < 0.75
    (whole sequence quiet) and must NOT enter the high stratum; likewise a
    min-entropy origin can exceed 0.35."""
    out = {}
    for name, lo, hi in STRATA:
        pool = [c for c in cands if lo <= c["H"] < hi]
        rng.shuffle(pool)
        out[name] = pool[:TARGET_PER_STRATUM]
    return out


def score_origin(c, seed):
    """Both arms, common random numbers, all horizons. Returns per-horizon dict
    with crps_A, crps_C and coverage indicators."""
    post, T, truth = c["post"], c["T"], c["truth"]
    res = {}
    # CRN: identical seeds for init and rollout across the two arms
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
            covC=R.coverage_position(rolls["C"][h], yh, LEVEL))
    return res


def paired_bootstrap(deltas, B=4000, seed=0):
    """Percentile bootstrap over origins. With one origin per sequence per
    stratum, origins within a stratum are independent, so this IS the honest CI
    (the cluster bootstrap below would return the same thing)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(1)
    return d.mean(), np.quantile(means, [0.025, 0.975])


def cluster_bootstrap(deltas, sids, B=4000, seed=0):
    """Resample whole sequences. Kept for the explicit check that, under the
    one-origin-per-sequence design, clusters are singletons and this coincides
    with paired_bootstrap. If sids within a stratum are all unique, the two CIs
    match up to bootstrap noise."""
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
                          "covA": [], "covC": [], "sid": []}
                      for h in HORIZONS} for name, _, _ in STRATA}

    ORDER = {name: i for i, (name, _, _) in enumerate(STRATA)}
    for name in results:
        for k, c in enumerate(strat[name]):
            r = score_origin(c, seed=10_000 * ORDER[name] + 13 * k)
            for h in HORIZONS:
                results[name][h]["d"].append(r[h]["crpsA"] - r[h]["crpsC"])
                results[name][h]["covA"].append(r[h]["covA"])
                results[name][h]["covC"].append(r[h]["covC"])
                results[name][h]["sid"].append(c["sid"])
                results[name][h]["crpsA"].append(r[h]["crpsA"])
                results[name][h]["crpsC"].append(r[h]["crpsC"])

    print(f"\n{'stratum':>6} {'n':>4} {'H':>4} "
          f"{'CRPS_A':>8} {'CRPS_C':>8} {'dCRPS':>10} {'cost%':>7} "
          f"{'95% CI':>22} {'covA':>6} {'covC':>6}")
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
            sig = "*" if (ci[1] < 0 or ci[0] > 0) else " "
            print(f"{name:>6} {len(d):>4} {h:>4} "
                  f"{cA:>8.4f} {cC:>8.4f} {m:>9.4f}{sig} {cost:>6.1f}% "
                  f"[{ci[0]:>8.4f},{ci[1]:>8.4f}] {covA:>6.2f} {covC:>6.2f}")
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
    print(f"\nelapsed {time.time()-t0:.1f}s")


