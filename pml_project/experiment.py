"""
For a stratified set of forecast origins. 
For each of them we use IMM and save, for each origin, posterior and new horizon.

We evaluate the required metrics: 
    - entropy
    - stratified origins divided by entropy
    - CRPS and coverage
    - mean (A-C) with respect to stratification, and with respect to horizon

Rk. we used double bootstrap because paired_bootstrap re-samples origins 
as if they were independent; cluster_bootstrap re-samples entire sequences.
We require both since we use 6 origins for sequence (ORIGINS x N_SEQ), 
so origins are not independent.
"""
import numpy as np
import slds
import rollout as R

SIGMA_R = 0.05
SEQ_LEN = 200
ORIGINS = [60, 80, 100, 120, 140, 160]     # all satisfy T + 40 <= SEQ_LEN
HORIZONS = [5, 20, 40]
N_PART = 2000
N_SEQ = 150
TARGET_PER_STRATUM = 100
LEVEL = 0.90
LOG3 = np.log(3)
STRATA = [("low", 0.0, 0.35), ("mid", 0.35, 0.75), ("high", 0.75, LOG3 + 1e-9)]


def entropy(mu):
    mu = np.clip(mu, 1e-12, 1.0)
    return float(-(mu * np.log(mu)).sum())


def collect_origins(rng):
    """Single IMM pass per sequence; snapshot posteriors + truth at origins.
    Each candidate carries its sequence id (sid) for the cluster bootstrap."""
    cands = []   # each: dict(H, post, truth{h:pos}, sid)
    for sid in range(N_SEQ):
        modes, x = slds.simulate(SEQ_LEN, rng)
        y = slds.observe(x, SIGMA_R, rng)
        posts = slds.run_imm_multi(y, SIGMA_R, ORIGINS)
        for T in ORIGINS:
            post = posts[T]
            cands.append(dict(
                H=entropy(post[0]), post=post, T=T, sid=sid,
                truth={h: x[T + h, :2].copy() for h in HORIZONS}))
    return cands


def stratify(cands, rng):
    out = {}
    for name, lo, hi in STRATA:
        pool = [c for c in cands if lo <= c["H"] < hi]
        rng.shuffle(pool)
        out[name] = pool[:TARGET_PER_STRATUM]
    return out


def score_origin(c, seed):
    """Both arms, CRN, all horizons. Returns per-horizon dict with crps_A,
    crps_C and coverage indicators."""
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
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(1)
    return d.mean(), np.quantile(means, [0.025, 0.975])


def cluster_bootstrap(deltas, sids, B=4000, seed=0):
    """Resample whole sequences (clusters) to respect within-sequence
    correlation from taking multiple origins per sequence."""
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
    Hs = np.array([c["H"] for c in cands])
    print(f"pool: {len(cands)} origins  (sigma_r={SIGMA_R}, N_part={N_PART})")
    print(f"entropy: mean {Hs.mean():.3f}  p90 {np.quantile(Hs,0.9):.3f}  "
          f"max {Hs.max():.3f}  (log3={LOG3:.3f})")

    strat = stratify(cands, rng)
    results = {name: {h: {"d": [], "crpsA": [], "crpsC": [],
                          "covA": [], "covC": [], "sid": []}
                      for h in HORIZONS} for name, _, _ in STRATA}
  
    ORDER = {"low": 0, "mid": 1, "high": 2}
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
          f"{'naive 95% CI':>22} {'covA':>6} {'covC':>6}")
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
    print(f"\n(* = naive 95% CI excludes 0.  Negative dCRPS => collapse hurts.)")

    # Primary endpoint with the CORRECT (clustered) CI: high stratum, H=20.
    pe = results["high"][20]
    if pe["d"]:
        mn, ci_n = paired_bootstrap(pe["d"])
        mc, ci_c = cluster_bootstrap(pe["d"], pe["sid"])
        print(f"\nPRIMARY ENDPOINT (high stratum, H=20):")
        print(f"  mean dCRPS = {mc:.4f}")
        print(f"  naive   95% CI [{ci_n[0]:.4f}, {ci_n[1]:.4f}]")
        print(f"  cluster 95% CI [{ci_c[0]:.4f}, {ci_c[1]:.4f}]  "
              f"(resamples whole sequences; the honest one)")
    print(f"\nelapsed {time.time()-t0:.1f}s")
