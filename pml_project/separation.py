"""
Separation / collapse-cost diagnostic at the calibrated difficulty.

Question this answers: at the calibrated sigma_r (at the forecast origin (h=0)), 
is there actually multimodality to lose when we collapse, 
and does the collapse cost track posterior mode entropy H(mu_T), 
or is it flat / non-monotone?

Two scalars per forecast origin (position components only):
  sep_ratio : max_{i<j} ||xhat_i - xhat_j|| / avg_component_pos_std
              (weight-agnostic geometric separation of the mode means.
              ignores mu, so it is large even if a mode has tiny weight).
  kl_collapse : KL( mixture || moment-matched Gaussian ), MC estimate
              (weight-aware; the actual information lost at h=0. Always >= 0,
               and = 0 iff the mixture is already a single Gaussian)
"""
import numpy as np
import slds

SIGMA_R = 0.05            # calibrated value
SEQ_LEN = 200
ORIGIN_LO, ORIGIN_HI = 60, 160
N_SEQ = 300
ORIGINS_PER_SEQ = 3       # diagnostic only (final test uses 1/seq)
N_MC = 2000               # MC samples for KL
LOG3 = np.log(3)

# entropy bins from the proposal (+ the cut middle bin, restored for the curve)
BINS = [("low [0,0.35)", 0.0, 0.35),
        ("mid [0.35,0.75)", 0.35, 0.75),
        ("high [0.75,log3]", 0.75, LOG3 + 1e-9)]


"""
Shannon entropy of the posterior mode probabilities mu_T.
It is the amount of mode uncertainty / multimodality:
    - 0 when we are certainly in the one mode regime
    - log(3) when all three are equally likely

This is the stratification variable the whole study is organised around.
"""
def entropy(mu):
    mu = np.clip(mu, 1e-12, 1.0)
    return float(-(mu * np.log(mu)).sum())

"""
Weight-agnostic geometric separation of the modes, on position only
    - Numerator: the largest pairwise distance between the mode-conditional position
      (xhat_i)
    - Denominator: the average per-mode position std (sqrt of the mean of the position
      variances). It is a length scale for "how spread is each component". 

So sep_ratio is approximately "how far apart are the modes, in units of their own width". 
It ignores mu entirely -- two well-separated modes give a large value even if one
carries almost no posterior weight.
"""
def sep_ratio(xhat, Phat):
    pos = xhat[:, :2]                       # (3,2) mode-conditional pos means
    s = np.sqrt(np.mean([(Phat[j, 0, 0] + Phat[j, 1, 1]) / 2
                         for j in range(3)]))
    dmax = max(np.linalg.norm(pos[i] - pos[j])
               for i in range(3) for j in range(i + 1, 3))
    return dmax / s

"""
Monte-Carlo KL( mixture || moment-matched Gaussian ) on position: the
weight-aware information lost by collapsing, at h=0.
  - mixture p = sum_j mu_j N(m_j, S_j)   (mode-conditional position Gaussians)
  - collapse q = N(xbar_pos, Pbar_pos)   (the IMM's moment-matched single Gaussian)

Estimate: draw z from p (pick component by mu, sample its Gaussian), then average
log p(z) - log q(z), with log p via logsumexp over the components. 

KL is always >= 0 and = 0 only if p is already Gaussian (components coincide or
a single mode dominates).

The inner logN is a standard multivariate-Gaussian log-density via Cholesky.
"""

def kl_collapse(mu, xhat, Phat, xbar, Pbar, rng):
    m = xhat[:, :2]                         # component pos means (3,2)
    S = Phat[:, :2, :2]                     # component pos covs (3,2,2)
    mq, Sq = xbar[:2], Pbar[:2, :2]
    # draw z ~ mixture
    comp = rng.choice(3, size=N_MC, p=mu)
    z = np.empty((N_MC, 2))
    chols = [np.linalg.cholesky(S[j] + 1e-12 * np.eye(2)) for j in range(3)]
    for j in range(3):
        idx = comp == j
        n = int(idx.sum())
        if n:
            z[idx] = m[j] + rng.standard_normal((n, 2)) @ chols[j].T

    def logN(z, mean, cov):
        d = z - mean
        L = np.linalg.cholesky(cov + 1e-12 * np.eye(2))
        sol = np.linalg.solve(L, d.T)
        quad = (sol ** 2).sum(0)
        logdet = 2 * np.log(np.diag(L)).sum()
        return -0.5 * (2 * np.log(2 * np.pi) + logdet + quad)

    # log p(z) for the mixture (logsumexp over components)
    logp_comp = np.stack([np.log(mu[j]) + logN(z, m[j], S[j])
                          for j in range(3)], axis=0)
    logp = np.logaddexp.reduce(logp_comp, axis=0)
    logq = logN(z, mq, Sq)
    return float(np.mean(logp - logq))


"""
- Build the (H, sep_ratio, kl_collapse) table over a pool of sequences at the
  calibrated sigma_r. Several origins per sequence (ORIGINS_PER_SEQ).
- For each origin, run the IMM up to T, read the mixture (mu, xhat, Phat) and its
  collapse (xbar, Pbar), and compute the three scalars.

"""
def run():
    rng = np.random.default_rng(2024)       # held-out diagnostic seed
    rows = []  # (H, sep, kl)
    for _ in range(N_SEQ):
        modes, x = slds.simulate(SEQ_LEN, rng)
        y = slds.observe(x, SIGMA_R, rng)
        # one full IMM pass; read mode-conditional mixture at several origins
        Ts = rng.choice(range(ORIGIN_LO, ORIGIN_HI + 1),
                        size=ORIGINS_PER_SEQ, replace=False)
        for T in Ts:
            mu, xh, Ph, xb, Pb, _ = slds.run_imm(y[:T + 1], SIGMA_R)
            H = entropy(mu)
            sep = sep_ratio(xh, Ph)
            kl = kl_collapse(mu, xh, Ph, xb, Pb, rng)
            rows.append((H, sep, kl))
    return np.array(rows)


if __name__ == "__main__":
    import time
    t0 = time.time()
    R = run()
    Hs, seps, kls = R[:, 0], R[:, 1], R[:, 2]
    print(f"n origins: {len(R)}   sigma_r={SIGMA_R}")
    print(f"KL >= 0 holds (min KL = {kls.min():.4f}; tiny negatives are MC noise)")
    print(f"\nEntropy distribution (max = log3 = {LOG3:.3f}):")
    print(f"  H: mean {Hs.mean():.3f}  median {np.median(Hs):.3f} "
          f" p90 {np.quantile(Hs,0.9):.3f}  max {Hs.max():.3f}")
    for name, lo, hi in BINS:
        m = (Hs >= lo) & (Hs < hi)
        frac = m.mean()
        print(f"  {name:>18}: {frac*100:5.1f}% of origins")

    print(f"\n{'entropy bin':>18} {'n':>5} {'sep_ratio':>20} {'KL_collapse':>20}")
    for name, lo, hi in BINS:
        m = (Hs >= lo) & (Hs < hi)
        n = int(m.sum())
        if n == 0:
            print(f"{name:>18} {n:>5} {'--':>20} {'--':>20}")
            continue
        sm, ss = seps[m].mean(), seps[m].std()
        km, ks = kls[m].mean(), kls[m].std()
        print(f"{name:>18} {n:>5} {sm:>10.3f} +/-{ss:<6.3f} "
              f"{km:>10.4f} +/-{ks:<7.4f}")

    # correlations: is collapse cost rising with entropy?
    print(f"\nSpearman-ish (Pearson on ranks):")
    def pr(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return np.corrcoef(ra, rb)[0, 1]
    print(f"  corr(H, sep_ratio)   = {pr(Hs, seps):+.3f}")
    print(f"  corr(H, KL_collapse) = {pr(Hs, kls):+.3f}")
    print(f"  corr(sep, KL)        = {pr(seps, kls):+.3f}")
    print(f"\nelapsed {time.time()-t0:.1f}s")
