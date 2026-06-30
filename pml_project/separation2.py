"""
Separation / collapse cost split by SUBSPACE: position, velocity, full state.

Same h=0 diagnostic as separation.py, but computed on each subspace separately
to reveal WHERE the multimodality lives. 
This matters because the scored endpoint is POSITION: if the modes separate in velocity
but not in position, the position A-C cost is ~0 at h=0 and can only build up over 
the horizon as velocity offsets integrate into position. 

So this file tells you where the propagated cost must come from. Same calibrated sigma_r.
"""
import numpy as np
import slds

SIGMA_R = 0.05
SEQ_LEN = 200
ORIGIN_LO, ORIGIN_HI = 60, 160
N_SEQ = 300
ORIGINS_PER_SEQ = 3
N_MC = 2000
LOG3 = np.log(3)
SUBSPACES = {"position": [0, 1], "velocity": [2, 3], "full": [0, 1, 2, 3]}
BINS = [("low", 0.0, 0.35), ("mid", 0.35, 0.75), ("high", 0.75, LOG3 + 1e-9)]


"""
Shannon entropy of the posterior mode probabilities mu_T.
It is the amount of mode uncertainty / multimodality:
    - 0 when we are certainly in the one-mode regime
    - log(3) when all three are equally likely
This is the stratification variable the whole study is organised around.
"""
def entropy(mu):
    mu = np.clip(mu, 1e-12, 1.0)
    return float(-(mu * np.log(mu)).sum())


"""
Multivariate-Gaussian log-density log N(z; mean, cov), Cholesky-based and
dimension-GENERIC (k = mean.shape[0]).
"""

def logN(z, mean, cov):
    d = z - mean
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(cov.shape[0]))
    sol = np.linalg.solve(L, d.T)
    quad = (sol ** 2).sum(0)
    logdet = 2 * np.log(np.diag(L)).sum()
    k = mean.shape[0]
    return -0.5 * (k * np.log(2 * np.pi) + logdet + quad)

"""
The two scalars (sep_ratio and kl_collapse) RESTRICTED to a subspace given by
`idx` ([0,1]=position, [2,3]=velocity, [0,1,2,3]=full). 
It slices the mode means, covariances and the collapse to those coordinates, then:
  - sep = max pairwise distance of the mode means / avg per-component std on idx
          (weight-agnostic geometric separation, in that subspace)
  - kl  = MC KL(mixture||collapse) on idx (weight-aware info lost there, >= 0)

Mixture and collapse are taken on the SAME coordinates, so kl is the
information the collapse loses WITHIN that subspace.
"""
def sep_and_kl(mu, xhat, Phat, xbar, Pbar, idx, rng):
    idx = np.array(idx)
    m = xhat[:, idx]                          # (3,k)
    S = Phat[:, idx][:, :, idx]               # (3,k,k)
    mq = xbar[idx]; Sq = Pbar[np.ix_(idx, idx)]
    k = len(idx)

    # separation ratio
    s = np.sqrt(np.mean([np.trace(S[j]) / k for j in range(3)]))
    dmax = max(np.linalg.norm(m[i] - m[j])
               for i in range(3) for j in range(i + 1, 3))
    sep = dmax / s

    # KL(mixture||collapse) via MC
    comp = rng.choice(3, size=N_MC, p=mu)
    z = np.empty((N_MC, k))
    chols = [np.linalg.cholesky(S[j] + 1e-10 * np.eye(k)) for j in range(3)]

    for j in range(3):
        sel = comp == j; n = int(sel.sum())
        if n:
            z[sel] = m[j] + rng.standard_normal((n, k)) @ chols[j].T

    logp = np.logaddexp.reduce(np.stack([np.log(mu[j]) + logN(z, m[j], S[j]) for j in range(3)]), 0)
    logq = logN(z, mq, Sq)

    return sep, float(np.mean(logp - logq))

"""
- Build the table over a pool of sequences at the calibrated sigma_r (several
  origins per sequence). 
- For each origin: run the IMM to T
- Read the mixture and its collapse, and compute (sep, kl) for EACH subspace.

"""
def run():
    rng = np.random.default_rng(2024)
    data = {name: {"H": [], "sep": [], "kl": []} for name in SUBSPACES}
    Hs_all = []

    for _ in range(N_SEQ):
        modes, x = slds.simulate(SEQ_LEN, rng)
        y = slds.observe(x, SIGMA_R, rng)
        for T in rng.choice(range(ORIGIN_LO, ORIGIN_HI + 1),
                            size=ORIGINS_PER_SEQ, replace=False):
            mu, xh, Ph, xb, Pb, _ = slds.run_imm(y[:T + 1], SIGMA_R)
            H = entropy(mu); Hs_all.append(H)
            for name, idx in SUBSPACES.items():
                sep, kl = sep_and_kl(mu, xh, Ph, xb, Pb, idx, rng)
                data[name]["H"].append(H)
                data[name]["sep"].append(sep)
                data[name]["kl"].append(kl)
                
    for name in SUBSPACES:
        for key in data[name]:
            data[name][key] = np.array(data[name][key])

    print(f"n origins {len(Hs_all)}  sigma_r={SIGMA_R}\n")
    print(f"{'subspace':>9} {'bin':>5} {'sep_ratio':>12} {'KL_collapse':>14}")
    for name in SUBSPACES:
        Hs = data[name]["H"]; seps = data[name]["sep"]; kls = data[name]["kl"]
        for bname, lo, hi in BINS:
            msk = (Hs >= lo) & (Hs < hi)
            if msk.sum() == 0:
                continue
            print(f"{name:>9} {bname:>5} {seps[msk].mean():>12.3f} "
                  f"{kls[msk].mean():>14.4f}")
        print()


if __name__ == "__main__":
    run()
