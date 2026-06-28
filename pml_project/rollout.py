"""
Multi-step forecast rollouts under mode uncertainty, two arms:
  A (Mixture):  particle i drawn from its mode-conditional Gaussian
                N(xhat^{j_i}, Phat^{j_i})  -- mode-state coupling kept.
  C (Collapse): particle i drawn from the single moment-matched Gaussian
                N(xbar, Pbar)              -- mode-state coupling destroyed.
Both keep the mode marginal mu_T and propagate identically. With common
random numbers (same component labels j_i, same init normals z, same mode
paths and process noise across arms), the paired difference CRPS_A - CRPS_C
isolates exactly the cost of collapsing the multimodal posterior.

Oracle propagator = true switching dynamics. (GP propagator: gp_rollout.py.)
Primary endpoint: CRPS on position. Lower CRPS = better; so a NEGATIVE
A - C means the collapse hurts.
"""
import numpy as np
import properscoring as ps
import slds

A_MATS = np.stack([slds._A(m) for m in range(3)])      # (3,4,4)
Q_MATS = np.stack([slds._Q(m) for m in range(3)])      # (3,4,4)


def _chol_batch(P):
    return np.linalg.cholesky(P + 1e-10 * np.eye(P.shape[-1]))


"""
Sample N particles:
    - draw component etiquettes j ~ mu and a matrix of normal z
    - A-arm: x = x_hat^j + chol(P^j)*z 
        mode-state coupling preserved
    - C-arm: x = x_bar + chol(P_bar)*z
        mode-state coupling destroyed
Same seed (Common Random Number setup)
"""

def init_particles(post, arm, N, rng):
    """Draw N particles for the given arm. Consumes randomness in a fixed
    order (component labels, then init normals) so arms A and C are paired."""
    mu, xhat, Phat, xbar, Pbar = post
    j = rng.choice(3, size=N, p=mu)            # shared component labels
    z = rng.standard_normal((N, 4))            # shared init normals
    if arm == "A":
        L = _chol_batch(Phat)                  # (3,4,4)
        x = xhat[j] + np.einsum("nij,nj->ni", L[j], z)
    elif arm == "C":
        Lb = _chol_batch(Pbar[None])[0]
        x = xbar + z @ Lb.T
    else:
        raise ValueError(arm)
    return x, j.copy()                         # state (N,4), mode (N,)


"""
Propagation. At each step:
    - p_{t+1} = p_t + v_t
    - update: v = f_m * v + c + w (c forcing argument for 2-mode and w process noise)
    - sample the next mode
The process is vectorized on the N particles.

Rk. by using same seed we have same etiquettes j, same normal z, same process noise,
and same uniform of transition. This is important so the two paths are identical.
The only difference is the starting state (coupled or collapsed). This is done so that
A-C isolate exactly the effect of the initial coupling
"""

def oracle_rollout(x, modes, T, horizons, rng):
    """Propagate particles with the TRUE switching dynamics from absolute time
    T. Returns {h: positions (N,2)} for each requested horizon h."""
    hmax = max(horizons)
    hset = set(horizons)
    out = {}
    for step in range(hmax):
        t = T + step                           # transition t -> t+1, mode = m_t
        pos, vel = x[:, :2], x[:, 2:]
        pos = pos + vel                        # dt = 1
        f = slds.F_DIAG[modes][:, None]
        sig = slds.SIGMA[modes][:, None]
        c = np.zeros((x.shape[0], 2))
        is3 = modes == 2
        c[is3] = slds.A_F * np.array([np.sin(slds.OMEGA * t),
                                      np.cos(slds.OMEGA * t)])
        w = rng.standard_normal((x.shape[0], 2)) * sig
        vel = f * vel + c + w
        x = np.concatenate([pos, vel], axis=1)
        # mode transition m_t -> m_{t+1}
        Prow = slds.PI[modes]                  # (N,3)
        u = rng.random(x.shape[0])
        modes = (u[:, None] > np.cumsum(Prow, axis=1)).sum(1)
        h = step + 1
        if h in hset:
            out[h] = x[:, :2].copy()
    return out



"""
CRPS evaluation. For each coordinate (x,y) we compute CRPS of the particle
ensemble against teh true value, then we average. 
Coverage: for each coordinate take the central interval at 'level'% of quantiles. 
We need to check wheter the true value fall inside of this interval.

CRPS measures quality, while Coverage measure calibration.
"""
def crps_position(samples_pos, truth_pos):
    """Mean over the 2 position coords of the ensemble CRPS."""
    return float(np.mean([ps.crps_ensemble(truth_pos[d], samples_pos[:, d])
                          for d in range(2)]))


def coverage_position(samples_pos, truth_pos, level):
    """Per-coord central-interval coverage indicator, averaged over coords."""
    lo = (1 - level) / 2
    hits = []
    for d in range(2):
        q = np.quantile(samples_pos[:, d], [lo, 1 - lo])
        hits.append(q[0] <= truth_pos[d] <= q[1])
    return float(np.mean(hits))


# ----------------------------------------------------------------------
# Closed-form H=1 predictive (for validation)
# ----------------------------------------------------------------------

"""
External validation. At H=1 the forecast is a closed form Gaussian Mixture 
with 3 components. For each mode j, we apply the dynamics A_j to the starting
Gaussian (((x_hat)^j ; P^j) for A and (x_bar, P_bar) for C). 

We obtain mean A_j x_hat^j + c_j and covarianc eA_j P^j A_j^T + Q_j

"""
def closed_form_h1(post, arm, T):
    """Exact position predictive of x_{T+1} as a 3-component Gaussian mixture.
    Arm A uses mode-conditional (xhat^j, Phat^j); arm C uses (xbar, Pbar)."""
    mu, xhat, Phat, xbar, Pbar = post
    means, covs = [], []
    for j in range(3):
        Aj, Qj = A_MATS[j], Q_MATS[j]
        cj = np.zeros(4)
        if j == 2:
            cj[2:] = slds.forcing_u(T)
        if arm == "A":
            mj = Aj @ xhat[j] + cj
            Cj = Aj @ Phat[j] @ Aj.T + Qj
        else:
            mj = Aj @ xbar + cj
            Cj = Aj @ Pbar @ Aj.T + Qj
        means.append(mj[:2]); covs.append(Cj[:2, :2])
    means = np.array(means); covs = np.array(covs)
    mbar = (mu[:, None] * means).sum(0)
    Cbar = sum(mu[j] * (covs[j] + np.outer(means[j] - mbar, means[j] - mbar))
               for j in range(3))
    return mbar, Cbar


if __name__ == "__main__":
    # ---- VALIDATION: particle rollout vs closed form at H=1 ----
    rng = np.random.default_rng(0)
    modes, x = slds.simulate(200, rng)
    y = slds.observe(x, 0.05, rng)
    T = 120
    post = slds.run_imm_multi(y, 0.05, [T])[T]
    N = 200_000
    print(f"H=1 validation (N={N} particles), origin T={T}:")
    for arm in ("A", "C"):
        xs, ms = init_particles(post, arm, N, np.random.default_rng(7))
        pos1 = oracle_rollout(xs, ms, T, [1], np.random.default_rng(7))[1]
        emp_m, emp_C = pos1.mean(0), np.cov(pos1.T)
        cf_m, cf_C = closed_form_h1(post, arm, T)
        print(f"  arm {arm}: |mean err| {np.abs(emp_m-cf_m).max():.4f}  "
              f"|cov err| {np.abs(emp_C-cf_C).max():.4f}  "
              f"(mean scale {np.abs(cf_m).max():.2f}, cov scale {np.abs(cf_C).max():.3f})")
