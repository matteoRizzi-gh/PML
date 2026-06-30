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
"""Draw N particles. Fixed randomness order so all arms stay paired:
    1. j        ~ Cat(mu)       component labels
    2. z        ~ N(0, I_4)     shared init normals
    3. m_indep  ~ Cat(mu)       independent propagation mode
Every arm draws all three (even if unused) so j, z, m_indep are IDENTICAL
across arms. What each USES:
  'mix'         : state from component j; propagate under independent mode.
  'collapse'    : state from moment-matched Gaussian; SAME independent mode.
                  mix - collapse = cost of collapsing the STATE posterior.
  'mix_coupled' : state from component j; propagate under j (coupled mode).
                  mix_coupled - mix = cost of losing mode-state coupling."""
def init_particles(post, arm, N, rng):

    mu, xhat, Phat, xbar, Pbar = post
    j = rng.choice(3, size=N, p=mu)
    z = rng.standard_normal((N, 4))
    m_indep = rng.choice(3, size=N, p=mu)
    if arm == "mix":
        L = _chol_batch(Phat)
        x = xhat[j] + np.einsum("nij,nj->ni", L[j], z)
        m = m_indep
    elif arm == "collapse":
        Lb = _chol_batch(Pbar[None])[0]
        x = xbar + z @ Lb.T
        m = m_indep
    elif arm == "mix_coupled":
        L = _chol_batch(Phat)
        x = xhat[j] + np.einsum("nij,nj->ni", L[j], z)
        m = j.copy()
    else:
        raise ValueError(arm)
    return x, m



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
    """Propagate particles with the TRUE switching dynamics from absolute time T"""
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
            out[h] = x.copy()                 # full state (N,4)
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


# Closed-form H=1 predictive (for validation)

"""
External validation. At H=1 the forecast is a closed form Gaussian Mixture 
with 3 components. For each mode j, we apply the dynamics A_j to the starting
Gaussian (((x_hat)^j ; P^j) for A and (x_bar, P_bar) for C). 

We obtain mean A_j x_hat^j + c_j and covarianc eA_j P^j A_j^T + Q_j

"""
def closed_form_h1(post, arm, T):
    """Exact position predictive of x_{T+1} as a 3-component Gaussian mixture"""
    mu, xhat, Phat, xbar, Pbar = post
    means, covs = [], []

    for j in range(3):
        Aj, Qj = A_MATS[j], Q_MATS[j]
        cj = np.zeros(4)
        if j == 2:
            cj[2:] = slds.forcing_u(T)
        if arm in ("mix", "mix_coupled"):
            mj = Aj @ xhat[j] + cj
            Cj = Aj @ Phat[j] @ Aj.T + Qj
        else:                               # collapse
            mj = Aj @ xbar + cj
            Cj = Aj @ Pbar @ Aj.T + Qj
        means.append(mj[:2]); covs.append(Cj[:2, :2])

    means = np.array(means); covs = np.array(covs)
    mbar = (mu[:, None] * means).sum(0)
    Cbar = sum(mu[j] * (covs[j] + np.outer(means[j] - mbar, means[j] - mbar))
               for j in range(3))
    return mbar, Cbar


def sharpness_position(samples_pos, level):
    lo = (1 - level) / 2
    widths = []
    for d in range(2):
        q = np.quantile(samples_pos[:, d], [lo, 1 - lo])
        widths.append(q[1] - q[0])
    return float(np.mean(widths))


# Velocity CRPS

def crps_velocity(samples_state, truth_vel):
    """Mean over the 2 velocity components of the ensemble CRPS"""
    samples_vel = samples_state[:, 2:]          # (N, 2)
    return float(np.mean([ps.crps_ensemble(truth_vel[d], samples_vel[:, d])
                          for d in range(2)]))


# PIT / rank histogram

def pit_rank(samples_1d, truth_1d, rng):

    S = len(samples_1d)
    n_below = int(np.sum(samples_1d < truth_1d))
    n_equal = int(np.sum(samples_1d == truth_1d))
    if n_equal == 0:
        return n_below
    # random tie-breaking: distribute ties uniformly
    return n_below + int(rng.integers(0, n_equal + 1))


def pit_ranks_origin(samples_state, truth_pos, truth_vel, rng):
    """Compute PIT ranks for one origin at one horizon"""
    names = ["px", "py", "vx", "vy"]
    truth = np.concatenate([truth_pos, truth_vel])   # (4,)
    return {name: pit_rank(samples_state[:, d], truth[d], rng)
            for d, name in enumerate(names)}


def pit_histogram(ranks, n_particles, n_bins=10):
    """Build a normalised rank histogram from a list of integer ranks """
    ranks = np.asarray(ranks)
    edges = np.linspace(0, n_particles + 1, n_bins + 1)
    counts, _ = np.histogram(ranks, bins=edges)
    return edges, counts / counts.sum()


# Energy score (multivariate generalisation of CRPS)
def energy_score(samples, truth):

    """Energy score for a multivariate ensemble forecast.

    ES(F, y) = E||X - y|| - 0.5 * E||X - X'||
    where X, X' are independent draws from the forecast distribution F
    and ||.|| is the Euclidean norm.

    Estimated from S particles via the unbiased U-statistic:
        ES = (1/S) sum_i ||x_i - y||
           - (1 / (2 * S*(S-1))) sum_{i != j} ||x_i - x_j||

    Works for any dimension k (position k=2, velocity k=2, full state k=4).
    The second term uses the O(S^2) pairwise distance — feasible for S<=2000.

    Note: the proposal explicitly EXCLUDED the energy score from primary
    endpoints (Section 7: "weakly sensitive to dependence structure, costly,
    no discriminative gain"). It is included here as an optional diagnostic
    only; the primary endpoint remains CRPS on position components.

    samples : (S, k)  ensemble particles
    truth   : (k,)    true value
    Returns a scalar float.
    """
    
    samples = np.asarray(samples)           # (S, k)
    truth   = np.asarray(truth)             # (k,)
    S = samples.shape[0]
    # first term: mean distance to truth
    term1 = np.mean(np.linalg.norm(samples - truth, axis=1))
    # second term: mean pairwise distance (U-statistic, excludes i==j)
    # use broadcasting: (S,1,k) - (1,S,k) -> (S,S,k)
    diff = samples[:, None, :] - samples[None, :, :]    # (S,S,k)
    pw = np.linalg.norm(diff, axis=2)                   # (S,S)
    # sum off-diagonal, divide by S*(S-1)
    term2 = (pw.sum() - np.diag(pw).sum()) / (S * (S - 1))
    return float(term1 - 0.5 * term2)


def energy_score_position(samples_state, truth_pos):

    return energy_score(samples_state[:, :2], truth_pos)


def energy_score_velocity(samples_state, truth_vel):

    return energy_score(samples_state[:, 2:], truth_vel)


# VALIDATION: particle rollout vs closed form at H=1 

if __name__ == "__main__":
    
    rng = np.random.default_rng(0)
    modes, x = slds.simulate(200, rng)
    y = slds.observe(x, 0.05, rng)
    T = 120
    post = slds.run_imm_multi(y, 0.05, [T])[T]
    N = 200_000
    print(f"H=1 validation (N={N} particles), origin T={T}:")
    for arm in ("mix", "collapse"):
        xs, ms = init_particles(post, arm, N, np.random.default_rng(7))
        state1 = oracle_rollout(xs, ms, T, [1], np.random.default_rng(7))[1]
        pos1 = state1[:, :2]               # slice position from full state (N,4)
        emp_m, emp_C = pos1.mean(0), np.cov(pos1.T)
        cf_m, cf_C = closed_form_h1(post, arm, T)
        print(f"  arm {arm}: |mean err| {np.abs(emp_m-cf_m).max():.4f}  "
              f"|cov err| {np.abs(emp_C-cf_C).max():.4f}  "
              f"(mean scale {np.abs(cf_m).max():.2f}, cov scale {np.abs(cf_C).max():.3f})")



"""
STATUS (aggiornato).

FATTO:
  - tre bracci: mix, collapse, mix_coupled con CRN corretto
  - oracle_rollout restituisce stato completo (N,4)
  - crps_position, crps_velocity, energy_score (pos/vel)
  - pit_rank, pit_ranks_origin, pit_histogram
  - closed_form_h1 con naming corretto (mix/mix_coupled/collapse)
  - validazione H=1 corretta con slice [:, :2]

RISULTATO PRINCIPALE (oracle, high stratum, H=20):
  state collapse   (mix - collapse)        = -0.0012  [-0.0040, +0.0015]  -> NULL
  coupling loss    (mix_coupled - mix)     = -0.0247  [-0.0495, +0.0003]  -> il vero costo
  total            (mix_coupled - collapse)= -0.0259  [-0.0517, -0.0007]
  additivity: state + coupling = total (diff 0.0000), esatta.

MANCA ANCORA:
  - H=2 exact validation (3^2=9 componenti): testa la coerenza pathwise
    lungo la traiettoria; la validazione H=1 non la copre.
  - energy score da integrare in experiment.py (funzione qui presente,
    da aggiungere a score_origin e al loop di raccolta risultati).
  - gp_support_check.py: arm "A" -> "mix" (crasherebbe a runtime).
"""
