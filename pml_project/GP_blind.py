#!/usr/bin/env python3
"""
gp_blind.py -- the PROPOSAL'S GP propagator: mode-BLIND.

Difference from mode-AWARE, off-spec gp_rollout:
  - TWo SVGPs total, one per velocity component (x,y). (In M-A we had 6)
  - Trained on ALL transitions POOLED across modes; the mode is neither a feature
    nor a selector. The map z -> dv is one-to-many across modes, so the GP can only
    learn the mode-averaged dynamics, and the irreducible between-mode spread  
    is absorbed into a single homoscedastic noise.
  - Target is the velocity INCREMENT dv = v_{t+1} - v_t. Position comes from 
    the exact kinematic relation (p += v). Input are standardized.

Rollout has NO mode variable at all. Each particle carries only its stae and
applies the same two GPs: dv = g(z) + eta, where g is ONE coherent Matheron 
function drawn per particle and eta is the idd noise (no mode chain).

Arm A and C share function draws, noise streams and initial normals (Common Random Numebrs)
and differ only in the initial state distribution (mode-conditional for A and 
collapsed for C). This contrast isolates the cost of "destroying the mode-sate
coupling, propagated through identical dynamics. 

"""
import warnings
import numpy as np
import torch
import gpytorch
from botorch.models import SingleTaskVariationalGP
from botorch.sampling.pathwise import draw_matheron_paths
from botorch.sampling.pathwise.prior_samplers import draw_kernel_feature_paths


import slds
import rollout as R
import experiment as E

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.double)

# ---- reduced rollout scale (the rollout is the slow part); bump for a real run
N_PART = 2000
TARGET = 100
HORIZONS = [5, 20, 40]
STRATA = [("low", 0.0, 0.35), ("high", 0.75, float(np.log(3)) + 1e-9)]
# ---- GP fit capacity (proposal: M=256, k-means init, minibatch 1024, up to 2e4 steps). 
M_IND = 256
STEPS = 3000
BATCH = 1024


# ====================== mode-blind training data ======================
"""
Build one pooled traing set from the True simulator. The mode is known, 
but not used. For each transition t -> t+1 we do:
    - input z_t  = (vx_t, vy_t, sin(omega t), cos(omega t))
    - target dv_t = v_{t+1} - v_t 
Then we subsample to n_keep points. Pooling across modes is what makes GP
model-blind: same z maps to different dv depending on the unobserved mode.
"""
def make_training_data_blind(K=200, T=300, n_keep=30000, seed=1):
    """Pooled (z, dv) from TRUE simulator states; mode known but NOT used.
    z = (vx, vy, sin wt, cos wt); dv = v_{t+1} - v_t. One dataset, two outputs."""
    rng = np.random.default_rng(seed)
    Zs, DVs = [], []
    for _ in range(K):
        modes, x = slds.simulate(T, rng)          # modes deliberately ignored
        v = x[:, 2:]
        t = np.arange(T)
        z = np.column_stack([v[:-1, 0], v[:-1, 1],
                             np.sin(slds.OMEGA * t), np.cos(slds.OMEGA * t)])
        dv = v[1:] - v[:-1]
        Zs.append(z); DVs.append(dv)
    Z = np.vstack(Zs); DV = np.vstack(DVs)
    idx = rng.permutation(len(Z))[:n_keep]
    return Z[idx], DV[idx]


# ============================ fit two SVGPs ============================
"""
Fit two SVGD, one per velocity component d in {x,y}.
We use ARD-SE kernel, standarized inputs, M inducing points (obtained with random init),
minibatch ELBO with Adam. We used RBF because BoTorch native.
"""

def fit_blind(Z, DV, M=M_IND, steps=STEPS, batch=BATCH, lr=0.01, seed=0):
    """Two independent SVGPs (one per velocity component), ARD-SE kernel,
    standardized inputs, minibatch ELBO. Returns (gps, (mu_z, sd_z))."""
    torch.manual_seed(seed)
    mu_z, sd_z = Z.mean(0), Z.std(0)
    sd_z[sd_z < 1e-8] = 1.0
    Zt = torch.as_tensor((Z - mu_z) / sd_z)
    n = Zt.shape[0]
    gps = []
    for d in range(2):
        yd = torch.as_tensor(DV[:, d:d + 1])
        ind = Zt[torch.randperm(n)[:M]].clone()   # random inducing init (k-means optional)
        model = SingleTaskVariationalGP(
            Zt, yd, inducing_points=ind,
            covar_module=gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=4)))
        mll = gpytorch.mlls.VariationalELBO(model.likelihood, model.model, num_data=n)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for it in range(steps):
            idx = torch.randint(0, n, (batch,))
            opt.zero_grad()
            loss = -mll(model(Zt[idx]), yd[idx].squeeze(-1))
            loss.backward(); opt.step()
            if it % 500 == 0:
                print(f" [comp {d}] step {it:>4d} -loss = {- loss.item():.4f}")
        model.eval()
        gps.append(model)
    return gps, (mu_z, sd_z)

"""
Just a sanity check
"""
def fit_sanity(gps, transform, seed=7):
    """Held-out RMSE / predictive log-density / learned noise per component.
    The residual std is LARGE on purpose: it is the between-mode variability
    that a mode-blind model must dump into homoscedastic noise."""
    mu_z, sd_z = transform
    Zte, DVte = make_training_data_blind(K=40, n_keep=8000, seed=seed)
    Zt = torch.as_tensor((Zte - mu_z) / sd_z)
    print("Mode-blind GP fit sanity (held-out):")
    print(f"  {'comp':>5} {'RMSE':>8} {'resid-std':>10} {'learned-noise':>14} "
          f"{'logdens/pt':>11}")
    for d in range(2):
        with torch.no_grad():
            post = gps[d].posterior(Zt)
            mean = post.mean.squeeze(-1).numpy()
            var = post.variance.squeeze(-1).numpy()
        resid = DVte[:, d] - mean
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        noise = float(gps[d].likelihood.noise.detach().sqrt())
        s2 = var + noise ** 2
        ll = float(-0.5 * np.mean(resid ** 2 / s2 + np.log(2 * np.pi * s2)))
        print(f"  {('x','y')[d]:>5} {rmse:>8.4f} {resid.std():>10.4f} "
              f"{noise:>14.4f} {ll:>11.3f}")
    print("  (resid-std >> process sigma is EXPECTED: it is the mode-marginalized")
    print("   spread the mode-blind GP cannot resolve. That is the design's point.)")


"""
The rollout's homoscedastic process noise eta, estimated FROM DATA: 
per component, the std of (dv - posterior mean) over the training set. Used in the
rollout INSTEAD of the SVGP likelihood variance (robust to SVGP noise
mis-estimation).

"""

def estimate_noise_blind(gps, transform, Z, DV):
    """Homoscedastic process noise for the rollout: data-driven residual std
    per component (robust to SVGP noise mis-estimation)."""
    mu_z, sd_z = transform
    Zt = torch.as_tensor((Z - mu_z) / sd_z)
    noise = []
    for d in range(2):
        with torch.no_grad():
            mean = gps[d].posterior(Zt).mean.squeeze(-1).numpy()
        noise.append(float(np.std(DV[:, d] - mean)))
    return noise


# ====================== pathwise mode-blind rollout ======================

"""
Draw ONE batch of N Matheron pathwise samples from each of the two GPs: 
sample i is the coherent function draw carried by particle i through its whole
rollout. 
Pathwise = a FIXED function sample, not a fresh marginal draw per step. 
Called once per origin; the SAME paths feed both arms A and C, which
is part of the CRN.
"""

def _rff(model, sample_shape, num_rff=4096):
    return draw_kernel_feature_paths(model, sample_shape=sample_shape,
                                     num_features=num_rff)

def draw_paths_blind(gps, N, seed):
    torch.manual_seed(seed)
    return [draw_matheron_paths(g, sample_shape=torch.Size([N]),
            prior_sampler=lambda model, sample_shape: _rff(model, sample_shape, 4096))
            for g in gps]



"""
Mode-blind propagation, there is NO mode variable. Per step:
  - p += v                  
  - build z from (v, absolute time t) and apply the training standardization
  - for each component: dv = g(z) [coherent per-particle pathwise draw] + eta
    [iid ~ N(0, noise^2)]
  - v += dv
No mode is sampled and no per-mode dispatch happens: every particle uses the
SAME two GPs regardless of any mode. 
"""

def blind_rollout(x, T, horizons, gps, transform, paths, noise, rng):
    """NO mode variable. Per step: dv = g(z) + eta (g = coherent per-particle
    function draw, eta fresh ~ N(0, noise^2)); v += dv; p += v (current v)."""
    mu_z, sd_z = transform
    N = x.shape[0]
    hset, out = set(horizons), {}
    for step in range(max(horizons)):
        t = T + step
        pos, vel = x[:, :2], x[:, 2:]
        pos = pos + vel                                   # p_{t+1} = p_t + v_t
        s, c = np.sin(slds.OMEGA * t), np.cos(slds.OMEGA * t)
        z = np.column_stack([vel[:, 0], vel[:, 1], np.full(N, s), np.full(N, c)])
        Zt = torch.as_tensor((z - mu_z) / sd_z).unsqueeze(-2)   # (N,1,4) ensemble
        dv = np.empty((N, 2))
        for d in range(2):
            with torch.no_grad():
                g = paths[d](Zt).squeeze(-1).numpy()      # path i @ input i
            dv[:, d] = g + noise[d] * rng.standard_normal(N)
        vel = vel + dv                                    # v_{t+1} = v_t + dv
        x = np.concatenate([pos, vel], axis=1)
        h = step + 1
        if h in hset:
            out[h] = x[:, :2].copy()
    return out


"""
Arms A and C under CRN sheres function deaws, noise strem and the initial normals.
The differ in 'init_partcile':
    - Arm A draws the initial state from the mode-conditional N(xhat^j, Phat^j),
    - Arm C draws the initial state from the collapsed N(x_bar, P_bar)
The mode label is discarded (the Mode-blind rollout never use it), so A-C is
purely 'initial state coupled to its mode vs collapsed', through identical dynamics.
"""

def score_blind(c, gps, transform, noise, horizons, n_part, seed):
    """Arms A/C with common random numbers (shared function draws + noise +
    init normals); only the initial state distribution differs."""
    post, T, truth = c["post"], c["T"], c["truth"]
    paths = draw_paths_blind(gps, n_part, seed + 99)      # shared A/C
    si, sr = seed, seed + 1
    rolls = {}
    for arm in ("A", "C"):
        xs, _ = R.init_particles(post, arm, n_part, np.random.default_rng(si))  # mode unused
        rolls[arm] = blind_rollout(xs, T, horizons, gps, transform, paths, noise,
                                   np.random.default_rng(sr))
    res = {}
    for h in horizons:
        yh = truth[h]
        res[h] = dict(
            d=R.crps_position(rolls["A"][h], yh) - R.crps_position(rolls["C"][h], yh),
            covA=R.coverage_position(rolls["A"][h], yh, 0.9),
            covC=R.coverage_position(rolls["C"][h], yh, 0.9))
    return res


def stratify_local(cands, strata, target, rng):
    out = {}
    for name, lo, hi in strata:
        pool = [c for c in cands if lo <= c["H"] < hi]
        rng.shuffle(pool)
        out[name] = pool[:target]
    return out


# ================================ main ================================
if __name__ == "__main__":
    import time
    t0 = time.time()
    print("=" * 70)
    print("GP-PROPAGATOR EXPERIMENT  --  mode-BLIND (the proposal's GP)")
    print("=" * 70)
    print("One SVGP per velocity component; mode is NOT an input. Compare with")
    print("gp_experiment.py (mode-AWARE). Here a null at long horizon is")
    print("INFORMATIVE (predicted GP under-dispersion may swamp the effect).\n")

    print(f"Training 2 mode-blind SVGPs (M={M_IND}, steps={STEPS}, batch={BATCH})...")
    Ztr, DVtr = make_training_data_blind()
    gps, transform = fit_blind(Ztr, DVtr)
    print(f"  trained in {time.time()-t0:.0f}s\n")
    fit_sanity(gps, transform)
    noise = estimate_noise_blind(gps, transform, Ztr, DVtr)
    print(f"\nrollout process-noise eta (data-driven): x={noise[0]:.4f}, y={noise[1]:.4f}\n")

    rng = np.random.default_rng(404)
    cands = E.collect_origins(rng)                        # reuse experiment.py pool
    strat = stratify_local(cands, STRATA, TARGET, rng)

    print(f"Mode-blind GP propagator (N_part={N_PART}, reduced scale):")
    print(f"{'stratum':>6} {'n':>4} {'H':>4} {'mean dCRPS (A-C)':>18} "
          f"{'95% CI':>22} {'covA':>6} {'covC':>6}")
    for name, _, _ in STRATA:
        acc = {h: {"d": [], "covA": [], "covC": [], "sid": []} for h in HORIZONS}
        for k, c in enumerate(strat[name]):
            r = score_blind(c, gps, transform, noise, HORIZONS, N_PART, seed=7 * k + 1)
            for h in HORIZONS:
                acc[h]["d"].append(r[h]["d"])
                acc[h]["covA"].append(r[h]["covA"])
                acc[h]["covC"].append(r[h]["covC"])
                acc[h]["sid"].append(c["sid"])
        for h in HORIZONS:
            m, ci = E.paired_bootstrap(acc[h]["d"])
            sig = "*" if (ci[1] < 0 or ci[0] > 0) else " "
            print(f"{name:>6} {len(acc[h]['d']):>4} {h:>4} {m:>17.4f}{sig} "
                  f"[{ci[0]:>8.4f},{ci[1]:>8.4f}] "
                  f"{np.mean(acc[h]['covA']):>6.2f} {np.mean(acc[h]['covC']):>6.2f}")
        # cluster CI on the primary endpoint, when this stratum is 'high'
        if name == "high" and 20 in acc:
            mc, cc = E.cluster_bootstrap(acc[20]["d"], acc[20]["sid"])
            print(f"   -> PRIMARY (high, H=20) cluster 95% CI "
                  f"[{cc[0]:.4f}, {cc[1]:.4f}]  mean {mc:.4f}")
    print("\n(* = 95% CI excludes 0.  Negative => collapse hurts.  Watch coverage:")
    print(" if covA/covC fall well below 0.90 at H=40, that is the predicted")
    print(" mode-blind under-dispersion.)")
    print(f"\n[gp_blind done in {time.time()-t0:.0f}s]")