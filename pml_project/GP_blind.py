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

# ---- reduced rollout scale 
N_PART = 2000     
HORIZONS = [5, 20, 40]

# ---- GP fit capacity. 
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
Fit two SVGP, one per velocity component d in {x,y}.
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
        ind = Zt[torch.randperm(n)[:M]].clone()
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
    print("  (resid-std >> process sigma is EXPECTED: the mode-marginalized")
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
        pos = pos + vel
        s, c = np.sin(slds.OMEGA * t), np.cos(slds.OMEGA * t)
        z = np.column_stack([vel[:, 0], vel[:, 1], np.full(N, s), np.full(N, c)])
        Zt = torch.as_tensor((z - mu_z) / sd_z).unsqueeze(-2)
        dv = np.empty((N, 2))
        for d in range(2):
            with torch.no_grad():
                g = paths[d](Zt).squeeze(-1).numpy()
            dv[:, d] = g + noise[d] * rng.standard_normal(N)
        vel = vel + dv
        x = np.concatenate([pos, vel], axis=1)
        h = step + 1
        if h in hset:
            out[h] = x.copy()          # full state (N,4); callers slice [:, :2] for position
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
    """Three arms for structural parity with oracle/aware. The blind rollout
    IGNORES the mode, so mix_coupled (state coupled, mode coupled) and mix
    (state coupled, mode independent) propagate IDENTICALLY -- mix_coupled is
    degenerate here and should coincide with mix up to MC noise. We compute it
    anyway as an explicit check. Only mix - collapse (state-collapse cost) is
    informative under the blind propagator."""
    post, T, truth_pos, truth_vel = (c["post"], c["T"],
                                     c["truth_pos"], c["truth_vel"])
    paths = draw_paths_blind(gps, n_part, seed + 99)
    si, sr = seed, seed + 1
    pit_rng = np.random.default_rng(seed + 2)
    rolls = {}
    for arm in ("mix", "collapse", "mix_coupled"):
        xs, _ = R.init_particles(post, arm, n_part, np.random.default_rng(si))
        rolls[arm] = blind_rollout(xs, T, horizons, gps, transform, paths, noise,
                                   np.random.default_rng(sr))
    res = {}
    for h in horizons:
        ypos = truth_pos[h]
        yvel = truth_vel[h]
        res[h] = {}
        for arm in ("mix", "collapse", "mix_coupled"):
            state = rolls[arm][h]           # (N, 4) full state
            res[h][arm] = dict(
                crps=R.crps_position(state[:, :2], ypos),
                crps_vel=R.crps_velocity(state, yvel),
                cov=R.coverage_position(state[:, :2], ypos, 0.9),
                shp=R.sharpness_position(state[:, :2], 0.9),
                pit=R.pit_ranks_origin(state, ypos, yvel, pit_rng))
    return res

# ================================ main ================================

if __name__ == "__main__":
    import time
    t0 = time.time()
    print("=" * 70)
    print("GP-PROPAGATOR EXPERIMENT  --  mode-BLIND (the proposal's GP)")
    print("=" * 70)

    print(f"Training 2 mode-blind SVGPs (M={M_IND}, steps={STEPS}, batch={BATCH})...")
    Ztr, DVtr = make_training_data_blind()
    gps, transform = fit_blind(Ztr, DVtr)
    print(f"  trained in {time.time()-t0:.0f}s\n")
    fit_sanity(gps, transform)
    noise = estimate_noise_blind(gps, transform, Ztr, DVtr)
    print(f"\nrollout process-noise eta (data-driven): x={noise[0]:.4f}, y={noise[1]:.4f}\n")

    ARMS = ("mix", "collapse", "mix_coupled")
    rng = np.random.default_rng(404)
    cands = E.collect_origins(rng)
    strat = E.stratify(cands, rng)

    print(f"Mode-blind GP propagator (N_part={N_PART}):")
    for name, _, _ in E.STRATA:
        print(f"  {name:>4}: {len(strat[name])} origins "
              f"({len({c['sid'] for c in strat[name]})} unique seqs)")

    results = {name: {h: {arm: {"crps": [], "cov": [], "shp": [],
                                "crps_vel": [],
                                "pit": {coord: [] for coord in ("px","py","vx","vy")}}
                           for arm in ARMS}
                      for h in HORIZONS} for name, _, _ in E.STRATA}
    sids = {name: [] for name, _, _ in E.STRATA}

    for name in results:
        for k, c in enumerate(strat[name]):
            r = score_blind(c, gps, transform, noise, HORIZONS, N_PART, seed=7 * k + 1)
            sids[name].append(c["sid"])
            for h in HORIZONS:
                for arm in ARMS:
                    results[name][h][arm]["crps"].append(r[h][arm]["crps"])
                    results[name][h][arm]["crps_vel"].append(r[h][arm]["crps_vel"])
                    results[name][h][arm]["cov"].append(r[h][arm]["cov"])
                    results[name][h][arm]["shp"].append(r[h][arm]["shp"])
                    for coord in ("px", "py", "vx", "vy"):
                        results[name][h][arm]["pit"][coord].append(
                            r[h][arm]["pit"][coord])

    print(f"\n{'stratum':>6} {'n':>4} {'H':>4} "
          f"{'CRPS_mix':>9} {'CRPS_col':>9} {'dCRPS':>10} {'95% CI':>22} "
          f"{'cov_m':>6} {'cov_c':>6} {'shp_m':>7} {'shp_c':>7}")
    for name, lo, hi in E.STRATA:
        for h in HORIZONS:
            cm = results[name][h]["mix"]["crps"]
            cc = results[name][h]["collapse"]["crps"]
            d = [a - b for a, b in zip(cm, cc)]
            m, ci = E.paired_bootstrap(d)
            sig = "*" if (ci[1] < 0 or ci[0] > 0) else " "
            print(f"{name:>6} {len(d):>4} {h:>4} "
                  f"{np.mean(cm):>9.4f} {np.mean(cc):>9.4f} {m:>9.4f}{sig} "
                  f"[{ci[0]:>8.4f},{ci[1]:>8.4f}] "
                  f"{np.mean(results[name][h]['mix']['cov']):>6.2f} "
                  f"{np.mean(results[name][h]['collapse']['cov']):>6.2f} "
                  f"{np.mean(results[name][h]['mix']['shp']):>7.2f} "
                  f"{np.mean(results[name][h]['collapse']['shp']):>7.2f}")
    print(f"\n(* = 95% CI excludes 0.)")

    hi_arm = results["high"][20]
    cm, cc, cf = (hi_arm["mix"]["crps"], hi_arm["collapse"]["crps"],
                  hi_arm["mix_coupled"]["crps"])
    mn, ci_n = E.paired_bootstrap([a - b for a, b in zip(cm, cc)])
    print(f"\nPRIMARY (high, H=20): state collapse under mode-blind")
    print(f"  mix - collapse = {mn:+.4f}  [{ci_n[0]:+.4f},{ci_n[1]:+.4f}]")
    # degeneracy check: mix_coupled should equal mix (blind ignores mode)
    deg = np.mean([a - b for a, b in zip(cf, cm)])
    print(f"  degeneracy check  mix_coupled - mix = {deg:+.5f}  (should be ~0: blind has no mode)")

    # ---- velocity CRPS table ----
    print(f"\n--- VELOCITY CRPS (mix - collapse, same structure as position) ---")
    print(f"\n{'stratum':>6} {'n':>4} {'H':>4} "
          f"{'vCRPS_mix':>10} {'vCRPS_col':>10} {'dvCRPS':>10} {'95% CI':>22}")
    for name, lo, hi in E.STRATA:
        for h in HORIZONS:
            cm = results[name][h]["mix"]["crps_vel"]
            cc = results[name][h]["collapse"]["crps_vel"]
            d = [a - b for a, b in zip(cm, cc)]
            m, ci = E.paired_bootstrap(d)
            sig = "*" if (ci[1] < 0 or ci[0] > 0) else " "
            print(f"{name:>6} {len(d):>4} {h:>4} "
                  f"{np.mean(cm):>10.4f} {np.mean(cc):>10.4f} {m:>9.4f}{sig} "
                  f"[{ci[0]:>8.4f},{ci[1]:>8.4f}]")
    print(f"(* = 95% CI excludes 0.  Negative dvCRPS => collapse hurts in velocity.)")

    # ---- PIT rank histograms ----
    print(f"\n--- PIT RANK HISTOGRAMS (10 bins, uniform = calibrated) ---")
    print(f"Flat histogram => well-calibrated. U-shape => under-dispersed.")
    for name, lo, hi in E.STRATA:
        for h in HORIZONS:
            for arm in ("mix_coupled", "collapse"):
                for coord in ("px", "py", "vx", "vy"):
                    ranks = results[name][h][arm]["pit"][coord]
                    _, freq = R.pit_histogram(ranks, N_PART, n_bins=10)
                    bar = " ".join(f"{f:.2f}" for f in freq)
                    print(f"  {name:>4} h={h:>2} {arm:>12} {coord}: [{bar}]")

    print(f"\n[gp_blind done in {time.time()-t0:.0f}s]")



"""
======================================================================
GP-PROPAGATOR EXPERIMENT  --  mode-BLIND (the proposal's GP)
======================================================================
Training 2 mode-blind SVGPs (M=256, steps=3000, batch=1024)...
 [comp 0] step    0 -loss = -50.3322
 [comp 0] step  500 -loss = 0.7018
 [comp 0] step 1000 -loss = 1.2856
 [comp 0] step 1500 -loss = 1.2875
 [comp 0] step 2000 -loss = 1.3226
 [comp 0] step 2500 -loss = 1.3231
 [comp 1] step    0 -loss = -50.2895
 [comp 1] step  500 -loss = 0.7161
 [comp 1] step 1000 -loss = 1.3273
 [comp 1] step 1500 -loss = 1.2896
 [comp 1] step 2000 -loss = 1.3738
 [comp 1] step 2500 -loss = 1.3165
  trained in 94s

Mode-blind GP fit sanity (held-out):
   comp     RMSE  resid-std  learned-noise  logdens/pt
      x   0.0649     0.0649         0.0597       1.312
      y   0.0659     0.0658         0.0623       1.300
  (resid-std >> process sigma is EXPECTED: the mode-marginalized
   spread the mode-blind GP cannot resolve. That is the design's point.)

rollout process-noise eta (data-driven): x=0.0642, y=0.0639

Mode-blind GP propagator (N_part=2000):
   low: 150 origins (150 unique seqs)
  high: 150 origins (150 unique seqs)

stratum    n    H  CRPS_mix  CRPS_col      dCRPS                 95% CI  cov_m  cov_c   shp_m   shp_c
   low  150    5    0.4265    0.4275   -0.0010* [ -0.0017, -0.0003]   0.72   0.74    1.60    1.61
   low  150   20    5.1728    5.1727    0.0001  [ -0.0021,  0.0025]   0.50   0.50   11.79   11.79
   low  150   40   13.2361   13.2339    0.0023  [ -0.0024,  0.0084]   0.53   0.53   33.08   33.15
  high  150    5    0.4474    0.4506   -0.0032* [ -0.0049, -0.0015]   0.78   0.81    1.75    1.76
  high  150   20    4.6108    4.6112   -0.0004  [ -0.0046,  0.0040]   0.60   0.60   11.36   11.35
  high  150   40   10.8701   10.8637    0.0065* [  0.0001,  0.0133]   0.64   0.64   29.27   29.24

(* = 95% CI excludes 0.)

PRIMARY (high, H=20): state collapse under mode-blind
  mix - collapse = -0.0004  [-0.0046,+0.0040]
  degeneracy check  mix_coupled - mix = +0.00000  (should be ~0: blind has no mode)

--- VELOCITY CRPS (mix - collapse, same structure as position) ---

stratum    n    H  vCRPS_mix  vCRPS_col     dvCRPS                 95% CI
   low  150    5     0.1786     0.1786   -0.0001  [ -0.0002,  0.0001]
   low  150   20     0.4604     0.4604    0.0000  [ -0.0001,  0.0001]
   low  150   40     0.5408     0.5407    0.0001  [ -0.0000,  0.0004]
  high  150    5     0.1712     0.1714   -0.0002  [ -0.0004,  0.0000]
  high  150   20     0.3801     0.3800    0.0001  [ -0.0001,  0.0002]
  high  150   40     0.4097     0.4097    0.0000  [ -0.0001,  0.0002]
(* = 95% CI excludes 0.  Negative dvCRPS => collapse hurts in velocity.)

--- PIT RANK HISTOGRAMS (10 bins, uniform = calibrated) ---
Flat histogram => well-calibrated. U-shape => under-dispersed.
   low h= 5  mix_coupled px: [0.09 0.09 0.11 0.09 0.11 0.11 0.09 0.09 0.10 0.11]
   low h= 5  mix_coupled py: [0.29 0.10 0.05 0.07 0.03 0.04 0.07 0.07 0.06 0.23]
   low h= 5  mix_coupled vx: [0.14 0.10 0.08 0.09 0.12 0.10 0.07 0.05 0.11 0.14]
   low h= 5  mix_coupled vy: [0.35 0.05 0.05 0.06 0.05 0.03 0.03 0.03 0.07 0.27]
   low h= 5     collapse px: [0.08 0.10 0.12 0.09 0.11 0.12 0.08 0.10 0.09 0.11]
   low h= 5     collapse py: [0.29 0.09 0.05 0.09 0.03 0.05 0.05 0.08 0.05 0.23]
   low h= 5     collapse vx: [0.15 0.09 0.08 0.09 0.12 0.10 0.07 0.05 0.11 0.14]
   low h= 5     collapse vy: [0.35 0.05 0.05 0.06 0.05 0.03 0.03 0.03 0.07 0.27]
   low h=20  mix_coupled px: [0.33 0.06 0.05 0.03 0.02 0.04 0.07 0.04 0.07 0.30]
   low h=20  mix_coupled py: [0.31 0.07 0.07 0.07 0.04 0.03 0.05 0.04 0.04 0.28]
   low h=20  mix_coupled vx: [0.31 0.08 0.07 0.02 0.03 0.03 0.04 0.05 0.03 0.35]
   low h=20  mix_coupled vy: [0.27 0.07 0.08 0.07 0.02 0.03 0.09 0.05 0.09 0.24]
   low h=20     collapse px: [0.33 0.05 0.05 0.03 0.01 0.05 0.07 0.04 0.07 0.30]
   low h=20     collapse py: [0.31 0.07 0.07 0.07 0.04 0.03 0.05 0.04 0.04 0.28]
   low h=20     collapse vx: [0.31 0.07 0.08 0.02 0.03 0.03 0.04 0.05 0.03 0.34]
   low h=20     collapse vy: [0.27 0.07 0.06 0.07 0.03 0.03 0.09 0.05 0.08 0.25]
   low h=40  mix_coupled px: [0.31 0.08 0.05 0.04 0.04 0.02 0.04 0.05 0.07 0.31]
   low h=40  mix_coupled py: [0.31 0.07 0.07 0.05 0.04 0.05 0.07 0.07 0.05 0.21]
   low h=40  mix_coupled vx: [0.27 0.07 0.07 0.08 0.07 0.03 0.07 0.07 0.08 0.19]
   low h=40  mix_coupled vy: [0.32 0.06 0.05 0.05 0.05 0.05 0.07 0.05 0.04 0.25]
   low h=40     collapse px: [0.31 0.08 0.05 0.04 0.04 0.02 0.04 0.05 0.07 0.31]
   low h=40     collapse py: [0.31 0.07 0.09 0.05 0.03 0.06 0.07 0.07 0.05 0.21]
   low h=40     collapse vx: [0.27 0.07 0.07 0.08 0.07 0.03 0.07 0.07 0.08 0.19]
   low h=40     collapse vy: [0.32 0.06 0.05 0.06 0.05 0.05 0.07 0.05 0.04 0.25]
  high h= 5  mix_coupled px: [0.12 0.07 0.14 0.08 0.07 0.13 0.08 0.14 0.10 0.07]
  high h= 5  mix_coupled py: [0.24 0.09 0.08 0.05 0.04 0.07 0.11 0.07 0.07 0.19]
  high h= 5  mix_coupled vx: [0.15 0.09 0.05 0.07 0.15 0.11 0.07 0.09 0.11 0.12]
  high h= 5  mix_coupled vy: [0.28 0.09 0.07 0.02 0.07 0.04 0.09 0.08 0.04 0.23]
  high h= 5     collapse px: [0.11 0.08 0.15 0.08 0.07 0.11 0.09 0.15 0.09 0.07]
  high h= 5     collapse py: [0.23 0.09 0.10 0.04 0.05 0.06 0.11 0.08 0.05 0.19]
  high h= 5     collapse vx: [0.14 0.10 0.05 0.07 0.15 0.09 0.09 0.09 0.11 0.12]
  high h= 5     collapse vy: [0.27 0.09 0.07 0.03 0.05 0.05 0.10 0.07 0.05 0.22]
  high h=20  mix_coupled px: [0.23 0.09 0.07 0.07 0.03 0.04 0.03 0.07 0.09 0.28]
  high h=20  mix_coupled py: [0.31 0.05 0.09 0.03 0.06 0.07 0.07 0.07 0.05 0.21]
  high h=20  mix_coupled vx: [0.25 0.09 0.07 0.04 0.03 0.03 0.03 0.07 0.08 0.31]
  high h=20  mix_coupled vy: [0.30 0.11 0.08 0.06 0.05 0.07 0.05 0.07 0.05 0.15]
  high h=20     collapse px: [0.24 0.09 0.06 0.07 0.03 0.04 0.03 0.07 0.08 0.28]
  high h=20     collapse py: [0.31 0.05 0.09 0.02 0.06 0.07 0.08 0.07 0.04 0.21]
  high h=20     collapse vx: [0.25 0.09 0.08 0.03 0.03 0.03 0.03 0.06 0.09 0.30]
  high h=20     collapse vy: [0.30 0.11 0.08 0.06 0.05 0.07 0.05 0.08 0.05 0.15]
  high h=40  mix_coupled px: [0.21 0.07 0.08 0.04 0.06 0.07 0.07 0.07 0.05 0.27]
  high h=40  mix_coupled py: [0.27 0.13 0.09 0.06 0.04 0.05 0.05 0.07 0.05 0.19]
  high h=40  mix_coupled vx: [0.15 0.04 0.11 0.05 0.06 0.05 0.09 0.10 0.09 0.25]
  high h=40  mix_coupled vy: [0.29 0.13 0.10 0.11 0.05 0.02 0.05 0.03 0.05 0.16]
  high h=40     collapse px: [0.20 0.09 0.08 0.04 0.06 0.07 0.05 0.08 0.06 0.27]
  high h=40     collapse py: [0.26 0.13 0.10 0.07 0.04 0.05 0.05 0.07 0.05 0.19]
  high h=40     collapse vx: [0.15 0.04 0.11 0.05 0.07 0.05 0.09 0.11 0.09 0.25]
  high h=40     collapse vy: [0.29 0.13 0.10 0.11 0.05 0.02 0.05 0.03 0.05 0.16]

[gp_blind done in 3459s]

"""

