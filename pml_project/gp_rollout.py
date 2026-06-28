"""
GP propagator -- mode-AWARE version

We fit one SVGP PER MODE and, at rollout, each particle uses its OWN mode's GP.
Giving the GP the mode makes this propagation similar to the oracle with a small
learning error: this is used for a robustness check, does not answer any other issue

Position is the deterministic intergal of the velocity (p += v), so ONLY the 
velocity map is learned. For each mode m and component d in {x,y} we fit an SVGP:
                phi = (vx, vy, sinn( omega * t), cos (omega *t))
them the next velocity is determined on simulated transitions WITH KNOWN modes.
The time feature let the forced mode recover its forcing.

At rollout each particle gets one coherent Matheron function drawn from its current mode's GP,
plus iid noise. Same CRN structure as for the oracle.

Note: the SVGP likelihood variance was unreliable (it was inflated), but the posterior
mean was accurate. So the rollout noise comes from the DATA-DRIVEN residual std (estimate_noise).
We used RBF kernel because BoTorch do not have native LinearKernel

"""
import numpy as np
import torch
import gpytorch
from botorch.models import SingleTaskVariationalGP
from botorch.sampling.pathwise import draw_matheron_paths
from botorch.sampling.pathwise.prior_samplers import draw_kernel_feature_paths
import slds

torch.set_default_dtype(torch.double)
DIMS = 2
MODES = 3


def _features(vel, t):
    """vel (...,2), scalar/array t -> (...,4) features."""
    s, c = np.sin(slds.OMEGA * t), np.cos(slds.OMEGA * t)
    n = vel.shape[0]
    return np.column_stack([vel[:, 0], vel[:, 1],
                            np.full(n, s), np.full(n, c)])


"""
Build PER-MODE training sets from the TRUE simulator. The mode is known here
and is used to GROUP the data by mode, this grouping is exactly what makes
the propagator mode-aware. For each transition t -> t+1 record
    - input  phi_t = (vx_t, vy_t, sin(omega t), cos(omega t))
    - target v_{t+1}                          
filed under m = modes[t]. 
"""

def make_training_data(K=300, T=200, per_mode=5000, seed=1):
    rng = np.random.default_rng(seed)
    feats = {m: [] for m in range(MODES)}
    targs = {m: [] for m in range(MODES)}
    for _ in range(K):
        modes, x = slds.simulate(T, rng)
        for t in range(T):
            m = modes[t]
            phi = np.array([x[t, 2], x[t, 3],
                            np.sin(slds.OMEGA * t), np.cos(slds.OMEGA * t)])
            feats[m].append(phi)
            targs[m].append(x[t + 1, 2:])          # next velocity (2,)
    data = {}
    for m in range(MODES):
        F = np.array(feats[m]); Y = np.array(targs[m])
        idx = rng.permutation(len(F))[:per_mode]
        data[m] = (F[idx], Y[idx])
    return data

"""
Fit ONE SVGP for a single (mode, component)

"""
def fit_gp(F, y, M=64, steps=700):
    Ft = torch.as_tensor(F); yt = torch.as_tensor(y).unsqueeze(-1)
    ind = Ft[torch.randperm(Ft.shape[0])[:M]].clone()
    # RBF (pathwise-compatible in BoTorch). Its posterior MEAN fits the
    # dynamics; the unreliable likelihood noise is replaced downstream by a
    # data-driven residual estimate (estimate_noise). NOTE BoTorch pathwise
    # does not support a LinearKernel, which would otherwise be ideal here.
    model = SingleTaskVariationalGP(
        Ft, yt, inducing_points=ind,
        covar_module=gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=4)))
    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model.model,
                                        num_data=yt.shape[0])
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = -mll(model(Ft), yt.squeeze(-1))
        loss.backward(); opt.step()
    model.eval()
    return model


"""
Fit ALL 6 GPs: 
one per (mode in {0,1,2}) x (component in {x,y}). This 6-model structure is the
mode-aware design: the mode is supplied to the GP by SELECTING the right model.
"""
def fit_all(data):
    gps = {}
    for m in range(MODES):
        F, Y = data[m]
        for d in range(DIMS):
            gps[(m, d)] = fit_gp(F, Y[:, d])
    return gps


def validate_fit(gps, data):
    """Check GP posterior mean recovers the known velocity map on a grid."""
    print("GP fit validation (RMSE of posterior mean vs true next-velocity map):")
    rng = np.random.default_rng(9)
    vgrid = rng.uniform(-2, 2, size=(400, 2))
    tgrid = rng.integers(0, 200, size=400)
    for m in range(MODES):
        phi = _features(vgrid, 0)
        phi[:, 2] = np.sin(slds.OMEGA * tgrid)
        phi[:, 3] = np.cos(slds.OMEGA * tgrid)
        Pt = torch.as_tensor(phi)
        rms = []
        for d in range(DIMS):
            with torch.no_grad():
                pred = gps[(m, d)].posterior(Pt).mean.squeeze(-1).numpy()
            c = (slds.A_F * (np.sin if d == 0 else np.cos)(slds.OMEGA * tgrid)
                 if m == 2 else 0.0)
            true = slds.F_DIAG[m] * vgrid[:, d] + c
            rms.append(np.sqrt(np.mean((pred - true) ** 2)))
        print(f"  mode {m+1}: RMSE dim-x {rms[0]:.4f}  dim-y {rms[1]:.4f}  "
              f"(velocity scale ~{np.abs(vgrid).mean():.2f})")
    print("Data-driven noise (residual std) vs true process-noise sigma_m:")
    est = estimate_noise(gps, data)
    for m in range(MODES):
        e = np.mean([est[(m, d)] for d in range(DIMS)])
        print(f"  mode {m+1}: residual-std ~{e:.4f}   true {slds.SIGMA[m]:.4f}")


"""
The rollout's process noise, estimated FROM DATA: 
per (mode, component), the std of (true next velocity - GP posterior mean) over the training set. 
Used INSTEAD of the SVGP likelihood variance, which variational training leaves
inflated. Because the posterior mean is accurate, this residual std recovers the true sigma_m.
"""

def estimate_noise(gps, data):
    est = {}
    for m in range(MODES):
        F, Y = data[m]
        Ft = torch.as_tensor(F)
        for d in range(DIMS):
            with torch.no_grad():
                mean = gps[(m, d)].posterior(Ft).mean.squeeze(-1).numpy()
            est[(m, d)] = float(np.std(Y[:, d] - mean))
    return est


def _rff(model, sample_shape, num_rff=4096):
    return draw_kernel_feature_paths(model, sample_shape=sample_shape,
                                     num_features=num_rff)

def draw_paths(gps, N, seed, noise_est):
    torch.manual_seed(seed)
    paths = {}
    for key, model in gps.items():
        paths[key] = draw_matheron_paths(
            model, sample_shape=torch.Size([N]),
            prior_sampler=lambda model, sample_shape: _rff(model, sample_shape, 4096))
    return paths, noise_est


"""
Same structure as oracle_rollout, with ONE change: the velocity update is the
LEARNED GP map instead of the true dynamics. Per step:
  - p += v                      
  - build phi from (v, absolute time t)
  - for each component d: evaluate EVERY mode's path at all particles, then
    KEEP, per particle, the draw from ITS current mode (sel = modes == m),
    and add fresh iid noise[(m,d)] 
  - sample the next mode from the TRUE chain PI (the mode is still tracked and
    evolved. This per-mode dispatch is what keeps it mode-aware)

    
"""

def gp_rollout(x, modes, T, horizons, gps, paths, noise, rng):
    N = x.shape[0]
    hset, out = set(horizons), {}
    for step in range(max(horizons)):
        t = T + step
        pos, vel = x[:, :2], x[:, 2:]
        pos = pos + vel
        phi = _features(vel, t)
        Phi = torch.as_tensor(phi).unsqueeze(-2)        # (N,1,4) ensemble eval
        vnew = np.empty_like(vel)
        for d in range(DIMS):
            cand = np.empty(N)
            for m in range(MODES):
                with torch.no_grad():
                    g = paths[(m, d)](Phi).squeeze(-1).numpy()   # path i @ input i
                sel = modes == m
                cand[sel] = g[sel] + noise[(m, d)] * rng.standard_normal(sel.sum())
            vnew[:, d] = cand
        x = np.concatenate([pos, vnew], axis=1)
        Prow = slds.PI[modes]
        u = rng.random(N)
        modes = (u[:, None] > np.cumsum(Prow, axis=1)).sum(1)
        h = step + 1
        if h in hset:
            out[h] = x[:, :2].copy()
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    print("Training per-mode velocity GPs...")
    data = make_training_data()
    gps = fit_all(data)
    print(f"  trained {len(gps)} SVGPs in {time.time()-t0:.0f}s\n")
    validate_fit(gps, data)
    print(f"\ntotal {time.time()-t0:.0f}s")
