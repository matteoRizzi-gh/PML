"""
Switching linear-Gaussian state-space model + IMM filter.

Used by both the separation diagnostic and (later) the full pipeline. 
The IMM is standard machinery (Blom &Bar-Shalom 1988) via filterpy;
the only custom part is per-mode control,
because mode 3 is forced in absolute time and modes 1-2 are not.

State x = (px, py, vx, vy). Observation y = H x + r, position only.
Dynamics (mode m governs the t -> t+1 transition):
    p_{t+1} = p_t + dt * v_t
    v_{t+1} = F_m v_t + c_m(t) + w_t,   w_t ~ N(0, Sigma_m)

Modes:
    - 0-mode: almost-constant velocity, F=I with 0.02 noise
    - 1-mode: damped, F= 0.9 * I, noise 0.05
    - 2-mode: forced, F= I (+ forced oscillations), noise = 0.05

PI: mode transition matrix, 
    0.95 on the diagonal (mode preserval)
    0.025 outside (20 steps permanency)
H_OBS selects the positioning, we only observe this quantity

Explaination for 20 steps:
Assume we are at a certain node. At each step we have a proabbility of
remaining at that node of 0.95, hence a probability of mode switching of 
1 - 0.95 = 0.05. Exists are independent Bernulli trials at each step.
The dwell time (time spent on each mode) is a geometrical variable:

                    P(escaping at step k) = (0.95)^(k-1) * 0.05

                    E[permanence] = 1/(1 - 0.95) = 1/0.05= 20

We have two modes different than the one we are at so instaed og 0.05 we have 0.025

To modify the permanence we should modify the 0.95 diagonal

Rk. 20 is a mean for independent Bernulli trials (memoryless),
    so we could have few permanence steps as a lot

    
What the A-C contrast measures, and what the horizon grid brackets:

DESIGN FACT (common random numbers). Arms A and C share the initial component
label j, the mode-transition uniforms and the process-noise normals. Their mode
trajectories are therefore IDENTICAL and the process noise CANCELS in the paired
difference A - C. Consequence: the mode-chain mixing time does NOT drive the
collapse cost -- both arms forget the initial mode in lockstep, symmetrically,
leaving no trace in A - C. The only thing the two arms differ in is the INITIAL
STATE: A samples it from the mode-conditional (xhat^j, P^j), C from the
collapsed moment-matched (xbar, Pbar).

STRUCTURAL FACT (where the modes differ). By construction the three modes differ
ONLY in their velocity dynamics (F_m and Sigma_m act on velocity; position is a
deterministic integrator p += v, and H observes position only). Hence the
mode-conditional posteriors separate primarily in their VELOCITY components: the
position is directly observed and thus pinned down by the data regardless of the
mode hypothesis, while the velocity is latent and mode-dependent. Any separation
in position is second-order, produced only by integrating velocity differences
over the filtering history. The A-C contrast is therefore driven by the
propagation of an INITIAL VELOCITY OFFSET -- the velocity that C mis-specifies by
averaging over modes instead of conditioning on the mode the particle is then
propagated under.

WHICH TIMESCALES THE GRID MUST BRACKET. The velocity offset propagates under the
shared dynamics, so the horizon profile of the POSITION cost is set by the
dynamical scales of the propagator, not by the posterior:
  - the offset integrates into position (p += v each step), so its effect on
    position ACCUMULATES with horizon;
  - it is damped during mode-2 sojourns (f = 0.9, characteristic decay
    -1/ln 0.9 ~ 9.5 steps; effectively longer once averaged over the fraction of
    time a trajectory spends in mode 2);
  - the forcing acts with period 2*pi/omega = 40 steps.
The grid H in {5, 20, 40} brackets these scales: H=5 is well inside the
velocity-damping scale (offset still largely intact), H=20 is around the mean
dwell time and a couple of damping constants in, H=40 is one full forcing
period. This resolves the position-cost profile across the relevant dynamical
scales WITHOUT presupposing its shape.

(For completeness, the mode chain has its own characteristic scales, which the
grid also brackets but which do NOT drive the paired contrast:
    PI = 0.925 * I + 0.025 * J,  J all-ones with eigenvalues (3,0,0),
 so PI has eigenvalues 1 (stationary) and 0.925 (double); mean dwell time
 1/(1-0.95) = 20, mode-marginal relaxation time -1/ln(0.925) ~ 13. These
 describe one chain's marginal, not the decay of the A-C cost.)
"""


import numpy as np

# Fixed benchmark parameters

DT = 1.0
N_MODES = 3
F_DIAG = np.array([1.0, 0.9, 1.0])          # F_m = f_m * I_2
SIGMA = np.array([0.02, 0.05, 0.05])        # Sigma_m = sigma_m^2 * I_2
A_F = 0.15                                  # forcing amplitude (mode 3)
OMEGA = 2.0 * np.pi / 40.0                  # forcing angular freq
PI_DIAG, PI_OFF = 0.95, 0.025               # transition matrix
H_OBS = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0]])

PI = np.full((N_MODES, N_MODES), PI_OFF)
np.fill_diagonal(PI, PI_DIAG)               # rows sum to 0.95 + 2*0.025 = 1


"""
Generative model: position is a deterministic integrator of the velocity
_A(mode): given x=(px, py, vx, vy), state transition matrix. 

p_{t+1} = p_t + v_t 
v_{t+1} = f_m * v_t

_Q: process noise, it enters only in the velocity

"""
def _A(mode):
    """State-transition matrix for a given mode"""
    f = F_DIAG[mode]
    return np.array([[1.0, 0, DT, 0],
                     [0, 1.0, 0, DT],
                     [0, 0, f, 0],
                     [0, 0, 0, f]])


def _Q(mode):
    """Process-noise covariance"""
    s2 = SIGMA[mode] ** 2
    return np.diag([0.0, 0.0, s2, s2])

"""
Forcing of the 2-mode: a_f(sin(w*t); cos(w*t))


clearly it depends on t and not the horizon. 
This is fundamental to allineate times in the filter and in the rollout
"""
def forcing_u(t):
    """forcing vector"""
    return A_F * np.array([np.sin(OMEGA * t), np.cos(OMEGA * t)])



# Simulator

"""
Generate a real sequence:
    - Initial state: position 0, velocity drawn from N(0, (0.5)^2)
    - Cycle: for each t we take the current m = modes[t]
             and we evaluate x_{t+1} = A_m * x_t + c + w (c is the forcing parameter for 2-mode; w = noise for velocity)
    - Sample the next mode from PI[m]

"""
def simulate(T, rng):
    """
    Generate one sequence of length T+1
    """
    modes = np.empty(T + 1, dtype=int)
    x = np.empty((T + 1, 4))
    modes[0] = rng.integers(N_MODES)
    v0 = rng.normal(0.0, 0.5, size=2)
    x[0] = np.array([0.0, 0.0, v0[0], v0[1]])

    for t in range(T):
        m = modes[t]
        c = np.zeros(4)
        if m == 2:                          # mode 3 (0-indexed -> 2) is forced
            c[2:] = forcing_u(t)
        w = np.zeros(4)
        w[2:] = rng.normal(0.0, SIGMA[m], size=2)
        x[t + 1] = _A(m) @ x[t] + c + w
        modes[t + 1] = rng.choice(N_MODES, p=PI[m])

    return modes, x


def observe(x, sigma_r, rng):
    """We add Gaussian noise to the position"""
    pos = x[:, :2]
    return pos + rng.normal(0.0, sigma_r, size=pos.shape)



# IMM filter 

"""
We build the IMM as 3 Kalman filter (one each mode), all of them with independent F; Q; H; R= (sigma_r)^2 * I

Note: filterpy passes the same control input u to each filter in predict(u).
      But forcing is only in the 2-mode.
      The solution was to give to each filter a different control matrix B

B_3 maps the 2D forcing vector to the velocity components
B_1 = B_2 = 0, so 0-mode and 1-mode ignore u.

"""


def _build_imm(sigma_r):
    from filterpy.kalman import IMMEstimator, KalmanFilter
    R = (sigma_r ** 2) * np.eye(2)
    # per-mode control transition: only mode 3 maps the 2-vector forcing
    # into the velocity components; modes 1-2 get B = 0 so u is ignored.
    B3 = np.array([[0.0, 0], [0, 0], [1.0, 0], [0, 1.0]])
    B0 = np.zeros((4, 2))
    filters = []
    for m in range(N_MODES):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = _A(m)
        kf.Q = _Q(m)
        kf.H = H_OBS.copy()
        kf.R = R.copy()
        kf.B = B3 if m == 2 else B0
        kf.x = np.zeros(4)
        kf.P = np.diag([1.0, 1.0, 0.5, 0.5])   
        filters.append(kf)
    mu0 = np.full(N_MODES, 1.0 / N_MODES)       # m0 ~ Uniform
    return IMMEstimator(filters, mu0, PI)

"""
We run IMM for a sequence.
    - Initialize with update(y[0])
    - Cycle:
            - predict(u = forcing u(t-1)) so that the transition t-1 -> t use forcing parameter at t-1
            - update(y[t])

    Returns, at the final time T:
        mu     : (3,)   posterior mode probs P(m_T = j | y_{1:T})
        x_hat  : (3, 4) mode-conditional posterior means
        P_hat  : (3, 4, 4) mode-conditional posterior covariances
        x_bar  : (4,)   moment-matched (collapsed) mean
        P_bar  : (4, 4) moment-matched (collapsed) covariance
    Also returns the full mode-probability trace mu_trace (T+1, 3).

IMM evaluated the collapsed as a combinatorial estimate:
x_bar = sum_j mu_j * (x_hat)^j
P_bar =sum_j mu_j * [P^j + ((x_hat)^j - x_bar) * (x_hat)^j - x_bar)^T]

"""
def run_imm(y, sigma_r):
    """
    Run the IMM over observations y 
    """
    imm = _build_imm(sigma_r)
    Tn = y.shape[0] - 1
    mu_trace = np.empty((Tn + 1, N_MODES))
    imm.update(y[0])
    mu_trace[0] = imm.mu.copy()

    for t in range(1, Tn + 1):
        imm.predict(u=forcing_u(t - 1))     # predict t-1 -> t uses forcing(t-1)
        imm.update(y[t])
        mu_trace[t] = imm.mu.copy()

    x_hat = np.array([f.x.copy() for f in imm.filters])
    P_hat = np.array([f.P.copy() for f in imm.filters])
    
    return imm.mu.copy(), x_hat, P_hat, imm.x.copy(), imm.P.copy(), mu_trace



def run_imm_multi(y, sigma_r, origins):
    """
    Same IMM run but it only passes once and it saves snapshots of the posterior
    """
    imm = _build_imm(sigma_r)
    want = sorted(int(o) for o in origins)
    out = {}
    imm.update(y[0])

    if want and want[0] == 0:
        out[0] = (imm.mu.copy(),
                  np.array([f.x.copy() for f in imm.filters]),
                  np.array([f.P.copy() for f in imm.filters]),
                  imm.x.copy(), imm.P.copy())
        
    Tmax = max(want)
    wset = set(want)

    for t in range(1, Tmax + 1):
        imm.predict(u=forcing_u(t - 1))
        imm.update(y[t])
        if t in wset:
            out[t] = (imm.mu.copy(),
                      np.array([f.x.copy() for f in imm.filters]),
                      np.array([f.P.copy() for f in imm.filters]),
                      imm.x.copy(), imm.P.copy())
            
    return out

