"""
Calibration of sigma_r (the only free parameter, which sets the DIFFICULTY of
the mode-inference problem).

Observations:
- Small observation noise:modes are easy to tell apart. We would see
  a near-degenerate posterior (one mode dominates, so low entropy)
- large obervation noise: modes are indistinguishable. We would see
  a diffuse posterior
We tune sigma_r to the interesting middle regime: modes identifiable, but
genuinely uncertain.

This file answer to the question: 'Is there a sigma_r such that our system
lands in the interesting regime (modes identifiable, but genuinely uncertain)?'
if no grid value landed in band, the system would not produce the
phenomenon and there would be nothing to study.

The method is:
Pre-registered grid, seed and rule fixed before experiment (no sweep).
We pick the grid sigma_r whose Terminal Mode Accuracy is closest to 0.85.
This value is called Goldilocks point:
    - near 1.0 the mode is almost known (no multimodality to collapse)
    - near 0.33 the mode posterior is approximately the uniform stationary law

The terminal mode accuracy is the fraction of pilot sequences where the 
most-probable posterior mode argmax_{x_j} (mu_j)^T (T is a temporal index not transposed) 
equals the true terminal mode m_T.
This is a hard mode-classification proxy for difficulty.
"""

import numpy as np
import time
import slds

GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
N_PILOT = 200
SEQ_LEN = 200
ORIGIN_LO, ORIGIN_HI = 60, 160   # past burn-in; matches test-origin window


"""
Terminal mode accuracy at a given sigma_r, over N_PILOT pilot sequences with a
fixed seed:

For each pilot sequence the forecast origin T is drawn UNIFORMLY in
[60,160]: 60 is past burn-in (the IMM has reached steady state and the prior P0
has washed out), and the window matches the test-origin window, so calibration
uses the same origin distribution as the real experiment. 
Run the IMM to T and count a hit when argmax mu_T equals the true mode m_T.
Returns the hit rate.
"""

def entropy(mu):
    mu = np.clip(mu, 1e-12, 1.0)
    return float(-(mu * np.log(mu)).sum())


def terminal_accuracy(sigma_r, seed):
    """Terminal mode accuracy AND fraction of origins with H(mu_T) >= 0.75
    (the 'high' stratum threshold), over N_PILOT pilot sequences."""
    rng = np.random.default_rng(seed)
    correct = 0
    n_high = 0
    for _ in range(N_PILOT):
        T = int(rng.integers(ORIGIN_LO, ORIGIN_HI + 1))
        modes, x = slds.simulate(T, rng)
        y = slds.observe(x, sigma_r, rng)
        mu, *_ = slds.run_imm(y, sigma_r)
        correct += int(mu.argmax() == modes[-1])
        n_high += int(entropy(mu) >= 0.75)
    return correct / N_PILOT, n_high / N_PILOT

if __name__ == "__main__":
    t0 = time.time()
    print(f"{'sigma_r':>8} {'term_acc':>9} {'frac_high':>10}")
    accs = {}
    for s in GRID:
        a, fh = terminal_accuracy(s, seed=12345)
        accs[s] = a
        print(f"{s:>8} {a:>9.3f} {fh:>10.3f}")
    print(f"elapsed {time.time()-t0:.1f}s")



"""

term_acc decresce monotòna con sigma_r, frac_high cresce monotòna. 
Sono anti-correlate per costruzione: più rumore → modi meno identificabili 
(accuracy giù) → posterior più diffuse → più entropia (frac_high su). 
Quindi "accuracy ≈ 0.85" e "tanta massa nell'high" sono obiettivi in 
conflitto diretto. Sotto 0.80 i modi non sono affidabilmente distinguibili 
e la posterior mode-conditional perde significato.
Il criterio "accuracy 0.85" non è il criterio giusto per questo studio.
Un criterio sano seleziona il valore che serve allo studio. 



Due vincoli, non un target:

- Vincolo di identificabilità: term_acc ≥ 0.80 (i modi devono essere 
    distinguibili, altrimenti non c'è una posterior multimodale ben 
    definita da collassare — è rumore, non multimodalità).
    Questo elimina 0.1, 0.2, 0.4, 0.8.
- Obiettivo dato il vincolo: massimizzare frac_high (massa sull'estimando).
    Tra i sopravvissuti {0.01: 0.055, 0.02: 0.100, 0.05: 0.140}, 
    il massimo è 0.05.

"""
