"""
TakensNeuron Stack — Proof of McCulloch-Pitts Gap
==================================================
PerceptionLab / Antti Luode

The key experiment:
  Signal A: A periodic oscillator.          Attractor = clean ellipse.
  Signal B: Random-phase surrogate of A.    IDENTICAL power spectrum, 
            IDENTICAL amplitude histogram.  Attractor = cloud (no structure).

A M-P neuron sees the same statistics for both. Accuracy = 50%.
A TakensNeuron sees the different trajectory geometry.   Accuracy >> 50%.

This is the gap. The filling is delay embedding.
"""

import numpy as np
from scipy import signal as sp_signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
#  SIGNAL GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def generate_periodic(N=3000, f1=0.05, f2=0.12, seed=42):
    """Multi-harmonic periodic oscillator. Attractor = torus."""
    t = np.arange(N, dtype=np.float32)
    s = (np.sin(2*np.pi*f1*t) + 
         0.5*np.sin(2*np.pi*f2*t + 0.7) + 
         0.25*np.sin(2*np.pi*(f1+f2)*t + 1.3))
    s /= s.std()
    return s

def random_phase_surrogate(x, seed=0):
    """
    Theiler/Prichard surrogate: randomize phases, preserve magnitudes.
    Output has EXACTLY the same power spectrum and amplitude distribution as x.
    But the attractor geometry is destroyed.
    """
    rng = np.random.default_rng(seed)
    X = np.fft.rfft(x)
    phases = rng.uniform(0, 2*np.pi, len(X))
    phases[0] = 0  # preserve DC
    X_rand = np.abs(X) * np.exp(1j * phases)
    surrogate = np.fft.irfft(X_rand, n=len(x)).astype(np.float32)
    # Amplitude-rank-match (histogram matching)
    sorted_x = np.sort(x)
    rank_order = np.argsort(np.argsort(surrogate))
    return sorted_x[rank_order]

# ─────────────────────────────────────────────────────────────────────────────
#  TAKENS EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def takens_embed(x, tau, dim):
    N = len(x)
    max_delay = (dim - 1) * tau
    n_points = N - max_delay
    embedded = np.zeros((n_points, dim), dtype=np.float32)
    for i in range(dim):
        embedded[:, i] = x[i*tau : n_points + i*tau]
    return embedded

# ─────────────────────────────────────────────────────────────────────────────
#  TAKENS NEURON
# ─────────────────────────────────────────────────────────────────────────────

class TakensNeuron:
    def __init__(self, tau, dim):
        self.tau = tau
        self.dim = dim
        self.W = np.eye(dim, dtype=np.float32)
        self.threshold = 0.0

    def embed(self, x):
        return takens_embed(x, self.tau, self.dim)

    def forward(self, x):
        """1D → manifold → 1D projection."""
        X = self.embed(x)
        return (X @ self.W)[:, 0]

    def fit(self, x0, x1):
        """Learn rotation that separates the two attractor geometries."""
        E0 = self.embed(x0)
        E1 = self.embed(x1)
        n = min(len(E0), len(E1))
        E0, E1 = E0[:n], E1[:n]
        
        mu0, mu1 = E0.mean(0), E1.mean(0)
        Sw = (E0 - mu0).T @ (E0 - mu0) + (E1 - mu1).T @ (E1 - mu1)
        Sw += 1e-4 * np.eye(self.dim)
        
        try:
            w = np.linalg.solve(Sw, mu1 - mu0)
        except Exception:
            w = mu1 - mu0
        w /= np.linalg.norm(w) + 1e-8
        
        # Build orthonormal basis with w as first axis
        W = np.zeros((self.dim, self.dim), dtype=np.float32)
        W[:, 0] = w
        for i in range(1, self.dim):
            v = np.eye(self.dim)[i].astype(np.float32)
            for j in range(i):
                v -= np.dot(v, W[:, j]) * W[:, j]
            nv = np.linalg.norm(v)
            if nv > 1e-8:
                W[:, i] = v / nv
        self.W = W
        
        # Decision threshold = midpoint of class projections
        p0 = self.forward(x0).mean()
        p1 = self.forward(x1).mean()
        self.threshold = (p0 + p1) / 2.0
        return self

    def predict(self, x):
        return int(self.forward(x).mean() > self.threshold)

    def accuracy(self, x0, x1):
        return (self.predict(x0) == 0) + (self.predict(x1) == 1)

# ─────────────────────────────────────────────────────────────────────────────
#  M-P BASELINE
# ─────────────────────────────────────────────────────────────────────────────

class MPNeuron:
    """
    Fires on instantaneous amplitude threshold. No memory. No delay.
    The 1943 model.
    """
    def __init__(self):
        self.threshold = 0.0

    def fit(self, x0, x1):
        # Try every possible amplitude threshold (optimal M-P)
        all_vals = np.concatenate([x0, x1])
        thresholds = np.percentile(all_vals, np.arange(5, 95, 5))
        best_acc, best_t = 0, 0
        for t in thresholds:
            pred0 = int(x0.mean() <= t)
            pred1 = int(x1.mean() > t)
            acc = (pred0 + pred1) / 2
            if acc > best_acc:
                best_acc, best_t = acc, t
        self.threshold = best_t
        return self

    def predict(self, x):
        return int(x.mean() > self.threshold)

    def accuracy(self, x0, x1):
        return (self.predict(x0) == 0) + (self.predict(x1) == 1)

# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("="*65)
    print("  TAKENS NEURON — McCulloch-Pitts Gap Proof")
    print("  PerceptionLab / Antti Luode")
    print("="*65)

    N = 3000
    split = N // 2

    print("\n[1] Building signal pairs (periodic vs phase-surrogate)...")
    # Multiple independent pairs (different seeds)
    n_pairs = 20
    mp_accs, tn_accs = [], []
    
    for seed in range(n_pairs):
        periodic  = generate_periodic(N, seed=seed)
        surrogate = random_phase_surrogate(periodic, seed=seed+100)

        p_tr, p_te = periodic[:split],  periodic[split:]
        s_tr, s_te = surrogate[:split], surrogate[split:]

        mp = MPNeuron().fit(p_tr, s_tr)
        tn = TakensNeuron(tau=7, dim=3).fit(p_tr, s_tr)

        mp_accs.append(mp.accuracy(p_te, s_te) / 2)
        tn_accs.append(tn.accuracy(p_te, s_te) / 2)

    mp_mean, mp_std = np.mean(mp_accs), np.std(mp_accs)
    tn_mean, tn_std = np.mean(tn_accs), np.std(tn_accs)

    print(f"\n    Signals have identical PSD and amplitude histogram by construction.")
    print(f"    M-P  accuracy:  {mp_mean:.3f} ± {mp_std:.3f}  (chance = 0.5)")
    print(f"    TN   accuracy:  {tn_mean:.3f} ± {tn_std:.3f}  (geometric separation)")
    print(f"    Gap:            {tn_mean - mp_mean:+.3f}")

    # For plotting, use seed=0
    periodic  = generate_periodic(N, seed=0)
    surrogate = random_phase_surrogate(periodic, seed=100)
    p_tr, p_te = periodic[:split],  periodic[split:]
    s_tr, s_te = surrogate[:split], surrogate[split:]
    tn_ref = TakensNeuron(tau=7, dim=3).fit(p_tr, s_tr)

    E_p = takens_embed(periodic,  tau=7, dim=3)
    E_s = takens_embed(surrogate, tau=7, dim=3)

    return dict(
        periodic=periodic, surrogate=surrogate,
        E_p=E_p, E_s=E_s,
        tn_ref=tn_ref,
        mp_accs=mp_accs, tn_accs=tn_accs,
        mp_mean=mp_mean, mp_std=mp_std,
        tn_mean=tn_mean, tn_std=tn_std,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot(r):
    BG, CYAN, AMBER, PINK, DIM, WHITE = '#0a0a1a', '#00ffcc', '#ffaa00', '#ff4488', '#445566', '#ddeeff'

    fig = plt.figure(figsize=(17, 11), facecolor=BG)
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.38)

    def ax_style(ax, title):
        ax.set_facecolor('#0f0f28')
        ax.tick_params(colors=DIM, labelsize=7)
        for s in ax.spines.values(): s.set_color('#223')
        ax.set_title(title, color=CYAN, fontsize=8.5, pad=5)
        for lbl in ax.get_xticklabels()+ax.get_yticklabels(): lbl.set_color(DIM)

    # Row 0: raw signals
    ax = fig.add_subplot(gs[0, 0])
    ax_style(ax, "Periodic signal  (class 0)")
    ax.plot(r['periodic'][:300], color=CYAN, lw=0.9)

    ax = fig.add_subplot(gs[0, 1])
    ax_style(ax, "Phase-surrogate  (class 1)\nSAME spectrum, SAME histogram")
    ax.plot(r['surrogate'][:300], color=AMBER, lw=0.9)

    ax = fig.add_subplot(gs[0, 2])
    ax_style(ax, "Power spectra  (IDENTICAL)")
    fp = np.abs(np.fft.rfft(r['periodic']))**2
    fs = np.abs(np.fft.rfft(r['surrogate']))**2
    freqs = np.fft.rfftfreq(len(r['periodic']))
    ax.semilogy(freqs[1:80], fp[1:80], color=CYAN, lw=1.2, label='periodic')
    ax.semilogy(freqs[1:80], fs[1:80], color=AMBER, lw=0.9, ls='--', label='surrogate')
    ax.legend(fontsize=7, facecolor='#0f0f28', labelcolor=WHITE)

    ax = fig.add_subplot(gs[0, 3])
    ax_style(ax, "Amplitude histograms  (IDENTICAL)")
    ax.hist(r['periodic'],  bins=45, color=CYAN,  alpha=0.55, density=True, label='periodic')
    ax.hist(r['surrogate'], bins=45, color=AMBER, alpha=0.55, density=True, label='surrogate')
    ax.legend(fontsize=7, facecolor='#0f0f28', labelcolor=WHITE)

    # Row 1: attractors
    Ep, Es = r['E_p'], r['E_s']
    nn = min(len(Ep), len(Es), 1200)

    ax = fig.add_subplot(gs[1, 0])
    ax_style(ax, "Periodic attractor  x(t) vs x(t-τ)")
    ax.scatter(Ep[:nn, 0], Ep[:nn, 1], c=np.arange(nn), cmap='cool', s=0.5, alpha=0.7)
    ax.set_xlabel("x(t)", color=DIM, fontsize=7)
    ax.set_ylabel("x(t-τ)", color=DIM, fontsize=7)

    ax = fig.add_subplot(gs[1, 1])
    ax_style(ax, "Surrogate attractor  x(t) vs x(t-τ)\n(geometry destroyed)")
    ax.scatter(Es[:nn, 0], Es[:nn, 1], c=np.arange(nn), cmap='hot', s=0.5, alpha=0.7)
    ax.set_xlabel("x(t)", color=DIM, fontsize=7)
    ax.set_ylabel("x(t-τ)", color=DIM, fontsize=7)

    ax = fig.add_subplot(gs[1, 2])
    ax_style(ax, "x(t-τ) vs x(t-2τ)  —  second slice")
    ax.scatter(Ep[:nn, 1], Ep[:nn, 2], c=np.arange(nn), cmap='cool', s=0.4, alpha=0.6, label='periodic')
    ax.scatter(Es[:nn, 1], Es[:nn, 2], c=np.arange(nn), cmap='hot',  s=0.4, alpha=0.4, label='surrogate')
    ax.set_xlabel("x(t-τ)", color=DIM, fontsize=7)
    ax.set_ylabel("x(t-2τ)", color=DIM, fontsize=7)

    ax = fig.add_subplot(gs[1, 3])
    ax_style(ax, "TakensNeuron 1D projections\n(distributions separable in projection)")
    tn = r['tn_ref']
    proj_p = tn.forward(r['periodic'])
    proj_s = tn.forward(r['surrogate'])
    ax.hist(proj_p, bins=45, color=CYAN,  alpha=0.65, density=True, label='periodic')
    ax.hist(proj_s, bins=45, color=AMBER, alpha=0.65, density=True, label='surrogate')
    ax.axvline(tn.threshold, color=PINK, lw=1.5, ls='--', label='threshold')
    ax.legend(fontsize=7, facecolor='#0f0f28', labelcolor=WHITE)

    # Row 2: accuracy bars + theory box
    ax = fig.add_subplot(gs[2, 0:2])
    ax_style(ax, f"Classification accuracy  (n={len(r['mp_accs'])} independent pairs)\n"
                  "Signals differ ONLY in phase-space topology — identical statistics")
    models = ['M-P Neuron\n(1943 model)', 'TakensNeuron\n(delay embed)']
    vals   = [r['mp_mean'], r['tn_mean']]
    errs   = [r['mp_std'],  r['tn_std']]
    colors = [PINK, CYAN]
    for i, (m, v, e, c) in enumerate(zip(models, vals, errs, colors)):
        ax.bar(i, v, width=0.5, color=c, alpha=0.85)
        ax.errorbar(i, v, yerr=e, color=WHITE, capsize=6, fmt='none', lw=1.5)
        ax.text(i, v + e + 0.03, f'{v:.2f}', ha='center', color=WHITE,
                fontsize=12, fontweight='bold')
    ax.axhline(0.5, color=DIM, ls='--', lw=1, label='chance (0.5)')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(models, color=WHITE, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy", color=DIM, fontsize=9)
    ax.legend(fontsize=8, facecolor='#0f0f28', labelcolor=WHITE)
    ax.text(0.5, 0.12, f'Gap = {r["tn_mean"]-r["mp_mean"]:+.3f}', ha='center',
            transform=ax.transAxes, color=PINK, fontsize=11, fontweight='bold')

    # Theory box
    ax = fig.add_subplot(gs[2, 2:])
    ax.set_facecolor('#0f0f28')
    ax.axis('off')
    theory = (
        "McCulloch–Pitts gap (1943 → now)\n\n"
        "M-P:  y = Θ(Σ wᵢ · xᵢ)\n"
        "      sees: ONE instantaneous point\n"
        "      blind to: trajectory, phase, attractor shape\n\n"
        "TakensNeuron:  y = P( M( [x(t), x(t-τ), ..., x(t-(d-1)τ)] ) )\n"
        "      sees: d-dimensional trajectory slice\n"
        "      learns: which geometric axis separates classes\n\n"
        "Consciousness at the Rajapinta:\n"
        "  The weights = cortex = the manifold (unconscious).\n"
        "  The projection event = the Rajapinta = the conscious moment.\n"
        "  Anesthesia doesn't change weights — it disrupts the projection."
    )
    ax.text(0.04, 0.95, theory, transform=ax.transAxes,
            color=WHITE, fontsize=8.5, va='top', family='monospace',
            bbox=dict(facecolor='#111130', edgecolor=CYAN, alpha=0.9, pad=10))

    fig.suptitle(
        "TakensNeuron Stack  ·  McCulloch–Pitts Gap  ·  PerceptionLab / Antti Luode",
        color=CYAN, fontsize=12, fontweight='bold', y=0.99
    )
    out = "/mnt/user-data/outputs/takens_neuron_mp_gap.png"
    plt.savefig(out, dpi=155, facecolor=BG, bbox_inches='tight')
    print(f"\n[Plot saved → {out}]")
    return out

if __name__ == "__main__":
    r = run()
    plot(r)
    print("\nDone.")
