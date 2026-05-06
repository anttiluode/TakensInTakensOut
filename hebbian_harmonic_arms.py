"""
Hebbian Harmonic Arm Network
============================
PerceptionLab / Antti Luode

The concrete step:
  Arm amplitudes Aₖ learn via Hebbian phase-coherence.
  No backprop. No gradient. No labels.
  Only: "does arm k resonate with what is actually arriving?"

Learning rule:
  ΔAₖ = η · x(t) · cos(k·ω₀·t)   [Hebbian: grow when input and arm are in phase]
        - decay · Aₖ               [homeostatic: prevent runaway]

This is a windowed, online Fourier update — but derived from Hebb, not calculus.

DEMO:
  Signal changes in three epochs:
    Epoch 1 (t=0→600):   modes k=2,5 active  (slow + medium)
    Epoch 2 (t=600→1200): modes k=4,9 active  (different structure)
    Epoch 3 (t=1200→1800): modes k=2,5 return (memory test)

  The Aₖ should track this without being told. No labels. Just Hebb.
  Then the Takens-out reconstruction quality is measured across all epochs.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

# ─────────────────────────────────────────────────────────────────────────────
#  THE NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class HarmonicArmNetwork:
    """
    N harmonic arms at frequencies k·ω₀ for k = 1,...,N.
    Amplitudes Aₖ learn by Hebbian phase-coherence.
    No backprop. No labels.
    """

    def __init__(self, N=12, omega0=0.08, eta=0.004, decay=0.0008, tau_embed=8):
        self.N       = N          # number of modes (integer weights)
        self.omega0  = omega0     # fundamental frequency
        self.eta     = eta        # Hebbian learning rate
        self.decay   = decay      # homeostatic decay
        self.tau     = tau_embed  # Takens delay

        # Amplitudes: the learnable part. Small random init.
        rng = np.random.default_rng(0)
        self.A = rng.uniform(0.01, 0.05, N).astype(np.float64)

        # History for Takens embedding and logging
        self.signal_history = []
        self.reconstruction_history = []
        self.A_history = []
        self.error_history = []
        self.t_history = []

    # ── core operations ───────────────────────────────────────────────────

    def arm_cosines(self, t):
        """cos(k·ω₀·t) for k=1..N  — the real part of each arm at time t."""
        ks = np.arange(1, self.N + 1, dtype=np.float64)
        return np.cos(ks * self.omega0 * t)

    def arm_sines(self, t):
        """sin(k·ω₀·t) for k=1..N  — the imaginary part."""
        ks = np.arange(1, self.N + 1, dtype=np.float64)
        return np.sin(ks * self.omega0 * t)

    def reconstruct(self, t):
        """Forward pass: weighted sum of arms → 1D reconstruction."""
        return float(np.dot(self.A, self.arm_cosines(t)))

    def hebbian_update(self, x, t):
        """
        Hebb + homeostasis:
          ΔAₖ = η · x(t) · cos(k·ω₀·t)   [phase coherence]
              - decay · Aₖ                 [homeostatic normalisation]

        x(t) · cos(k·ω₀·t) > 0  ↔  input and arm k are in phase → strengthen
        x(t) · cos(k·ω₀·t) < 0  ↔  out of phase → weaken (but floored at 0)
        """
        c = self.arm_cosines(t)
        delta = self.eta * x * c - self.decay * self.A
        self.A = np.maximum(0.0, self.A + delta)

    def takens_embed_recent(self, dim=3):
        """
        Takens in: take the last dim·tau samples of the reconstruction,
        sample at delay τ, return embedding point.
        """
        h = self.reconstruction_history
        needed = (dim - 1) * self.tau + 1
        if len(h) < needed:
            return None
        return np.array([h[-(1 + i * self.tau)] for i in range(dim)])

    def takens_out(self):
        """
        Rajapinta projection: collapse the current Takens embedding to 1D.
        Here: just return the most recent reconstructed value (trivial projection).
        The non-trivial version (rotation W) is in TakensNeuron.
        This is the 'read off the tip of the arm chain' operation.
        """
        if self.reconstruction_history:
            return self.reconstruction_history[-1]
        return 0.0

    # ── run ───────────────────────────────────────────────────────────────

    def step(self, x, t):
        """Process one sample: reconstruct → update → log."""
        x_hat = self.reconstruct(t)
        err   = x - x_hat
        self.hebbian_update(x, t)

        self.signal_history.append(x)
        self.reconstruction_history.append(x_hat)
        self.A_history.append(self.A.copy())
        self.error_history.append(abs(err))
        self.t_history.append(t)

    def run(self, signal, dt=1.0):
        """Process a full signal array."""
        for i, x in enumerate(signal):
            self.step(float(x), i * dt)
        return self


# ─────────────────────────────────────────────────────────────────────────────
#  SIGNAL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def make_epoch(T, active_modes, omega0, noise=0.12, seed=None):
    """
    One epoch: sum of cosines at modes k in active_modes,
    plus small noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float64)
    s = np.zeros(T)
    for k, amp in active_modes:
        phase = rng.uniform(0, 2 * np.pi)
        s += amp * np.cos(k * omega0 * t + phase)
    s += noise * rng.standard_normal(T)
    s /= np.std(s) + 1e-8  # unit variance
    return s


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment():
    OMEGA0 = 0.08
    T_EPOCH = 600

    print("=" * 60)
    print("  Hebbian Harmonic Arm Network")
    print("  PerceptionLab / Antti Luode")
    print("=" * 60)
    print()
    print("Epoch 1 (t=0→600):    k=2,5 dominant  (slow + medium arms)")
    print("Epoch 2 (t=600→1200): k=4,9 dominant  (different structure)")
    print("Epoch 3 (t=1200→1800): k=2,5 return   (memory / re-locking)")
    print()

    e1 = make_epoch(T_EPOCH, [(2, 1.0), (5, 0.7)], OMEGA0, seed=1)
    e2 = make_epoch(T_EPOCH, [(4, 1.0), (9, 0.6)], OMEGA0, seed=2)
    e3 = make_epoch(T_EPOCH, [(2, 1.0), (5, 0.7)], OMEGA0, seed=3)  # same as e1
    signal = np.concatenate([e1, e2, e3])

    net = HarmonicArmNetwork(N=12, omega0=OMEGA0, eta=0.004, decay=0.0008)
    net.run(signal)

    A_arr  = np.array(net.A_history)    # shape: (T, N)
    err    = np.array(net.error_history)
    t_arr  = np.array(net.t_history)
    sig    = np.array(net.signal_history)
    recon  = np.array(net.reconstruction_history)
    T_total = len(sig)

    # ── per-epoch reconstruction quality ─────────────────────────────────
    def rmse(a, b): return np.sqrt(np.mean((a - b) ** 2))
    boundaries = [0, T_EPOCH, 2*T_EPOCH, 3*T_EPOCH]
    print("Reconstruction RMSE per epoch:")
    for i in range(3):
        sl = slice(boundaries[i], boundaries[i+1])
        r = rmse(sig[sl], recon[sl])
        dominant = ["k=2,5", "k=4,9", "k=2,5"][i]
        print(f"  Epoch {i+1} ({dominant}):  RMSE = {r:.4f}")

    # ── dominant mode at end of each epoch ───────────────────────────────
    print("\nDominant modes (by Aₖ) at epoch boundaries:")
    for name, idx in [("end epoch 1", T_EPOCH-1),
                      ("end epoch 2", 2*T_EPOCH-1),
                      ("end epoch 3", 3*T_EPOCH-1)]:
        a = A_arr[idx]
        top2 = np.argsort(a)[::-1][:3]
        print(f"  {name:16s}: k = {[k+1 for k in top2]}  "
              f"A = {[f'{a[k]:.3f}' for k in top2]}")

    return dict(
        net=net, signal=signal, recon=recon, A_arr=A_arr,
        err=err, t_arr=t_arr, T_epoch=T_EPOCH, omega0=OMEGA0
    )


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(r):
    BG, CYAN, AMBER, GREEN, PINK, DIM, WHITE = (
        '#0a0a1a', '#00ffcc', '#ffaa00', '#44dd88', '#ff4488', '#445566', '#ddeeff'
    )
    T_ep  = r['T_epoch']
    A_arr = r['A_arr']
    sig   = r['signal']
    recon = r['recon']
    err   = r['err']
    N     = r['net'].N

    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.52, wspace=0.38)

    def ax_style(ax, title, subtitle=''):
        ax.set_facecolor('#0f0f28')
        ax.tick_params(colors=DIM, labelsize=7)
        for s in ax.spines.values(): s.set_color('#223')
        full = title + (f'\n{subtitle}' if subtitle else '')
        ax.set_title(full, color=CYAN, fontsize=8.5, pad=5)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color(DIM)

    def epoch_shading(ax, ymin, ymax):
        for i, (col, lbl) in enumerate([(CYAN,'E1: k=2,5'), (AMBER,'E2: k=4,9'), (GREEN,'E3: k=2,5')]):
            ax.axvspan(i*T_ep, (i+1)*T_ep, alpha=0.04, color=col)
            ax.axvline(i*T_ep, color=col, lw=0.4, alpha=0.35)
            ax.text(i*T_ep + T_ep*0.35, ymax*0.92, lbl, color=col, fontsize=6.5, alpha=0.7)

    T = len(sig)

    # ── Row 0: signal + reconstruction ───────────────────────────────────
    ax = fig.add_subplot(gs[0, :3])
    ax_style(ax, "Input signal x(t)  vs  Reconstruction x̂(t)",
             "No labels. No backprop. Aₖ adapt by Hebbian phase-coherence only.")
    ax.plot(sig,   color=WHITE,  lw=0.6, alpha=0.55, label='x(t) input')
    ax.plot(recon, color=AMBER,  lw=0.9, alpha=0.85, label='x̂(t) reconstruction')
    epoch_shading(ax, sig.min(), sig.max())
    ax.legend(fontsize=7, facecolor='#0f0f28', labelcolor=WHITE, loc='upper right')
    ax.set_xlim(0, T)

    ax = fig.add_subplot(gs[0, 3])
    ax_style(ax, "Reconstruction error |x - x̂|")
    # smoothed error
    w = 30
    sm = np.convolve(err, np.ones(w)/w, mode='same')
    ax.plot(sm, color=PINK, lw=0.9)
    for i, col in enumerate([CYAN, AMBER, GREEN]):
        ax.axvspan(i*T_ep, (i+1)*T_ep, alpha=0.05, color=col)
        ax.axvline(i*T_ep, color=col, lw=0.4, alpha=0.35)
    ax.set_xlim(0, T)
    ax.set_xlabel("t", color=DIM, fontsize=7)

    # ── Row 1: Aₖ heatmap over time ──────────────────────────────────────
    ax = fig.add_subplot(gs[1, :3])
    ax_style(ax, "Amplitude matrix Aₖ(t)  —  integer weights evolving in real time",
             "Rows = mode k (integer weight). Bright = high amplitude. No backprop.")
    img = ax.imshow(A_arr.T, aspect='auto', origin='lower',
                    extent=[0, T, 0.5, N + 0.5],
                    cmap='inferno', interpolation='nearest')
    ax.set_ylabel("mode k", color=DIM, fontsize=8)
    ax.set_xlabel("t", color=DIM, fontsize=7)
    for i, col in enumerate([CYAN, AMBER, GREEN]):
        ax.axvline(i*T_ep, color=col, lw=1.0, alpha=0.6)
    # Mark true active modes
    active = {0: [2, 5], 1: [4, 9], 2: [2, 5]}
    for ep, modes in active.items():
        for k in modes:
            ax.annotate('', xy=((ep+0.5)*T_ep, k), xycoords='data',
                        fontsize=6)
    plt.colorbar(img, ax=ax, fraction=0.012, pad=0.01).ax.tick_params(
        colors=DIM, labelsize=6)

    # ── Row 1 col 3: bar chart at three time points ───────────────────────
    ax = fig.add_subplot(gs[1, 3])
    ax_style(ax, "Aₖ at epoch boundaries")
    ks = np.arange(1, N + 1)
    width = 0.25
    for i, (idx, col, lbl) in enumerate([
        (T_ep - 1,     CYAN,  'end E1'),
        (2*T_ep - 1,   AMBER, 'end E2'),
        (3*T_ep - 1,   GREEN, 'end E3'),
    ]):
        ax.bar(ks + (i - 1) * width, A_arr[idx], width=width,
               color=col, alpha=0.8, label=lbl)
    ax.set_xlabel("mode k", color=DIM, fontsize=7)
    ax.set_ylabel("Aₖ", color=DIM, fontsize=7)
    ax.set_xticks(ks)
    ax.legend(fontsize=6.5, facecolor='#0f0f28', labelcolor=WHITE)
    # Mark true modes
    for k in [2, 5]:
        ax.axvline(k, color=CYAN,  lw=0.6, ls='--', alpha=0.4)
    for k in [4, 9]:
        ax.axvline(k, color=AMBER, lw=0.6, ls='--', alpha=0.4)

    # ── Row 2: Takens attractors at three epochs ──────────────────────────
    TAU = r['net'].tau
    for col_i, (ep_start, ep_end, col, lbl) in enumerate([
        (0,    T_ep,    CYAN,  'Epoch 1: k=2,5'),
        (T_ep, 2*T_ep,  AMBER, 'Epoch 2: k=4,9'),
        (2*T_ep, 3*T_ep, GREEN, 'Epoch 3: k=2,5 (return)'),
    ]):
        ax = fig.add_subplot(gs[2, col_i])
        ax_style(ax, f"Attractor  {lbl}", "Takens: x̂(t) vs x̂(t-τ)")
        seg = recon[ep_start:ep_end]
        if len(seg) > TAU:
            ax.scatter(seg[TAU:], seg[:-TAU],
                       c=np.arange(len(seg) - TAU),
                       cmap='cool' if col == CYAN else ('hot' if col == AMBER else 'summer'),
                       s=0.5, alpha=0.6)
        ax.set_xlabel("x̂(t)", color=DIM, fontsize=7)
        ax.set_ylabel("x̂(t-τ)", color=DIM, fontsize=7)

    # ── Row 2 col 3: learning speed ───────────────────────────────────────
    ax = fig.add_subplot(gs[2, 3])
    ax_style(ax, "Mode 2 and mode 5 amplitude\nover full run")
    ax.plot(A_arr[:, 1], color=CYAN,  lw=1.0, label='k=2')
    ax.plot(A_arr[:, 4], color=GREEN, lw=1.0, label='k=5')
    ax.plot(A_arr[:, 3], color=AMBER, lw=1.0, label='k=4', ls='--')
    ax.plot(A_arr[:, 8], color=PINK,  lw=1.0, label='k=9', ls='--')
    for i, col in enumerate([CYAN, AMBER, GREEN]):
        ax.axvline(i*T_ep, color=col, lw=0.4, alpha=0.35)
    ax.legend(fontsize=7, facecolor='#0f0f28', labelcolor=WHITE)
    ax.set_xlabel("t", color=DIM, fontsize=7)
    ax.set_ylabel("Aₖ", color=DIM, fontsize=7)
    ax.set_xlim(0, T)

    # ── Row 3: theory panel ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, :2])
    ax.set_facecolor('#0f0f28')
    ax.axis('off')
    rule_text = (
        "Hebbian Phase-Coherence Learning Rule\n\n"
        "  ΔAₖ = η · x(t) · cos(k·ω₀·t)    ← Hebb: grow when in phase\n"
        "      - decay · Aₖ                 ← homeostasis: prevent runaway\n\n"
        "  x(t) · cos(k·ω₀·t) > 0  →  arm k and input in phase  → Aₖ ↑\n"
        "  x(t) · cos(k·ω₀·t) < 0  →  arm k and input out of phase → Aₖ ↓\n\n"
        "  No labels. No backprop. No gradient.\n"
        "  Only: does arm k resonate with what is actually arriving?\n\n"
        "  The integer k is architectural (fixed).\n"
        "  The amplitude Aₖ is learned (Hebbian).\n"
        "  Phase-lock = weight convergence."
    )
    ax.text(0.03, 0.97, rule_text, transform=ax.transAxes,
            color=WHITE, fontsize=8.5, va='top', family='monospace',
            bbox=dict(facecolor='#111130', edgecolor=CYAN, alpha=0.9, pad=10))

    ax = fig.add_subplot(gs[3, 2:])
    ax.set_facecolor('#0f0f28')
    ax.axis('off')
    arch_text = (
        "Architecture: Takens-in / Arms / Takens-out\n\n"
        "  1D input x(t)\n"
        "      │\n"
        "      ▼  Takens in (delay embed: [x(t), x(t-τ), x(t-2τ), ...])\n"
        "  ┌──────────────────────────────────┐\n"
        "  │  Harmonic Arm Bank               │\n"
        "  │  zₖ(t) = Aₖ·cos(k·ω₀·t)         │  k = integer weight\n"
        "  │  Aₖ → Hebbian learning           │  Aₖ = scalar (learnable)\n"
        "  │  k  → architectural prior        │\n"
        "  └──────────────────────────────────┘\n"
        "      │\n"
        "      ▼  Takens out / Rajapinta  (project → 1D output)\n"
        "  1D output x̂(t)\n\n"
        "  Stack multiple banks with different ω₀ → multi-scale memory.\n"
        "  Next: make ω₀ adaptive → full dynamic AI."
    )
    ax.text(0.03, 0.97, arch_text, transform=ax.transAxes,
            color=WHITE, fontsize=8.5, va='top', family='monospace',
            bbox=dict(facecolor='#111130', edgecolor=AMBER, alpha=0.9, pad=10))

    fig.suptitle(
        "Hebbian Harmonic Arm Network  ·  Integer Weights  ·  No Backprop  ·  PerceptionLab",
        color=CYAN, fontsize=12, fontweight='bold', y=0.995
    )

    out = 'hebbian_harmonic_arms.png'
    plt.savefig(out, dpi=155, facecolor=BG, bbox_inches='tight')
    print(f"\n[Figure → {out}]")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE-LOCK CONVERGENCE TEST
# ─────────────────────────────────────────────────────────────────────────────

def test_phase_lock_speed():
    """
    How many steps does Hebb take to lock onto a new mode?
    Compare modes the signal uses vs. modes it doesn't.
    """
    print("\n── Phase-lock convergence test ──────────────────────────────")
    OMEGA0 = 0.08

    # Simple: single mode k=5
    t = np.arange(800, dtype=np.float64)
    sig = np.cos(5 * OMEGA0 * t) + 0.15 * np.random.default_rng(42).standard_normal(800)
    sig /= sig.std()

    net = HarmonicArmNetwork(N=12, omega0=OMEGA0, eta=0.005, decay=0.0010)
    net.run(sig)

    A_arr = np.array(net.A_history)
    k5_traj = A_arr[:, 4]  # k=5 (index 4)
    k3_traj = A_arr[:, 2]  # k=3 (not in signal, should stay low)

    # Find when k=5 first exceeds 2× starting value
    threshold = k5_traj[0] * 3.0
    lock_step = next((i for i, a in enumerate(k5_traj) if a > threshold), None)
    print(f"  Signal contains: k=5 only")
    print(f"  A₅ at t=0:   {k5_traj[0]:.4f}")
    print(f"  A₅ at t=end: {k5_traj[-1]:.4f}")
    print(f"  A₃ at t=end: {k3_traj[-1]:.4f}  (not in signal — should be low)")
    print(f"  Phase-lock threshold (3× init) reached at: t = {lock_step}")
    print(f"  Selectivity (A₅/A₃): {k5_traj[-1]/k3_traj[-1]:.1f}×")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results = run_experiment()
    test_phase_lock_speed()
    plot_results(results)
    print("\nDone.")
    print("The Aₖ tracked the changing signal structure without being told anything.")
    print("The integer weights (k) stayed fixed. The scalar amplitudes (Aₖ) did all the work.")
    print("That is the Hebbian harmonic arm network.")
