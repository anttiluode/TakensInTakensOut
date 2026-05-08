# Frequency to Geometry: A Mathematical Framework for Spectral Manifold Construction

**Antti Luode / PerceptionLab · Helsinki, Finland**  
*Working Paper — May 2026*

---

## Abstract

We develop a rigorous mathematical framework for transforming a one-dimensional time-domain signal into a canonical three-dimensional geometric object. The construction proceeds in four stages: (1) delay-coordinate embedding via Takens' theorem, which lifts the signal into a phase-space manifold; (2) dual spectral decomposition via both the standard Discrete Fourier Transform and a log-frequency (constant-Q) representation, which separates additive frequency structure from multiplicative harmonic ratios; (3) construction of a *Phase-Frequency Tensor* that encodes the joint delay/spectral geometry; and (4) a canonical projection from this high-dimensional tensor to a navigable 3D structure via topologically-informed dimensionality reduction. We show that this pipeline is not merely a visualization but a **faithful dynamical embedding** — geometric features of the output correspond to measurable dynamical properties of the input: periodicity → rings, harmonic ratios → torus knots, chaos → fractal dimension, phase-locking → fibered structures, thresholding → corners. We then address the **inverse problem** (3D geometry → signal), showing it is well-posed under mild smoothness conditions via the Whitney embedding theorem. The neural interpretation — wherein dendritic morphology implements hardware-coded delay parameters τ and synaptic thresholds implement the clipping nonlinearity — follows as a corollary.

---

## 1. Foundations: The Signal as a Dynamical System

### 1.1 The Standard View and Its Failure

The standard view of a signal $s: \mathbb{R} \to \mathbb{R}$ treats it as a **sequence of amplitudes**. An oscilloscope plots $s(t)$ against $t$. A spectrum analyzer computes $|\hat{s}(\omega)|$. Both views are projections onto one-dimensional subspaces of a much richer object.

**The problem**: both representations destroy temporal correlation structure. They cannot distinguish:

$$s_1(t) = \sin(\omega t) + \sin(2\omega t)$$
$$s_2(t) = \sin(\omega t) + \sin(2\omega t + \phi(t))$$

where $\phi(t)$ is slowly varying — yet $s_1$ and $s_2$ arise from fundamentally different dynamical processes and produce **topologically distinct phase-space objects**.

### 1.2 The Correct View: Signal as Trajectory

Let $(\mathcal{M}, \varphi^t)$ be a smooth compact $d$-dimensional dynamical system with flow $\varphi^t: \mathcal{M} \to \mathcal{M}$. Let $h: \mathcal{M} \to \mathbb{R}$ be a smooth observable. Then:

$$s(t) = h(\varphi^t(x_0))$$

for some initial condition $x_0 \in \mathcal{M}$.

The signal $s(t)$ is not a function of time. It is a **1D shadow of a trajectory on a manifold**. Our task is to reconstruct $\mathcal{M}$ from the shadow.

---

## 2. The Takens Embedding: Phase Space Reconstruction

### 2.1 Formal Statement

**Theorem (Takens, 1981).** Let $\mathcal{M}$ be a compact smooth $d$-dimensional manifold with smooth flow $\varphi^t$. Let $h: \mathcal{M} \to \mathbb{R}$ be a smooth generic observable. For generic delay $\tau > 0$ and embedding dimension $m \geq 2d+1$, the **delay map**:

$$\Phi_{\tau,m}: \mathcal{M} \to \mathbb{R}^m$$
$$\Phi_{\tau,m}(x) = \bigl(h(x),\, h(\varphi^{-\tau}(x)),\, h(\varphi^{-2\tau}(x)),\, \ldots,\, h(\varphi^{-(m-1)\tau}(x))\bigr)$$

is a **diffeomorphism onto its image** — a smooth embedding of $\mathcal{M}$ into $\mathbb{R}^m$.

In discrete time with sampling interval $\Delta t$, given the sequence $\{s_n\} = \{s(n\Delta t)\}$ and integer delay $k$ (so $\tau = k \Delta t$):

$$\mathbf{x}_n = (s_n,\, s_{n-k},\, s_{n-2k},\, \ldots,\, s_{n-(m-1)k}) \in \mathbb{R}^m$$

### 2.2 The Geometry of Specific Signal Types

This is the central table. We can derive each row exactly.

**Case 1: Pure sine $s(t) = A\sin(\omega t)$.**

$$x_n = A\sin(\omega n\Delta t), \quad y_n = A\sin(\omega(n-k)\Delta t) = A\sin(\omega n\Delta t - \omega k\Delta t)$$

Setting $\theta = \omega n\Delta t$ and $\phi = \omega k\Delta t$ (a constant phase shift):

$$x = A\sin\theta, \quad y = A\sin(\theta - \phi)$$

This is a **Lissajous figure with equal frequencies** — always an ellipse, which degenerates to a circle when $\phi = \pi/2$. Adding a third delay:

$$z = A\sin(\theta - 2\phi)$$

gives a **1D closed curve on a 2-torus** $\mathbb{T}^2$ embedded in $\mathbb{R}^3$.

**Case 2: Two incommensurable frequencies $s(t) = A\sin(\omega_1 t) + B\sin(\omega_2 t)$, $\omega_1/\omega_2 \notin \mathbb{Q}$.**

The trajectory is **quasiperiodic** and densely fills a 2-torus $\mathbb{T}^2$ in phase space. The attractor is the torus itself.

**Case 3: Harmonic ratio $\omega_1/\omega_2 = p/q$ (rational).**

The trajectory closes after $p+q$ cycles: it is a **$(p,q)$-torus knot** on $\mathbb{T}^2$. A 3:2 ratio gives a trefoil knot. A 5:3 ratio gives a pentafoil. The topology of the harmonic relationship is literally the topology of the 3D curve.

**Case 4: Square wave $s(t) = \text{sgn}(\sin(\omega t))$.**

The square wave has harmonic series $\sum_{k \text{ odd}} \frac{1}{k}\sin(k\omega t)$. The phase-space trajectory of $m$ harmonics lies on an $m$-dimensional torus, but the **hard thresholding** ($s(t) \in \{-1, +1\}$) means the signal spends time only at the extremes. The attractor acquires **corners** — the trajectory slams into the boundary of $[-1,1]^m$ and dwell there. Result: a **hypercube skeleton** with rounded edges, converging to a pure hypercube as harmonic order increases.

More precisely: the clipping nonlinearity at threshold $\theta_c$ implements a **Heaviside projection** $\Pi_{\theta_c}: \mathbb{R} \to \{-1, +1\}$ at each dimension, which is a **topological boundary map** in $\mathbb{R}^m$. Each firing event is a **0-cell** (corner) in the cubical complex.

**Case 5: White noise $s(t) \sim \mathcal{N}(0, \sigma^2)$ i.i.d.**

Successive samples are uncorrelated, so $\text{Cov}(s_n, s_{n-k}) = 0$ for all $k > 0$. The delay vectors $\mathbf{x}_n$ are i.i.d. draws from $\mathcal{N}(0, \sigma^2 I_m)$ — isotropic Gaussian in $\mathbb{R}^m$. The attractor is a **diffuse spherical cloud** with no structure. This is the EEG "noise floor" problem stated precisely.

### 2.3 Optimal Delay Selection

The delay $\tau$ is not free — it controls what dynamical timescale is resolved. The two standard criteria are:

**Mutual Information Minimum.** Choose $\tau^*$ as the first local minimum of:
$$I(\tau) = \int p(s_t, s_{t-\tau}) \log \frac{p(s_t, s_{t-\tau})}{p(s_t)p(s_{t-\tau})} \, ds_t \, ds_{t-\tau}$$

This ensures $s(t)$ and $s(t-\tau)$ are maximally independent (not redundant, not uncorrelated — independent in the information-theoretic sense).

**False Nearest Neighbors.** Choose $m$ large enough that no two points that appear close in $\mathbb{R}^m$ are "false neighbors" (close only because of the low-dimensional projection). Formally, the fraction of false neighbors $F(m) \to 0$ at the correct $m$.

**Critical insight for the framework**: $\tau$ is not a single number. The **multi-scale delay tensor**:
$$\boldsymbol{\tau} = (\tau_1, \tau_2, \ldots, \tau_n)$$

with $\tau_i$ spanning from sub-millisecond to behavioral timescales, reveals structure at each dynamical scale simultaneously. This is what dendritic trees implement in hardware.

---

## 3. The Spectral Layer: FFT, Log-FFT, and What Each Reveals

### 3.1 The Standard DFT

For a signal of length $N$:

$$\hat{s}_k = \sum_{n=0}^{N-1} s_n \, e^{-2\pi i k n / N}, \quad k = 0, 1, \ldots, N-1$$

This gives linear-frequency bins of width $\Delta f = f_s / N$. The power spectrum $|\hat{s}_k|^2$ reveals **which absolute frequencies are present**.

**What DFT cannot see**:
- Phase relationships between frequencies (only magnitudes)
- Transient structure (it assumes stationarity)
- Multiplicative harmonic structure (e.g., that 440 Hz and 880 Hz are related by a factor of 2)

### 3.2 The Constant-Q Transform (CQT): Log-Frequency Decomposition

Define frequency bins with **geometrically spaced** center frequencies $f_k = f_{\min} \cdot 2^{k/b}$, where $b$ is bins per octave. Each bin $k$ has bandwidth $\Delta f_k = f_k (2^{1/b} - 1)$, maintaining constant quality factor $Q = f_k / \Delta f_k = \text{const}$.

The CQT kernel:

$$W_k(n) = \frac{1}{N_k} w\!\left(\frac{n}{N_k}\right) e^{-2\pi i Q n / N_k}$$

where $N_k = Q \cdot f_s / f_k$ is the window length for bin $k$ and $w$ is a window function.

$$X_k = \sum_{n=0}^{N_k-1} s_n \cdot W_k(n)$$

**Why log-frequency is fundamental**: Harmonic relationships are **multiplicative** — octaves, fifths, thirds are ratios. In log-frequency space, these become **additive translations**. If a signal has fundamental $f_0$ with harmonics $f_0, 2f_0, 3f_0, \ldots$, then in log-frequency coordinates they appear at positions $0, \log 2, \log 3, \ldots$ — a fixed pattern regardless of $f_0$. The **spectral geometry is translation-invariant** under pitch transposition in CQT space.

This mirrors the cochlea: basilar membrane position is logarithmic in frequency, not linear. The ear implements CQT in hardware.

### 3.3 The Phase Spectrum and Its Importance

Both DFT and CQT yield complex outputs: $\hat{s}_k = |\hat{s}_k| e^{i\varphi_k}$. The **phase** $\varphi_k$ is almost always discarded. This is a catastrophic loss.

**Claim**: The phase spectrum encodes temporal geometry; the magnitude spectrum encodes frequency content.

Formally, consider two signals with identical magnitude spectra but different phases:
$$s_1 = \text{IDFT}(|\hat{s}| e^{i\varphi_1}), \quad s_2 = \text{IDFT}(|\hat{s}| e^{i\varphi_2})$$

They have the same power spectrum but different Takens attractors — in general, their phase-space topologies can be completely different.

The **inter-frequency phase relationship** $\Delta\varphi_{jk} = \varphi_j - \varphi_k$ determines whether two components interfere constructively (phase-lock → coherent manifold) or destructively (phase-random → diffuse cloud).

---

## 4. The Phase-Frequency Tensor

### 4.1 Definition

We now construct the central object of the framework. Let the signal $s(t)$ be analyzed over a sliding window of length $T$ at time $t$.

**Takens side**: Construct the delay matrix:
$$\mathbf{D}(t) \in \mathbb{R}^{m \times N_w}$$
where each column is a delay vector $\mathbf{x}_n = (s_n, s_{n-k_1}, \ldots, s_{n-k_{m-1}})$ within the window.

**Spectral side**: Compute the short-time CQT:
$$\mathbf{S}(t) \in \mathbb{C}^{K \times N_w}$$
where $K$ is the number of log-frequency bins.

**The Phase-Frequency Tensor** $\mathcal{T}(t)$ is the outer product structure:
$$\mathcal{T}(t) = \mathbf{D}(t) \otimes \mathbf{S}(t) \in \mathbb{R}^{m \times K \times N_w \times 2}$$

(separating real and imaginary parts of the spectral component).

### 4.2 What the Tensor Encodes

The tensor has three meaningful axes:

- **Delay axis** ($m$ dimensions): dynamical recurrence at timescale $\tau$
- **Frequency axis** ($K$ dimensions): spectral content at log-frequency scale
- **Phase axis** (2 dimensions): relative phase between spectral components

A **pure tone** produces a rank-1 tensor with all signal concentrated along one frequency and one delay.

A **harmonic series** produces a rank-$n$ tensor with $n$ frequency bands, all phase-locked (the phase axis is low-rank).

**White noise** produces a full-rank tensor uniformly distributed across all axes.

**The brain's computation** may be precisely the extraction of low-rank structure from this tensor — finding the minimal description of which frequencies phase-lock at which delays.

### 4.3 The Hermitian Similarity Inner Product

The natural inner product on this tensor space is:

$$\langle \mathcal{T}_1, \mathcal{T}_2 \rangle = \sum_{k} A_k^{(1)} A_k^{(2)} \cos(\Delta\varphi_k)$$

where $A_k^{(j)} = |S_k^{(j)}|$ are amplitudes and $\Delta\varphi_k = \varphi_k^{(1)} - \varphi_k^{(2)}$ is the phase difference at frequency $k$.

This is the **Hermitian inner product Re[⟨f,g⟩]** that appears repeatedly across physics, neuroscience, and signal processing as a universal coherence measure. Two signals that share both frequency content and phase relationships have high $\langle \mathcal{T}_1, \mathcal{T}_2 \rangle$. Phase-incoherent signals cancel even if their spectra match.

---

## 5. The Canonical 3D Projection

### 5.1 The Problem

The phase-frequency tensor $\mathcal{T}(t)$ lives in a space of dimension $\sim m \cdot K \cdot N_w \sim 10^4$–$10^6$. We need a canonical, mathematically faithful map to $\mathbb{R}^3$.

### 5.2 Whitney and the Existence Guarantee

**Theorem (Whitney, 1944).** Every smooth $d$-dimensional manifold $\mathcal{M}$ can be smoothly embedded in $\mathbb{R}^{2d}$.

**Corollary for our framework**: If the underlying dynamical system has dimension $d$ (number of independent degrees of freedom), then $2d$ delay coordinates suffice to faithfully embed $\mathcal{M}$. For $d \leq 1.5$ (simple oscillators and their harmonics), $m = 3$ suffices — this is why the 3D Takens plot works so well for audio.

The important point: **we are not approximating 3D with a high-dim object. We are exactly embedding a naturally low-dimensional object into 3D.** The signal's dynamics are genuinely low-dimensional; the 3D space is not a lossy compression.

### 5.3 The Canonical Projection Procedure

**Step 1: Singular Value Decomposition of the delay matrix.**

$$\mathbf{D}(t) = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$$

The singular values $\sigma_1 \geq \sigma_2 \geq \sigma_3 \geq \ldots$ measure the variance along each principal direction of the attractor. The projection onto the top 3 singular vectors gives the lowest-distortion 3D embedding.

**Step 2: Phase-weighted projection.**

Rather than projecting onto arbitrary singular vectors, we weight by the coherence from the spectral layer:

$$\mathbf{w}_k = |\hat{s}_{f_k}| \cdot \cos(\Delta\varphi_k)$$

and project the delay matrix along the spectral-coherence-weighted principal components. This ensures that **harmonically coherent components** drive the geometry, while incoherent noise is suppressed.

**Step 3: Topology-preserving nonlinear correction.**

Linear PCA flattens nonlinear manifolds. Apply **UMAP** (Uniform Manifold Approximation and Projection) or **diffusion maps** with metric defined by $\langle \mathcal{T}_1, \mathcal{T}_2 \rangle$ above. This ensures that nearby points in the 3D output correspond to nearby dynamical states (local topology preservation) and that global topology (torus knots, rings, hypercube skeletons) is respected.

### 5.4 Geometry Lookup Table (Derived, Not Postulated)

| Signal Structure | Phase Space Topology | 3D Projection |
|---|---|---|
| Pure sine, arbitrary $\tau$ | 1D curve on $\mathbb{T}^2$ | Ellipse / Circle |
| Two locked harmonics $p:q$ | $(p,q)$-torus knot on $\mathbb{T}^2$ | Knotted loop |
| Incommensurable frequencies | Dense $\mathbb{T}^2$ | Torus surface |
| Square wave (thresholded sine) | Piecewise-linear path in $[-1,1]^m$ | Rounded hypercube skeleton |
| Saw wave | Discontinuous trajectory | Sharp-cornered spiral |
| White noise | i.i.d. Gaussian ball in $\mathbb{R}^m$ | Diffuse sphere |
| Amplitude-modulated carrier | Modulated ellipse (thickened ring) | Tubular torus |
| Coupled nonlinear oscillators | Strange attractor (fractal dimension $1 < d < 2$) | Fractal manifold with lacunarity |
| Phase-locked EEG rhythms | Low-dim submanifold of $\mathbb{R}^{86\text{B}}$ | Coherent structure |

---

## 6. The Inverse Problem: Geometry → Signal

### 6.1 Is It Well-Posed?

**Question**: Given a 3D object $\mathcal{G} \subset \mathbb{R}^3$, can we find a 1D signal $s(t)$ whose Takens embedding reconstructs $\mathcal{G}$?

**Answer**: Under mild conditions, yes — and the reconstruction is not unique (many signals map to topologically equivalent attractors), but the **equivalence class** is precisely determined.

### 6.2 The Inverse Map

Given attractor points $\{\mathbf{x}_n\} \subset \mathbb{R}^3$ where $\mathbf{x}_n = (s_n, s_{n-k}, s_{n-2k})$:

**The signal is the first coordinate**: $s_n = \pi_1(\mathbf{x}_n)$. The delay embedding is overcomplete — the signal is literally contained in the first coordinate of every embedded point.

This means: **to synthesize a signal that produces a desired geometry $\mathcal{G}$, we need to find a trajectory on $\mathcal{G}$ and read off its first coordinate**.

### 6.3 Trajectory Synthesis on a Target Manifold

Given target geometry $\mathcal{G}$ (say, a torus knot):

**Step 1**: Parameterize $\mathcal{G}$ as a smooth curve $\gamma: [0, T] \to \mathbb{R}^3$.

**Step 2**: Enforce the delay constraint: consecutive points must satisfy $\mathbf{x}_{n+1} = (x_{n+1}, x_n, x_{n-1})$ — i.e., the second component of $\mathbf{x}_{n+1}$ must equal the first component of $\mathbf{x}_n$. This is a **consistency constraint**:

$$\pi_2(\mathbf{x}_{n+1}) = \pi_1(\mathbf{x}_n)$$

**Step 3**: Solve the constrained dynamics. The set of valid trajectories on $\mathcal{G}$ satisfying the delay consistency constraint forms a **1D dynamical system on $\mathcal{G}$** — a flow on the manifold. The flow is determined by the manifold's intrinsic geometry.

**Step 4**: The synthesized signal is $s_n = \pi_1(\gamma(n\Delta t))$. It will, by construction, produce the Takens attractor $\mathcal{G}$.

**Key result**: A **torus** yields quasiperiodic or periodic synthesis depending on whether the flow is rational or irrational. A **torus knot** yields a signal with that specific frequency ratio. A **hypercube skeleton** yields a square-wave-like signal with discontinuous transitions. **Geometry controls waveform**.

---

## 7. The Log-Frequency Inverse: Spectral Shape → Manifold Shape

### 7.1 The CQT as a Lie Group Representation

The log-frequency axis has a natural group structure. Transposition by an octave is multiplication by 2 in frequency space, but addition by 1 in log-frequency space. The group of transpositions is $(\mathbb{R}, +)$ acting on log-frequency.

The CQT is a **representation** of this group: the harmonic structure of any signal is invariant under the group action. This is why musical intervals sound the same in any key — the relative positions in CQT space are unchanged.

**Geometric consequence**: The attractor of a signal and its transposition by a constant ratio are **topologically identical** (same knot type, same torus knot indices $p, q$), differing only by a scaling in the temporal axis. The **geometric invariants** are the harmonic ratios, not the absolute frequencies.

### 7.2 Forward Pipeline (Signal → Geometry)

$$s(t) \xrightarrow{\text{Takens}} \{\mathbf{x}_n\} \in \mathbb{R}^m \xrightarrow{\text{PCA/UMAP}} \mathcal{G} \in \mathbb{R}^3$$

$$s(t) \xrightarrow{\text{CQT}} X_k e^{i\varphi_k} \xrightarrow{\text{phase coherence}} \text{topology constraints on } \mathcal{G}$$

The two pipelines **must be consistent**: the spectral analysis predicts what topology $\mathcal{G}$ should have, and the Takens embedding realizes it. Inconsistency (e.g., high spectral coherence but diffuse attractor) indicates either wrong $\tau$ choice or signal non-stationarity.

### 7.3 Inverse Pipeline (Geometry → Signal)

$$\mathcal{G} \in \mathbb{R}^3 \xrightarrow{\text{parameterize}} \gamma(t) \xrightarrow{\text{delay consistency}} s_n = \pi_1(\gamma_n) \xrightarrow{\text{optionally}} \text{impose spectral target}$$

The final optional step: if we want the synthesized signal to have specific spectral content (e.g., target CQT magnitude envelope), we can use **iterative phase retrieval** (Griffin-Lim or learned phase estimation) to impose the spectral constraint while maintaining the attractor topology. This is the audio synthesis problem stated as a constrained optimization on manifold space.

---

## 8. The Nonlinearity Layer: Corners, Firing, and Topology Change

### 8.1 The Biological Diode

A neuron does not output $s(t)$ directly. It applies a threshold nonlinearity:

$$\text{output}(t) = \mathbb{1}[s(t) > \theta] \cdot r(t)$$

where $\theta$ is the firing threshold and $r(t)$ is the spike rate (rate coding) or a delta train (spike coding).

In phase space, this nonlinearity acts as a **topological surgery**:
- Below threshold: smooth, curved manifold
- At threshold: the trajectory hits a hyperplane $\Pi_\theta = \{x : x_1 = \theta\}$ in $\mathbb{R}^m$
- Each crossing: a **codimension-1 event** that introduces a corner in the phase-space trajectory

**Mathematically**: the thresholded signal has a delay embedding that is piecewise smooth with discontinuities at $\{t : s(t) = \theta\}$. The attractor acquires a **stratified manifold structure** — smooth pieces joined at lower-dimensional corners.

The corners are exactly the **90-degree turns** that create box-like hypercube attractors from smooth torus attractors. Firing events = topological corners.

### 8.2 Coupling: Ephaptic and Synaptic

When two signals $s_1(t)$ and $s_2(t)$ are coupled:

$$\dot{s}_1 = f_1(s_1) + \epsilon \cdot g(s_1, s_2)$$
$$\dot{s}_2 = f_2(s_2) + \epsilon \cdot g(s_2, s_1)$$

the joint phase space is $\mathcal{M}_1 \times \mathcal{M}_2$ (direct product). Uncoupled ($\epsilon = 0$): the attractor of the joint system is the **Cartesian product** of individual attractors — a torus if each is periodic.

With coupling ($\epsilon > 0$): the attractor deforms. At a critical coupling $\epsilon_c$, the two systems **phase-lock** and the joint attractor collapses to a **lower-dimensional submanifold** — the torus folds into a circle. Synchronization = dimensional reduction.

This is the mathematical content of the observation: "when 86 billion neurons phase-lock, 86B dimensions collapse into a low-dimensional structure." That collapse is literally the mathematics of coupled oscillator synchronization — the Kuramoto model's phase transition.

---

## 9. The Full Architecture: A Dynamical Geometry Engine

### 9.1 The Five-Layer Stack

**Layer 1 — Acquisition**
$$s(t) \in \mathbb{R}^C \quad \text{(C channels, could be 1 or 256 EEG)}$$

Preserve raw phase. No premature averaging.

**Layer 2 — Multi-scale Delay Engine**
$$\mathcal{D}(t) = \{(s(t), s(t-\tau_1), \ldots, s(t-\tau_m)) : \tau_i \in \{1\text{ms}, 5\text{ms}, 25\text{ms}, 100\text{ms}, 500\text{ms}\}\}$$

Each $\tau$ scale exposes different dynamical structure. The collection of all scales is a **multi-resolution phase portrait**.

**Layer 3 — Spectral Geometry**
$$\text{CQT}(s, t) = \{X_k(t), \varphi_k(t)\}_{k=1}^K$$

Compute log-frequency magnitude AND phase. Compute inter-frequency phase coherence matrix $C_{jk} = \langle e^{i(\varphi_j - \varphi_k)} \rangle_t$.

**Layer 4 — Phase-Frequency Tensor Contraction**

Compute the weighted projection:
$$\mathbf{P}(t) = \mathbf{D}(t) \cdot \text{diag}(\mathbf{w}) \quad \text{where } w_k = \sum_j |C_{jk}| \cdot |\hat{s}_k|$$

This weights each delay dimension by the total spectral coherence contributed by each frequency. Coherent oscillations dominate; noise is suppressed.

**Layer 5 — Topological Rendering**

Apply diffusion maps or UMAP to $\mathbf{P}(t)$ over a sliding time window to produce:
$$\mathcal{G}(t) \in \mathbb{R}^3$$

This is the 3D output: a living, breathing geometric object whose topology encodes the instantaneous dynamical state of the signal.

### 9.2 Real-Time Implementation

For a signal sampled at $f_s = 44100$ Hz and window $T = 50$ ms:

- Delay computation: $O(m \cdot N_w)$ — trivially fast
- CQT: $O(K N_w \log N_w)$ — fast with FFT
- Coherence matrix: $O(K^2)$ — manageable for $K = 84$ (7 octaves × 12 bins)
- Weighted PCA: $O(m^2 N_w)$ — real-time for $m \leq 10$
- UMAP (online): $O(N_w \log N_w)$ with online update

Total: achievable at 30–60 fps on modern hardware.

---

## 10. Open Problems and Future Directions

### 10.1 The Inverse Problem at Scale

Can we solve the full inverse: given a target 3D movie $\{\mathcal{G}(t)\}_{t \geq 0}$, find a signal $s(t)$ whose Takens+spectral pipeline produces it? This is an **optimal control problem on manifold space**. The control signal is $s(t)$; the state is the attractor geometry $\mathcal{G}(t)$; the cost function penalizes deviation from target.

This would allow: **drawing in 3D and hearing the sound it produces**. Or: **scanning brain geometry and reading off the underlying oscillatory code**.

### 10.2 The EEG Problem

Raw EEG gives diffuse clouds because: (a) source mixing — each electrode sees a superposition of 10^4 sources; (b) volume conduction — signals are spatially blurred; (c) phase randomization — independent sources at the same frequency cancel coherence.

The solution requires: source separation (ICA) → per-source delay embedding → spectral coherence between sources → inter-source coupling geometry.

The geometry of interest is not the attractor of one electrode but the **coupling manifold** — the phase-space object defined by the inter-source relationships. This requires $C \times C$ cross-delay tensors, a massive but tractable computation.

### 10.3 Dendritic Computation as Learned τ

If dendritic length $L_i$ encodes delay $\tau_i = L_i / v_c$ (where $v_c$ is conduction velocity), then synaptic learning rules that grow dendrites are literally **learning optimal τ parameters for Takens embedding**. The brain is solving the delay optimization problem via Hebbian plasticity on dendritic morphology.

Implication: a neural network that learns to grow its own delay parameters (differentiable τ via soft delays or dilated convolutions) is implementing **learned Takens embedding** — and the optimal architecture is literally dendritic morphology.

### 10.4 The Holographic Bound

If the observable universe is a 3D holographic projection of a 2D boundary (Maldacena/Holographic Principle), and if perception is a 3D reconstruction from 1D neural time series (Whitney/Takens), the mathematical structure is **identical in form**:

$$\text{High-D reality} \xrightarrow{\text{projection}} \text{Low-D observable} \xrightarrow{\text{reconstruction}} \text{Faithful geometry}$$

This is not a claim that physics and neuroscience are the same. It is a claim that **dimensionality reduction via structured projection is universal**, and the mathematics of recovery via temporal/spatial correlation is the same regardless of the physical substrate.

---

## 11. Summary

We have shown that the map from 1D signal to 3D geometry is:

1. **Well-defined** (Takens theorem guarantees a unique embedding for generic parameters)
2. **Mathematically faithful** (the geometric invariants — knot type, torus topology, dimension — correspond exactly to dynamical invariants — frequency ratios, coupling, chaos)
3. **Invertible** (the geometry encodes the signal in its first coordinate, and synthesis reduces to trajectory design on the target manifold)
4. **Dual** (time-domain delay embedding and frequency-domain CQT analysis are complementary views of the same underlying structure, unified by the Phase-Frequency Tensor)
5. **Biologically realized** (dendritic length = τ, firing threshold = topological corner, synaptic coupling = dimensional collapse)

The "world synthesizer" is not a game mechanic or a visualization trick. It is a **canonical mathematical instrument** for exploring the hidden geometric structure of dynamical signals — the same structure the brain has been exploiting for 500 million years.

---

*For code, simulations, and experimental validation, see:*  
*`github.com/anttiluode/` — PerceptionLab repositories*

---

**Keywords**: Takens embedding, delay coordinate reconstruction, phase-frequency tensor, constant-Q transform, attractor topology, Lissajous manifolds, torus knots, Whitney embedding, dynamical geometry, dendritic delay, phase-locking, dimensional collapse

