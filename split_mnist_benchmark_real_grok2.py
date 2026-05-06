"""
Real Split-MNIST Continual Learning Benchmark
============================================
Hebbian Harmonic Arms (Geometry-based) vs baseline
"""

import numpy as np
from urllib.request import urlretrieve
import gzip
import os

np.random.seed(42)

# ========================== LOAD MNIST ==========================
def load_mnist():
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"]
    for f in files:
        if not os.path.exists(f):
            urlretrieve(base + f, f)
    
    def extract_images(f):
        with gzip.open(f, 'rb') as fb:
            data = np.frombuffer(fb.read(), np.uint8, offset=16)
        return data.reshape(-1, 784).astype(np.float32) / 255.0
    
    def extract_labels(f):
        with gzip.open(f, 'rb') as fb:
            return np.frombuffer(fb.read(), np.uint8, offset=8)
    
    X = extract_images(files[0])
    y = extract_labels(files[1])
    return X, y

X_all, y_all = load_mnist()
print("MNIST loaded:", X_all.shape)

tasks = [(0,1), (2,3), (4,5), (6,7), (8,9)]

def get_task_data(task_id, train=True):
    a, b = tasks[task_id]
    mask = np.isin(y_all, [a, b])
    X = X_all[mask]
    y = (y_all[mask] == b).astype(np.int32)  # 0 = first class, 1 = second
    
    n = len(X)
    split = int(0.85 * n)
    if train:
        return X[:split], y[:split]
    else:
        return X[split:], y[split:]

# ========================== IMPROVED HEBBIAN ARMS ==========================
class HarmonicArmClassifier:
    def __init__(self, n_arms=128, eta=0.15, decay=2e-5):
        self.n_arms = n_arms
        self.eta = eta
        self.decay = decay
        self.protos = {}  # task_id -> (2, n_features)

    def _features(self, X):
        """Improved features for real images"""
        # Basic statistics
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + 1e-8
        
        # Frequency content (phase-invariant power)
        fft_abs = np.abs(np.fft.rfft(X, axis=1))[:, :self.n_arms]
        fft_power = fft_abs ** 2
        
        # Simple spatial features
        left = X[:, :392].mean(axis=1, keepdims=True)
        right = X[:, 392:].mean(axis=1, keepdims=True)
        top = X[:, :392].mean(axis=1, keepdims=True)   # rough
        bottom = X[:, 392:].mean(axis=1, keepdims=True)
        
        return np.concatenate([mean, std, left, right, top, bottom, fft_power], axis=1)

    def train_task(self, task_id, X, y, epochs=8):
        F = self._features(X)
        
        if task_id not in self.protos:
            self.protos[task_id] = np.vstack([
                F[y == 0].mean(axis=0),
                F[y == 1].mean(axis=0)
            ])
        
        P = self.protos[task_id]
        for epoch in range(epochs):
            idx = np.random.permutation(len(X))
            for i in idx:
                c = y[i]
                P[c] += self.eta * (F[i] - P[c])
            P *= (1.0 - self.decay)

    def eval_task(self, task_id, X, y):
        if task_id not in self.protos:
            return 0.5
        F = self._features(X)
        P = self.protos[task_id]
        d0 = np.sum((F - P[0])**2, axis=1)
        d1 = np.sum((F - P[1])**2, axis=1)
        pred = (d1 < d0).astype(int)
        return (pred == y).mean()

# ========================== BENCHMARK ==========================
print("\n=== Running Real Split-MNIST ===\n")

heb = HarmonicArmClassifier(n_arms=128)

history = []

for t in range(5):
    X_tr, y_tr = get_task_data(t, train=True)
    X_te, y_te = get_task_data(t, train=False)
    
    print(f"Task {t} ({tasks[t][0]}/{tasks[t][1]}) - {len(X_tr)} samples")
    heb.train_task(t, X_tr, y_tr)
    
    # Evaluate all previous tasks
    accs = []
    for k in range(t + 1):
        X_te_k, y_te_k = get_task_data(k, train=False)
        a = heb.eval_task(k, X_te_k, y_te_k)
        accs.append(a)
    history.append(accs)
    
    print(f"  Accuracies: {[f'{a:.4f}' for a in accs]}")

print("\n=== FINAL RESULTS ===")
for t, accs in enumerate(history):
    print(f"After task {t}: {[f'{a:.4f}' for a in accs]}")

print(f"\nFinal Task 0 accuracy: {history[-1][0]:.4f}")