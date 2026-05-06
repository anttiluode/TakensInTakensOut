import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def takens_embed(signal, delay=7, dim=3):
    N = len(signal) - (dim - 1) * delay
    emb = np.zeros((N, dim))
    for i in range(dim):
        emb[:, i] = signal[i*delay : i*delay + N]
    return emb

def takens_reconstruct(emb, delay=7, orig_len=None):
    """Better diagonal averaging reconstruction"""
    if orig_len is None:
        orig_len = len(emb) + (emb.shape[1]-1)*delay
    recon = np.zeros(orig_len)
    counts = np.zeros(orig_len, dtype=int)
    
    for i in range(emb.shape[1]):
        start = i * delay
        end = start + len(emb)
        recon[start:end] += emb[:, i]
        counts[start:end] += 1
    recon /= np.maximum(counts, 1)
    return recon

class TakensProcessor:
    def __init__(self, delay=7, dim=3):
        self.delay = delay
        self.dim = dim
    
    def process(self, signal):
        emb = takens_embed(signal, self.delay, self.dim)
        
        # === Core: Process the attractor ===
        # Simple but meaningful: denoise by keeping principal directions
        # (This is where Rajapinta / arm banks / resonance would go deeper)
        U, S, Vt = np.linalg.svd(emb, full_matrices=False)
        # Keep top components (strongest geometry)
        rank = min(2, len(S))  # keep dominant structure
        cleaned_emb = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
        
        # Optional: boost stable parts
        cleaned_emb *= 1.1
        
        recon = takens_reconstruct(cleaned_emb, self.delay, len(signal))
        return recon, emb, cleaned_emb

# ====================== RUN ======================
if __name__ == "__main__":
    # === Load your signal (audio or EEG flattened) ===
    fs, data = wavfile.read("your_audio.wav")
    if len(data.shape) > 1:
        data = np.mean(data, axis=1).astype(float)
    signal = data / (np.max(np.abs(data)) + 1e-8)
    
    # Trim for clarity
    signal = signal[:min(44100*12, len(signal))]
    
    processor = TakensProcessor(delay=8, dim=3)
    reconstructed, original_emb, cleaned_emb = processor.process(signal)
    
    # === Plot ===
    fig = plt.figure(figsize=(15, 10))
    
    plt.subplot(3,1,1)
    plt.plot(signal, label='Original', alpha=0.75)
    plt.title("Takens In → Process → Takens Out")
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.plot(reconstructed, label='Reconstructed (Takens Out)', color='cyan')
    plt.legend()
    
    # Attractor comparison
    ax = plt.subplot(3,1,3, projection='3d')
    step = 8
    ax.scatter(original_emb[::step,0], original_emb[::step,1], original_emb[::step,2],
               s=1, alpha=0.4, label='Original Attractor', color='lime')
    ax.scatter(cleaned_emb[::step,0], cleaned_emb[::step,1], cleaned_emb[::step,2],
               s=1, alpha=0.6, label='Processed Attractor', color='red')
    ax.set_title("Phase Space: Original vs Processed Attractor")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("takens_in_out_improved.png", dpi=180)
    plt.show()
    
    print("Done. Compare plots. Red points = processed geometry.")