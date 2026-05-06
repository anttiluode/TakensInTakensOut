import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Environment Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"

class AttractorSculptorWASD:
    def __init__(self, root):
        self.root = root
        self.root.title("Attractor Sculptor VST (Keyboard Interactivity)")
        self.root.geometry("1400x900")
        self.root.configure(bg="#0b0b12")

        # Core Parameters
        self.tau = 15
        self.warp_strength = 0.0
        # Initialize the warp center (The X) in the middle of phase space
        self.warp_center = np.array([0.5, 0.5, 0.5])
        
        self.signal_1d = None
        self.original_image = None
        self.img_dims = (128, 128)
        self.attractor_raw = None
        self.move_step = 0.02 # How fast the X moves

        self.setup_gui()
        self.bind_keys() # Attach keyboard listeners

    def setup_gui(self):
        main_frame = tk.Frame(self.root, bg="#0b0b12")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Top: Controls ---
        ctrl_frame = tk.Frame(main_frame, bg="#11111a", bd=1, relief="ridge")
        ctrl_frame.pack(fill=tk.X, side=tk.TOP, pady=5)

        tk.Button(ctrl_frame, text="📂 Load Image", command=self.load_image, bg="#4A90E2", fg="white").pack(side=tk.LEFT, padx=10, pady=10)
        
        # Display the current numerical status (like VST readouts)
        self.lbl_readout = tk.Label(ctrl_frame, text=f"Topological State: τ={self.tau} | X:{self.warp_center[0]:.2f}, Y:{self.warp_center[1]:.2f}, Z:{self.warp_center[2]:.2f}", fg="#00ffcc", bg="#11111a", font=("Consolas", 10))
        self.lbl_readout.pack(side=tk.LEFT, padx=20)

        # --- Center: Displays ---
        display_frame = tk.Frame(main_frame, bg="#0b0b12")
        display_frame.pack(fill=tk.BOTH, expand=True)

        self.panel_orig = self._create_panel(display_frame, "1. Hilbert Space (1D Signal Source)", 0, 0)
        
        # 3D Sculpting Canvas
        self.panel_3d = tk.Frame(display_frame, bg="#0b0b12", bd=1, relief="sunken")
        self.panel_3d.grid(row=0, column=1, sticky="nsew", padx=5)
        self._setup_canvas_3d()
        
        self.panel_recon = self._create_panel(display_frame, "3. Rajapinta Output (Warped Perception)", 1, 0)
        
        self.lbl_status = tk.Label(display_frame, text="Use WASD/QE to move Warp Center (Red X)\nUse Arrow keys for τ and Strength", fg="#00ffcc", bg="#0b0b12", font=("Consolas", 12))
        self.lbl_status.grid(row=1, column=1)

        display_frame.grid_columnconfigure(0, weight=1)
        display_frame.grid_columnconfigure(1, weight=1)
        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_rowconfigure(1, weight=1)

    def _setup_canvas_3d(self):
        self.fig = Figure(figsize=(5, 4), facecolor='#0b0b12')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#0b0b12')
        self.canvas_3d = FigureCanvasTkAgg(self.fig, master=self.panel_3d)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_panel(self, parent, title, r, c):
        f = tk.LabelFrame(parent, text=title, bg="#11111a", fg="#8888aa", font=("Consolas", 10, "bold"))
        f.grid(row=r, column=c, sticky="nsew", padx=5, pady=5)
        lbl = tk.Label(f, bg="#000000")
        lbl.pack(expand=True, fill=tk.BOTH)
        return lbl

    def bind_keys(self):
        # Bind keys to the root window
        self.root.bind('<w>', lambda event: self._move_center(0, self.move_step)) # Y up
        self.root.bind('<s>', lambda event: self._move_center(0, -self.move_step)) # Y down
        self.root.bind('<a>', lambda event: self._move_center(-self.move_step, 0)) # X left
        self.root.bind('<d>', lambda event: self._move_center(self.move_step, 0)) # X right
        
        self.root.bind('<q>', lambda event: self._move_center(0, 0, self.move_step)) # Z up
        self.root.bind('<e>', lambda event: self._move_center(0, 0, -self.move_step)) # Z down

        # Arrows for parameters
        self.root.bind('<Left>', lambda event: self._change_tau(-1))
        self.root.bind('<Right>', lambda event: self._change_tau(1))
        self.root.bind('<Up>', lambda event: self._change_strength(0.05))
        self.root.bind('<Down>', lambda event: self._change_strength(-0.05))

    def _move_center(self, dx, dy, dz=0):
        self.warp_center[0] = np.clip(self.warp_center[0] + dx, 0, 1)
        self.warp_center[1] = np.clip(self.warp_center[1] + dy, 0, 1)
        self.warp_center[2] = np.clip(self.warp_center[2] + dz, 0, 1)
        self.process()

    def _change_tau(self, delta):
        self.tau = np.clip(self.tau + delta, 1, 100)
        self.process()

    def _change_strength(self, delta):
        self.warp_strength = np.clip(self.warp_strength + delta, -2.0, 2.0)
        self.process()

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.original_image = cv2.resize(img, self.img_dims)
            # Flatten image into a 1D continuous signal (Hilbert Tape)
            self.signal_1d = self.original_image.flatten().astype(np.float32) / 255.0
            self._update_img_label(self.panel_orig, self.original_image)
            self.process()

    def process(self):
        if self.signal_1d is None: return
        
        # 1. TAKENS ENCODING (Takens-In)
        s = self.signal_1d
        length = len(s) - 2*self.tau
        # Create vectors S(t), S(t-tau), S(t-2tau)
        x = s[2*self.tau : 2*self.tau + length]
        y = s[1*self.tau : 1*self.tau + length]
        z = s[0     : 0 + length]
        attractor = np.stack([x, y, z], axis=1)

        # 2. SCULPTING (The Warp - Neuroplasticity model)
        # Apply gravitational pull toward the warp_center (Attention)
        diff = attractor - self.warp_center
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        # Squeeze or explode points based on strength
        warp_effect = diff * (self.warp_strength / (dist + 0.1))
        attractor_sculpted = attractor + warp_effect

        # 3. RAJAPINTA PROJECTION (Takens-Out)
        # Collapse the complex 3D geometry back into a 1D sequence
        # We take the X-axis of the warped space
        reconstructed_1d = attractor_sculpted[:, 0]
        
        padded = np.zeros_like(self.signal_1d)
        padded[:len(reconstructed_1d)] = np.clip(reconstructed_1d, 0, 1)
        recon_img = (padded.reshape(self.img_dims) * 255).astype(np.uint8)

        # 4. UPDATE VISUALS
        self._update_plot_3d(attractor_sculpted)
        self._update_img_label(self.panel_recon, recon_img)
        
        # Update Readout
        self.lbl_readout.config(text=f"Topological State: τ={self.tau} | Str:{self.warp_strength:.2f} | X:{self.warp_center[0]:.2f}, Y:{self.warp_center[1]:.2f}, Z:{self.warp_center[2]:.2f}")

    def _update_plot_3d(self, data):
        self.ax.clear()
        step = 8 # Downsample for interactivity
        self.ax.scatter(data[::step, 0], data[::step, 1], data[::step, 2], 
                        c=data[::step, 2], cmap='winter', s=1, alpha=0.4)
        # Draw the warp center (The X)
        self.ax.scatter(self.warp_center[0], self.warp_center[1], self.warp_center[2], 
                        color='red', s=50, marker='X')
        self.ax.set_axis_off()
        self.canvas_3d.draw()

    def _update_img_label(self, label, img_array):
        img = Image.fromarray(img_array).resize((350, 350), Image.NEAREST)
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk)
        label.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = AttractorSculptorWASD(root)
    root.mainloop()