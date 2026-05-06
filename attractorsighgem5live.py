import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Environment Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"

class AttractorManipulator:
    """Helper class containing the geometric phase space transformations."""
    @staticmethod
    def drop_information(X, drop_ratio):
        mask = np.random.rand(len(X)) > drop_ratio
        return mask 

    @staticmethod
    def coarse_grain(X, pixel_size):
        if pixel_size <= 0.01: return X
        return np.round(X / pixel_size + 1e-8) * pixel_size

    @staticmethod
    def stretch(X, dim, scale):
        X[:, dim] *= scale
        return X

    @staticmethod
    def flatten(X, dim, factor):
        X[:, dim] *= factor
        return X

    @staticmethod
    def skew(X, target_dim, source_dim, factor):
        X[:, target_dim] += X[:, source_dim] * factor
        return X

class LiveAttractorSculptor:
    def __init__(self, root):
        self.root = root
        self.root.title("LIVE Attractor Sculptor VST (Real-Time Rajapinta)")
        self.root.geometry("1500x900")
        self.root.configure(bg="#0b0b12")

        # Handle window close to release webcam
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Core Parameters[cite: 15]
        self.tau = 15
        self.warp_strength = 0.0
        self.warp_center = np.array([0.5, 0.5, 0.5])
        
        self.signal_1d = None
        self.original_image = None
        self.img_dims = (128, 128)
        self.move_step = 0.02
        self.frame_counter = 0

        # UI Variables for sliders
        self.var_drop = tk.DoubleVar(value=0.0)
        self.var_pixel = tk.DoubleVar(value=0.0)
        self.var_stretch = tk.DoubleVar(value=1.0)
        self.var_flatten = tk.DoubleVar(value=1.0)
        self.var_skew = tk.DoubleVar(value=0.0)

        # Webcam Setup
        self.cap = cv2.VideoCapture(0)
        self.is_streaming = True

        self.setup_gui()
        self.bind_keys()
        
        # Start the live video loop
        self.video_loop()

    def setup_gui(self):
        main_frame = tk.Frame(self.root, bg="#0b0b12")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Top: Controls ---
        ctrl_frame = tk.Frame(main_frame, bg="#11111a", bd=1, relief="ridge")
        ctrl_frame.pack(fill=tk.X, side=tk.TOP, pady=5)

        tk.Button(ctrl_frame, text="⏸ Toggle Camera", command=self.toggle_camera, bg="#4A90E2", fg="white").pack(side=tk.LEFT, padx=10, pady=10)
        
        self.lbl_readout = tk.Label(ctrl_frame, text="Topological State: Ready", fg="#00ffcc", bg="#11111a", font=("Consolas", 10))
        self.lbl_readout.pack(side=tk.LEFT, padx=20)

        # --- Left: Slider Controls ---
        slider_frame = tk.Frame(main_frame, bg="#11111a", bd=1, relief="ridge", width=250)
        slider_frame.pack(fill=tk.Y, side=tk.LEFT, padx=5, pady=5)
        
        tk.Label(slider_frame, text="GEOMETRIC WARP", fg="#ff5555", bg="#11111a", font=("Consolas", 12, "bold")).pack(pady=10)

        self._create_slider(slider_frame, "Drop Info (%)", self.var_drop, 0.0, 0.99, 0.01)
        self._create_slider(slider_frame, "Pixelate / Coarse", self.var_pixel, 0.0, 0.5, 0.01)
        self._create_slider(slider_frame, "Stretch X", self.var_stretch, 0.1, 3.0, 0.1)
        self._create_slider(slider_frame, "Flatten Z", self.var_flatten, 0.0, 1.0, 0.05)
        self._create_slider(slider_frame, "Skew X-Y", self.var_skew, -2.0, 2.0, 0.1)

        # --- Center: Displays ---
        display_frame = tk.Frame(main_frame, bg="#0b0b12")
        display_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        self.panel_orig = self._create_panel(display_frame, "1. Hilbert Space (Live Source)", 0, 0)
        
        # 3D Sculpting Canvas
        self.panel_3d = tk.Frame(display_frame, bg="#0b0b12", bd=1, relief="sunken")
        self.panel_3d.grid(row=0, column=1, sticky="nsew", padx=5)
        self._setup_canvas_3d()
        
        self.panel_recon = self._create_panel(display_frame, "3. Rajapinta Output (Warped)", 1, 0)
        
        self.lbl_status = tk.Label(display_frame, text="Use WASD/QE to move Warp Center (Red X)\nUse Arrow keys for τ and Strength", fg="#00ffcc", bg="#0b0b12", font=("Consolas", 12))
        self.lbl_status.grid(row=1, column=1)

        display_frame.grid_columnconfigure(0, weight=1)
        display_frame.grid_columnconfigure(1, weight=1)
        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_rowconfigure(1, weight=1)

    def _create_slider(self, parent, text, variable, from_, to, res):
        frame = tk.Frame(parent, bg="#11111a")
        frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(frame, text=text, fg="#aaaaaa", bg="#11111a", font=("Consolas", 9)).pack(anchor="w")
        slider = tk.Scale(frame, from_=from_, to=to, resolution=res, orient=tk.HORIZONTAL, 
                          variable=variable, bg="#11111a", fg="#00ffcc", 
                          troughcolor="#0b0b12", highlightthickness=0)
        slider.pack(fill=tk.X)

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
        self.root.bind('<w>', lambda event: self._move_center(0, self.move_step))
        self.root.bind('<s>', lambda event: self._move_center(0, -self.move_step))
        self.root.bind('<a>', lambda event: self._move_center(-self.move_step, 0))
        self.root.bind('<d>', lambda event: self._move_center(self.move_step, 0))
        
        self.root.bind('<q>', lambda event: self._move_center(0, 0, self.move_step))
        self.root.bind('<e>', lambda event: self._move_center(0, 0, -self.move_step))

        self.root.bind('<Left>', lambda event: self._change_tau(-1))
        self.root.bind('<Right>', lambda event: self._change_tau(1))
        self.root.bind('<Up>', lambda event: self._change_strength(0.05))
        self.root.bind('<Down>', lambda event: self._change_strength(-0.05))

    def _move_center(self, dx, dy, dz=0):
        self.warp_center[0] = np.clip(self.warp_center[0] + dx, 0, 1)
        self.warp_center[1] = np.clip(self.warp_center[1] + dy, 0, 1)
        self.warp_center[2] = np.clip(self.warp_center[2] + dz, 0, 1)

    def _change_tau(self, delta):
        self.tau = np.clip(self.tau + delta, 1, 100)

    def _change_strength(self, delta):
        self.warp_strength = np.clip(self.warp_strength + delta, -2.0, 2.0)

    def toggle_camera(self):
        self.is_streaming = not self.is_streaming

    def video_loop(self):
        """Continuously grab frames from the webcam and process them."""
        if self.is_streaming and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Convert to grayscale and resize for the buffer
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.original_image = cv2.resize(gray, self.img_dims)
                self.signal_1d = self.original_image.flatten().astype(np.float32) / 255.0
                
                self._update_img_label(self.panel_orig, self.original_image)
                self.process_frame()
                
        # Schedule the next frame update (~30 fps)
        self.root.after(30, self.video_loop)

    def process_frame(self):
        """Perform the Takens math and manipulate the geometry."""
        if self.signal_1d is None: return
        
        # 1. TAKENS ENCODING (Takens-In)[cite: 15]
        s = self.signal_1d
        length = len(s) - 2*self.tau
        x = s[2*self.tau : 2*self.tau + length]
        y = s[1*self.tau : 1*self.tau + length]
        z = s[0     : 0 + length]
        attractor = np.stack([x, y, z], axis=1)

        # 2. GRAVITY SCULPTING[cite: 15]
        diff = attractor - self.warp_center
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        warp_effect = diff * (self.warp_strength / (dist + 0.1))
        attractor = attractor + warp_effect

        # 3. GEOMETRIC MANIPULATIONS[cite: 15]
        attractor = AttractorManipulator.coarse_grain(attractor, self.var_pixel.get())
        attractor = AttractorManipulator.skew(attractor, target_dim=0, source_dim=1, factor=self.var_skew.get())
        attractor = AttractorManipulator.stretch(attractor, dim=0, scale=self.var_stretch.get())
        attractor = AttractorManipulator.flatten(attractor, dim=2, factor=self.var_flatten.get())
        keep_mask = AttractorManipulator.drop_information(attractor, self.var_drop.get())
        
        # Zero out dropped points
        attractor[~keep_mask] = 0

        # 4. RAJAPINTA PROJECTION (Takens-Out)[cite: 15]
        reconstructed_1d = attractor[:, 0]
        padded = np.zeros_like(self.signal_1d)
        padded[:len(reconstructed_1d)] = np.clip(reconstructed_1d, 0, 1)
        recon_img = (padded.reshape(self.img_dims) * 255).astype(np.uint8)

        # 5. UPDATE VISUALS
        self._update_img_label(self.panel_recon, recon_img)
        
        # Throttle 3D rendering to prevent UI lag (update every 5 frames)
        self.frame_counter += 1
        if self.frame_counter % 5 == 0:
            plot_attractor = attractor[keep_mask]
            self._update_plot_3d(plot_attractor)
        
        self.lbl_readout.config(text=f"Topological State: τ={self.tau} | Str:{self.warp_strength:.2f} | X:{self.warp_center[0]:.2f}, Y:{self.warp_center[1]:.2f}, Z:{self.warp_center[2]:.2f}")

    def _update_plot_3d(self, data):
        self.ax.clear()
        step = max(1, len(data) // 3000) # Heavy downsample for live 3D speed
        if len(data) > 0:
            self.ax.scatter(data[::step, 0], data[::step, 1], data[::step, 2], 
                            c=data[::step, 2], cmap='winter', s=2, alpha=0.5)
            
        self.ax.scatter(self.warp_center[0], self.warp_center[1], self.warp_center[2], 
                        color='red', s=50, marker='X')
        self.ax.set_axis_off()
        self.canvas_3d.draw()

    def _update_img_label(self, label, img_array):
        img = Image.fromarray(img_array).resize((350, 350), Image.NEAREST)
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk)
        label.image = img_tk

    def on_closing(self):
        """Release the webcam when the window closes."""
        self.is_streaming = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LiveAttractorSculptor(root)
    root.mainloop()