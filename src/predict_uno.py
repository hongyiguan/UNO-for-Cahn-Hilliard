"""
MIT License - Copyright (c) 2026 Hongyi Guan
See LICENSE file for full license text
"""

"""
Long-horizon rollout of a single trained UNO checkpoint against the solver.
 
Loads a checkpoint from a grid-search run directory, warms up a random initial
condition by 10 spectral steps to obtain a common starting field, then advances
two trajectories from that identical state for 5000 solver steps: one with the
FFT solver, one by applying the surrogate autoregressively 500 times, each
application covering a 10-step macro-interval. Normalisation and
de-normalisation are handled by the dataset's data processor, so the model
always sees inputs with the statistics it was trained on. The script prints the
value range of both fields periodically, which is a useful early warning, since
a diverging surrogate usually escapes [-1, 1] well before the pattern starts to
look visibly wrong. Outputs are FFT.mp4, UNO.mp4, and a three-panel figure
showing the two final states, their difference, and the relative L2 error. Set
RUN_DIR to choose which checkpoint to evaluate.
"""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from neuralop.training.training_state import load_training_state
from CH_FFT import step_cahn_hilliard

from train_uno import build_uno_model, build_dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN_CHANNELS = 32
BATCH_SIZE = 50
RESOLUTION = 64
#RUN_DIR = "./grid_search_checkpoints/run_4_C32_LR0.004_WD0.00012"
RUN_DIR = "./grid_search_pi_roll_checkpoints/run_5_C32_LR0.004_WD0.00012_B0.01_K5"
SAVE_NAME = "best_model"

N_GRID = RESOLUTION
DX = 2.0 * torch.pi / N_GRID
DT = 0.01
EPS = 0.1
A = 2.0
N_TOTAL_STEPS = 5000
STRIDE = 10
N_MACRO = N_TOTAL_STEPS // STRIDE
SEED = 42


def make_fft_constants(N_grid, dx, dt, eps, a):
    freqs = torch.fft.fftfreq(N_grid, d=dx, device=DEVICE, dtype=torch.float32) * 2.0 * torch.pi
    KY, KX = torch.meshgrid(freqs, freqs, indexing="ij")
    k2 = KX**2 + KY**2
    k4 = k2**2
    denom = 1.0 + dt * (eps**2 * k4 + a * k2)
    return k2, denom
 
 
def step_10_fft(c, dt, eps, a, k2, denom):
    curr = c
    for _ in range(STRIDE):
        curr = step_cahn_hilliard(curr, dt, eps, a, k2, denom).clone()
    return curr
 
 
@torch.no_grad()
def step_10_uno(c, model, data_processor):
    x = c.view(1, 1, *c.shape).to(DEVICE)
    batch = {"x": x, "y": torch.zeros_like(x)}
    batch = data_processor.preprocess(batch)
    out = model(batch["x"])
    out, _ = data_processor.postprocess(out, batch)
    return out[0, 0].detach()
 
 
def save_video(frames, out_path, title, fps=30):
    arr = torch.stack(frames).cpu().numpy()
    vlim = max(abs(float(arr.min())), abs(float(arr.max())))
 
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(arr[0], vmin=-vlim, vmax=vlim, cmap="coolwarm", origin="lower")
    ax.axis("off")
    ttl = ax.set_title(f"{title} - step {STRIDE}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
 
    def update(i):
        im.set_array(arr[i])
        ttl.set_text(f"{title} - step {(i + 1) * STRIDE}")
        return [im, ttl]
 
    ani = animation.FuncAnimation(
        fig, update, frames=len(arr), interval=1000 / fps, blit=False
    )
    ani.save(out_path, writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(f"Saved {out_path}  (color range +-{vlim:.3f})")
 
 
def plot_final_comparison(fft_final, uno_final, out_path):
    err = uno_final - fft_final
    rel_l2 = float(
        ((err ** 2).sum() / (fft_final ** 2).sum()) ** 0.5
    )
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vlim = max(abs(fft_final).max(), abs(uno_final).max())
 
    im0 = axes[0].imshow(fft_final, vmin=-vlim, vmax=vlim, cmap="coolwarm", origin="lower")
    axes[0].set_title(f"FFT at step {N_TOTAL_STEPS}")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
 
    im1 = axes[1].imshow(uno_final, vmin=-vlim, vmax=vlim, cmap="coolwarm", origin="lower")
    axes[1].set_title(f"UNO at step {N_TOTAL_STEPS}")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
 
    err_lim = float(abs(err).max())
    im2 = axes[2].imshow(err, vmin=-err_lim, vmax=err_lim, cmap="coolwarm", origin="lower")
    axes[2].set_title(f"UNO - FFT   max |.| = {err_lim:.3f}   rel L2 = {rel_l2:.3f}")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
 
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}  (relative L2 = {rel_l2:.4f})")
 
 
def main():
    torch.manual_seed(SEED)
 
    k2, denom = make_fft_constants(N_GRID, DX, DT, EPS, A)
 
    dataset = build_dataset(RESOLUTION, BATCH_SIZE)
    data_processor = dataset.data_processor.to(DEVICE)
 
    model = build_uno_model(HIDDEN_CHANNELS).to(DEVICE)
    model, _, _, _, epoch = load_training_state(
        save_dir=RUN_DIR,
        save_name=SAVE_NAME,
        model=model,
    )
    model.eval()
    data_processor.eval()
    print(f"Loaded UNO from '{RUN_DIR}' (epoch: {epoch}).")
    print(f"FFT grid: dx = 2*pi/{N_GRID} = {DX:.5f}, dt = {DT}, eps = {EPS}, a = {A}")
 
    c0 = (torch.rand(N_GRID, N_GRID, device=DEVICE, dtype=torch.float32) - 0.5) * 0.2
    c_init = step_10_fft(c0, DT, EPS, A, k2, denom)
    print(
        f"Warmed-up c_10:  mean={c_init.mean().item():+.4f}  "
        f"std={c_init.std().item():.4f}  "
        f"range=[{c_init.min().item():+.4f}, {c_init.max().item():+.4f}]"
    )
 
    fft_frames = [c_init.detach().cpu().clone()]
    uno_frames = [c_init.detach().cpu().clone()]
    curr_fft = c_init.clone()
    curr_uno = c_init.clone()
 
    for i in range(1, N_MACRO):
        curr_fft = step_10_fft(curr_fft, DT, EPS, A, k2, denom)
        curr_uno = step_10_uno(curr_uno, model, data_processor)
        fft_frames.append(curr_fft.detach().cpu().clone())
        uno_frames.append(curr_uno.detach().cpu().clone())
        if i % 50 == 0 or i == N_MACRO - 1:
            sim_step = (i + 1) * STRIDE
            print(
                f"  step {sim_step:5d}  "
                f"FFT [{curr_fft.min().item():+.3f}, {curr_fft.max().item():+.3f}]  "
                f"UNO [{curr_uno.min().item():+.3f}, {curr_uno.max().item():+.3f}]"
            )
 
    save_video(fft_frames, "FFT.mp4", "FFT", fps=30)
    save_video(uno_frames, "UNO.mp4", "UNO", fps=30)
 
    plot_final_comparison(
        fft_final=fft_frames[-1].numpy(),
        uno_final=uno_frames[-1].numpy(),
        out_path="final_comparison.png",
    )


if __name__ == "__main__":
    main()
