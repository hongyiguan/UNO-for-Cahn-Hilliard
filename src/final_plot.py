"""
MIT License - Copyright (c) 2026 Hongyi Guan
See LICENSE file for full license text
"""

"""
Renders the side-by-side comparison video used as the headline result.
 
Rolls out three trajectories from one shared initial condition, namely the
spectral solver as ground truth, the baseline UNO trained on one-step
supervision, and the UNO trained with the unrolled barrier-augmented objective,
for 5000 Cahn-Hilliard steps sampled every 10 steps, giving 500 animation
frames. Both checkpoints are loaded from their respective grid-search
directories and share the same data processor, so the only difference between
the two right-hand panels is the training objective. The colour scale is pinned
to [-1, +1] rather than fitted per frame, which is deliberate: a surrogate whose
field drifts outside the physical range saturates visibly instead of being
silently rescaled into looking plausible. Output is comparison_rollout.mp4 at
1920x1080 and 30 fps.
"""


import math
import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch
from matplotlib.patheffects import withStroke
from matplotlib.cm import ScalarMappable

from neuralop.training.training_state import load_training_state
from CH_FFT import step_cahn_hilliard

from train_uno import build_uno_model, build_dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VANILLA_RUN_DIR = "./grid_search_checkpoints/run_4_C32_LR0.004_WD0.00012"
OPTIMIZED_RUN_DIR = "./grid_search_pi_roll_checkpoints/run_5_C32_LR0.004_WD0.00012_B0.01_K5"
SAVE_NAME = "best_model"

VANILLA_HIDDEN_CHANNELS = 32
OPT_HIDDEN_CHANNELS = 32

BATCH_SIZE = 50
RESOLUTION = 64
N_GRID = RESOLUTION
L = 2.0 * math.pi
DX = L / N_GRID
DT = 0.01
EPS = 0.1
A = 2.0
N_TOTAL_STEPS = 5000
STRIDE = 10
N_MACRO = N_TOTAL_STEPS // STRIDE
SEED = 42

OUT_PATH = "comparison_rollout.mp4"
FIG_SIZE = (16, 9)
DPI = 120
FPS = 30

BG_COLOR = "#FFFFFF"
TITLE_COLOR = "#1A1A1A"
PROGRESS_FG = "#2266CC"
PROGRESS_BG = "#E8ECF1"

TITLE_FONT = {"family": "DejaVu Sans", "weight": "bold"}


def make_fft_constants(N_grid, dx, dt, eps, a):
    freqs = torch.fft.fftfreq(N_grid, d=dx, device=DEVICE, dtype=torch.float32) * 2.0 * torch.pi
    KY, KX = torch.meshgrid(freqs, freqs, indexing="ij")
    k2 = KX ** 2 + KY ** 2
    k4 = k2 ** 2
    denom = 1.0 + dt * (eps ** 2 * k4 + a * k2)
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


def run_rollout(label, stepper, c_init, **stepper_kwargs):
    frames = [c_init.detach().cpu().clone()]
    curr = c_init.clone()
    for i in range(1, N_MACRO):
        curr = stepper(curr, **stepper_kwargs)
        frames.append(curr.detach().cpu().clone())
        if i % 100 == 0 or i == N_MACRO - 1:
            sim_step = (i + 1) * STRIDE
            print(
                f"  [{label}] step {sim_step:5d}  "
                f"range=[{curr.min().item():+.3f}, {curr.max().item():+.3f}]"
            )
    return torch.stack(frames).numpy()


def build_cmap():
    stops = [
        (0.00, "#1F5FA8"),
        (0.25, "#6FA8DC"),
        (0.50, "#F2EFE9"),
        (0.75, "#E8825E"),
        (1.00, "#B83A1F"),
    ]
    return LinearSegmentedColormap.from_list("ch_diverge_light", stops)


def style_axes(ax):
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    torch.manual_seed(SEED)
    cmap = build_cmap()

    print(f"Device: {DEVICE}")
    print(f"Macro frames: {N_MACRO} (simulation steps {STRIDE} .. {N_TOTAL_STEPS})")

    k2, denom = make_fft_constants(N_GRID, DX, DT, EPS, A)

    dataset = build_dataset(RESOLUTION, BATCH_SIZE)
    data_processor = dataset.data_processor.to(DEVICE)
    data_processor.eval()

    print(f"Loading vanilla UNO from {VANILLA_RUN_DIR}")
    model_vanilla = build_uno_model(VANILLA_HIDDEN_CHANNELS).to(DEVICE)
    model_vanilla, _, _, _, ep_v = load_training_state(
        save_dir=VANILLA_RUN_DIR, save_name=SAVE_NAME, model=model_vanilla,
    )
    model_vanilla.eval()
    print(f"  loaded (epoch {ep_v})")

    print(f"Loading optimized UNO from {OPTIMIZED_RUN_DIR}")
    model_opt = build_uno_model(OPT_HIDDEN_CHANNELS).to(DEVICE)
    model_opt, _, _, _, ep_o = load_training_state(
        save_dir=OPTIMIZED_RUN_DIR, save_name=SAVE_NAME, model=model_opt,
    )
    model_opt.eval()
    print(f"  loaded (epoch {ep_o})")

    c0 = (torch.rand(N_GRID, N_GRID, device=DEVICE, dtype=torch.float32) - 0.5) * 0.2
    c_init = step_10_fft(c0, DT, EPS, A, k2, denom)

    print("Rolling out Ground Truth (FFT)...")
    fft_frames = run_rollout(
        "FFT", step_10_fft, c_init,
        dt=DT, eps=EPS, a=A, k2=k2, denom=denom,
    )
    print("Rolling out Vanilla UNO...")
    van_frames = run_rollout(
        "Vanilla", step_10_uno, c_init,
        model=model_vanilla, data_processor=data_processor,
    )
    print("Rolling out Optimized UNO...")
    opt_frames = run_rollout(
        "Optimized", step_10_uno, c_init,
        model=model_opt, data_processor=data_processor,
    )

    vlim = 1.0
    print(f"Color range pinned to +-{vlim:.2f}; out-of-range values saturate.")

    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI, facecolor=BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)

    gs = fig.add_gridspec(
        nrows=1, ncols=3,
        wspace=0.06,
        left=0.03, right=0.97, top=0.93, bottom=0.22,
    )

    panel_titles = ["Ground Truth", "Vanilla UNO", "Optimized UNO"]
    ax_panels = [fig.add_subplot(gs[0, i]) for i in range(3)]

    images = []
    for ax, ttl, stack in zip(
        ax_panels, panel_titles, [fft_frames, van_frames, opt_frames],
    ):
        style_axes(ax)
        im = ax.imshow(
            stack[0], vmin=-vlim, vmax=vlim, cmap=cmap,
            origin="lower", interpolation="bilinear", aspect="equal",
        )
        images.append(im)
        ax.set_title(
            ttl, color=TITLE_COLOR, pad=14,
            fontdict={**TITLE_FONT, "size": 20},
        )

    cbar_ax = fig.add_axes([0.35, 0.15, 0.30, 0.020])
    sm = ScalarMappable(norm=Normalize(vmin=-vlim, vmax=vlim), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([-vlim, 0.0, vlim])
    cbar.set_ticklabels(["-1.0", "0.0", "+1.0"])
    cbar.ax.tick_params(colors=TITLE_COLOR, labelsize=10, length=0, pad=4)
    cbar.outline.set_visible(False)

    ax_progress = fig.add_axes([0.03, 0.04, 0.94, 0.050])
    ax_progress.axis("off")
    bar_height = 0.60
    bar_y = 0.20
    ax_progress.add_patch(
        FancyBboxPatch(
            (0.02, bar_y), 0.96, bar_height,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            transform=ax_progress.transAxes,
            facecolor=PROGRESS_BG, edgecolor="none",
        )
    )
    progress_fill = FancyBboxPatch(
        (0.02, bar_y), 0.0, bar_height,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        transform=ax_progress.transAxes,
        facecolor=PROGRESS_FG, edgecolor="none",
    )
    ax_progress.add_patch(progress_fill)
    step_text = ax_progress.text(
        0.5, bar_y + bar_height / 2.0,
        f"Step {STRIDE} / {N_TOTAL_STEPS}",
        ha="center", va="center",
        transform=ax_progress.transAxes,
        color=TITLE_COLOR,
        fontdict={**TITLE_FONT, "size": 13},
        path_effects=[withStroke(linewidth=3, foreground=BG_COLOR)],
    )

    def update(i):
        """Advance the figure to macro frame i (zero-indexed)."""
        sim_step = (i + 1) * STRIDE
        images[0].set_array(fft_frames[i])
        images[1].set_array(van_frames[i])
        images[2].set_array(opt_frames[i])
        frac = (i + 1) / N_MACRO
        progress_fill.set_width(0.96 * frac)
        step_text.set_text(f"Step {sim_step} / {N_TOTAL_STEPS}")
        return images + [progress_fill, step_text]

    print(f"Rendering {N_MACRO} frames at {FPS} fps to {OUT_PATH}...")
    ani = animation.FuncAnimation(
        fig, update, frames=N_MACRO, interval=1000 / FPS, blit=False,
    )
    ani.save(
        OUT_PATH, writer="ffmpeg", fps=FPS, dpi=DPI,
        savefig_kwargs={"facecolor": BG_COLOR},
    )
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
