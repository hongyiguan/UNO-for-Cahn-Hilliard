# Physics-Informed Neural Operator for Cahn-Hilliard Dynamics

A U-shaped Neural Operator (UNO) surrogate for 2D Cahn-Hilliard phase separation, trained with **unrolled autoregressive supervision** (targeting accumulative rollout error) and a **tangent barrier loss** (targeting phase boundaries).

---

## Governing equation
 
The Cahn-Hilliard equation is the $H^{-1}$ gradient flow of the Ginzburg-Landau free energy
 
$$F[c] = \int_\Omega \left[ \frac{1}{4}(c^2 - 1)^2 + \frac{\varepsilon^2}{2}|\nabla c|^2 \right] d\mathbf{x}$$
 
giving, with unit mobility,
 
$$\frac{\partial c}{\partial t} = \nabla^2 \mu \qquad \mu = \frac{\delta F}{\delta c} = c^3 - c - \varepsilon^2 \nabla^2 c$$
 
Here $c(\mathbf{x},t) \in [-1, 1]$ is the conserved phase field and $\varepsilon$ sets the interface width.
 
### Numerical scheme
 
Solved pseudo-spectrally on a periodic square with semi-implicit **convexity splitting** (Eyre). Adding and subtracting $a\nabla^2 c$, then treating $N(c) = c^3 - (1+a)c$ explicitly and the linear terms implicitly:
 
$$\hat{c}^{n+1} = \frac{\hat{c}^n - \Delta t k^2 \widehat{N(c^n)}}{1 + \Delta t(\varepsilon^2 k^4 + a k^2)} \qquad k^2 = k_x^2 + k_y^2$$
 
Unconditionally stable for sufficiently large $a$, at two FFTs per step.

## System settings

| Parameter | Symbol | Value |
|---|---|---|
| Domain | $\Omega$ | $[0, 2\pi)^2$, periodic |
| Grid | $N$ | $64 \times 64$ (data), $128 \times 128$ (solver demo) |
| Interface width | $\varepsilon$ | $0.1$ |
| Splitting parameter | $a$ | $2.0$ |
| Time step | $\Delta t$ | $0.01$ |
| Macro-step (learned map) | $S$ | $10$ steps, i.e. $\Delta T = 0.1$ |
| Initial condition | $c_0$ | $\mathcal{U}(-0.05, 0.05)$ |

The surrogate learns the **macro-step operator** $\mathcal{G}: c(t) \mapsto c(t + 10\Delta t)$, so one network call replaces ten solver steps.

## Training objective

Standard one-step supervision is replaced by a $K$-step unrolled loss.

$$\mathcal{L}(\theta) = \sum_{k=1}^{K} \Big[ \underbrace{\|\hat{c}_k - c_{n+k}\|_{H^1}}_{\text{gradient-aware fidelity}} + \beta \underbrace{\left\langle \tan^2 \left(\tfrac{\pi}{4}\left(\hat{c}_k - c_{n+k}\right)\right)\right\rangle}_{\text{barrier}} \Big]$$

The barrier is $O(\Delta c^2)$ for small errors but diverges as $|\Delta c| \to 2$, the full phase span, so predictions are softly confined to the physically admissible range. The argument is clamped to $\pm(\pi/2 - \delta)$ with $\delta = 0.1$, capping the per-pixel penalty at $\cot^2\delta \approx 100$ and zeroing the gradient beyond it.

Optimiser: AdamW, $\eta = 4\times 10^{-3}$, weight decay $1.2\times 10^{-4}$, cosine annealing over 500 epochs, gradient-norm clipping at 1.0. Grid search sweeps $\beta \in \{0, 10^{-2}\}$ and $K \in \{1, 3, 5\}$. The selected model is $\beta = 10^{-2}$, $K = 5$, 32 hidden channels.

## Installation and requirements
 
Follow the install instructions for PyTorch and [`neuraloperator`](https://github.com/neuraloperator/neuraloperator). FFmpeg is required for video export.

## Pipeline

| File | Role |
|---|---|
| `CH_FFT.py` | Spectral Cahn-Hilliard solver; batched, GPU-ready |
| `generate_data.py` | Trajectory generation, 5000 train / 1000 test pairs |
| `train_uno.py` | Model and dataset builders; baseline training |
| `train_pi_uno_roll.py` | `UnrolledTrainer`, `BarrierLoss`, grid search |
| `predict_uno.py` | 5000-step rollout vs. solver; relative $L^2$ error |
| `final_plot.py` | Publication figure and comparison video |

Checkpoints are written to `grid_search_checkpoints/` (baseline) and `grid_search_pi_roll_checkpoints/` (physics-informed), one directory per configuration.

## Results

Autoregressive rollout to 5000 solver steps (500 network calls) from a shared initial condition. Colour scale is pinned to $[-1, +1]$, so any excursion outside the physical range saturates rather than being rescaled away.

https://github.com/user-attachments/assets/58921985-0d0f-44b8-8f64-fb93b010885b

The baseline, trained only on one-step targets, accumulates error and loses the coarsening morphology over the horizon. The unrolled, barrier-trained model tracks the spectral reference and stays within the physical bounds throughout.

## References

- Cahn, J. W. & Hilliard, J. E. *Free Energy of a Nonuniform System*. J. Chem. Phys. 28, 258 (1958).
- Fisher, R. *Comparing Numerical Solution Methods for the Cahn-Hilliard Equation* (2021).
- Rahman, M. A., Ross, Z. E. & Azizzadenesheli, K. *U-NO: U-shaped Neural Operators*. TMLR (2023).
- Kossaifi, J., Kovachki, N., Li, Z., Pitt, D., Liu-Schiaffini, M., Duruisseaux, V., George, R., Bonev, B., Azizzadenesheli, K., Berner, J., and Anandkumar, A., *A Library for Learning Neural Operators*. arXiv:2412.10354 (2024).
