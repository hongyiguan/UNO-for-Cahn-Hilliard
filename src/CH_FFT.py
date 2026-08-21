"""
MIT License - Copyright (c) 2026 Hongyi Guan
See LICENSE file for full license text
"""

"""
Spectral solver for the 2D Cahn-Hilliard equation on a periodic square domain.
 
Integrates du/dt = lap(u^3 - u - eps^2 lap(u)) with a semi-implicit Fourier
pseudo-spectral scheme (Eyre convexity splitting). The nonlinear part
N(u) = u^3 - (1 + a)u is treated explicitly while the linear terms a*lap(u) and
-eps^2 * lap^2(u) are treated implicitly, giving the Fourier-space update
 
    u_hat^{n+1} = (u_hat^n - dt * k^2 * N_hat^n) / (1 + dt * (eps^2 k^4 + a k^2)),
 
which is unconditionally stable for a large enough splitting parameter a and
costs two FFTs per step. Everything is written against PyTorch tensors, so the
same routine runs on CPU or GPU and accepts a batch of fields unchanged. That is
what lets it serve two roles downstream: offline data generator, and on-the-fly
target oracle inside the unrolled training loop. Executing the file directly
runs a standalone phase-separation demo that saves snapshots at a fixed interval
and halts once the field stops changing.
"""

import torch
import matplotlib.pyplot as plt
import os

def step_cahn_hilliard(u_n, dt, eps, a, k2, denom):
    """
    Advances the 2D Cahn-Hilliard equation by one time step using the 
    semi-implicit Fast Fourier Transform method (Eyre's CS method).
    
    Reference: Fisher, Riley. “Comparing Numerical Solution Methods 
    for the Cahn-Hilliard Equation.” (2021).
    
    Args:
        u_n: 2D PyTorch tensor (real space field at time n).
        dt: Time step size.
        eps: Gradient energy coefficient.
        a: Convexity splitting parameter.
        k2: 2D PyTorch tensor of squared wave numbers (k^2).
        denom: 2D PyTorch tensor, precomputed denominator for the update.
        
    Returns:
        u_new: 2D PyTorch tensor (real space field at time n+1).
    """
    nonlin_term = (u_n ** 3) - (1.0 + a) * u_n
    
    u_hat = torch.fft.fft2(u_n)
    nonlin_hat = torch.fft.fft2(nonlin_term)
    
    num = u_hat - dt * k2 * nonlin_hat
    
    u_hat_new = num / denom
    
    u_new = torch.fft.ifft2(u_hat_new).real
    
    return u_new


def plot_state_and_save(u, step, output_dir):
    plt.figure(figsize=(6, 5))
    plt.imshow(u.cpu().numpy(), cmap='coolwarm', origin='lower')
    plt.colorbar(label='Phase Field (u)')
    plt.title(f'Cahn-Hilliard Phase Separation - Step {step}')
    
    filepath = os.path.join(output_dir, f'ch_step_{step:05d}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath



def test_cahn_hilliard(N, eps, a, dt, tol, plot_interval=100, max_steps=10000):
    """
    A simple 2D Cahn-Hilliard equation simulation.

    Args:
        N: Grid size
        eps: Gradient coefficient
        a: Convexity splitting parameter
        dt: Time step size
        max_steps: Maximum number of steps to prevent infinite execution
        plot_interval: Interval for saving plots
        tol: Convergence tolerance
    """

    # Assuming a square periodic domain length of L = 2*pi
    L = 2.0 * torch.pi
    dx = L / N

    freqs = torch.fft.fftfreq(N, d=dx, device=device, dtype=torch.float32) * 2.0 * torch.pi
    KY, KX = torch.meshgrid(freqs, freqs, indexing='ij')
    k2 = KX**2 + KY**2
    k4 = k2**2

    denom = 1.0 + dt * (eps**2 * k4 + a * k2)
    
    u_n = 0.1 * (torch.rand((N, N), dtype=torch.float32, device=device) - 0.5)

    output_dir = "ch_results"
    os.makedirs(output_dir, exist_ok=True)

    for step in range(max_steps + 1):
        
        if step % plot_interval == 0 or step == 10:
            filepath = plot_state_and_save(u_n, step, output_dir)
            print(f"Saved: {filepath}")
        
        u_new = step_cahn_hilliard(u_n, dt, eps, a, k2, denom)
        
        max_diff = torch.max(torch.abs(u_new - u_n)).item()
        
        if max_diff < tol:
            print(f"Simulation converged at step {step} (Max diff: {max_diff:.2e})")
            filepath = plot_state_and_save(u_new, step, output_dir)
            print(f"Saved converged state: {filepath}")
            break
            
        u_n = u_new
    else:
        print(f"Simulation reached maximum steps ({max_steps}) without strict convergence.")    



if __name__ == "__main__":
    N = 128            
    eps = 0.1          
    a = 2.0           
    dt = 0.01          
    max_steps = 15000  
    plot_interval = 100 
    tol = 1e-3        
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Cahn-Hilliard Simulator on: {device}")

    test_cahn_hilliard(N, eps, a, dt, tol, plot_interval, max_steps)
    
