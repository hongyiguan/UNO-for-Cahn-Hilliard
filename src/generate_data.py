"""
MIT License - Copyright (c) 2026 Hongyi Guan
See LICENSE file for full license text
"""

"""
Builds the supervised dataset for the neural-operator surrogate.
 
Runs the spectral solver from CH_FFT.py on N_systems independent random initial
conditions (uniform noise of amplitude 0.1 about zero mean) on a 64x64 periodic
grid, recording input/target pairs separated by STRIDE = 10 solver steps, i.e.
the map u(t) -> u(t + 10*dt) with dt = 0.01. The first 10-step window of every 
trajectory is discarded so that samples begin after the initial noise has 
organised into an incipient interface pattern rather than from pure noise. All 
pairs from all trajectories are pooled, shuffled once, and split into 5000 training 
and 1000 test samples, then written to data/ch_train.pt and data/ch_test.pt as 
dictionaries of stacked tensors of shape (N, 1, 64, 64). 
Run this once before any training script.
"""

import torch
import matplotlib.pyplot as plt
import os
from CH_FFT import step_cahn_hilliard

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_CH_data(N_samples=6000, N_grid=64, N_systems=20):
    eps = 0.1          
    a = 2.0           
    dt = 0.01   
    
    X = torch.zeros((N_samples, 1, N_grid, N_grid))
    Y = torch.zeros((N_samples, 1, N_grid, N_grid))

    L = L = 2.0 * torch.pi
    dx = L / N_grid

    freqs = torch.fft.fftfreq(N_grid, d=dx, device=device, dtype=torch.float32) * 2.0 * torch.pi
    KY, KX = torch.meshgrid(freqs, freqs, indexing='ij')
    k2 = KX**2 + KY**2
    k4 = k2**2

    denom = 1.0 + dt * (eps**2 * k4 + a * k2)
    
    N_steps = (N_samples // N_systems) * 10 + 10

    ptr = 0

    for s in range(N_systems):
        print(f"Generating data with the {s+1}-th system.")
        u_n = 0.1 * (torch.rand((N_grid, N_grid), dtype=torch.float32, device=device) - 0.5)
        u_prev_10 = u_n.clone()
        for nstep in range(N_steps):
            u_n = step_cahn_hilliard(u_n, dt, eps, a, k2, denom)
            if nstep > 0 and (nstep+1) % 10 == 0:
                if nstep > 10:
                    X[ptr,0,:,:] = u_prev_10.cpu()
                    Y[ptr,0,:,:] = u_n.cpu()
                    ptr += 1
                
                u_prev_10 = u_n.clone()
    
    return X, Y

def save_datasets(train_size=5000, test_size=1000):
    total_size = train_size + test_size
    X, Y = generate_CH_data(N_samples=total_size)

    indices = torch.randperm(total_size)

    X = X[indices]
    Y = Y[indices]

    X_train, X_test = X[:train_size].clone(), X[train_size:].clone()
    Y_train, Y_test = Y[:train_size].clone(), Y[train_size:].clone()

    os.makedirs('data', exist_ok=True)

    torch.save({'x': X_train, 'y': Y_train}, 'data/ch_train.pt')
    torch.save({'x': X_test, 'y': Y_test}, 'data/ch_test.pt')

if __name__ == "__main__":
    save_datasets()






    
