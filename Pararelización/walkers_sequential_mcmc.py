# experiment_sequential_mcmc.py

import time
import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
from tqdm import tqdm
import pandas as pd

from model.mcmc import log_probability
from model.simulation import simulate_daisyworld
from model.plotting import plot_recovered_simulations

walker_values = [50, 100, 200, 300, 400, 500, 700]
n_dim = 6
n_burn_in = 200
n_samples = 700

fixed_T_opt = 50
fixed_T_tol = 17.5
fixed_A_bare = 0.5
target_black = 0.4
target_white = 0.4

results = []

for n_walkers in tqdm(walker_values, desc="Evaluando número de walkers"):
    
    initial_positions = np.column_stack([
        np.random.uniform(0.1, 3.0, n_walkers),    # L
        np.random.uniform(0.01, 0.9, n_walkers),    # y_mort
        np.random.uniform(0, 1, n_walkers),         # a_black_init
        np.random.uniform(0, 1, n_walkers),         # a_white_init
        np.random.uniform(0, 1, n_walkers),         # A_black
        np.random.uniform(0, 1, n_walkers)          # A_white
    ])

    start_time = time.time()
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability)
    
    state = sampler.run_mcmc(initial_positions, n_burn_in, progress=True)
    sampler.reset()
    sampler.run_mcmc(state, n_samples, progress=True)
    end_time = time.time()
    
    comp_time = end_time - start_time
        
    samples = sampler.get_chain(flat=True)
    theta_median = np.median(samples, axis=0)
        
    sim = simulate_daisyworld(
            a_black_init=theta_median[2],
            a_white_init=theta_median[3],
            t_max=100,
            y_mort=theta_median[1],
            A_black=theta_median[4],
            A_white=theta_median[5],
            A_bare=fixed_A_bare,
            L=theta_median[0],
            q_prime=20,
            T_opt=fixed_T_opt,
            T_tol=fixed_T_tol,
            num_points=100
        )
    final_a_black = sim[1][-1]
    final_a_white = sim[2][-1]
    error_black = abs(final_a_black - target_black)
    error_white = abs(final_a_white - target_white)
    avg_error = (error_black + error_white) / 2.0

    results.append({
        'n_walkers': n_walkers,
        'time': comp_time,
        'error_black': error_black,
        'error_white': error_white,
        'avg_error': avg_error
    })

    tqdm.write(f"Walkers: {n_walkers} | Tiempo: {comp_time:.2f} s | Error medio: {avg_error:.4f}")

fig, ax1 = plt.subplots()
ax1.plot(walker_values, [r['time'] for r in results], marker='o', color='blue', label='Tiempo de ejecución (s)')
ax1.set_xlabel('Número de Walkers')
ax1.set_ylabel('Tiempo (s)', color='blue')
ax1.tick_params('y', colors='blue')
ax2 = ax1.twinx()
ax2.plot(walker_values, [r['avg_error'] for r in results], marker='s', color='red', label='Error Medio')
ax2.set_ylabel('Error Medio', color='red')
ax2.tick_params('y', colors='red')
fig.suptitle('Tiempo y Error vs. Número de Walkers (Secuencial)')
fig.tight_layout()
fig.savefig("results_sequential.png", dpi=300)
plt.show()

df = pd.DataFrame(results)
df.to_csv("results_sequential.csv", index=False)

fig_corner = corner.corner(samples, labels=["L", "y_mort", "a_black_init", "a_white_init", "A_black", "A_white"],
                           bins=30, show_titles=True, quantiles=[0.16, 0.5, 0.84])
fig_corner.suptitle("Distribución Posterior MCMC (Secuencial, 700 Walkers)", fontsize=20)
fig_corner.savefig("corner_sequential.png", dpi=300)
plt.show()

plot_recovered_simulations(theta_median, theta_median,
                           labels=["L", "y_mort", "a_black_init", "a_white_init", "A_black", "A_white"],
                           fixed_T_opt=fixed_T_opt, fixed_T_tol=fixed_T_tol, fixed_A_bare=fixed_A_bare,
                           target_black=target_black, target_white=target_white,
                           save_fig=True, filename="recovered_simulation_sequential.png")
