# experiment_mpi_mcmc_gather.py

from mpi4py import MPI
import numpy as np
import emcee
import time
import matplotlib.pyplot as plt
import corner
import pandas as pd

# Importamos funciones del proyecto
from model.mcmc import log_probability
from model.simulation import simulate_daisyworld
from model.plotting import plot_recovered_simulations

# Inicialización MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print(f"Soy el proceso {rank} de {size} procesos")

# --- Parámetros del MCMC y de la simulación ---
n_dim = 6
n_total_walkers = 200  # Total de walkers deseados para toda la cadena
# Distribuimos equitativamente entre procesos
walkers_per_proc = n_total_walkers // size

n_burn_in = 200
n_samples = 700

# Parámetros fijos y objetivos para la simulación de Daisyworld
fixed_T_opt = 50     
fixed_T_tol = 17.5   
fixed_A_bare = 0.5   
target_black = 0.4   
target_white = 0.4   

# --- 1. Root genera las posiciones iniciales y las reparte ---
if rank == 0:
    np.random.seed(42)
    initial_positions = np.column_stack([
        np.random.uniform(0.1, 3.0, n_total_walkers),    # L
        np.random.uniform(0.01, 0.9, n_total_walkers),    # y_mort
        np.random.uniform(0, 1, n_total_walkers),         # a_black_init
        np.random.uniform(0, 1, n_total_walkers),         # a_white_init
        np.random.uniform(0, 1, n_total_walkers),         # A_black
        np.random.uniform(0, 1, n_total_walkers)          # A_white
    ])
    # Particiona la matriz en subgrupos para cada proceso
    positions_split = np.array_split(initial_positions, size)
else:
    positions_split = None

# Cada proceso recibe su subgrupo de posiciones iniciales
local_positions = comm.scatter(positions_split, root=0)

# --- 2. Cada proceso corre su MCMC local ---
n_local_walkers = local_positions.shape[0]

start_local = MPI.Wtime()
sampler = emcee.EnsembleSampler(n_local_walkers, n_dim, log_probability)
state = sampler.run_mcmc(local_positions, n_burn_in, progress=(rank==0))
sampler.reset()
sampler.run_mcmc(state, n_samples, progress=(rank==0))
end_local = MPI.Wtime()

local_time = end_local - start_local

# Cada proceso obtiene su cadena (flattened: de forma 2D: (n_samples*n_local_walkers, n_dim))
local_chain = sampler.get_chain(flat=True)

# --- 3. Gather de las cadenas completas ---
# Cada proceso envía su cadena al proceso raíz
all_chains = comm.gather(local_chain, root=0)
all_times = comm.gather(local_time, root=0)

if rank == 0:
    # Concatenamos todas las cadenas obtenidas (por ejemplo, a lo largo de la primera dimensión)
    big_chain = np.concatenate(all_chains, axis=0)
    # Calculamos la mediana global para cada parámetro (a lo largo de las muestras)
    global_theta_median = np.median(big_chain, axis=0)
    
    # Imprimimos los tiempos individuales
    print("Tiempos de cada proceso (s):", all_times)
    print("Tiempo total estimado (máximo):", np.max(all_times))
    print("Parámetros recuperados (global, mediana):", global_theta_median)
    
    # --- 4. Ejecutamos la simulación con los parámetros globales ---
    sim = simulate_daisyworld(
         a_black_init=global_theta_median[2],
         a_white_init=global_theta_median[3],
         t_max=100,
         y_mort=global_theta_median[1],
         A_black=global_theta_median[4],
         A_white=global_theta_median[5],
         A_bare=fixed_A_bare,
         L=global_theta_median[0],
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
    print(f"Error final: Margaritas Negras = {error_black:.4f}, Blancas = {error_white:.4f}, Promedio = {avg_error:.4f}")
    
    # --- 5. Generamos y guardamos gráficos ---
    # Corner plot de la gran cadena
    fig_corner = corner.corner(big_chain, labels=["L", "y_mort", "a_black_init", "a_white_init", "A_black", "A_white"],
                               bins=30, show_titles=True, quantiles=[0.16, 0.5, 0.84])
    fig_corner.suptitle("Distribución Posterior MCMC (MPI, Gran Cadena)", fontsize=20)
    fig_corner.savefig("corner_mpi_gran_cadena.png", dpi=300)
    plt.show()
    
    # Gráfico de evolución de coverage and temperature usando global_theta_median
    plot_recovered_simulations(global_theta_median, global_theta_median,
                               labels=["L", "y_mort", "a_black_init", "a_white_init", "A_black", "A_white"],
                               fixed_T_opt=fixed_T_opt, fixed_T_tol=fixed_T_tol, fixed_A_bare=fixed_A_bare,
                               target_black=target_black, target_white=target_white,
                               save_fig=True, filename="recovered_simulation_mpi.png")
    
    # Opcional: guardar resultados en CSV
    results = {
       'n_total_walkers': n_total_walkers,
       'n_walkers_por_proc': walkers_per_proc,
       'tiempo_max': np.max(all_times),
       'avg_error': avg_error
    }
    df = pd.DataFrame([results])
    df.to_csv("results_mpi.csv", index=False)
    
comm.Barrier()
if rank == 0:
    print("Proceso MPI COMPLETO.")
