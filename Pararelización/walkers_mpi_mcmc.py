# experiment_mpi_incremental.py

from mpi4py import MPI
import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
import pandas as pd
import time
import os

from model.mcmc import log_probability
from model.simulation import simulate_daisyworld
from model.plotting import plot_recovered_simulations

# Inicialización MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print(f"Soy el proceso {rank} de {size} procesos")

# ------------------------------------
# CONFIGURACIÓN MANUAL:
# ------------------------------------
n_dim = 6
# Define manualmente el total de walkers para esta corrida
n_total_walkers = 50    # 50, 100, 200, 300, 400, 500, 700
walkers_per_proc = n_total_walkers // size
n_burn_in = 200
n_samples = 700

# Parámetros fijos para la simulación de Daisyworld
fixed_T_opt = 50     
fixed_T_tol = 17.5   
fixed_A_bare = 0.5   
target_black = 0.4   
target_white = 0.4   

# Archivo CSV a actualizar (se agrega una fila en cada ejecución)
csv_filename = "results_mpi_incremental.csv"
if rank == 0 and not os.path.isfile(csv_filename):
    df_init = pd.DataFrame(columns=[
        "run_date", "n_total_walkers", "walkers_per_proc", "time", 
        "error_black", "error_white", "avg_error",
        "L", "y_mort", "a_black_init", "a_white_init", "A_black", "A_white"
    ])
    df_init.to_csv(csv_filename, index=False)

# ------------------------------------
# 1. Generación y Scatter de Posiciones Iniciales
# ------------------------------------
if rank == 0:
    np.random.seed(42)  # Fijamos semilla para reproducibilidad
    initial_positions = np.column_stack([
        np.random.uniform(0.1, 3.0, n_total_walkers),    # L
        np.random.uniform(0.01, 0.9, n_total_walkers),    # y_mort
        np.random.uniform(0, 1, n_total_walkers),         # a_black_init
        np.random.uniform(0, 1, n_total_walkers),         # a_white_init
        np.random.uniform(0, 1, n_total_walkers),         # A_black
        np.random.uniform(0, 1, n_total_walkers)          # A_white
    ])
    positions_split = np.array_split(initial_positions, size)
else:
    positions_split = None

local_positions = comm.scatter(positions_split, root=0)
n_local_walkers = local_positions.shape[0]

# ------------------------------------
# 2. Ejecución Local del MCMC
# ------------------------------------
comm.Barrier()
start_time = MPI.Wtime()

sampler = emcee.EnsembleSampler(n_local_walkers, n_dim, log_probability)
state = sampler.run_mcmc(local_positions, n_burn_in, progress=(rank==0))
sampler.reset()
sampler.run_mcmc(state, n_samples, progress=(rank==0))

end_time = MPI.Wtime()
local_time = end_time - start_time

local_chain = sampler.get_chain(flat=True)

# ------------------------------------
# 3. Recolección de Cadenas y Tiempos
# ------------------------------------
all_chains = comm.gather(local_chain, root=0)
all_times = comm.gather(local_time, root=0)

if rank == 0:
    # Concatenamos las cadenas locales para formar la gran cadena global
    global_chain = np.concatenate(all_chains, axis=0)
    # Calculamos la mediana global de cada parámetro
    global_theta_median = np.median(global_chain, axis=0)
    # Tomamos el tiempo total como el máximo de los tiempos locales
    cycle_time = np.max(all_times)

    # ------------------------------------
    # 4. Simulación y Cálculo del Error
    # ------------------------------------
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

    print("Tiempo total:", cycle_time, "s")
    print("Error final: Negras =", error_black, ", Blancas =", error_white, ", Promedio =", avg_error)
    print("Parámetros recuperados:", global_theta_median)

    # ------------------------------------
    # 5. Guardar Resultados en CSV
    # ------------------------------------
    new_row = {
      "run_date": pd.Timestamp.now(),
      "n_total_walkers": n_total_walkers,
      "walkers_per_proc": walkers_per_proc,
      "time": cycle_time,
      "error_black": error_black,
      "error_white": error_white,
      "avg_error": avg_error,
      "L": global_theta_median[0],
      "y_mort": global_theta_median[1],
      "a_black_init": global_theta_median[2],
      "a_white_init": global_theta_median[3],
      "A_black": global_theta_median[4],
      "A_white": global_theta_median[5]
    }
    df_row = pd.DataFrame([new_row])
    df_existing = pd.read_csv(csv_filename)
    df_new = pd.concat([df_existing, df_row], ignore_index=True)
    df_new.to_csv(csv_filename, index=False)
    print("Datos guardados en", csv_filename)

    # ------------------------------------
    # 6. Generar Gráficos (Opcional)
    # ------------------------------------
    # Corner plot de la gran cadena
    fig_corner = corner.corner(global_chain, 
        labels=["L", "y_mort", "a_black_init", "a_white_init", "A_black", "A_white"],
        bins=30, show_titles=True, quantiles=[0.16, 0.5, 0.84])
    fig_corner.suptitle("Distribución Posterior MCMC (MPI, Manual)", fontsize=20)
    fig_corner.savefig("corner_mpi_manual.png", dpi=300)
    plt.show()

    # Gráfico de evolución de coverage and temps usando global_theta_median
    plot_recovered_simulations(global_theta_median, global_theta_median,
        labels=["L", "y_mort", "a_black_init", "a_white_init", "A_black", "A_white"],
        fixed_T_opt=fixed_T_opt, fixed_T_tol=fixed_T_tol, fixed_A_bare=fixed_A_bare,
        target_black=target_black, target_white=target_white,
        save_fig=True, filename="recovered_simulation_mpi_manual.png")
    
comm.Barrier()
if rank == 0:
    print("Proceso MPI COMPLETO.")
