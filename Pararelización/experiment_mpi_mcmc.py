# experiment_mpi_mcmc.py

from mpi4py import MPI
import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
import pandas as pd
import time
from tqdm import tqdm

# Importamos funciones de nuestro proyecto
from model.mcmc import log_probability
from model.simulation import simulate_daisyworld
from model.plotting import plot_recovered_simulations

# Inicialización de MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print(f"Soy el proceso {rank} de {size} procesos")

# Lista de distintos números totales de walkers a evaluar
walker_values = [50, 100, 200, 300, 400, 500, 700]
n_dim = 6
n_burn_in = 200
n_samples = 700

# Parámetros fijos y objetivos para la simulación de Daisyworld
fixed_T_opt = 50    # Temperatura óptima
fixed_T_tol = 17.5  # Tolerancia
fixed_A_bare = 0.5  # Albedo del suelo desnudo
target_black = 0.4  # Cobertura deseada para margaritas negras
target_white = 0.4  # Cobertura deseada para margaritas blancas

results = []  # Almacena los resultados por cada configuración de walkers

# Usamos tqdm solo en el proceso raíz para mostrar el progreso
if rank == 0:
    pbar = tqdm(walker_values, desc="Evaluando número de walkers (MPI)")
else:
    pbar = walker_values

for n_total_walkers in pbar:
    # Distribuimos equitativamente los walkers entre los procesos
    walkers_per_proc = n_total_walkers // size
    # El proceso raíz genera todas las posiciones iniciales
    if rank == 0:
        np.random.seed(42)  # Para reproducibilidad
        initial_positions = np.column_stack([
            np.random.uniform(0.1, 3.0, n_total_walkers),    # L
            np.random.uniform(0.01, 0.9, n_total_walkers),    # y_mort
            np.random.uniform(0, 1, n_total_walkers),         # a_black_init
            np.random.uniform(0, 1, n_total_walkers),         # a_white_init
            np.random.uniform(0, 1, n_total_walkers),         # A_black
            np.random.uniform(0, 1, n_total_walkers)          # A_white
        ])
        # Particionamos la matriz en tantos subgrupos como procesos
        positions_split = np.array_split(initial_positions, size)
    else:
        positions_split = None

    # Cada proceso recibe su subgrupo de posiciones iniciales
    local_positions = comm.scatter(positions_split, root=0)
    n_local_walkers = local_positions.shape[0]

    # Sincronizamos antes de iniciar la medición
    comm.Barrier()
    start_time = MPI.Wtime()

    # Cada proceso ejecuta su MCMC local
    sampler = emcee.EnsembleSampler(n_local_walkers, n_dim, log_probability)
    state = sampler.run_mcmc(local_positions, n_burn_in, progress=(rank==0))
    sampler.reset()
    sampler.run_mcmc(state, n_samples, progress=(rank==0))
    end_time = MPI.Wtime()
    local_time = end_time - start_time

    # Cada proceso obtiene su cadena local (flattened)
    local_chain = sampler.get_chain(flat=True)  # Dimensión: (n_samples*n_local_walkers, n_dim)

    # Recolectamos todas las cadenas locales y tiempos en el proceso raíz
    all_chains = comm.gather(local_chain, root=0)
    all_times = comm.gather(local_time, root=0)

    if rank == 0:
        # Concatenamos para formar la gran cadena del MCMC
        global_chain = np.concatenate(all_chains, axis=0)
        # Calculamos la mediana global para cada parámetro
        global_theta_median = np.median(global_chain, axis=0)

        # Ejecutamos la simulación con los parámetros globales
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
        # Extraemos la cobertura final para cada especie
        final_a_black = sim[1][-1]
        final_a_white = sim[2][-1]
        error_black = abs(final_a_black - target_black)
        error_white = abs(final_a_white - target_white)
        avg_error = (error_black + error_white) / 2.0

        # Guardamos los resultados en la lista
        results.append({
            'n_total_walkers': n_total_walkers,
            'walkers_per_proc': walkers_per_proc,
            'max_time': np.max(all_times),
            'avg_time': np.mean(all_times),
            'error_black': error_black,
            'error_white': error_white,
            'avg_error': avg_error
        })
        
        # Imprimimos información para este número de walkers
        pbar.write(f"Walkers: {n_total_walkers} | Tiempo: {np.max(all_times):.2f} s (max) | Error medio: {avg_error:.4f}")

# --- Final del bucle: el proceso raíz genera y guarda gráficos y CSV ---
if rank == 0:
    # Convertimos los resultados en un DataFrame y los guardamos en un CSV
    df = pd.DataFrame(results)
    df.to_csv("results_mpi.csv", index=False)

    # Graficamos Tiempo y Error vs. Número de Walkers
    fig, ax1 = plt.subplots()
    ax1.plot([r['n_total_walkers'] for r in results],
             [r['max_time'] for r in results], marker='o', color='blue', label='Tiempo (s)')
    ax1.set_xlabel('Número de Walkers')
    ax1.set_ylabel('Tiempo (s)', color='blue')
    ax1.tick_params(axis='y', colors='blue')
    ax2 = ax1.twinx()
    ax2.plot([r['n_total_walkers'] for r in results],
             [r['avg_error'] for r in results], marker='s', color='red', label='Error Medio')
    ax2.set_ylabel('Error Medio', color='red')
    ax2.tick_params(axis='y', colors='red')
    fig.suptitle('Tiempo y Error vs. Número de Walkers (MPI)')
    fig.tight_layout()
    fig.savefig("time_error_mpi.png", dpi=300)
    plt.show()

    # También podemos generar un corner plot para el último experimento (por ejemplo, con n_total_walkers igual al último valor)
    fig_corner = corner.corner(global_chain, labels=["L", "y_mort", "a_black_init", "a_white_init", "A_black", "A_white"],
                               bins=30, show_titles=True, quantiles=[0.16, 0.5, 0.84])
    fig_corner.suptitle("Distribución Posterior MCMC (MPI)", fontsize=20)
    fig_corner.savefig("corner_mpi.png", dpi=300)
    plt.show()

    # Graficamos la evolución de coverage and temperature usando global_theta_median
    plot_recovered_simulations(global_theta_median, global_theta_median,
                               labels=["L", "y_mort", "a_black_init", "a_white_init", "A_black", "A_white"],
                               fixed_T_opt=fixed_T_opt, fixed_T_tol=fixed_T_tol, fixed_A_bare=fixed_A_bare,
                               target_black=target_black, target_white=target_white,
                               save_fig=True, filename="recovered_simulation_mpi.png")
    print("Proceso MPI COMPLETO. Resultados guardados.")

comm.Barrier()
