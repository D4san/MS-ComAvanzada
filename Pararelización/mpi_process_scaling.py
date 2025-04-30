# mpi_process_scaling.py

from mpi4py import MPI
import numpy as np
import emcee
import pandas as pd
import time
import os

from model.mcmc import log_probability

# Inicialización MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() # Número de procesos MPI

if rank == 0:
    print(f"--- Ejecutando con {size} procesos MPI ---")

# ------------------------------------
# CONFIGURACIÓN FIJA:
# ------------------------------------
n_dim = 6
n_total_walkers = 500  # Número fijo de walkers para este experimento
walkers_per_proc = n_total_walkers // size
if n_total_walkers % size != 0 and rank == 0:
    print(f"Advertencia: {n_total_walkers} walkers no es divisible exactamente por {size} procesos.")
    print(f"Cada proceso manejará aproximadamente {walkers_per_proc} walkers.")
    # Ajustar walkers_per_proc para el último proceso si es necesario, aunque scatter lo maneja.
n_burn_in = 200
n_samples = 700

# Directorio y archivo CSV para guardar resultados de escalabilidad
data_dir = "data"
csv_filename = os.path.join(data_dir, "results_mpi_scaling.csv")

# Crear directorio 'data' si no existe y el archivo CSV si es la primera vez
if rank == 0:
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.isfile(csv_filename):
        df_init = pd.DataFrame(columns=["run_date", "n_processes", "n_total_walkers", "time"])
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
    # Dividir las posiciones iniciales entre los procesos
    positions_split = np.array_split(initial_positions, size)
else:
    positions_split = None

# Cada proceso recibe su parte de las posiciones iniciales
local_positions = comm.scatter(positions_split, root=0)
n_local_walkers = local_positions.shape[0]

# ------------------------------------
# 2. Ejecución Local del MCMC
# ------------------------------------
comm.Barrier() # Sincronizar antes de medir el tiempo
start_time = MPI.Wtime()

sampler = emcee.EnsembleSampler(n_local_walkers, n_dim, log_probability)
# Ejecutar burn-in
state = sampler.run_mcmc(local_positions, n_burn_in, progress=False) # No mostrar progreso por proceso
sampler.reset()
# Ejecutar muestreo principal
sampler.run_mcmc(state, n_samples, progress=False) # No mostrar progreso por proceso

end_time = MPI.Wtime()
local_time = end_time - start_time

# No necesitamos la cadena completa, solo el tiempo
# local_chain = sampler.get_chain(flat=True)

# ------------------------------------
# 3. Recolección de Tiempos
# ------------------------------------
# Recolectar los tiempos de ejecución de cada proceso en el proceso 0
all_times = comm.gather(local_time, root=0)

# ------------------------------------
# 4. Guardar Resultados (Solo en Proceso 0)
# ------------------------------------
if rank == 0:
    # Tomamos el tiempo total como el máximo de los tiempos locales
    cycle_time = np.max(all_times)

    print(f"Tiempo total para {size} procesos: {cycle_time:.4f} s")

    # Crear nueva fila para el CSV
    new_row = {
      "run_date": pd.Timestamp.now(),
      "n_processes": size,
      "n_total_walkers": n_total_walkers,
      "time": cycle_time
    }
    df_row = pd.DataFrame([new_row])

    # Leer CSV existente y añadir la nueva fila
    try:
        df_existing = pd.read_csv(csv_filename)
        df_new = pd.concat([df_existing, df_row], ignore_index=True)
    except FileNotFoundError:
        # Si por alguna razón el archivo no se creó antes (aunque debería)
        print(f"Archivo {csv_filename} no encontrado, creando uno nuevo.")
        df_new = df_row

    df_new.to_csv(csv_filename, index=False)
    print(f"Resultados de tiempo guardados en {csv_filename}")

# Sincronización final antes de terminar
comm.Barrier()
if rank == 0:
    print(f"--- Ejecución con {size} procesos COMPLETA ---")