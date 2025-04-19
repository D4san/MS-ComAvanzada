import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# Lee los archivos CSV
df_seq = pd.read_csv("results_sequential.csv")
df_job = pd.read_csv("results_joblib.csv")
df_mpi = pd.read_csv("results_mpi.csv").sort_values(by='n_walkers', ascending=True)


# Crear subplots compartiendo eje X para comparar Tiempo y Error Promedio versus Walkers
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10), sharex=True)

# Gráfico 1: Tiempo de ejecución vs. Número de Walkers
ax1.plot(df_seq["n_walkers"], df_seq["time"], 'o-', label="Sequential", color="blue", markersize=8)
ax1.plot(df_job["n_walkers"], df_job["time"], 's-', label="Joblib", color="green", markersize=8)
ax1.plot(df_mpi["n_walkers"], df_mpi["time"], '^-', label="MPI", color="red", markersize=8)
ax1.set_ylabel("Tiempo (s)", fontsize=12)
ax1.legend(fontsize=10)
ax1.set_title("Tiempo de ejecución vs. Número de Walkers", fontsize=14)

# Gráfico 2: Error Promedio vs. Número de Walkers
ax2.plot(df_seq["n_walkers"], df_seq["avg_error"], 'o-', label="Sequential", color="blue", markersize=8)
ax2.plot(df_job["n_walkers"], df_job["avg_error"], 's-', label="Joblib", color="green", markersize=8)
ax2.plot(df_mpi["n_walkers"], df_mpi["avg_error"], '^-', label="MPI", color="red", markersize=8)
ax2.set_xlabel("Número de Walkers", fontsize=12)
ax2.set_ylabel("Error Promedio", fontsize=12)
ax2.set_title("Error Promedio vs. Número de Walkers", fontsize=14)

fig.suptitle("Comparación de estrategias MCMC", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("comparacion_estrategias_mcmc.png", dpi=300)
plt.show()
