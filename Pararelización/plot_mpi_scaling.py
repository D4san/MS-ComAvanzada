import pandas as pd
import matplotlib.pyplot as plt
import os

# Define las rutas relativas al script
script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, 'results', 'data', 'results_mpi_scaling.csv')
plots_dir = os.path.join(script_dir, 'results', 'plots')
output_plot_path = os.path.join(plots_dir, 'mpi_scaling_time_vs_processes.png')

# Asegurarse de que el directorio de plots exista
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Directorio creado: {plots_dir}")

# Estilo del gráfico
plt.style.use('ggplot')

# Leer los datos CSV
try:
    df_mpi_scale = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de datos en {data_path}")
    exit()
except Exception as e:
    print(f"Error al leer el archivo CSV {data_path}: {e}")
    exit()

# Verificar columnas necesarias
required_columns = ['n_processes', 'time']
if not all(col in df_mpi_scale.columns for col in required_columns):
    print(f"Error: El archivo CSV debe contener las columnas: {required_columns}")
    exit()

# Ordenar por número de procesos para el gráfico
df_mpi_scale = df_mpi_scale.sort_values(by='n_processes')

# Crear el gráfico
plt.figure(figsize=(10, 6))
plt.plot(df_mpi_scale['n_processes'], df_mpi_scale['time'], marker='o', linestyle='-', color='red', markersize=8)

# Configurar etiquetas y título
plt.xlabel("Número de Procesos MPI", fontsize=12)
plt.ylabel("Tiempo de Ejecución (s)", fontsize=12)
plt.title("Escalabilidad MPI: Tiempo vs. Número de Procesos", fontsize=14)

# Añadir ticks explícitos si hay pocos puntos
unique_processes = df_mpi_scale['n_processes'].unique()
if len(unique_processes) < 15:
    plt.xticks(unique_processes)

plt.grid(True)
plt.tight_layout()

# Guardar el gráfico
try:
    plt.savefig(output_plot_path, dpi=300)
    print(f"Gráfico guardado en: {output_plot_path}")
except Exception as e:
    print(f"Error al guardar el gráfico en {output_plot_path}: {e}")

# plt.show()