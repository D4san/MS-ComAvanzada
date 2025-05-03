# model/plotting.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Añadido
import os           # Añadido
from model.simulation import simulate_daisyworld

def plot_recovered_simulations(theta_median, labels,
                               fixed_T_opt, fixed_T_tol, fixed_A_bare,
                               target_black, target_white,
                               save_fig=False, filename="recovered_simulation.png"):
    """
    Grafica las simulaciones obtenidas usando los parámetros recuperados (mediana y último)
    y agrega líneas horizontales en la gráfica de cobertura para indicar los valores deseados.
    
    Args:
        theta_median (array_like): Parámetros recuperados (mediana) en el orden
            [L, y_mort, a_black_init, a_white_init, A_black, A_white].
        labels (list): Lista de nombres de parámetros.
        fixed_T_opt, fixed_T_tol, fixed_A_bare: Valores fijos para la simulación.
        target_black, target_white: Valores deseados de cobertura para margaritas negras y blancas.
        save_fig (bool): Si True, guarda la figura.
        filename (str): Nombre del archivo donde se guarda la figura.
    
    Returns:
        La figura generada (para poder mostrarla o guardarla).
    """
    # Extraer parámetros para la mediana
    (L_med, y_mort_med, a_b_init_med, a_w_init_med, A_black_med, A_white_med) = theta_median

    # Ejecutar simulación con los parámetros recuperados
    sim_med = simulate_daisyworld(
        a_black_init=a_b_init_med,
        a_white_init=a_w_init_med,
        t_max=100, y_mort=y_mort_med,
        A_black=A_black_med, A_white=A_white_med,
        A_bare=fixed_A_bare, L=L_med,
        q_prime=20, P=1, p=1, S=780,
        T_opt=fixed_T_opt, T_tol=fixed_T_tol, num_points=100
    )

    t_med, a_black_med, a_white_med, T_eff_med, _, _, _ = sim_med

    # Crear figura con dos subplots: uno para cobertura, otro para temperatura
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfica de cobertura con líneas horizontales en los objetivos
    axes[0].plot(t_med, a_black_med, color="k", linewidth=2, label="Margaritas Negras (Mediana)")
    axes[0].plot(t_med, a_white_med, color="gold", linewidth=2, label="Margaritas Blancas (Mediana)")
    axes[0].axhline(y=target_black, color="k", linestyle=":", alpha=0.6, label="Objetivo Negras")
    axes[0].axhline(y=target_white, color="gold", linestyle=":", alpha=0.6, label="Objetivo Blancas")
    axes[0].set_xlabel("Tiempo")
    axes[0].set_ylabel("Fracción de Cobertura")
    axes[0].set_title("Evolución de Cobertura")
    axes[0].legend()

    # Gráfica de temperatura
    axes[1].plot(t_med, T_eff_med, color="darkred", linewidth=2, label="T_eff")
    axes[1].axhline(y=fixed_T_opt, color="forestgreen", linestyle="-", alpha=0.7, linewidth=1.5, label="T_opt")
    axes[1].fill_between(t_med, fixed_T_opt - fixed_T_tol, fixed_T_opt + fixed_T_tol, color="forestgreen",
                         alpha=0.15, label="± T_tol")
    axes[1].set_xlabel("Tiempo")
    axes[1].set_ylabel("Temperatura (°C)")
    axes[1].set_title("Evolución de Temperatura")
    axes[1].legend()
    
    fig.suptitle("Simulación con Parámetros Recuperados", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_fig:
        fig.savefig(filename, dpi=300)
        
    plt.show()
    
    return fig


def plot_scaling_results(base_dir='..'): # Añadir base_dir para flexibilidad
    """
    Genera gráficos comparando el tiempo de ejecución de diferentes métodos
    (Secuencial, Joblib, MPI) vs. número de walkers, y la escalabilidad de MPI
    vs. número de procesos.

    Lee datos desde archivos CSV ubicados relativemente a base_dir.

    Args:
        base_dir (str): Directorio base desde donde se buscan 'data' y 'plots'.
                        Por defecto es '..', asumiendo que plotting.py está en 'model/'.
    """
    data_dir = os.path.join(base_dir, "data")
    plots_dir = os.path.join(base_dir, "plots")
    sequential_csv = os.path.join(base_dir, "results_sequential.csv") # Ajustado
    joblib_csv = os.path.join(base_dir, "results_joblib.csv")       # Ajustado
    mpi_scaling_csv = os.path.join(data_dir, "results_mpi_scaling.csv")
    # Usar results_mpi.csv si existe para comparar walkers MPI
    mpi_incremental_csv = os.path.join(base_dir, "results_mpi.csv") 

    # Crear directorio de plots si no existe
    if not os.path.exists(plots_dir):
        try:
            os.makedirs(plots_dir)
            print(f"Directorio creado: {plots_dir}")
        except OSError as e:
            print(f"Error creando directorio {plots_dir}: {e}")
            return # Salir si no se puede crear el directorio

    plt.style.use('seaborn-v0_8-darkgrid') # Estilo de gráfico

    # --- Gráfico 1: Tiempo vs. Número de Walkers --- 
    plt.figure(figsize=(10, 6))
    plot1_generated = False

    try:
        df_seq = pd.read_csv(sequential_csv)
        plt.plot(df_seq['n_walkers'], df_seq['time'], marker='o', linestyle='-', label='Secuencial')
        plot1_generated = True
    except FileNotFoundError:
        print(f"Advertencia: No se encontró {sequential_csv}")

    try:
        df_joblib = pd.read_csv(joblib_csv)
        plt.plot(df_joblib['n_walkers'], df_joblib['time'], marker='s', linestyle='--', label='Joblib (CPU)')
        plot1_generated = True
    except FileNotFoundError:
        print(f"Advertencia: No se encontró {joblib_csv}")

    # Intentar añadir datos de MPI si existen y tienen variación de walkers (desde results_mpi.csv)
    try:
        df_mpi_inc = pd.read_csv(mpi_incremental_csv)
        # Asumiendo que results_mpi.csv tiene 'n_walkers' y 'time'
        if 'n_walkers' in df_mpi_inc.columns and 'time' in df_mpi_inc.columns:
            # Agrupar por n_walkers y tomar la media del tiempo (si hay múltiples corridas por walker)
            mpi_time_vs_walkers = df_mpi_inc.groupby('n_walkers')['time'].mean().reset_index()
            if len(mpi_time_vs_walkers) > 1: # Solo graficar si hay más de un punto
                plt.plot(mpi_time_vs_walkers['n_walkers'], mpi_time_vs_walkers['time'], marker='^', linestyle=':', label='MPI (Variable Walkers)')
                plot1_generated = True
        else:
             print(f"Advertencia: {mpi_incremental_csv} no tiene las columnas 'n_walkers' o 'time'.")
    except FileNotFoundError:
        print(f"Advertencia: No se encontró {mpi_incremental_csv} para comparación de walkers MPI.")
    except KeyError as e:
        print(f"Advertencia: Error de clave '{e}' al procesar {mpi_incremental_csv}.")

    if plot1_generated:
        plt.xlabel("Número Total de Walkers")
        plt.ylabel("Tiempo de Ejecución (segundos)")
        plt.title("Comparación de Tiempo de Ejecución vs. Número de Walkers")
        plt.legend()
        plt.grid(True)
        plot1_filename = os.path.join(plots_dir, "time_vs_walkers_comparison.png")
        try:
            plt.savefig(plot1_filename, dpi=300)
            print(f"Gráfico 1 guardado en: {plot1_filename}")
        except Exception as e:
            print(f"Error guardando gráfico 1 ({plot1_filename}): {e}")
        # plt.show() # Descomentar si se quiere mostrar interactivamente
        plt.close()
    else:
        print("No se generó el Gráfico 1 porque no se encontraron datos suficientes.")
        plt.close() # Cerrar la figura vacía

    # --- Gráfico 2: Tiempo vs. Número de Procesos MPI (para 700 walkers) ---
    try:
        df_mpi_scale = pd.read_csv(mpi_scaling_csv)
        # Asegurarse que las columnas necesarias existen
        if not all(col in df_mpi_scale.columns for col in ['n_total_walkers', 'n_processes', 'time']):
             raise KeyError("Faltan columnas requeridas en results_mpi_scaling.csv")

        # Filtrar por 700 walkers (o el valor que se usó en mpi_process_scaling.py)
        # Podríamos hacerlo más genérico si hubiera múltiples corridas de escalabilidad
        target_walkers = 700 # Asumiendo que la escalabilidad se midió con 700
        df_mpi_target = df_mpi_scale[df_mpi_scale['n_total_walkers'] == target_walkers]

        if not df_mpi_target.empty:
            plt.figure(figsize=(10, 6))
            # Ordenar por número de procesos para el gráfico
            df_mpi_target = df_mpi_target.sort_values(by='n_processes')
            plt.plot(df_mpi_target['n_processes'], df_mpi_target['time'], marker='o', linestyle='-', color='green')
            plt.xlabel("Número de Procesos MPI")
            plt.ylabel("Tiempo de Ejecución (segundos)")
            plt.title(f"Escalabilidad MPI: Tiempo vs. Número de Procesos ({target_walkers} Walkers)")
            # Asegurar ticks en los números de procesos usados, si son pocos
            unique_processes = df_mpi_target['n_processes'].unique()
            if len(unique_processes) < 15: # Poner ticks explícitos si no son demasiados
                 plt.xticks(unique_processes)
            plt.grid(True)
            plot2_filename = os.path.join(plots_dir, "mpi_scaling_time_vs_processes.png")
            try:
                plt.savefig(plot2_filename, dpi=300)
                print(f"Gráfico 2 guardado en: {plot2_filename}")
            except Exception as e:
                print(f"Error guardando gráfico 2 ({plot2_filename}): {e}")
            # plt.show() # Descomentar si se quiere mostrar interactivamente
            plt.close()
        else:
            print(f"Advertencia: No se encontraron datos para {target_walkers} walkers en {mpi_scaling_csv}")

    except FileNotFoundError:
        print(f"Advertencia: No se encontró el archivo de escalabilidad MPI: {mpi_scaling_csv}")
    except KeyError as e:
        print(f"Error: Falta la columna '{e}' o estructura incorrecta en {mpi_scaling_csv}")
    except Exception as e:
        print(f"Error inesperado procesando {mpi_scaling_csv}: {e}")

    print("Proceso de graficación de escalabilidad completado.")

# --- Fin de la nueva función ---
