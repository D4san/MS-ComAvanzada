# model/plotting.py
import numpy as np
import matplotlib.pyplot as plt
from model.simulation import simulate_daisyworld

def plot_recovered_simulations(theta_median, theta_last, labels,
                               fixed_T_opt, fixed_T_tol, fixed_A_bare,
                               target_black, target_white,
                               save_fig=False, filename="recovered_simulation.png"):
    """
    Grafica las simulaciones obtenidas usando los parámetros recuperados (mediana y último)
    y agrega líneas horizontales en la gráfica de cobertura para indicar los valores deseados.
    
    Args:
        theta_median (array_like): Parámetros recuperados (mediana) en el orden
            [L, y_mort, a_black_init, a_white_init, A_black, A_white].
        theta_last (array_like): (No necesariamente usado si se quiere solo la mediana).
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
