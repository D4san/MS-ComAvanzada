# main_joblib_mcmc.py
import numpy as np
import corner
import matplotlib.pyplot as plt

# Se importa run_mcmc, que utiliza JoblibPool internamente, y la función de gráficas.
from model.mcmc import run_mcmc
from model.plotting import plot_recovered_simulations

def main():
    n_dim = 6
    n_walkers = 500
    np.random.seed(42)
    initial_positions = np.column_stack([
        np.random.uniform(0.1, 3.0, n_walkers),    # L
        np.random.uniform(0.01, 0.9, n_walkers),    # y_mort
        np.random.uniform(0, 1, n_walkers),         # a_black_init
        np.random.uniform(0, 1, n_walkers),         # a_white_init
        np.random.uniform(0, 1, n_walkers),         # A_black
        np.random.uniform(0, 1, n_walkers)          # A_white
    ])
    
    n_burn_in = 200
    n_samples = 700
    
    print("Ejecutando MCMC con Joblib...")
    sampler = run_mcmc(initial_positions, n_walkers, n_dim, n_burn_in, n_samples, cores=-1)
    
    samples = sampler.get_chain(flat=True)
    labels = ["L", "y_mort", "a_black_init", "a_white_init", "A_black", "A_white"]
    theta_median = np.median(samples, axis=0)
    print("Resultados (Joblib):", dict(zip(labels, theta_median)))
    
    # --- Corner Plot ---
    fig_corner = corner.corner(samples, labels=labels, bins=30,
                               show_titles=True, quantiles=[0.16, 0.5, 0.84])
    fig_corner.suptitle("Distribución Posterior MCMC (Joblib)", fontsize=20)
    fig_corner.savefig("corner_joblib.png", dpi=300)
    plt.show()
    
    # --- Parámetros fijos y objetivos para la simulación ---
    fixed_T_opt = 50     # Ejemplo: Temperatura óptima fija
    fixed_T_tol = 17.5   # Tolerancia
    fixed_A_bare = 0.5   # Albedo del suelo desnudo
    target_black = 0.4   # Cobertura deseada para margaritas negras
    target_white = 0.4   # Cobertura deseada para margaritas blancas
    
    # --- Gráfico de cobertura y temperatura con los parámetros recuperados ---
    # Se asume que plot_recovered_simulations muestra ambas gráficas y guarda la figura.
    plot_recovered_simulations(theta_median, theta_median, labels,
                               fixed_T_opt, fixed_T_tol, fixed_A_bare,
                               target_black, target_white,
                               save_fig=True, filename="recovered_simulation_joblib.png")

if __name__ == "__main__":
    main()