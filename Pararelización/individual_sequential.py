# main_sequential_mcmc.py
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

# Se importan las funciones definidas en model/mcmc.py y model/plotting.py
from model.mcmc import log_probability
from model.plotting import plot_recovered_simulations

def main():
    n_dim = 6
    n_walkers = 500
    initial_positions = np.column_stack([
        np.random.uniform(0.1, 3.0, n_walkers),    # L
        np.random.uniform(0.01, 0.9, n_walkers),    # y_mort
        np.random.uniform(0, 1, n_walkers),         # a_black_init
        np.random.uniform(0, 1, n_walkers),         # a_white_init
        np.random.uniform(0, 1, n_walkers),         # A_black
        np.random.uniform(0, 1, n_walkers)          # A_white
    ])
    
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability)
    
    n_burn_in = 200
    n_samples = 700
    
    print("Ejecutando burn-in (secuencial)...")
    state = sampler.run_mcmc(initial_positions, n_burn_in, progress=True)
    sampler.reset()
    
    print("Ejecutando muestreo principal (secuencial)...")
    sampler.run_mcmc(state, n_samples, progress=True)
    
    # Extraer la cadena y calcular la mediana de cada parámetro
    samples = sampler.get_chain(flat=True)
    labels = ["L", "y_mort", "a_black_init", "a_white_init", "A_black", "A_white"]
    theta_median = np.median(samples, axis=0)
    print("Resultados (Secuencial):")
    print(dict(zip(labels, theta_median)))
    
    # --- Generar y guardar el corner plot ---
    fig_corner = corner.corner(samples, labels=labels, bins=30,
                          show_titles=True, quantiles=[0.16, 0.5, 0.84])
    fig_corner.suptitle("Distribución Posterior MCMC (Secuencial)", fontsize=20)
    fig_corner.savefig("corner_secuencial.png", dpi=300)
    plt.show()
    
    # --- Parámetros para los gráficos de cobertura y temperatura ---
    # Estos valores fijos y objetivos se pueden definir según tus necesidades.
    fixed_T_opt   = 50     # Temperatura óptima fija
    fixed_T_tol   = 17.5   # Tolerancia fija
    fixed_A_bare  = 0.5    # Albedo del suelo desnudo fijo
    target_black  = 0.4    # Cobertura deseada para margaritas negras
    target_white  = 0.4    # Cobertura deseada para margaritas blancas

    # --- Generar y guardar el gráfico de coverage y temperature con los parámetros recuperados ---
    # La función plot_recovered_simulations se encarga de:
    # - Realizar una nueva simulación usando los parámetros recuperados.
    # - Graficar la evolución de la cobertura y la temperatura.
    # - Añadir en la gráfica de cobertura una línea horizontal en el valor deseado.
    fig_recovered = plot_recovered_simulations(theta_median, theta_median, labels,
                                               fixed_T_opt, fixed_T_tol, fixed_A_bare,
                                               target_black, target_white,
                                               save_fig=True, filename="recovered_simulation_secuencial.png")
    

if __name__ == "__main__":
    main()
