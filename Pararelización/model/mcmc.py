# model/mcmc.py
import numpy as np
import emcee
from joblib import Parallel, delayed
from model.simulation import simulate_daisyworld

# Parámetros fijos para MCMC (pueden ajustarse desde main_mcmc.py)
fixed_T_opt   = 50    
fixed_T_tol   = 17.5  
fixed_A_bare  = 0.5   
target_black  = 0.4   
target_white  = 0.4   
sigma_area    = 0.02  

def log_prior(theta):
    """Calcula el logaritmo de la probabilidad a priori para un conjunto de parámetros.

    Define los rangos permitidos para cada parámetro. Retorna -inf si algún parámetro
    está fuera de su rango permitido, o 0.0 si todos están dentro.

    Args:
        theta (array_like): Vector de parámetros [L, y_mort, a_b_init, a_w_init, A_black, A_white].

    Returns:
        float: Logaritmo de la probabilidad a priori (0.0 o -inf).
    """
    L, y_mort, a_b_init, a_w_init, A_black, A_white = theta
    if not (0.1 <= L <= 3.0):
        return -np.inf
    if not (0.01 <= y_mort <= 0.9):
        return -np.inf
    if not (0 <= a_b_init <= 1.0 and 0 <= a_w_init <= 1.0):
        return -np.inf
    if not (0 <= A_black <= 1.0 and 0 <= A_white <= 1.0):
        return -np.inf
    if not (A_white > A_black):
        return -np.inf
    return 0.0

def log_likelihood(theta):
    """Calcula el logaritmo de la verosimilitud basado en la simulación de Daisyworld.

    Ejecuta una simulación con los parámetros dados y compara las coberturas finales
    con los valores objetivo. Retorna -inf si la simulación falla.

    Args:
        theta (array_like): Vector de parámetros [L, y_mort, a_b_init, a_w_init, A_black, A_white].

    Returns:
        float: Logaritmo de la verosimilitud.
    """
    L, y_mort, a_b_init, a_w_init, A_black, A_white = theta
    try:
        sim = simulate_daisyworld(
            a_black_init=a_b_init,
            a_white_init=a_w_init,
            t_max=100,
            y_mort=y_mort,
            A_black=A_black,
            A_white=A_white,
            A_bare=fixed_A_bare,
            L=L,
            q_prime=20,
            P=1, p=1, S=780,
            T_opt=fixed_T_opt, T_tol=fixed_T_tol,
            num_points=100
        )
        a_black_final = sim[1][-1]
        a_white_final = sim[2][-1]
    except Exception:
        return -np.inf

    error_area = ((a_black_final - target_black)**2 + (a_white_final - target_white)**2)
    return -0.5 * (error_area / sigma_area**2)

def log_probability(theta):
    """Calcula el logaritmo de la probabilidad posterior (prior * likelihood).

    Combina el logaritmo de la probabilidad a priori y el logaritmo de la verosimilitud.
    Retorna -inf si alguna de las dos es -inf.

    Args:
        theta (array_like): Vector de parámetros.

    Returns:
        float: Logaritmo de la probabilidad posterior.
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

class JoblibPool:
    """Un pool de procesos compatible con la interfaz de emcee que utiliza Joblib para paralelización.

    Permite usar Joblib (útil para paralelización basada en hilos o procesos en una sola máquina)
    con emcee, que espera una interfaz similar a `multiprocessing.Pool`.

    Attributes:
        n_jobs (int): Número de trabajos paralelos a usar por Joblib. -1 usa todos los cores disponibles.
    """
    def __init__(self, n_jobs=-1):
        """Inicializa el JoblibPool.

        Args:
            n_jobs (int): Número de trabajos paralelos. Por defecto -1 (todos los cores).
        """
        self.n_jobs = n_jobs
    def map(self, func, iterable):
        """Aplica una función a cada elemento de un iterable en paralelo usando Joblib.

        Args:
            func (callable): La función a aplicar.
            iterable (iterable): El iterable sobre el cual mapear la función.

        Returns:
            list: Lista de resultados de aplicar la función a cada elemento.
        """
        return Parallel(n_jobs=self.n_jobs)(delayed(func)(i) for i in iterable)
    def close(self):
        """Método ficticio para compatibilidad con la interfaz de Pool. No hace nada."""
        pass
    def join(self):
        """Método ficticio para compatibilidad con la interfaz de Pool. No hace nada."""
        pass

def run_mcmc(initial_positions, n_walkers, n_dim, n_burn_in, n_samples, cores=1):
    """Ejecuta la cadena de Markov Monte Carlo (MCMC) usando emcee y Joblib para paralelización.

    Args:
        initial_positions (array_like): Posiciones iniciales para los walkers. Shape (n_walkers, n_dim).
        n_walkers (int): Número de walkers (cadenas).
        n_dim (int): Número de dimensiones (parámetros a ajustar).
        n_burn_in (int): Número de pasos para la fase de burn-in (descarte inicial).
        n_samples (int): Número de pasos para la fase de muestreo principal.
        cores (int): Número de cores a utilizar para la paralelización con Joblib. Por defecto 1.

    Returns:
        emcee.EnsembleSampler: El objeto sampler que contiene los resultados del MCMC.
    """
    pool = JoblibPool(n_jobs= cores)
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, pool=pool)
    print("Ejecutando fase de burn-in...")
    state = sampler.run_mcmc(initial_positions, n_burn_in, progress=True)
    sampler.reset()
    print("Ejecutando fase de muestreo principal...")
    sampler.run_mcmc(state, n_samples, progress=True)
    return sampler
