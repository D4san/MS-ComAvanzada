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
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

class JoblibPool:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
    def map(self, func, iterable):
        return Parallel(n_jobs=self.n_jobs)(delayed(func)(i) for i in iterable)
    def close(self):
        pass
    def join(self):
        pass

def run_mcmc(initial_positions, n_walkers, n_dim, n_burn_in, n_samples, cores = 1):
    pool = JoblibPool(n_jobs= cores)
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, pool=pool)
    print("Ejecutando fase de burn-in...")
    state = sampler.run_mcmc(initial_positions, n_burn_in, progress=True)
    sampler.reset()
    print("Ejecutando fase de muestreo principal...")
    sampler.run_mcmc(state, n_samples, progress=True)
    return sampler
