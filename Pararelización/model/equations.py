# model/equations.py
import numpy as np
from numba import njit
from typing import List

@njit
def growth_rate(temperature: float, t_opt: float = 22.5, tolerance: float = 17.5) -> float:
    value = 1.0 - (t_opt - temperature) ** 2 / tolerance ** 2
    return max(0.0, value)

@njit
def global_albedo(a_black: float, a_white: float, a_bare: float,
                  A_black: float, A_white: float, A_bare: float) -> float:
    if a_black < 0.0:
        a_black = 0.0
    if a_white < 0.0:
        a_white = 0.0
    if a_bare < 0.0:
        a_bare = 0.0
    suma = a_black + a_white + a_bare
    if suma == 0.0:
        return 0.0
    a_black_norm = a_black / suma
    a_white_norm = a_white / suma
    a_bare_norm = a_bare / suma
    return (a_black_norm * A_black + a_white_norm * A_white +
            a_bare_norm * A_bare)

@njit
def effective_temperature(L: float, albedo_global: float,
                          S: float = 780,
                          sigma: float = 5.67e-8) -> float:
    power_in = S * L * (1.0 - albedo_global)
    if power_in < 0.0:
        power_in = 0.0
    T_kelvin = (power_in / sigma) ** 0.25
    return T_kelvin - 273.0

@njit
def local_temperature(T_e: float, A_global: float, A_local: float, q_prime: float) -> float:
    return T_e + q_prime * (A_global - A_local)

@njit
def daisyworld_ode(y_vec: List[float], t: float, y_mort: float,
                   A_black: float, A_white: float, A_bare: float,
                   L: float, q_prime: float, P: float = 1, p: float = 1,
                   S: float = 780, T_opt: float = 22.5, T_tol: float = 17.5) -> List[float]:
    a_black, a_white = y_vec
    x_bare = P - a_black - a_white
    if x_bare < 0:
        x_bare = 0.0
    A_global = global_albedo(a_black, a_white, x_bare, A_black, A_white, A_bare)
    T_eff = effective_temperature(L, A_global, S)
    T_black = local_temperature(T_eff, A_global, A_black, q_prime)
    T_white = local_temperature(T_eff, A_global, A_white, q_prime)
    grow_black = growth_rate(T_black, T_opt, T_tol)
    grow_white = growth_rate(T_white, T_opt, T_tol)
    da_black_dt = a_black * (p * x_bare * grow_black) - y_mort * a_black
    da_white_dt = a_white * (p * x_bare * grow_white) - y_mort * a_white
    return [da_black_dt, da_white_dt]
