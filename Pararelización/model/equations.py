# model/equations.py
import numpy as np
from numba import njit
from typing import List

@njit
def growth_rate(temperature: float, t_opt: float = 22.5, tolerance: float = 17.5) -> float:
    """Calcula la tasa de crecimiento parabólica basada en la temperatura.

    Args:
        temperature (float): Temperatura local.
        t_opt (float): Temperatura óptima para el crecimiento. Por defecto 22.5.
        tolerance (float): Tolerancia o rango de temperatura para el crecimiento. Por defecto 17.5.

    Returns:
        float: Tasa de crecimiento (entre 0.0 y 1.0).
    """
    value = 1.0 - (t_opt - temperature) ** 2 / tolerance ** 2
    return max(0.0, value)

@njit
def global_albedo(a_black: float, a_white: float, a_bare: float,
                  A_black: float, A_white: float, A_bare: float) -> float:
    """Calcula el albedo global promedio ponderado por área.

    Args:
        a_black (float): Fracción de área cubierta por margaritas negras.
        a_white (float): Fracción de área cubierta por margaritas blancas.
        a_bare (float): Fracción de área descubierta.
        A_black (float): Albedo de las margaritas negras.
        A_white (float): Albedo de las margaritas blancas.
        A_bare (float): Albedo del suelo desnudo.

    Returns:
        float: Albedo global.
    """
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
    """Calcula la temperatura efectiva del planeta en grados Celsius.

    Args:
        L (float): Factor de luminosidad solar.
        albedo_global (float): Albedo global del planeta.
        S (float): Constante solar (W/m^2). Por defecto 780.
        sigma (float): Constante de Stefan-Boltzmann (W/m^2/K^4). Por defecto 5.67e-8.

    Returns:
        float: Temperatura efectiva en grados Celsius.
    """
    power_in = S * L * (1.0 - albedo_global)
    if power_in < 0.0:
        power_in = 0.0
    T_kelvin = (power_in / sigma) ** 0.25
    return T_kelvin - 273.0

@njit
def local_temperature(T_e: float, A_global: float, A_local: float, q_prime: float) -> float:
    """Calcula la temperatura local de una superficie (margarita o suelo).

    Args:
        T_e (float): Temperatura efectiva del planeta (°C).
        A_global (float): Albedo global.
        A_local (float): Albedo de la superficie local.
        q_prime (float): Factor de transferencia de calor.

    Returns:
        float: Temperatura local en grados Celsius.
    """
    return T_e + q_prime * (A_global - A_local)

@njit
def daisyworld_ode(y_vec: List[float], t: float, y_mort: float,
                   A_black: float, A_white: float, A_bare: float,
                   L: float, q_prime: float, P: float = 1, p: float = 1,
                   S: float = 780, T_opt: float = 22.5, T_tol: float = 17.5) -> List[float]:
    """Define el sistema de Ecuaciones Diferenciales Ordinarias (EDO) para Daisyworld.

    Calcula las tasas de cambio de las fracciones de área de margaritas negras y blancas.

    Args:
        y_vec (List[float]): Vector de estado [a_black, a_white].
        t (float): Tiempo actual (no usado explícitamente en las ecuaciones, pero requerido por el solver).
        y_mort (float): Tasa de mortalidad constante.
        A_black (float): Albedo de las margaritas negras.
        A_white (float): Albedo de las margaritas blancas.
        A_bare (float): Albedo del suelo desnudo.
        L (float): Factor de luminosidad solar.
        q_prime (float): Factor de transferencia de calor.
        P (float): Fracción total del planeta habitable. Por defecto 1.
        p (float): Tasa de propagación. Por defecto 1.
        S (float): Constante solar. Por defecto 780.
        T_opt (float): Temperatura óptima para el crecimiento. Por defecto 22.5.
        T_tol (float): Tolerancia de temperatura para el crecimiento. Por defecto 17.5.

    Returns:
        List[float]: Lista con las derivadas [da_black/dt, da_white/dt].
    """
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
