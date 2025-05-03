# model/simulation.py
import numpy as np
from model.equations import global_albedo, effective_temperature, local_temperature, daisyworld_ode

def simulate_daisyworld(a_black_init: float, a_white_init: float,
                        t_max: float, y_mort: float,
                        A_black: float, A_white: float, A_bare: float,
                        L: float, q_prime: float,
                        P: float = 1, p: float = 1, S: float = 780,
                        T_opt: float = 22.5, T_tol: float = 17.5,
                        num_points: int = 200):
    """Simula la evolución del modelo Daisyworld a lo largo del tiempo.

    Integra las EDOs de Daisyworld usando un método de Euler simple.

    Args:
        a_black_init (float): Fracción inicial de margaritas negras.
        a_white_init (float): Fracción inicial de margaritas blancas.
        t_max (float): Tiempo máximo de simulación.
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
        num_points (int): Número de puntos de tiempo en la simulación. Por defecto 200.

    Returns:
        tuple: Una tupla conteniendo:
            - t_array (np.ndarray): Array de tiempos.
            - a_black_arr (np.ndarray): Array de fracción de margaritas negras en cada tiempo.
            - a_white_arr (np.ndarray): Array de fracción de margaritas blancas en cada tiempo.
            - T_eff_arr (np.ndarray): Array de temperatura efectiva en cada tiempo.
            - T_black_arr (np.ndarray): Array de temperatura local de margaritas negras.
            - T_white_arr (np.ndarray): Array de temperatura local de margaritas blancas.
            - A_global_arr (np.ndarray): Array de albedo global en cada tiempo.
    """
    t_array = np.linspace(0.0, t_max, num_points)
    dt = t_array[1] - t_array[0]
    a_black_arr = np.zeros(num_points)
    a_white_arr = np.zeros(num_points)
    T_eff_arr = np.zeros(num_points)
    T_black_arr = np.zeros(num_points)
    T_white_arr = np.zeros(num_points)
    A_global_arr = np.zeros(num_points)
    a_black_arr[0] = a_black_init
    a_white_arr[0] = a_white_init

    for i in range(num_points - 1):
        a_b = a_black_arr[i]
        a_w = a_white_arr[i]
        x_bare = max(0.0, P - a_b - a_w)
        A_g = global_albedo(a_b, a_w, x_bare, A_black, A_white, A_bare)
        T_e = effective_temperature(L, A_g, S)
        T_b = local_temperature(T_e, A_g, A_black, q_prime)
        T_w = local_temperature(T_e, A_g, A_white, q_prime)
        A_global_arr[i] = A_g
        T_eff_arr[i] = T_e
        T_black_arr[i] = T_b
        T_white_arr[i] = T_w
        derivatives = daisyworld_ode([a_b, a_w], t_array[i],
                                     y_mort, A_black, A_white, A_bare, L,
                                     q_prime, P, p, S, T_opt, T_tol)
        a_black_arr[i + 1] = a_b + dt * derivatives[0]
        a_white_arr[i + 1] = a_w + dt * derivatives[1]

    # Último punto
    a_b = a_black_arr[-1]
    a_w = a_white_arr[-1]
    x_bare = max(0.0, P - a_b - a_w)
    A_g = global_albedo(a_b, a_w, x_bare, A_black, A_white, A_bare)
    T_e = effective_temperature(L, A_g, S)
    T_b = local_temperature(T_e, A_g, A_black, q_prime)
    T_w = local_temperature(T_e, A_g, A_white, q_prime)
    A_global_arr[-1] = A_g
    T_eff_arr[-1] = T_e
    T_black_arr[-1] = T_b
    T_white_arr[-1] = T_w
    return (t_array, a_black_arr, a_white_arr, T_eff_arr, T_black_arr, T_white_arr, A_global_arr)
