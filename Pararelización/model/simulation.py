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
