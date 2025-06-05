import numpy as np

# 1. Convertir log10 -> fracción lineal:
def log_to_frac(logX):
    return 10.0 ** (logX)

# 2. Calcular masa molecular media:
def calc_m_mean(logH2O, logHCN, logNH3):
    # Convertir a fracciones lineales:
    X_H2O = log_to_frac(logH2O)
    X_HCN = log_to_frac(logHCN)
    X_NH3 = log_to_frac(logNH3)
    # Fracción remanente H2+He:
    X_rem = 1.0 - (X_H2O + X_HCN + X_NH3)
    X_H2  = 0.85 * X_rem
    X_He  = 0.15 * X_rem

    # Masas molares (kg):
    m_amu  = 1.6605e-27
    mu_H2O = 18.01528 * m_amu
    mu_HCN = 27.02655 * m_amu
    mu_NH3 = 17.03052 * m_amu
    mu_H2  =  2.01588 * m_amu
    mu_He  =  4.00260 * m_amu

    # Masa molecular media:
    return (
        X_H2O * mu_H2O
      + X_HCN * mu_HCN
      + X_NH3 * mu_NH3
      + X_H2  * mu_H2
      + X_He  * mu_He
    )

# 3. Calcular grupos Pi1, Pi2, Pi3 (Pi4 opcional):
def calc_pis(T, logH2O, logHCN, logNH3, logkcl, kapp_tables, lambda_i):
    """
    - T: temperatura (K)
    - logH2O, logHCN, logNH3: arrays de log10 mixing ratios
    - logkcl: log10 de opacidad gray cloud (m^2/kg)
    - kapp_tables: diccionario con kappa_j[lam_i] pre‐cargado
    - lambda_i: índice de la banda i ∈ [0..12]
    """
    # Constantes fijas:
    R0 = 1.79 * 7.1492e7    # m (1.79 R_Jup)
    RS = 1.57 * 6.957e8     # m (1.57 R_Sol)
    g  = 9.77              # m/s^2
    P0 = 1e6               # Pa (10 bar)
    kB = 1.380649e-23       # J/K

    # 1. Masa molecular media:
    m_mean = calc_m_mean(logH2O, logHCN, logNH3)  # array (N,)

    # 2. Escala de altura:
    H = (kB * T) / (m_mean * g)  # array (N,) en metros

    # 3. Pi1:
    Pi1 = (R0 * m_mean * g) / (kB * T)  # equiv. R0 / H

    # 4. Opacidad molecular a lambda_i:
    #    kappa_eff = Σ_j(X_j * kappa_j(λ_i)) + kappa_cl
    X_H2O = log_to_frac(logH2O)
    X_HCN = log_to_frac(logHCN)
    X_NH3 = log_to_frac(logNH3)
    # kappa_tables debe proveer: kappa_H2O[i,:], kappa_HCN[i,:], kappa_NH3[i,:] arrays (N,):
    # Esta parte requiere que kapp_tables esté bien estructurado.
    # Ejemplo: kapp_tables = {"H2O": np.array([...]), "HCN": np.array([...]), "NH3": np.array([...])}
    # donde cada array interno tiene forma (n_lambdas, n_samples) o se accede por lambda_i
    kapp_H2O_i = kapp_tables["H2O"][lambda_i]
    kapp_HCN_i = kapp_tables["HCN"][lambda_i]
    kapp_NH3_i = kapp_tables["NH3"][lambda_i]
    kappa_cl = 10.0 ** (logkcl)  # array (N,)

    kappa_eff = (X_H2O * kapp_H2O_i 
                + X_HCN * kapp_HCN_i 
                + X_NH3 * kapp_NH3_i) 
                + kappa_cl

    # 5. Pi2:
    Pi2 = (kappa_eff * P0) / g  # array (N,)

    # 6. Pi3 (constante):
    Pi3 = RS / R0

    # 7. Pi4 (opcional):
    Pi4 = (m_mean * kappa_eff * P0) / (kB * T * g)

    return Pi1, Pi2, Pi3, Pi4