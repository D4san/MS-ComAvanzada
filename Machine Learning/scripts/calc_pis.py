import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.interpolate import RegularGridInterpolator

# —————— Utilidades de composición química ——————

def log_to_frac(logX):
    """Convierte log10(mixing ratio) a fracción lineal."""
    return 10.0 ** logX


def calc_m_mean(logH2O, logHCN, logNH3):
    """Calcula la masa molecular media (kg) de la atmósfera mezclada."""
    X_H2O = log_to_frac(logH2O)
    X_HCN = log_to_frac(logHCN)
    X_NH3 = log_to_frac(logNH3)
    X_rem = 1.0 - (X_H2O + X_HCN + X_NH3)
    X_H2, X_He = 0.85 * X_rem, 0.15 * X_rem

    m_amu = 1.6605e-27  # kg
    mu = {
        "H2O": 18.01528 * m_amu,
        "HCN": 27.02655 * m_amu,
        "NH3": 17.03052 * m_amu,
        "H2":   2.01588 * m_amu,
        "He":   4.00260 * m_amu,
    }

    return (X_H2O*mu["H2O"] + X_HCN*mu["HCN"] +
            X_NH3*mu["NH3"] + X_H2*mu["H2"] + X_He*mu["He"])

# —————— Carga e interpolación de opacidades ExoTransmit ——————

def load_exotransmit_opacity(filename):
    """
    Lee un .dat de ExoTransmit y devuelve:
      λ_grid [nλ] (m), P_grid [nP] (Pa), T_grid [nT] (K),
      sigma [nλ,nP,nT] (m²/molécula).
    """
    with open(filename, 'r') as f:
        T_grid      = np.array(next(f).split(), dtype=float)
        P_grid_bar  = np.array(next(f).split(), dtype=float)
        P_grid      = P_grid_bar * 1e5
        lines       = f.readlines()

    nP, nT = len(P_grid), len(T_grid)
    nλ = len(lines) // (nP + 1)
    sigma   = np.zeros((nλ, nP, nT))
    λ_grid  = np.zeros(nλ)
    idx     = 0

    for i in range(nλ):
        λ_grid[i] = float(lines[idx].strip()) * 1e-6
        idx += 1
        for j in range(nP):
            vals = np.array(lines[idx].split()[1:], dtype=float)
            sigma[i, j, :] = vals
            idx += 1

    return λ_grid, P_grid, T_grid, sigma


def make_interpolator(λ_grid, P_grid, T_grid, sigma):
    """Construye un interpolador RegularGridInterpolator sobre (λ,P,T)."""
    return RegularGridInterpolator(
        (λ_grid, P_grid, T_grid),
        sigma,
        bounds_error=False,
        fill_value=0.0
    )


def setup_opacity_interpolators(h2o_path, hcn_path, nh3_path):
    """Carga tablas y crea interpoladores para H2O, HCN, NH3."""
    λ_H2O, P_grid, T_grid, σ_H2O = load_exotransmit_opacity(h2o_path)
    _,      _,      _,        σ_HCN = load_exotransmit_opacity(hcn_path)
    _,      _,      _,        σ_NH3 = load_exotransmit_opacity(nh3_path)

    return {
        "H2O": (λ_H2O, make_interpolator(λ_H2O, P_grid, T_grid, σ_H2O)),
        "HCN": (λ_H2O, make_interpolator(λ_H2O, P_grid, T_grid, σ_HCN)),
        "NH3": (λ_H2O, make_interpolator(λ_H2O, P_grid, T_grid, σ_NH3)),
    }

# —————— Cálculo de los grupos Pi ——————

def calc_pis_for_sample(T, logH2O, logHCN, logNH3, logkcl,
                        lambda_i, interps,
                        P0=1e6, T_max_NH3=1600):
    """Calcula Pi1, Pi2, Pi3, Pi4 según Matchev+22 para N muestras."""
    R0, R_S = 1.79*7.1492e7, 1.57*6.957e8
    g, kB    = 9.77, 1.380649e-23
    N = T.size

    # Pi1
    m_mean = calc_m_mean(logH2O, logHCN, logNH3)
    Pi1 = (R0 * m_mean * g) / (kB * T)

    # κ_j en λ_i, P0, T
    λ_val = interps["H2O"][0][lambda_i]
    pts = np.vstack([np.full(N, λ_val),
                     np.full(N, P0),
                     T]).T
    κ_H2O = interps["H2O"][1](pts)
    κ_HCN = interps["HCN"][1](pts)
    κ_NH3 = interps["NH3"][1](pts)

    # Umbral NH3
    mask = T > T_max_NH3
    if mask.any():
        κ_NH3[mask] = 0.0
        logNH3 = logNH3.copy()
        logNH3[mask] = -13.0

    # Fracciones & cloud
    X_H2O = log_to_frac(logH2O)
    X_HCN = log_to_frac(logHCN)
    X_NH3 = log_to_frac(logNH3)
    κ_cl  = 10.0 ** logkcl
    κ_eff = X_H2O*κ_H2O + X_HCN*κ_HCN + X_NH3*κ_NH3 + κ_cl

    # Pi2, Pi3, Pi4
    Pi2 = (κ_eff * P0) / g
    Pi3 = R0 / R_S
    Pi4 = (R0**2) / (m_mean * κ_eff)

    return Pi1, Pi2, Pi3, Pi4

# —————— Procesamiento y entrenamiento ——————
def process_files(file_list, opacity_files, lambda_i):
    """Procesa muchos archivos de entrada y calcula los Pi para cada uno."""
    interps = setup_opacity_interpolators(
        opacity_files["H2O"],
        opacity_files["HCN"],
        opacity_files["NH3"]
    )

    features = {}
    for path in file_list:
        arr = np.load(path)
        T, logH2O, logHCN, logNH3, logkcl = arr.T
        pi1, pi2, pi3, pi4 = calc_pis_for_sample(
            T, logH2O, logHCN, logNH3, logkcl,
            lambda_i, interps
        )
        features[path] = np.vstack([pi1, pi2, pi3, pi4]).T
    return features

# —————— Plot estilo Figura 2 ——————
def plot_training_pi_with_contour(training_path, opacity_files, lambda_index):
    """Genera contour+scatter estilo Figura 2 y guarda en results/figures."""
    data   = np.load(training_path)
    T      = data[:,13];  logH2O = data[:,14]
    logHCN = data[:,15];  logNH3 = data[:,16]
    logkcl = data[:,17]

    interps = setup_opacity_interpolators(
        opacity_files["H2O"],
        opacity_files["HCN"],
        opacity_files["NH3"]
    )
    Pi1, Pi2, Pi3, _ = calc_pis_for_sample(
        T, logH2O, logHCN, logNH3, logkcl,
        lambda_index, interps
    )
    inv_pi1 = 1.0 / Pi1
    log_pi2 = np.log10(Pi2)

    # M(λ) límite τ→∞ en %
    γ_E = 0.57721566
    M = Pi3**2 * (1 + (1.0/Pi1)*(γ_E + np.log(Pi2 * np.sqrt(2*np.pi*Pi1))))**2
    M_pct = 100.0 * M

    outdir = "results/figures"
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_facecolor('black'); ax.set_facecolor('black')
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    ax.tick_params(colors='white'); ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    triang = mtri.Triangulation(inv_pi1, log_pi2)
    levels = np.linspace(np.nanmin(M_pct), np.nanmax(M_pct), 12)
    cf = ax.tricontourf(triang, M_pct, levels=levels,
                       cmap='nipy_spectral', alpha=1.0)
    ax.scatter(inv_pi1, log_pi2, c='black', s=1, alpha=0.3)

    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color='white'); plt.setp(cbar.ax.get_yticklabels(), color='white')
    cbar.set_label('M(λ) [%]', color='white')

    ax.set_xlabel('1/π₁ = H/R₀', fontsize=12)
    ax.set_ylabel('log₁₀(π₂) = log₁₀(P₀ κ/g)', fontsize=12)
    ax.set_title('1/π₁ vs log₁₀(π₂) coloreado por M(λ)', pad=12)

    plt.tight_layout()
    outpath = os.path.join(outdir, f"figure_pi_lambda_{lambda_index}.png")
    fig.savefig(outpath, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"Figura guardada en: {outpath}")

# —————— __main__ ——————
if __name__ == "__main__":
    # Rutas a archivos de opacidades
    opacity = {
        "H2O": "data/opacH2O.dat",
        "HCN": "data/opacHCN.dat",
        "NH3": "data/opacNH3.dat"
    }
    # Índice de banda, por ejemplo 4 => 0.978 µm
    lambda_index = 4

    # 1) Plot Figura 2
    plot_training_pi_with_contour("data/training.npy", opacity, lambda_index)

    # 2) Procesar archivos de espectros (para entrenamiento futuro)
    # files = ["spec1.npy", "spec2.npy", ...]
    # features = process_files(files, opacity, lambda_index)
    # Aquí podrías pasar 'features' al módulo de entrenamiento de tu modelo
