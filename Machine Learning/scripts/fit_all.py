#!/usr/bin/env python3
"""
Script para calcular grupos π de espectros de tránsito y ajustar una expresión simbólica
usando SymbolFit + PySRRegressor, entrenando por MSE (loss) en lugar de RMSE.
Incluye M(τ→∞) analítico y dependencia explícita de λ en 13 canales HST WFC3.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from symbolfit.symbolfit import SymbolFit
from pysr import PySRRegressor
from calc_pis import setup_opacity_interpolators, calc_pis_for_sample
from sklearn.metrics import mean_squared_error
import pandas as pd

# Parámetros fijos de HST/WFC3 (µm) — 13 sub­bandas G102+G141
wavelengths = np.array([
    0.838, 0.867, 0.904, 0.941, 0.978,
    1.015, 1.052, 1.118, 1.208, 1.305,
    1.409, 1.520, 1.666
])
N_channels = wavelengths.size  # 13

# Datos de entrenamiento
TRAIN_FILE   = 'data/training.npy'   # devé tener shape (N_samples, 13 + 5)
NUM_SAMPLES  = 80_000                # máximo de filas tras expandir
SEED         = None                  # reproducibilidad

# 1) Cargar matriz de datos
if not os.path.isfile(TRAIN_FILE):
    raise RuntimeError(f"No se encontró {TRAIN_FILE}")
arr = np.load(TRAIN_FILE)
N, total_cols = arr.shape
if total_cols - 5 != N_channels:
    raise ValueError(f"Esperaba {N_channels} bandas, pero shape es {total_cols-5}")
print(f"{N} muestras cargadas con {N_channels} bandas y 5 parámetros extras.")

# 2) Desempaquetar parámetros atmosféricos
T      = arr[:, -5]
logH2O = arr[:, -4]
logHCN = arr[:, -3]
logNH3 = arr[:, -2]
logkcl = arr[:, -1]

# 3) Crear interpoladores de opacidad
interps = setup_opacity_interpolators(
    'data/opacH2O.dat',
    'data/opacHCN.dat',
    'data/opacNH3.dat',
)

# 4) Calcular π-grupos y λ para cada banda
pi1_list = []; pi2_list = []; pi3_list = []; pi4_list = []; lam_list = []
for j in range(N_channels):
    p1, p2, p3, p4 = calc_pis_for_sample(
        T, logH2O, logHCN, logNH3, logkcl, j, interps
    )
    p1 = np.atleast_1d(p1)
    p2 = np.atleast_1d(p2)
    if np.isscalar(p3) or p3.shape != p1.shape:
        p3 = np.full_like(p1, p3)
    p4 = np.atleast_1d(p4)

    pi1_list.append(p1)
    pi2_list.append(p2)
    pi3_list.append(p3)
    pi4_list.append(p4)
    lam_list.append(np.full_like(p1, wavelengths[j]))

# Concatenar en un único dataset
pi1 = np.concatenate(pi1_list)
pi2 = np.concatenate(pi2_list)
pi3 = np.concatenate(pi3_list)
pi4 = np.concatenate(pi4_list)
lam = np.concatenate(lam_list)
print(f"Dataset expandido: {pi1.size} filas con 5 variables (π₁–π₄, λ).")

# 5) Calcular M en el límite τ→∞ (%)
γ_E = 0.57721566
M = pi3**2 * (
    1.0 + (1.0/pi1) * (γ_E + np.log(pi2 * np.sqrt(2*np.pi*pi1)))
)**2
print(f"M calculado: min={M.min():.3e}, max={M.max():.3e}")


# 7) Preparar X,y para regresión
X = np.vstack([pi1, pi2, pi3, pi4, lam]).T
y = np.sqrt(2 * M)

# 8) Filtrar no finitos
mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
if not mask.all():
    removed = np.count_nonzero(~mask)
    print(f"Eliminando {removed} filas no finitas")
    X, y = X[mask], y[mask]

# 9) Submuestreo
rng = np.random.default_rng(SEED)
if X.shape[0] > NUM_SAMPLES:
    idx = rng.choice(X.shape[0], size=NUM_SAMPLES, replace=False)
    X, y = X[idx], y[idx]
print(f"Entrenamiento: X.shape={X.shape}, y.shape={y.shape}")

# 10) Configurar PySRRegressor (MSE)
cores = multiprocessing.cpu_count()
pysr_config = PySRRegressor(
    model_selection = 'accuracy',
    niterations = 400,
    maxsize = 40,
    elementwise_loss='loss(y, y_pred, weights) = (y - y_pred)^2 * weights',
)

# 10) Configurar SymbolFit
model = SymbolFit(
    x=X, y=y,
    y_up=np.zeros_like(y), y_down=np.zeros_like(y),
    pysr_config = pysr_config,
    max_complexity=15,
    input_rescale=False,
    scale_y_by=None,
    fit_y_unc=False,
    loss_weights = None
)
# 12) Entrenar y guardar resultados
print("Entrenando regresión simbólica…")
model.fit()
os.makedirs('results/fit', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)
model.save_to_csv(output_dir='results/fit')
model.plot_to_pdf(output_dir='results/figures', plot_logy=False, plot_logx=False)

# 13) Extraer la mejor fórmula y calcular MSE de entrenamiento
# Importaciones necesarias
import pandas as pd
from sklearn.metrics import mean_squared_error

# Cargar el archivo de candidatos de forma robusta
candidates_csv_file = os.path.join('results/fit', 'candidates_reduced.csv')
if not os.path.isfile(candidates_csv_file):
    raise FileNotFoundError(f"Archivo de candidatos no encontrado: {candidates_csv_file}")
df = pd.read_csv(candidates_csv_file)

# Encontrar la columna de la expresión
expr_col_options = [
    'Parameterized equation, unscaled',
    'expression',
    'equation',
    'Equation',
    'symbol',
]
expr_col = None
for col in expr_col_options:
    if col in df.columns:
        expr_col = col
        break
if expr_col is None:
    print(f"Columnas disponibles en {candidates_csv_file}: {df.columns.tolist()}")
    raise KeyError(f"No se encontró la columna de expresión. Buscadas: {expr_col_options}")

best = df.iloc[0]
expr = best[expr_col]

# Obtener el RMSE directamente del CSV
mse_from_csv = best.get('RMSE', np.nan)
if pd.isna(mse_from_csv):
    print(f"Advertencia: 'RMSE' no encontrado o es NaN en la fila del mejor candidato.")

# Calcular MSE con sklearn para verificación
try:
    y_pred = model.predict(X)
    mse_sklearn_verification = mean_squared_error(y, y_pred)
except Exception as e:
    print(f"Advertencia: No se pudo calcular MSE con sklearn para verificación: {e}")
    mse_sklearn_verification = np.nan

# Guardar resultados
formula_out_path = os.path.join('results/fit', 'formula.txt')
mse_out_path     = os.path.join('results/fit', 'mse.txt')
with open(formula_out_path, 'w') as f:
    f.write(expr)
with open(mse_out_path, 'w') as f:
    f.write(str(mse_from_csv))

print(f"Mejor fórmula ({expr_col}): {expr}")
print(f"MSE (desde CSV columna 'RMSE'): {mse_from_csv if not pd.isna(mse_from_csv) else 'No disponible o NaN'}")
if not pd.isna(mse_sklearn_verification):
    print(f"MSE (sklearn para verificación): {mse_sklearn_verification:.4e}")
