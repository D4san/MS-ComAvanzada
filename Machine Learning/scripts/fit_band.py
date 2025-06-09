#!/usr/bin/env python3
"""
Script para calcular grupos π de espectros de tránsito y ajustar una expresión simbólica
usando SymbolFit + PySRRegressor, entrenando por MSE (loss) en lugar de RMSE.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from symbolfit.symbolfit import SymbolFit
from pysr import PySRRegressor
from calc_pis import setup_opacity_interpolators, calc_pis_for_sample
from sklearn.metrics import mean_squared_error

# Parámetros
TRAIN_FILE    = 'data/training.npy'
OPACITY_FILES = {
    'H2O': 'data/opacH2O.dat',
    'HCN': 'data/opacHCN.dat',
    'NH3': 'data/opacNH3.dat',
}
LAMBDA_I    = 4      # índice de la banda (0-based)
NUM_SAMPLES = 100_000   # número de muestras para entrenar
SEED        = None     # reproducibilidad

# 1) Cargar datos
if not os.path.isfile(TRAIN_FILE):
    raise RuntimeError(f"No se encontró {TRAIN_FILE}")
arr = np.load(TRAIN_FILE)
n_bands = arr.shape[1] - 5  # asumimos 5 columnas finales: T, logH2O, logHCN, logNH3, logkcl
if not (0 <= LAMBDA_I < n_bands):
    raise ValueError(f"LAMBDA_I debe estar entre 0 y {n_bands-1}")
print(f"Cargadas {arr.shape[0]} muestras, {n_bands} bandas de profundidad")

# 2) Extraer M (profundidad de tránsito) de la columna correspondiente
M = arr[:, LAMBDA_I]
print(f"Profundidad M: min={M.min():.3e}, max={M.max():.3e}")

# 3) Desempaquetar parámetros atmosféricos (últimas 5 columnas)
T      = arr[:, -5]
logH2O = arr[:, -4]
logHCN = arr[:, -3]
logNH3 = arr[:, -2]
logkcl = arr[:, -1]

# 4) Calcular π-grupos
interps = setup_opacity_interpolators(
    OPACITY_FILES['H2O'],
    OPACITY_FILES['HCN'],
    OPACITY_FILES['NH3'],
)
pi1, pi2, pi3, pi4 = calc_pis_for_sample(
    T, logH2O, logHCN, logNH3, logkcl,
    LAMBDA_I, interps
)
pi1 = np.atleast_1d(pi1)
pi2 = np.atleast_1d(pi2)
if np.isscalar(pi3) or pi3.shape != pi1.shape:
    pi3 = np.full_like(pi1, pi3)
pi4 = np.atleast_1d(pi4)
print(f"π shapes: π1{pi1.shape}, π2{pi2.shape}, π3{pi3.shape}, π4{pi4.shape}")

# 5) Depuración rápida (opcional)
plt.figure(figsize=(5,4))
plt.scatter(1/pi1, np.log10(pi2), c=M, s=4, cmap='viridis')
plt.colorbar(label='Profundidad M')
plt.xlabel('1/π₁'); plt.ylabel('log₁₀ π₂')
plt.title(f'Banda {LAMBDA_I}: π₁ vs log₁₀ π₂')
plt.tight_layout()
plt.show()

# 6) Preparar X, y
X = np.vstack([pi1, pi2, pi3, pi4]).T
y = np.sqrt(2*M)

# 7) Filtrar no finitos
mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
if not mask.all():
    print(f"Eliminando {np.count_nonzero(~mask)} filas no finitas")
    X, y = X[mask], y[mask]

# 8) Submuestreo aleatorio
rng = np.random.default_rng(SEED)
if X.shape[0] > NUM_SAMPLES:
    idx = rng.choice(X.shape[0], size=NUM_SAMPLES, replace=False)
    X, y = X[idx], y[idx]
print(f"Entrenamiento: X.shape={X.shape}, y.shape={y.shape}")

# 9) Configurar PySRRegressor para optimizar MSE (loss)
cores = multiprocessing.cpu_count()
pysr_config = PySRRegressor(
    model_selection = 'accuracy',
    niterations = 400,
    maxsize = 40,
    binary_operators = [
        '+', '*', '/'
                     ],
    unary_operators = [
        'log',
        'sqrt',
    ],
    nested_constraints = {
        'sqrt': {'sqrt': 0, 'log': 0, '*': 2},
        'log':  {'sqrt': 2, 'log': 0, '*': 2},
        '*':    {'sqrt': 2, 'log': 2, '*': 2},
    },
    elementwise_loss='loss(y, y_pred, weights) = (y - y_pred)^2 * weights',
)

# 10) Configurar SymbolFit
model = SymbolFit(
    x=X, y=y,
    y_up=np.zeros_like(y), y_down=np.zeros_like(y),
    pysr_config = pysr_config,
    max_complexity=20,
    input_rescale=False,
    scale_y_by=None,
    fit_y_unc=False,
    loss_weights = None
)

# 11) Entrenar y guardar resultados
print("Entrenando regresión simbólica optimizando MSE…")
model.fit()
os.makedirs('results/fit', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)
model.save_to_csv(output_dir='results/fit')
model.plot_to_pdf(output_dir='results/figures', plot_logy=False, plot_logx=False)

# 12) Extraer la mejor fórmula y calcular MSE de entrenamiento
import pandas as pd
from sklearn.metrics import mean_squared_error # Idealmente, esta importación va al inicio del archivo.

# Cargar el archivo de candidatos de forma robusta
candidates_csv_file = os.path.join('results/fit', 'candidates_reduced.csv')
if not os.path.isfile(candidates_csv_file):
    raise FileNotFoundError(f"Archivo de candidatos no encontrado: {candidates_csv_file}")
df = pd.read_csv(candidates_csv_file)

# Encontrar la columna de la expresión
# 'Parameterized equation, unscaled' es la esperada según el CSV de ejemplo.
expr_col_options = ['Parameterized equation, unscaled', 'expression', 'equation', 'Equation', 'symbol']
expr_col = None
for col in expr_col_options:
    if col in df.columns:
        expr_col = col
        break
if expr_col is None:
    print(f"Columnas disponibles en {candidates_csv_file}: {df.columns.tolist()}") # Ayuda para depuración
    raise KeyError(f"No se encontró la columna de expresión en {df.columns.tolist()}. Se buscaron: {expr_col_options}")

best = df.iloc[0]  # Asumir que la primera fila es la mejor
expr = best[expr_col]

# Obtener el RMSE directamente del CSV.
# La columna 'RMSE' está presente en el archivo candidates_reduced.csv.
mse_from_csv = best.get('RMSE', np.nan) # Usar np.nan de numpy que debe estar importado
if pd.isna(mse_from_csv):
    print(f"Advertencia: 'RMSE' no encontrado o es NaN en la fila del mejor candidato de {candidates_csv_file}.")
    # Considerar un fallback si es necesario, ej: mse_from_csv = best.get('loss', np.nan) / X.shape[0] # X debe estar en ámbito

# Calcular MSE con sklearn para verificación (model, X, y deben estar en el ámbito)
try:
    y_pred = model.predict(X) # 'model' y 'X' deben estar en el ámbito
    mse_sklearn_verification = mean_squared_error(y, y_pred) # 'y' debe estar en el ámbito
except Exception as e:
    print(f"Advertencia: No se pudo calcular MSE con sklearn para verificación: {e}")
    mse_sklearn_verification = np.nan # Marcar como no disponible si falla

# Guardar resultados
formula_out_path = os.path.join('results/fit', 'formula.txt')
mse_out_path = os.path.join('results/fit', 'mse.txt')

with open(formula_out_path, 'w') as f:
    f.write(expr)
with open(mse_out_path, 'w') as f:
    f.write(str(mse_from_csv)) # Guardar el MSE del CSV

print(f"Mejor fórmula ({expr_col}): {expr}")
print(f"MSE (desde CSV columna 'RMSE'): {mse_from_csv if not pd.isna(mse_from_csv) else 'No disponible o NaN'}")
if not pd.isna(mse_sklearn_verification):
    print(f"MSE (sklearn para verificación): {mse_sklearn_verification:.4e}")
