#!/usr/bin/env python3
"""
Script para calcular grupos π de espectros de tránsito y ajustar una expresión simbólica
usando SymbolFit + PySRRegressor, entrenando por MSE (loss) en lugar de RMSE.
Ahora M se define analíticamente en el límite τ→∞.
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

# Parámetros
TRAIN_FILE    = 'data/training.npy'
OPACITY_FILES = {
    'H2O': 'data/opacH2O.dat',
    'HCN': 'data/opacHCN.dat',
    'NH3': 'data/opacNH3.dat',
}
LAMBDA_I     = 0         # índice de la banda (0-based)
NUM_SAMPLES  = 10_000   # número de muestras para entrenar
SEED         = None      # reproducibilidad

# 1) Cargar datos
if not os.path.isfile(TRAIN_FILE):
    raise RuntimeError(f"No se encontró {TRAIN_FILE}")
arr = np.load(TRAIN_FILE)
n_bands = arr.shape[1] - 5  # asumimos 5 columnas finales: T, logH2O, logHCN, logNH3, logkcl
if not (0 <= LAMBDA_I < n_bands):
    raise ValueError(f"LAMBDA_I debe estar entre 0 y {n_bands-1}")
print(f"Cargadas {arr.shape[0]} muestras, {n_bands} bandas de profundidad")

# 2) Desempaquetar parámetros atmosféricos (últimas 5 columnas)
T      = arr[:, -5]
logH2O = arr[:, -4]
logHCN = arr[:, -3]
logNH3 = arr[:, -2]
logkcl = arr[:, -1]

# 3) Calcular π-grupos
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

# 4) Definir M analíticamente (límite τ→∞ en %)
γ_E = 0.57721566
# ojo: pi2 * sqrt(2π pi1) debe ser positivo siempre si pi1,pi2>0
M = pi3**2 * (1.0 + (1.0/pi1) * (γ_E + np.log(pi2 * np.sqrt(2*np.pi*pi1))))**2
print(f"M calculado: min={M.min():.3e}, max={M.max():.3e}")

# 5) Depuración rápida (opcional)
plt.figure(figsize=(5,4))
plt.scatter(1/pi1, np.log10(pi2), c=M, s=4, cmap='viridis')
plt.colorbar(label='M (%)')
plt.xlabel('1/π₁')
plt.ylabel('log₁₀ π₂')
plt.title(f'Banda {LAMBDA_I}: π₁ vs log₁₀ π₂ (color = M)')
plt.tight_layout()
plt.show()

# 6) Preparar X, y (mantenemos y = sqrt(2 M) como target)
X = np.vstack([pi1, pi2, pi3, pi4]).T
y = np.sqrt(2 * M)

# 7) Filtrar no finitos
mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
if not mask.all():
    removed = np.count_nonzero(~mask)
    print(f"Eliminando {removed} filas no finitas")
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
    model_selection='accuracy',
    niterations=400,
    maxsize=40,
    binary_operators=['+', '*', '/'],
    unary_operators=['log', 'sqrt'],
    nested_constraints={
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
    pysr_config=pysr_config,
    max_complexity=20,
    input_rescale=False,
    scale_y_by=None,
    fit_y_unc=False,
    loss_weights=None,
)

# 11) Entrenar y guardar resultados
print("Entrenando regresión simbólica optimizando MSE…")
model.fit()
os.makedirs('results/fit', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)
model.save_to_csv(output_dir='results/fit')
model.plot_to_pdf(output_dir='results/figures', plot_logy=False, plot_logx=False)

# 12) Extraer la mejor fórmula y calcular MSE de entrenamiento
candidates_csv = os.path.join('results/fit', 'candidates_reduced.csv')
if not os.path.isfile(candidates_csv):
    raise FileNotFoundError(f"Archivo de candidatos no encontrado: {candidates_csv}")
df = pd.read_csv(candidates_csv)

# Detectar columna de la expresión
for name in ['Parameterized equation, unscaled','expression','equation','symbol']:
    if name in df.columns:
        expr_col = name
        break
else:
    raise KeyError(f"No se encontró columna de expresión en {df.columns.tolist()}")

best = df.iloc[0]
expr = best[expr_col]
mse_csv = best.get('RMSE', np.nan)

# Verificación con sklearn
try:
    y_pred = model.predict(X)
    mse_check = mean_squared_error(y, y_pred)
except Exception as e:
    print(f"Advertencia al calcular MSE de verificación: {e}")
    mse_check = np.nan

# Guardar resultados
with open('results/fit/formula.txt', 'w') as f:
    f.write(expr)
with open('results/fit/mse.txt', 'w') as f:
    f.write(str(mse_csv))

print(f"Mejor fórmula ({expr_col}): {expr}")
print(f"MSE (CSV RMSE): {mse_csv}")
if np.isfinite(mse_check):
    print(f"MSE (verificación sklearn): {mse_check:.4e}")
