import numpy as np
from symbolfit import SymbolicRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from utils import load_data, load_metadata
from calc_pis import calc_pis

# --- Configuración ---
LAMBDA_INDEX = 0  # Índice de la banda a ajustar (e.g., 0 para la primera banda)
N_SPLITS_KFOLD = 5
MAXSIZE_REGRESSOR = 12
TIMEOUT_REGRESSOR = 3600  # 1 hora

# --- Cargar datos y metadatos ---
print("Cargando datos...")
# Asegúrate de que los archivos .npy y metadata.json estén en el directorio correcto
# o ajusta las rutas.
TRAINING_DATA_PATH = "../training.npy"
TESTING_DATA_PATH = "../testing.npy"
METADATA_PATH = "../metadata.json"

training_data = load_data(TRAINING_DATA_PATH)
testing_data = load_data(TESTING_DATA_PATH)
metadata = load_metadata(METADATA_PATH)

# Extraer flujos y parámetros físicos del conjunto de entrenamiento
F_train = training_data[:, :13]
T_train = training_data[:, 13]
logH2O_train = training_data[:, 14]
logHCN_train = training_data[:, 15]
logNH3_train = training_data[:, 16]
logkcl_train = training_data[:, 17]

# Extraer flujos y parámetros físicos del conjunto de prueba
F_test = testing_data[:, :13]
T_test = testing_data[:, 13]
logH2O_test = testing_data[:, 14]
logHCN_test = testing_data[:, 15]
logNH3_test = testing_data[:, 16]
logkcl_test = testing_data[:, 17]

# --- Cargar tablas de opacidad (Placeholder) ---
# Esta parte es crucial y depende de cómo tengas almacenadas tus tablas de opacidad.
# Debes cargar kappa_j(lambda_i) para H2O, HCN, NH3.
# Ejemplo de estructura para kapp_tables:
# kapp_tables = {
#     "H2O": np.random.rand(13, len(T_train)), # (n_lambdas, n_samples)
#     "HCN": np.random.rand(13, len(T_train)),
#     "NH3": np.random.rand(13, len(T_train))
# }
# Por ahora, usaremos un placeholder. ¡DEBES REEMPLAZAR ESTO!
print("ADVERTENCIA: Usando tablas de opacidad placeholder. Debes reemplazarlas.")
kapp_tables_train = {
    "H2O": np.random.rand(len(T_train)), # Para una sola banda, necesitamos (n_samples,)
    "HCN": np.random.rand(len(T_train)),
    "NH3": np.random.rand(len(T_train))
}
kapp_tables_test = {
    "H2O": np.random.rand(len(T_test)),
    "HCN": np.random.rand(len(T_test)),
    "NH3": np.random.rand(len(T_test))
}

# --- Calcular grupos Pi para el conjunto de entrenamiento ---
print(f"Calculando grupos Pi para la banda {LAMBDA_INDEX} (entrenamiento)...")
Pi1_train, Pi2_train, Pi3_train, _ = calc_pis(
    T_train, logH2O_train, logHCN_train, logNH3_train, logkcl_train, 
    kapp_tables_train, lambda_i=LAMBDA_INDEX
)

# Construir matriz de características X_train y vector objetivo y_train
X_train = np.column_stack([Pi1_train, Pi2_train, np.full_like(Pi1_train, Pi3_train)])
# Asumimos que F_train[:, LAMBDA_INDEX] es la profundidad de tránsito adimensional
y_train = F_train[:, LAMBDA_INDEX] 

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# --- Configurar y entrenar SymbolicRegressor ---
print("Configurando SymbolicRegressor...")
model = SymbolicRegressor(
    operators=["+", "-", "*", "/", "log", "pow"],
    loss="mse",
    maxsize=MAXSIZE_REGRESSOR,
    uncertainties=True,
    n_jobs=-1, # Usar todos los cores disponibles
    timeout=TIMEOUT_REGRESSOR
)

print("Entrenando modelo...")
model.fit(X_train, y_train)
expr_best = model.get_best()
print("Mejor expresión simbólica encontrada:", expr_best)
if hasattr(expr_best, 'sympy_format'):
    print("Formato Sympy:", expr_best.sympy_format)
if hasattr(expr_best, 'numpy_format'):
    print("Formato Numpy:", expr_best.numpy_format)

# --- Evaluar en el conjunto de prueba ---
print(f"Calculando grupos Pi para la banda {LAMBDA_INDEX} (prueba)...")
Pi1_test, Pi2_test, Pi3_test, _ = calc_pis(
    T_test, logH2O_test, logHCN_test, logNH3_test, logkcl_test, 
    kapp_tables_test, lambda_i=LAMBDA_INDEX
)
X_test = np.column_stack([Pi1_test, Pi2_test, np.full_like(Pi1_test, Pi3_test)])
y_test = F_test[:, LAMBDA_INDEX]

print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

y_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
print(f"MSE en el conjunto de prueba: {mse_test:.6e}")

# --- Validación cruzada (opcional pero recomendado) ---
print(f"Realizando validación cruzada con {N_SPLITS_KFOLD} folds...")
kf = KFold(n_splits=N_SPLITS_KFOLD, shuffle=True, random_state=42)
mse_cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"  Fold {fold+1}/{N_SPLITS_KFOLD}")
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
    
    model_cv = SymbolicRegressor(
        operators=["+", "-", "*", "/", "log", "pow"],
        loss="mse",
        maxsize=MAXSIZE_REGRESSOR,
        uncertainties=False, # Más rápido para CV
        n_jobs=-1,
        timeout=TIMEOUT_REGRESSOR // N_SPLITS_KFOLD # Reducir timeout por fold
    )
    model_cv.fit(X_fold_train, y_fold_train)
    y_fold_pred = model_cv.predict(X_fold_val)
    mse_fold = mean_squared_error(y_fold_val, y_fold_pred)
    mse_cv_scores.append(mse_fold)
    print(f"    MSE del fold: {mse_fold:.6e}")

print(f"MSE promedio de validación cruzada: {np.mean(mse_cv_scores):.6e} +/- {np.std(mse_cv_scores):.6e}")

# --- Guardar resultados y gráficos (Placeholder) ---
print("Guardando resultados y gráficos (placeholder)...")
# Aquí deberías guardar la mejor expresión, los scores, y generar gráficos como el Pareto frontier.
# Ejemplo: model.plot_pareto(output_file="../figures/pareto_band_" + str(LAMBDA_INDEX) + ".png")

print("Script fit_band.py completado.")