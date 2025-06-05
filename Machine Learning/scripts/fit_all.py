import numpy as np
from symbolfit import SymbolicRegressor
from sklearn.model_selection import train_test_split # Usaremos split simple para este ejemplo
from sklearn.metrics import mean_squared_error

from utils import load_data, load_metadata
from calc_pis import calc_pis # Asumimos que calc_pis puede manejar múltiples lambdas o ser adaptado

# --- Configuración ---
N_LAMBDAS = 13 # Número total de bandas de longitud de onda
MAXSIZE_REGRESSOR = 14 # Puede necesitar ser mayor para incluir lambda
TIMEOUT_REGRESSOR = 7200  # 2 horas, puede ser largo
TEST_SIZE_SPLIT = 0.2

# --- Cargar datos y metadatos ---
print("Cargando datos...")
TRAINING_DATA_PATH = "../training.npy"
# No usaremos testing.npy directamente aquí, sino que dividiremos training_data
METADATA_PATH = "../metadata.json"

training_data_full = load_data(TRAINING_DATA_PATH)
metadata = load_metadata(METADATA_PATH)

# --- Preparar datos para regresión multi-banda ---
print("Preparando datos para regresión multi-banda...")

# Longitudes de onda (ejemplo, necesitas los valores reales en micrones)
# Estos valores están basados en la sección 1 del README.
LAMBDAS_WFC3 = np.array([
    0.838, 0.867, 0.904, 0.941, 0.978, 1.015, 1.052, 
    1.118, 1.208, 1.305, 1.409, 1.520, 1.666
])

F_full = training_data_full[:, :N_LAMBDAS]
T_full = training_data_full[:, N_LAMBDAS]
logH2O_full = training_data_full[:, N_LAMBDAS+1]
logHCN_full = training_data_full[:, N_LAMBDAS+2]
logNH3_full = training_data_full[:, N_LAMBDAS+3]
logkcl_full = training_data_full[:, N_LAMBDAS+4]

num_samples_original = len(T_full)

# --- Cargar tablas de opacidad (Placeholder) ---
# Esto es CRUCIAL. kappa_tables debe ser un diccionario donde cada valor es un array
# de forma (num_samples_original, N_LAMBDAS) o (N_LAMBDAS, num_samples_original)
# que contenga las opacidades para cada muestra y cada longitud de onda.
print("ADVERTENCIA: Usando tablas de opacidad placeholder. Debes reemplazarlas.")
kapp_tables_full = {
    "H2O": np.random.rand(num_samples_original, N_LAMBDAS), 
    "HCN": np.random.rand(num_samples_original, N_LAMBDAS),
    "NH3": np.random.rand(num_samples_original, N_LAMBDAS) 
}

# --- Construir la matriz de características extendida X_tilde y el vector y_tilde ---
# X_tilde tendrá forma (N_samples_original * N_LAMBDAS, d_features + 1)
# y_tilde tendrá forma (N_samples_original * N_LAMBDAS,)

X_tilde_list = []
y_tilde_list = []

print("Construyendo X_tilde y y_tilde...")
for i_lambda in range(N_LAMBDAS):
    lambda_val = LAMBDAS_WFC3[i_lambda]
    print(f"  Procesando lambda = {lambda_val:.3f} um (banda {i_lambda+1}/{N_LAMBDAS})")
    
    # Adaptar kapp_tables para la función calc_pis si espera una sola lambda_i
    # Aquí asumimos que kapp_tables_full[mol] tiene forma (num_samples, N_LAMBDAS)
    # y pasamos solo la columna correspondiente a la lambda actual.
    current_kapp_tables = {
        mol: kapp_tables_full[mol][:, i_lambda] for mol in kapp_tables_full
    }

    Pi1_lam, Pi2_lam, Pi3_lam, _ = calc_pis(
        T_full, logH2O_full, logHCN_full, logNH3_full, logkcl_full,
        current_kapp_tables, lambda_i=i_lambda # lambda_i aquí es solo un índice para la tabla
    )
    
    # Características para esta lambda: Pi1, Pi2, Pi3, lambda_val
    # Pi3 es constante, pero lo incluimos por consistencia con el README
    features_lam = np.column_stack([
        Pi1_lam, 
        Pi2_lam, 
        np.full_like(Pi1_lam, Pi3_lam), 
        np.full_like(Pi1_lam, lambda_val) # Añadir lambda como característica
    ])
    
    X_tilde_list.append(features_lam)
    y_tilde_list.append(F_full[:, i_lambda]) # Flujo para esta lambda

X_tilde = np.concatenate(X_tilde_list, axis=0)
y_tilde = np.concatenate(y_tilde_list, axis=0)

print(f"X_tilde shape: {X_tilde.shape}, y_tilde shape: {y_tilde.shape}")

# --- Dividir en entrenamiento y prueba ---
print(f"Dividiendo datos: test_size = {TEST_SIZE_SPLIT}")
X_train, X_test, y_train, y_test = train_test_split(
    X_tilde, y_tilde, test_size=TEST_SIZE_SPLIT, random_state=42
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# --- Configurar y entrenar SymbolicRegressor ---
print("Configurando SymbolicRegressor...")
# Nombres de las características para el regresor
feature_names = ["Pi1", "Pi2", "Pi3", "lambda"]

model = SymbolicRegressor(
    operators=["+", "-", "*", "/", "log", "pow"], # Podrías añadir más si es necesario
    loss="mse",
    maxsize=MAXSIZE_REGRESSOR,
    feature_names=feature_names,
    uncertainties=True,
    n_jobs=-1,
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
print("Evaluando en el conjunto de prueba...")
y_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
print(f"MSE en el conjunto de prueba: {mse_test:.6e}")

# --- Guardar resultados y gráficos (Placeholder) ---
print("Guardando resultados y gráficos (placeholder)...")
# Aquí deberías guardar la mejor expresión, los scores, y generar gráficos.
# Ejemplo: model.plot_pareto(output_file="../figures/pareto_all_bands.png")

print("Script fit_all.py completado.")