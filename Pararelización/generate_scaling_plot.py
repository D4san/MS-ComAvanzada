# generate_scaling_plot.py

import os
from model.plotting import plot_scaling_results

# Asegurarse de que el directorio 'plots' exista
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Directorio '{plots_dir}' creado.")

# Llamar a la función para generar los gráficos de escalabilidad
# La función buscará los archivos CSV necesarios en '../data' y '../'
# y guardará los gráficos en '../plots'
print("Generando gráficos de escalabilidad...")
plot_scaling_results(base_dir='.') # Usar '.' como base_dir ya que ejecutamos desde la raíz
print("Gráficos generados.")