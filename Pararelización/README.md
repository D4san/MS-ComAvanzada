# Proyecto 2: Comparación de Estrategias de Paralelización en Daisy World

## Introducción
El objetivo de este proyecto es analizar y comparar el desempeño de diferentes estrategias de paralelización aplicadas al problema de Daisy World. Específicamente, se busca evaluar cómo la paralelización puede acelerar los experimentos de ajuste de parámetros mediante MCMC (Markov Chain Monte Carlo) para encontrar las condiciones que lleven a una cobertura estable de margaritas blancas y negras cercana al 40% (0.4) cada una.

## Descripción del Problema
Daisy World es un modelo conceptual que explora la interacción entre organismos (margaritas de diferentes colores) y su entorno, mostrando cómo la vida puede influir en el clima planetario. El modelo se resuelve mediante simulaciones y técnicas de ajuste de parámetros como MCMC para encontrar los parámetros que reproducen un comportamiento deseado (en este caso, cobertura de 0.4).

## Estrategias a Comparar
1.  **Ejecución Secuencial**: Todo el proceso MCMC se realiza en un solo núcleo.
2.  **Paralelización con Joblib**: Uso de la librería Joblib para distribuir las evaluaciones de la función de probabilidad del MCMC entre múltiples núcleos de la CPU.
3.  **Paralelización con MPI (Message Passing Interface)**: Implementación de paralelización distribuida usando MPI, permitiendo la ejecución en múltiples nodos o clústeres.

## Motivación
La simulación de modelos complejos y el ajuste de parámetros mediante MCMC pueden ser computacionalmente costosos. Comparar distintas estrategias de paralelización permite identificar la más eficiente y adecuada para acelerar la exploración del espacio de parámetros y encontrar soluciones específicas como la cobertura objetivo de 0.4.

## Estructura del Proyecto

El repositorio se organiza de la siguiente manera:

-   `README.md`: Este archivo.
-   `Daisy World_MCMC.ipynb`: Notebook original de exploración y base del modelo.
-   `requirements.txt`: Dependencias de Python necesarias.
-   `model/`: Contiene el código modularizado del modelo Daisyworld.
    -   `__init__.py`: Inicializador del módulo.
    -   `equations.py`: Ecuaciones fundamentales del modelo (temperatura, albedo, crecimiento).
    -   `simulation.py`: Función principal para ejecutar una simulación de Daisyworld.
    -   `mcmc.py`: Funciones relacionadas con el MCMC (log_probability, ejecución con Joblib).
    -   `plotting.py`: Funciones para generar gráficos específicos del modelo.
-   `walkers_sequential_mcmc.py`: Script principal para ejecutar la batería de experimentos MCMC de forma **secuencial** con diferente número de walkers (50, 100, 200, 300, 400, 500, 700). Registra tiempos y errores, y genera gráficos y CSV de resultados (`results_sequential.csv`, `results_sequential.png`, `corner_sequential.png`, `recovered_simulation_sequential.png`).
-   `walkers_joblib_mcmc.py`: Script principal para ejecutar la batería de experimentos MCMC usando **Joblib** para paralelizar en CPU, con diferente número de walkers. Registra tiempos y errores, y genera gráficos y CSV de resultados (`results_joblib.csv`, `results_joblib.png`, `corner_joblib.png`, `recovered_simulation_joblib.png`).
-   `walkers_mpi_mcmc.py`: Script principal para ejecutar un experimento MCMC usando **MPI** para paralelización distribuida. Este script está diseñado para ejecutarse con un número específico de walkers (configurable dentro del script) y varios procesos MPI. Guarda los resultados de forma incremental en `results_mpi_incremental.csv` y genera gráficos (`corner_mpi_gran_cadena.png`, `recovered_simulation_mpi.png`).
-   `individual_sequential.py`, `individual_joblib.py`, `individual_mpi.py`: Scripts de prueba para ejecutar una única simulación MCMC con un número fijo de walkers para cada estrategia, útiles para depuración o pruebas rápidas.
-   `plot_times.py`: Script para generar un gráfico comparativo (`comparacion_estrategias_mcmc.png`) de los tiempos de ejecución de las tres estrategias, leyendo los archivos CSV generados por los scripts `walkers_*`.
-   `results/`: Carpeta que contiene los resultados generados.
    -   `data/`: Archivos CSV con los datos numéricos de los experimentos.
    -   `plots/`: Archivos PNG con los gráficos generados (comparativas, corner plots, simulaciones recuperadas).

## Metodología Experimental

1.  **Definición del Modelo**: Se utilizan las ecuaciones implementadas en `model/equations.py`.
2.  **Simulación Base**: La función `model/simulation.py:simulate_daisyworld` ejecuta una simulación.
3.  **Ajuste MCMC**: Se utiliza `emcee` con la función `model/mcmc.py:log_probability` para evaluar la probabilidad de los parámetros, buscando aquellos que resulten en coberturas cercanas a 0.4 para ambas margaritas.
4.  **Ejecución de Experimentos**: Se ejecutan los scripts `walkers_*.py` para cada estrategia:
    -   Secuencial: `python walkers_sequential_mcmc.py`
    -   Joblib: `python walkers_joblib_mcmc.py`
    -   MPI: `mpiexec -n <numero_procesos> python walkers_mpi_mcmc.py` (ajustando `n_total_walkers` dentro del script).
5.  **Análisis de Resultados**: Se comparan los tiempos de ejecución y la calidad del ajuste (errores respecto a la cobertura objetivo) almacenados en los archivos `.csv` y visualizados en los `.png` dentro de `results/`. El script `plot_times.py` genera la comparación final de tiempos.

## Criterios de Evaluación

-   **Tiempo de ejecución**: Tiempo total para completar la batería de MCMC para cada estrategia.
-   **Escalabilidad**: Cómo varía el tiempo de ejecución al aumentar el número de walkers (para secuencial y Joblib) o procesos (para MPI).
-   **Calidad del Ajuste**: Error promedio entre la cobertura final obtenida y la cobertura objetivo (0.4).
-   **Facilidad de implementación**: Complejidad relativa de configurar y ejecutar cada estrategia.
