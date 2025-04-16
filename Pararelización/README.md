# Proyecto 2: Comparación de Estrategias de Paralelización en Daisy World

## Introducción
El objetivo de este proyecto es analizar y comparar el desempeño de diferentes estrategias de paralelización aplicadas al problema de Daisy World, específicamente en el contexto de la simulación y ajuste de parámetros mediante MCMC. Se busca evaluar cómo la paralelización puede acelerar los experimentos y qué ventajas ofrece cada enfoque.

## Descripción del Problema
Daisy World es un modelo conceptual que explora la interacción entre organismos (margaritas de diferentes colores) y su entorno, mostrando cómo la vida puede influir en el clima planetario. El modelo se resuelve mediante simulaciones y técnicas de ajuste de parámetros como MCMC.

## Estrategias a Comparar
1. **Ejecución Secuencial**
   - Todo el proceso se realiza en un solo núcleo, sin paralelización.
2. **Paralelización con Joblib**
   - Uso de la librería Joblib para distribuir tareas entre múltiples núcleos de la CPU de manera sencilla.
3. **Paralelización con MPI (Message Passing Interface)**
   - Implementación de paralelización distribuida usando MPI, permitiendo mayor flexibilidad y escalabilidad, incluso en clústeres.

## Motivación
La simulación de modelos complejos y el ajuste de parámetros mediante MCMC pueden ser computacionalmente costosos. Comparar distintas estrategias de paralelización permite identificar la más eficiente y adecuada para este tipo de problemas.

## Criterios de Evaluación
- **Tiempo de ejecución**
- **Escalabilidad**
- **Facilidad de implementación**
- **Flexibilidad y portabilidad**

## Resultados Esperados
Se documentarán los resultados de cada estrategia, incluyendo gráficos de desempeño y análisis comparativo.

## Metodología Experimental
El análisis práctico se basa en el notebook "Daisy World_MCMC.ipynb", donde se implementa y explora el modelo Daisy World. El flujo experimental consiste en:
- Definición de las ecuaciones principales del modelo y sus parámetros.
- Simulación de la dinámica de margaritas negras y blancas sobre una grilla, considerando el acoplamiento entre albedo, temperatura local y global.
- Implementación de funciones optimizadas para el cálculo de crecimiento, albedo y temperatura usando Numba.
- Ejecución de simulaciones extensas para observar la evolución de coberturas y temperatura bajo diferentes condiciones.
- Uso de técnicas de ajuste de parámetros mediante MCMC (Markov Chain Monte Carlo) para explorar el espacio de parámetros y ajustar el modelo a datos sintéticos o condiciones específicas.
- Visualización de resultados mediante gráficos y animaciones que muestran la dinámica espacial y temporal del sistema.

Esta base experimental permite comparar el desempeño de las estrategias de paralelización propuestas, ya que el flujo de trabajo es representativo de problemas reales de simulación y ajuste en modelos ecológicos complejos.

## Conclusiones
Se discutirán las ventajas y desventajas de cada enfoque, así como recomendaciones para futuros trabajos.