# Proyecto Daisy World

Este repositorio contiene un proyecto de simulación y análisis del modelo **Daisyworld**, basado en las ideas de A. J. Watson y J. E. Lovelock (1983). El objetivo principal es demostrar cómo dos especies de margaritas, con albedos distintos, pueden regular la temperatura global de un planeta ficticio mediante retroalimentaciones entre la temperatura y la cobertura vegetal.

El proyecto integra tres grandes bloques de trabajo:

- **Modelo ODE**: Se resuelven las ecuaciones diferenciales que describen la evolución global de las coberturas de margaritas negras y blancas, junto con el balance radiativo que determina la temperatura global.
- **Análisis Bayesiano con MCMC**: Se utiliza el muestreo MCMC para ajustar parámetros (como luminosidad, tasa de mortalidad, albedos y condiciones iniciales) con el fin de alcanzar objetivos de cobertura específicos. 
- **Simulación Espacial Monte Carlo**: Se modela el planeta como una grilla 2D en la que cada celda puede estar ocupada por una margarita negra, una blanca o permanecer desnuda. Se definen reglas de colonización y muerte, con la posibilidad de incorporar efectos de crecimiento local (dependiente de la influencia de vecinos), lo que conduce a la formación de parches o clusters.

---

## 1. Contenido de la Carpeta

- **Daisy World.ipynb**  
  Notebook principal que contiene:
  - La implementación del modelo Daisyworld mediante ODE.
  - El ajuste de parámetros mediante MCMC para alcanzar coberturas objetivo (por ejemplo, 40% para margaritas negras y 40% para blancas).
  - La simulación espacial Monte Carlo en grilla (con y sin crecimiento local).
  - Visualizaciones de la evolución temporal de las coberturas y la temperatura.

- **DaisyWorld_Montecarlo_Informe.pdf**  
  Informe detallado del proyecto que incluye la fundamentación teórica, metodología, resultados y conclusiones.

- **AAREADME.md**  
  Este documento.

### Carpeta de Visualizaciones: Images and Videos

- **Animaciones GIF**  
  - `GIF_Estaciones.gif`: Muestra la evolución del sistema con variaciones estacionales en la luminosidad.
  - `GIF_Global.gif`: Simulación con distribución aleatoria (sin efectos locales).
  - `GIF_Local.gif`: Simulación con efectos de crecimiento local.
  - `GIF_Nucleos.gif`: Formación de núcleos o clusters de margaritas.

- **Imágenes Estáticas**  
  - `Daisy_Calientes.png`, `Daisy_Frías.png`, `Daisy_Templadas.png`:  
    Resultados de simulaciones obtenidas a través del análisis MCMC, donde se observa la evolución temporal de las coberturas y la temperatura para escenarios de alta, baja e intermedia temperatura óptima.
  - `Corner_Calientes.png`, `Corner_Frías.png`, `Corner_Templadas.png`:  
    Corner plots que muestran la distribución posterior de los parámetros ajustados mediante MCMC.
  - `Daisy_Global.png`, `Daisy_Local.png`, `Daisy_middle.png`:  
    Comparativas entre diferentes configuraciones de simulación espacial.

---

## 2. Descripción General del Proyecto

El proyecto se inspira en el modelo Daisyworld de Watson y Lovelock (1983), que ejemplifica cómo la vida puede autorregular el ambiente planetario a través de la hipótesis de Gaia. En este modelo:

- **Margaritas Negras**: Tienen un albedo bajo (típicamente 0.25), absorbiendo más radiación y calentando su entorno local.
- **Margaritas Blancas**: Poseen un albedo alto (típicamente 0.75), reflejando la radiación y enfriando la zona donde crecen.
- La **tasa de crecimiento** de cada especie es función de la temperatura local, siguiendo una curva parabólica con un óptimo alrededor de 22.5°C, generando un ciclo de retroalimentación que permite estabilizar la temperatura global.

El proyecto explora este mecanismo mediante tres enfoques complementarios:

1. **Modelo ODE**: Resuelve las ecuaciones diferenciales que rigen la dinámica global de cobertura y temperatura, permitiendo analizar la evolución temporal del sistema y sus estados de equilibrio.

2. **Ajuste de Parámetros con MCMC**: Emplea técnicas bayesianas para ajustar los parámetros del modelo (luminosidad, tasa de mortalidad, albedos, etc.), asegurando que las coberturas finales se acerquen a los objetivos establecidos. Los resultados se analizan a través de corner plots que muestran la distribución posterior de los parámetros.

3. **Simulación Espacial Monte Carlo**: Representa el sistema en una grilla 2D, permitiendo estudiar la dinámica espacial y la formación de patrones (clusters) cuando se incorpora el efecto de crecimiento local. Esta aproximación permite visualizar fenómenos emergentes que no son capturados por el modelo ODE global.

---

## 3. Ejecución y Experimentos

### Requerimientos

Para ejecutar el proyecto se necesita tener instalado Python 3.x y las siguientes librerías:

- **NumPy**
- **Matplotlib**
- **Numba** (para aceleración de cálculos)
- **SciPy** (para integración de ODEs)
- **ipywidgets** (para interfaces interactivas)
- **emcee** (para MCMC)
- **corner** (para visualización de resultados MCMC)
- **joblib** (opcional, para paralelización)

Se recomienda utilizar Jupyter Notebook o JupyterLab para abrir y ejecutar el archivo `Daisy World.ipynb`.

### Ejecución del Notebook

1. **Clonar o descargar** este repositorio.
2. **Instalar** las librerías requeridas, por ejemplo:
   ```bash
   pip install numpy matplotlib scipy numba emcee corner ipywidgets joblib
   ```
3. **Abrir** `Daisy World.ipynb` en Jupyter Notebook o JupyterLab.
4. **Ejecutar** las celdas en secuencia para:
   - Definir los parámetros globales y las funciones principales del modelo ODE.
   - Ejecutar la simulación ODE y observar la evolución temporal de las coberturas y la temperatura.
   - Realizar el ajuste de parámetros con MCMC y analizar los resultados mediante corner plots y gráficos de evolución temporal.
   - Ejecutar la simulación espacial Monte Carlo, tanto sin efecto local (distribución aleatoria) como incorporando crecimiento local, y visualizar las diferencias en la formación de patrones.
   - Explorar diferentes escenarios: variación en los parámetros (por ejemplo, tasas de mortalidad, luminosidad, temperatura óptima) y análisis de distribuciones iniciales específicas o la introducción de estacionalidad en la luminosidad.

--- 
## 4. Referencias

- A. J. Watson y J. E. Lovelock (1983). *Biological homeostasis of the global environment: the parable of Daisyworld*. Tellus B, 35(4), 284–289.
- Lovelock, J. E. (1979). *Gaia: A New Look at Life on Earth*. Oxford University Press.
- Wood, A. J., Ackland, G. J., Dyke, J. G., Williams, H. T. P., & Lenton, T. M. (2008). *Daisyworld: A review*. Reviews of Geophysics, 46(1).

