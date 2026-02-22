# Proyecto 3: Regresión Simbólica de Espectros de Tránsito

## Introducción

El objetivo de este proyecto es demostrar la capacidad de aplicar modelos de _Machine Learning_ a la astrofísica, empleando específicamente el método de **Regresión Simbólica**. Este proyecto busca replicar la metodología y hallazgos del influyente artículo de Matchev _et al._ (2022) en el modelado analítico de espectros de transmisión planetaria.

## Descripción del Problema

Analizar cómo interactúan los parámetros físicos (temperatura, gravedad superficial, masa molecular media y nivel de opacidad por nubes) en el espectro de tránsito de un exoplaneta plantea un desafío importante debido a las degeneraciones paramétricas intrínsecas a las observaciones astrofísicas.
Matchev _et al._ (2022) demostraron que es posible utilizar Regresión Simbólica (ajustando algebraicamente un conjunto de variables observacionales, sin asunciones empíricas heurísticas directas) para redescubrir expresiones analíticas asintóticas subyacentes a simulaciones y datos obtenidos.

## Contexto Teórico: Matchev et al. (2022)

El artículo de _Matchev et al. (2022)_, titulado _"Analytical Modeling of Exoplanet Transit Spectroscopy with Dimensional Analysis and Symbolic Regression"_, proporciona el hilo conductor de este proyecto. Sus pasos fundamentales son:

1.  **Análisis Dimensional:** Basados en el teorema Pi de Buckingham, colapsan los parámetros atmosféricos del exoplaneta WASP-12b en un conjunto drásticamente paramétrico y reducido de grupos funcionales adimensionales ($\Pi_i$).
2.  **Regresión Simbólica:** En lugar de entrenar arquitecturas de "caja negra" (como redes neuronales densas), se utiliza un motor de búsqueda lógica algorítmica (basado en `PySR` o SymbolFit). Este busca heurísticamente el árbol matemático óptimo construyendo combinaciones recursivas elementales de operadores ($+, -, \times, \div, \log$) sobre los grupos de variables $\Pi$.
3.  **Redescubrimiento Físico:** El modelo halla matemáticamente sin injerencia top-down la forma exacta funcional asintótica teórica planteada varios años antes por Heng & Kitzmann (2017).

## Estructura del Proyecto

El avance programático implementado en este directorio cuenta con los siguientes archivos pivote:

- `README.md`: Este documento.
- `calc_pis.py`: Script central para calcular propiedades fundamentales como la composición química o fracciones molares limitantes y convertirlas numéricamente como escalares constructivos de los grupos adimensionales $\Pi$.
- `evaluate.py`: Suite de evaluación que mide residuales de sesgos (MSE, MAE), generador de mapas calóricos relacionales contra los formulismos literarios verdaderos para diagnosticar si la parametrización rompe con degeneraciones físicas irreales.
- `fit_all.py`: Rutina matriz con un _SymbolFit_ que calcula iterativamente las variables proxy para precesar y entrenar globalmente un regresor computacional sobre la suma total paralela de las 13 bandas espectrofotométricas emuladas simultáneamente.
- `fit_band.py` y `fit_band M_model.py`: Módulos reducidos que exploran el paisaje regresivo delimitando de forma univariada la búsqueda paramétrica aislando una banda discreta.
- `utils.py`: Herramienta subyacente para las interfaces rápidas, pre-carga inteligente o filtrados indexados de los inmensos volúmenes en crudo `.npy` derivados de bancos simulados ajenos (típicamente metadatos generados desde simulaciones base ajenas).

> **Aviso Importante sobre los Datos:** El andamiaje inicial asume el flujo propiciado por el banco de _Márquez-Neila et al. (2018)_, conteniendo típicamente 80,000 espectros sintéticos pesados alojados en `training.npy`, `testing.npy` y `metadata.json`. Por cuestiones de alojamiento distribuido liviano en control de versiones en repositorios públicos, es probable que se precisen descargarse por separado.

## Metodología Experimental y Ejecución

_Fase preparatoria de ejecuciones_:

1.  **Definición Físico-Atmosférica**: Pre-parsear e inicializar presiones, densidades o variables sintéticas isotérmicas con logaritmos acotados de mezclas.
2.  **Ajuste y Entrenamiento Automático (`PySRRegressor`)**: Cederle la ejecución bruta a motores genéticos para testear e incentivar miles de genes algorítmicos midiendo siempre exactitud y concisión métrica (Ecuación más simple/más exacta de frente de Pareto lograda).
    - _Nota:_ Exige contar en el OS con un perfil base de `Julia` en _PATH_, siendo su núcleo C-level requerido para acelerar las ramas de prueba inmensas en segundos.
3.  **Análisis Comparativo Directo**: Visualizar convergencias, ver recuperaciones explícitas de $\approx - \gamma_E$ de Euler-Mascheroni u otros terminos, y contrastarlos cuantitativamente con $f_{\rm Matchev}(\Pi_1,\Pi_2) = \ln(4.4645\,\Pi_1) - \gamma_E - 2\ln2$.

**Para testeo local**:

Requerimientos primarios (Python >= 3.8):

```bash
pip install numpy pandas sympy matplotlib scikit-learn pysr symbolfit
```

Comandos de test principal del regresor multidimensional:

```bash
python fit_all.py
```

## Referencias Bibliográficas

- **Matchev, K., Matchev, K., & Roman, A. (2022).** _Analytical Modeling of Exoplanet Transit Spectroscopy with Dimensional Analysis and Symbolic Regression_. The Astrophysical Journal, 930(1), 33. [Link Repositorio arXiv:2203.09200](https://arxiv.org/abs/2203.09200)
- **Márquez-Neila, P., Fisher, C., Sznitman, R., & Heng, K. (2018).** _Supervised machine learning for analysing spectra of exoplanetary atmospheres_. Nature Astronomy, 2(9), 719-724. [Link Repositorio arXiv:1806.03944](https://arxiv.org/abs/1806.03944v1)
- **Heng, K., & Kitzmann, D. (2017).** _Analytical transmission spectra for exoplanet atmospheres_. Monthly Notices of the Royal Astronomical Society, 470(3), 2972-2981.
