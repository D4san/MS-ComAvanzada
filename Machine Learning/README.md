#README: Regresión Simbólica de Espectros de Tránsito (Márquez‐Neila et al. + Matchev et al.)

Este documento describe paso a paso cómo reproducir el flujo de trabajo para construir grupos adimensionales (π), aplicar regresión simbólica a espectros de tránsito de exoplanetas y comparar resultados con los publicados en el influyente trabajo de Matchev et al. (2022), "Analytical Modeling of Exoplanet Transit Spectroscopy with Dimensional Analysis and Symbolic Regression". <mcreference link="https://arxiv.org/abs/2203.09200" index="1">1</mcreference> <mcreference link="https://iopscience.iop.org/article/10.3847/1538-4357/ac658c" index="2">2</mcreference> Dicho estudio utiliza el análisis dimensional y la regresión simbólica para derivar expresiones analíticas para los espectros de tránsito de exoplanetas, ofreciendo una comprensión más profunda de las degeneraciones de los parámetros. <mcreference link="https://arxiv.org/abs/2203.09200" index="1">1</mcreference> <mcreference link="https://iopscience.iop.org/article/10.3847/1538-4357/ac658c" index="2">2</mcreference>
Este proyecto se basa en el dataset de Márquez‐Neila et al. (2018), que contiene espectros sintéticos de “hot Jupiters” generados con TauREx o un método similar, y busca replicar y extender los hallazgos de Matchev et al. (2022).

---

## Descripción de los Scripts del Proyecto

El proyecto se compone de varios scripts de Python, cada uno con una función específica en el flujo de trabajo:

- **`calc_pis.py`**: Contiene funciones esenciales para utilidades de composición química (conversión de logaritmos a fracciones, cálculo de la masa molecular media) y para cargar e interpolar datos de opacidad de ExoTransmit. Su función principal es calcular los grupos Pi adimensionales (Π₁, Π₂, Π₃, Π₄) y graficar los datos de entrenamiento. Este script es fundamental para preparar los datos para la regresión simbólica.
- **`evaluate.py`**: Incluye funciones y marcadores de posición para evaluar los modelos de regresión simbólica (Aproximaciones A y B) contra los datos de prueba. Calcula el Error Cuadrático Medio (MSE), compara los resultados con los de Matchev et al. (2022) y grafica los residuales. Destaca la necesidad de tablas de opacidad reales y una evaluación segura de las expresiones simbólicas.
- **`fit_all.py`**: Un script diseñado para calcular los grupos Pi y la longitud de onda λ para las 13 bandas espectrales. Calcula la M analítica en el límite τ→∞, prepara los datos para la regresión, configura y entrena un modelo `PySRRegressor` optimizando el MSE, guarda los resultados y extrae la mejor fórmula y el MSE del archivo CSV de salida. Este script automatiza el proceso de ajuste para todas las bandas.
- **`fit_band M_model.py`**: Similar a `fit_band.py`, pero calcula específicamente la M analítica en el límite τ→∞ para una sola banda. Prepara los datos, configura y entrena un modelo `PySRRegressor` optimizando el MSE, guarda los resultados y extrae la mejor fórmula y el MSE. Útil para análisis detallados de bandas individuales con el modelo M.
- **`fit_band.py`**: Un script para calcular los grupos Pi para una sola banda espectral. Extrae la profundidad de tránsito M de los datos, prepara los datos para la regresión, configura y entrena un modelo `PySRRegressor` optimizando el MSE, guarda los resultados y extrae la mejor fórmula y el MSE del archivo CSV de salida. Permite un análisis enfocado en una banda específica.
- **`utils.py`**: Contiene funciones de utilidad para cargar datos desde archivos `.npy` y metadatos desde un archivo JSON. Estas funciones son utilizadas por otros scripts para manejar la entrada y salida de datos de manera eficiente.

---

## 1. Estructura general del proyecto

## 1. Estructura general del proyecto

1. **Dataset de Márquez‐Neila et al. (2018)**
   - Carpetas:  
     - `training.npy` (79 999 muestras × 18 columnas)  
     - `testing.npy` (también 18 columnas)  
     - `metadata.json` indicando nombres de columnas, rangos y colores.
   - Columnas por fila (índice 0–17):  
     1–13. Valores de flujo normalizado \(M(\lambda_i)\) en las 13 bandas WFC3 (desde 0.838 μm hasta 1.666 µm).  
     14. \(T\) (K) – temperatura isoterma.  
     15. \(\log_{10} X_{\mathrm{H_2O}}\) (rango \([-13,0]\)).  
     16. \(\log_{10} X_{\mathrm{HCN}}\) (rango \([-13,0]\)).  
     17. \(\log_{10} X_{\mathrm{NH_3}}\) (rango \([-13,0]\)).  
     18. \(\log_{10} \kappa_{\rm cl}\) (opacidad “gray cloud”, rango \([-13,0]\)).

2. **Objetivos principales**  
   1. Construir los **grupos adimensionales \(\Pi\)** a partir de los parámetros físicos originales y constantes asociadas al exoplaneta prototipo (WASP-12b).  
   2. Usar **regresión simbólica** (SymbolFit / PySR) para reencontrar la forma analítica asintótica de Matchev et al. (2022).  
   3. Probar dos alternativas de regresión simbólica:  
      - **(A)** Regresión para una única banda (p. ej. \(\lambda_1 = 0.867\,\mu\text{m}\)).  
      - **(B)** Incluir la longitud de onda \(\lambda\) como entrada adicional en los grupos \(\Pi\).  
   4. Proponer métricas/pruebas para **comparar** con los resultados originales de Matchev et al. (2022).

---

## 2. Definición del exoplaneta prototipo y parámetros fijos

Para reproducir los grupos \(\Pi\) de Matchev et al. (2022), se asume que todos los espectros de Márquez‐Neila et al. (2018) provienen de un mismo “hot Jupiter” (WASP-12b) con parámetros atmosféricos fijos. Esos valores aparecen en la sección 2.2 de Márquez‐Neila et al. (2018):

- Radio planetario  
  \[
    R_0 = 1.79\,R_{\rm J} \quad\bigl(\approx 1.28\times10^8\,\mathrm{m}\bigr).
  \]
- Radio estelar  
  \[
    R_S = 1.57\,R_\odot \quad\bigl(\approx 1.09\times10^9\,\mathrm{m}\bigr).
  \]
- Gravedad superficial  
  \[
    g = 9.77\;\mathrm{m/s^2}.
  \]
- Presión de referencia  
  \[
    P_0 = 10\,\mathrm{bar} = 1\times10^6\,\mathrm{Pa}.
  \]
- Masa molecular media (asumida “solar” H₂–He)  
  \[
    m_{\rm H_2\text{+}He} \;\approx\; 2.3\;m_{\rm amu}, 
    \quad m_{\rm amu} = 1.6605\times10^{-27}\,\mathrm{kg}.
  \]
  En la práctica se calcula a partir de las fracciones molares de H₂ y He (se asume \(X_{\mathrm{H_2}}:X_{\mathrm{He}}=0.85:0.15\) en el remanente de la atmósfera).

> **Observación:** Aunque Matchev et al. (2022) a veces citan \(m=2.4\,m_{\rm amu}\) para simplificar, lo más común en “hot Jupiters” es usar \(m\approx2.3\,m_{\rm amu}\). Nosotros recomiendamos calcularlo directamente con el procedimiento explicado en la sección siguiente, partiendo de las abundancias (mixing ratios) de H₂O, HCN, NH₃ y asignando el remanente H₂+He en proporción 0.85/0.15.

---

## 3. Cálculo de la masa molecular media \(\bar m_k\)

Para cada fila \(k\) del dataset (`training.npy`), extraemos:

1. \(\log_{10}X_{\mathrm{H_2O},k}\), \(\log_{10}X_{\mathrm{HCN},k}\), \(\log_{10}X_{\mathrm{NH_3},k}\).  
2. \(\log_{10}\kappa_{\rm cl,k}\) (opacidad de nubes grises).

### 3.1. Paso 1: conversiones de logarítmicos a fracciones lineales

Sea \(\log_{10}X_{j,k}\) la última matriz (columnas 15–17 en cero‐base):

```python
X_H2O_k = 10**(data[k, 14])
X_HCN_k = 10**(data[k, 15])
X_NH3_k = 10**(data[k, 16])
````

* Estas tres fracciones $X_j$ cubren solo las moléculas menores.
* Deben satisfacerse $0 \le X_j \le 1$, pero en `training.npy` están acotadas a $[10^{-13},1]$ antes de promediar.

### 3.2. Paso 2: asignar fracciones de H₂ y He

Definimos el remanente de la mezcla como:

$$
X_{\mathrm{rem}} = 1 - \bigl(X_{\mathrm{H_2O},k} + X_{\mathrm{HCN},k} + X_{\mathrm{NH_3},k}\bigr).
$$

Luego, dentro de este remanente distribuimos:

$$
X_{\mathrm{H_2},k} = 0.85 \times X_{\mathrm{rem}},
\quad
X_{\mathrm{He},k} = 0.15 \times X_{\mathrm{rem}}.
$$

**Verificación**:

$$
X_{\mathrm{H_2O},k} + X_{\mathrm{HCN},k} + X_{\mathrm{NH_3},k} + X_{\mathrm{H_2},k} + X_{\mathrm{He},k} = 1.
$$

### 3.3. Paso 3: masas molares de cada especie (en kg)

* $\mu_{\mathrm{H_2O}} = 18.01528\,m_{\rm amu} = 18.01528 \times 1.6605\times10^{-27}\,\mathrm{kg}.$
* $\mu_{\mathrm{HCN}} = 27.02655\,m_{\rm amu} = 27.02655 \times 1.6605\times10^{-27}\,\mathrm{kg}.$
* $\mu_{\mathrm{NH_3}} = 17.03052\,m_{\rm amu} = 17.03052 \times 1.6605\times10^{-27}\,\mathrm{kg}.$
* $\mu_{\mathrm{H_2}} =  2.01588\,m_{\rm amu} =  2.01588 \times 1.6605\times10^{-27}\,\mathrm{kg}.$
* $\mu_{\mathrm{He}} =  4.00260\,m_{\rm amu} =  4.00260 \times 1.6605\times10^{-27}\,\mathrm{kg}.$

### 3.4. Paso 4: fórmula de la masa molecular media

$$
\bar m_k
\;=\; 
X_{\mathrm{H_2O},k}\,\mu_{\mathrm{H_2O}}
\;+\; X_{\mathrm{HCN},k}\,\mu_{\mathrm{HCN}}
\;+\; X_{\mathrm{NH_3},k}\,\mu_{\mathrm{NH_3}}
\;+\; X_{\mathrm{H_2},k}\,\mu_{\mathrm{H_2}}
\;+\; X_{\mathrm{He},k}\,\mu_{\mathrm{He}}.
$$

> En Python:
>
> ```python
> m_amu = 1.6605e-27  # kg
> mu_H2O = 18.01528 * m_amu
> mu_HCN = 27.02655 * m_amu
> mu_NH3 = 17.03052 * m_amu
> mu_H2  =  2.01588 * m_amu
> mu_He  =  4.00260 * m_amu
>
> # Supongamos ya convertidas las X_j:
> X_H2O = ...
> X_HCN = ...
> X_NH3 = ...
> X_rem = 1.0 - (X_H2O + X_HCN + X_NH3)
> X_H2  = 0.85 * X_rem
> X_He  = 0.15 * X_rem
>
> m_mean = (
>     X_H2O * mu_H2O 
>   + X_HCN * mu_HCN
>   + X_NH3 * mu_NH3
>   + X_H2  * mu_H2
>   + X_He  * mu_He
> )
> ```

---

## 4. Construcción de los grupos adimensionales $\Pi$

Matchev et al. (2022) muestran que, para atmósferas isotermas e hidrostáticas, el espectro de tránsito $R_T(\lambda)$ se describe mediante cuatro combinaciones adimensionales de las variables físicas originales ($R_0,\,R_S,\,g,\,P_0,\,m,\,k_B,\,T,\;\kappa_{\rm eff}(\lambda)$). De ellas, en el límite asintótico relevante sólo intervienen tres:

1. **Escala de altura**

   $$
     H_k = \frac{k_B\,T_k}{\,\bar m_k\,g\,}.
   $$

2. **Primer grupo adimensional**

   $$
     \Pi_{1,k} = \frac{R_0}{H_k}
                   = \frac{R_0\,\bar m_k\,g}{\,k_B\,T_k\,}.
   $$

3. **Segundo grupo adimensional (opacidad total)**

   $$
     \Pi_{2,k}(\lambda_i) 
     = \frac{\kappa_{\rm eff,k}(\lambda_i)\;P_0}{\,g\,}.
   $$

   Donde

   $$
     \kappa_{\rm eff,k}(\lambda_i) 
       = \sum_{j\in\{\mathrm{H_2O,HCN,NH_3}\}} 
         \Bigl[X_{j,k}\,\kappa_j(\lambda_i)\Bigr] 
       \;+\; 10^{\,\log_{10}\kappa_{\rm cl,k}}.
   $$

   * $\kappa_j(\lambda_i)$ se obtiene de tablas de Heng & Kitzmann (2017) o bases compatibles (H₂O, HCN, NH₃).
   * $\kappa_{\rm cl,k} = 10^{\,\log_{10}\kappa_{\rm cl,k}}$ es la opacidad “gray cloud”.

4. **Tercer grupo adimensional (radio estelar vs. radio planetario)**

   $$
     \Pi_3 = \frac{R_S}{R_0}\quad\text{(constante para todo el dataset)}.
   $$

5. **Cuarto grupo adimensional (combinación redundante)**

   $$
     \Pi_{4,k}(\lambda_i) 
     = \frac{\bar m_k\,\kappa_{\rm eff,k}(\lambda_i)\,P_0}{\,k_B\,T_k\,g\,}.
   $$

   Observación: en la forma asintótica final de $R_T(\lambda)$ sólo aparecen realmente $\Pi_{1}$ y $\Pi_{2}$; $\Pi_3$ se anula al pasar a profundidad de tránsito y $\Pi_4$ resulta algebraicamente dependiente de $\Pi_1$ y $\Pi_2$.

> **Resumen para cada fila $k$:**
>
> 1. Calcular $\bar m_k$ según la sección anterior.
> 2. Calcular $H_k = k_B\,T_k/(\bar m_k\,g)$.
> 3. Obtener $\Pi_{1,k} = R_0/H_k$.
> 4. Para cada banda $\lambda_i$:
>
>    * Cargar $\kappa_j(\lambda_i)$ de las tablas moleculares (p. ej. de Heng & Kitzmann).
>    * Definir $\kappa_{\rm eff,k}(\lambda_i) = \sum_j X_{j,k}\,\kappa_j(\lambda_i) + 10^{\log_{10}\kappa_{\rm cl,k}}$.
>    * Calcular $\Pi_{2,k}(\lambda_i) = (\kappa_{\rm eff,k}(\lambda_i)\,P_0)/g$.
>    * (Opcional) Calcular $\Pi_{4,k}(\lambda_i) = (\bar m_k\,\kappa_{\rm eff,k}(\lambda_i)\,P_0)/(k_B\,T_k\,g)$.
> 5. Tomar $\Pi_3 = R_S/R_0$ (mismo valor para todas las filas).

---

## 5. Ecuación semianalítica asintótica (Heng & Kitzmann 2017)

Para atmósferas isoterma e hidrostática, con opacidad total $\kappa_{\rm eff}\gg 1$, el radio de tránsito $R_T(\lambda)$ satisface en primera aproximación:

$$
R_T(\lambda) 
= R_0 
  + H\,\Bigl[
      \ln\bigl(\sqrt{\,2\pi R_0 / H\,}\bigr) 
      + \gamma_E 
      + \ln\bigl(\kappa_{\rm eff}(\lambda)\,P_0/g\bigr)
    \Bigr],
$$

donde $\gamma_E$ es la constante de Euler–Mascheroni ($\approx0.5772$). Equivalentes formas:

1. **Profundidad de tránsito**

   $$
   \Delta F(\lambda) 
   = \Bigl[\tfrac{R_T(\lambda)}{R_S}\Bigr]^2
   \;\approx\; \Bigl[\tfrac{R_0}{R_S} 
     + \tfrac{H}{R_S}\bigl(\ln C + \ln\kappa_{\rm eff}(\lambda)\bigr)\Bigr]^2,
   $$

   con $C = \sqrt{2\pi R_0/H}\,(P_0/g)$.

2. **Forma adimensional simplificada**
   Definiendo

   $$
     f(\Pi_1,\Pi_2) 
     = \ln\bigl(\alpha\,\Pi_1\bigr) - \gamma_E - 2\ln 2,
   $$

   se recupera la relación que Matchev et al. demuestran simbólicamente (ver Matchev et al. 2022, ecuación (26)).

---

## 6. Aproximaciones y estrategias de regresión simbólica

Se proponen dos aproximaciones principales:

### 6.1. Aproximación A: regresión simbólica para UNA banda $\lambda_i$

1. **Elegir una banda fija** (p. ej. $\lambda_1 = 0.867\,\mu\mathrm{m}$, tal como en Matchev et al.).
2. **Construir matriz de características** $X$ y vector objetivo $y$:

   * $X$ de dimensión $(N,\,d)$ con columnas

     $$
       \bigl[\Pi_{1,k},\;\Pi_{2,k}(\lambda_i),\;\Pi_3,\;\Pi_{4,k}(\lambda_i)\bigr].
     $$

     Normalmente $\Pi_4$ es redundante y se puede omitir, así que basta con $\Pi_{1,k}$ y $\Pi_{2,k}(\lambda_i)$.
   * $y$ de dimensión $(N,)$ = valores de profundidad adimensional de tránsito
     $\;\Delta F_k(\lambda_i)= (R_T(\lambda_i)/R_S)^2\;-\;(\text{offset})\;\approx (R_T/R_S -1)$ según indique el paper.
3. **Ejecutar SymbolicRegressor** (SymbolFit o PySR) con operadores básicos:

   ```
   operators = ["+", "-", "*", "/", "log", "pow"]
   loss      = "mse"
   maxsize   = 12  # o 9 para replicar Matchev et al.
   uncertainties = True  # opcional, para errores de coeficientes
   ```
4. **Ajustar y seleccionar** la fórmula de menor error/MSE, verificando que coincida (hasta tolerancia) con
   $\displaystyle f(\Pi_1,\Pi_2) = \ln(C'\,\Pi_1) - \gamma_E - 2\ln 2.$
5. **Validación cruzada**: usar k‐fold (k=5) para evitar sobreajuste y graficar el Pareto frontier (complejidad vs. error).

### 6.2. Aproximación B: incluir $\lambda$ como input adicional

1. **Concatenar filas para todas las 13 bandas**

   * En vez de entrenar 13 regresiones independientes, se construye

     $$
       \widetilde X = 
       \begin{pmatrix}
         \Pi_{1,1} & \Pi_{2,1}(\lambda_1) & \Pi_3 & \Pi_{4,1}(\lambda_1) & \lambda_1 \\
         \Pi_{1,2} & \Pi_{2,2}(\lambda_1) & \Pi_3 & \Pi_{4,2}(\lambda_1) & \lambda_1 \\
         \;\vdots   & \vdots\;& \vdots & \vdots\; & \vdots \\
         \Pi_{1,1} & \Pi_{2,1}(\lambda_{13}) & \Pi_3 & \Pi_{4,1}(\lambda_{13}) & \lambda_{13} \\
         \vdots    & \vdots\;& \vdots & \vdots\; & \vdots \\
         \Pi_{1,N} & \Pi_{2,N}(\lambda_{13}) & \Pi_3 & \Pi_{4,N}(\lambda_{13}) & \lambda_{13}
       \end{pmatrix},
     $$

     donde $N$ es el número de filas originales (≈80 000).
   * El vector objetivo $\widetilde y$ corresponde a todos los flujos/depths combinados en un solo vector de longitud $13\times N$.
2. **Incluir $\lambda$ como entrada**: esta variable puede tratarse como numérica (en micrómetros) o como escala adimensional (por ejemplo, $\lambda/\lambda_0$, donde $\lambda_0$ es una longitud de onda de referencia).
3. **Entrenar SymbolicRegressor** con el mismo conjunto de operadores. La búsqueda simbólica ahora podrá escoger incluir términos en $\lambda$ (p. ej. $\ln\lambda$, $\lambda^2$, $\lambda \times \Pi_1$, etc.).
4. **Seleccionar** la mejor fórmula que minimice el MSE global, evaluando si efectivamente usa $\lambda$ o si la dependencia en $\lambda$ queda “oculta” en $\Pi_2(\lambda)$.

> **Nota:** al incluir $\lambda$ como variable, puede crecer la complejidad de la expresión. Se sugiere usar un parámetro `maxsize` moderado (p. ej. 14–16) y validar que el resultado sea físicamente interpretable.

---

## 7. Propuestas de comparación con Matchev et al. (2022)

Una vez obtenidas las fórmulas simbólicas para cada enfoque, se recomienda:

1. **Evaluar el Error de Ajuste (MSE/MAE)**

   * Para la Aproximación A (única banda), comparar el MSE entre
     $\Delta F_{\rm fit}(\Pi_1,\Pi_2)$ y $\Delta F_{\rm true}$ en el test set.
   * Para la Aproximación B (multi‐banda + $\lambda$), calcular MSE global en las 13 bandas.
   * Reportar MSE promedio y desviación estándar sobre k‐fold = 5.

2. **Comparar coeficientes y forma de la expresión**

   * Verificar si la expresión recuperada coincide (hasta constantes numéricas) con

     $$
       f(\Pi_1,\Pi_2) = \ln\!\bigl(\alpha\,\Pi_1\bigr) - \gamma_E - 2\ln2,
     $$

     donde idealmente $\alpha = \sqrt{2\pi R_0/H} \times (P_0/g)$.
   * Medir la diferencia relativa entre el coeficiente simbólico hallado y
     $\sqrt{2\pi R_0/H}\,(P_0/g)$ usando, por ejemplo,
     $\bigl|\alpha_{\rm fit} - \alpha_{\rm theo}\bigr| / \alpha_{\rm theo}$.

3. **Mapa de calor de residuales**

   * Graficar la diferencia $\Delta F_{\rm fit} - \Delta F_{\rm true}$ en función de $(\Pi_1,\,\Pi_2)$, tal como en la Fig. 6 de Matchev et al.
   * Mostrar que, para la Aproximación A, los errores quedan distribuidos cerca de cero y sin sesgos sistemáticos.

4. **Análisis de degeneraciones**

   * Tomar sub‐conjuntos de muestras donde $\Pi_1$ se mantenga constante pero cambien $(T,\,m,\,g)$ o donde $\Pi_2$ permanezca constante pero varíe $\kappa_{\rm eff}$.
   * Verificar que la fórmula simbólica produzca el mismo $\Delta F$ dentro del margen de ruido.
   * Con ello, demostrar que las degeneraciones físicas predichas (por ejemplo, la compensación $T \times (m\,g)^{-1}$ fija $H$) se cumplen.

5. **Comparación directa con Matchev et al. (2022)**

   * Extraer la expresión final que ellos reportan:

     $$
       f_{\rm Matchev}(\Pi_1,\Pi_2) 
       = \ln\!\bigl(4.4645\,\Pi_1\bigr) - \gamma_E - 2\ln2
     $$

     (complejidad 9, error ≈10⁻¹⁵).
   * Ajustar un “ajuste forzado” de tu $\Pi_1,\Pi_2$ a esa forma y medir el error resultante en tu dataset.
   * Comparar ese error con el MSE que tu regresión simbólica libre arroja. Si tus datos y tablas de $\kappa_j(\lambda)$ coinciden con los suyos, el error debería ser muy pequeño (<10⁻⁵).

---

## 8. Pasos detallados de implementación

### 8.1. Preparar el entorno

1. Crear un entorno virtual (por ejemplo, con conda o venv) con Python ≥3.8.
2. Instalar dependencias mínimas:

   ```bash
   pip install numpy pandas symbolfit sympy matplotlib scikit-learn pysr
   ```
   **Nota sobre PySR:** PySR requiere Julia para funcionar. Asegúrese de tener Julia instalado y accesible en su PATH. Puede encontrar instrucciones de instalación en el <mcurl name="sitio web oficial de Julia" url="https://julialang.org/downloads/"></mcurl> y en la <mcurl name="documentación de PySR" url="https://github.com/MilesCranmer/PySR#installation"></mcurl>. <mcreference link="https://github.com/MilesCranmer/PySR" index="3">3</mcreference> La primera vez que importe `PySRRegressor`, se instalarán automáticamente las dependencias de Julia necesarias. <mcreference link="https://github.com/MilesCranmer/PySR" index="3">3</mcreference>
3. Conseguir las tablas de opacidad molecular (H₂O, HCN, NH₃) compatibles con Heng & Kitzmann (2017). Puede usarse el repositorio oficial de Heng & Kitzmann o datos provistos en el suplemento de Márquez‐Neila et al. (2018).

### 8.2. Cargar datos y metadatos

```python
import numpy as np
import json

# Cargar JSON de metadata (opcional, para nombres y rangos):
with open("metadata.json", "r") as f:
    meta = json.load(f)

# Cargar datos de entrenamiento / test:
training = np.load("training.npy")   # shape: (79999, 18)
testing  = np.load("testing.npy")    # shape: (n_test, 18)

# Extraer flujos y parámetros físicos:
F_train = training[:, :13]     # flujos (79 999 × 13)
T_train = training[:, 13]      # (79 999,)
logH2O   = training[:, 14]
logHCN   = training[:, 15]
logNH3   = training[:, 16]
logkcl   = training[:, 17]
```

### 8.3. Funciones auxiliares

```python
import numpy as np

# 1. Convertir log10 -> fracción lineal:
def log_to_frac(logX):
    return 10.0 ** (logX)

# 2. Calcular masa molecular media:
def calc_m_mean(logH2O, logHCN, logNH3):
    # Convertir a fracciones lineales:
    X_H2O = log_to_frac(logH2O)
    X_HCN = log_to_frac(logHCN)
    X_NH3 = log_to_frac(logNH3)
    # Fracción remanente H2+He:
    X_rem = 1.0 - (X_H2O + X_HCN + X_NH3)
    X_H2  = 0.85 * X_rem
    X_He  = 0.15 * X_rem

    # Masas molares (kg):
    m_amu  = 1.6605e-27
    mu_H2O = 18.01528 * m_amu
    mu_HCN = 27.02655 * m_amu
    mu_NH3 = 17.03052 * m_amu
    mu_H2  =  2.01588 * m_amu
    mu_He  =  4.00260 * m_amu

    # Masa molecular media:
    return (
        X_H2O * mu_H2O
      + X_HCN * mu_HCN
      + X_NH3 * mu_NH3
      + X_H2  * mu_H2
      + X_He  * mu_He
    )

# 3. Calcular grupos Pi1, Pi2, Pi3 (Pi4 opcional):
def calc_pis(T, logH2O, logHCN, logNH3, logkcl, kapp_tables, lambda_i):
    """
    - T: temperatura (K)
    - logH2O, logHCN, logNH3: arrays de log10 mixing ratios
    - logkcl: log10 de opacidad gray cloud (m^2/kg)
    - kapp_tables: diccionario con kappa_j[lam_i] pre‐cargado
    - lambda_i: índice de la banda i ∈ [0..12]
    """
    # Constantes fijas:
    R0 = 1.79 * 7.1492e7    # m (1.79 R_Jup)
    RS = 1.57 * 6.957e8     # m (1.57 R_Sol)
    g  = 9.77              # m/s^2
    P0 = 1e6               # Pa (10 bar)
    kB = 1.380649e-23       # J/K

    # 1. Masa molecular media:
    m_mean = calc_m_mean(logH2O, logHCN, logNH3)  # array (N,)

    # 2. Escala de altura:
    H = (kB * T) / (m_mean * g)  # array (N,) en metros

    # 3. Pi1:
    Pi1 = (R0 * m_mean * g) / (kB * T)  # equiv. R0 / H

    # 4. Opacidad molecular a lambda_i:
    #    kappa_eff = Σ_j(X_j * kappa_j(λ_i)) + kappa_cl
    X_H2O = log_to_frac(logH2O)
    X_HCN = log_to_frac(logHCN)
    X_NH3 = log_to_frac(logNH3)
    # kappa_tables debe proveer: kappa_H2O[i,:], kappa_HCN[i,:], kappa_NH3[i,:] arrays (N,):
    kapp_H2O_i = kapp_tables["H2O"][lambda_i]
    kapp_HCN_i = kapp_tables["HCN"][lambda_i]
    kapp_NH3_i = kapp_tables["NH3"][lambda_i]
    kappa_cl = 10.0 ** (logkcl)  # array (N,)

    kappa_eff = (X_H2O * kapp_H2O_i 
                + X_HCN * kapp_HCN_i 
                + X_NH3 * kapp_NH3_i) 
                + kappa_cl

    # 5. Pi2:
    Pi2 = (kappa_eff * P0) / g  # array (N,)

    # 6. Pi3 (constante):
    Pi3 = RS / R0

    # 7. Pi4 (opcional):
    Pi4 = (m_mean * kappa_eff * P0) / (kB * T * g)

    return Pi1, Pi2, Pi3, Pi4
```

---

## 9. Procedimiento de regresión simbólica

A continuación se esquematiza el flujo completo de entrenamiento usando SymbolFit (o PySR), tanto para la Aproximación A como para la B.

### 9.1. Aproximación A: regresión simbólica para UNA banda

1. **Elegir banda** (e.g. `lambda_i = 0` para la primera banda WFC3).
2. **Calcular los grupos $\Pi_{1,k}$, $\Pi_{2,k}(\lambda_i)$, $\Pi_3$ y (opcional) $\Pi_{4,k}(\lambda_i)$** usando `calc_pis(...)`.
3. **Construir matriz de características**

   ```python
   Pi1, Pi2, Pi3, Pi4 = calc_pis(T_train, logH2O, logHCN, logNH3, logkcl, kapp_tables, lambda_i=0)
   # Omitir Pi4 si se desea:
   X_train = np.column_stack([Pi1, Pi2, np.full_like(Pi1, Pi3)])
   # Vector objetivo: profundidad de tránsito adimensional (≈ RT/RS - 1)
   # Asumimos que F_train[:, 0] ya es profundidad adimensional o se convierte:
   y_train = F_train[:, 0]
   ```
4. **Configurar y entrenar SymbolicRegressor**

   ```python
   from symbolfit import SymbolicRegressor

   model = SymbolicRegressor(
       operators=["+", "-", "*", "/", "log", "pow"],
       loss="mse",
       maxsize=12,
       uncertainties=True,
       n_jobs=-1,
       timeout=3600  # 1 hora, ajustar según recursos
   )
   model.fit(X_train, y_train)
   expr_best = model.get_best()
   print("Mejor expresión simbólica:", expr_best)
   ```
5. **Evaluar en test set**

   * Calcular $\Pi_{1,k},\Pi_{2,k},\Pi_3$ para los datos de prueba (`testing.npy`).
   * Construir `X_test` y `y_test = F_test[:, 0]`.
   * Calcular `y_pred = model.predict(X_test)` y medir MSE, MAE.
6. **Validación cruzada**

   * Repetir el ajuste con k‐fold = 5 para obtener MSE promedio y desviación estándar.
   * Graficar Pareto frontier (complejidad vs. error) con `model.plot_pareto()`.

### 9.2. Aproximación B: incluir $\lambda$ como variable

1. **Preparar datos agrupados para 13 bandas**

   ```python
   # Para cada banda i ∈ [0..12]:
   all_X = []
   all_y = []
   for i in range(13):
       Pi1, Pi2, Pi3, Pi4 = calc_pis(T_train, logH2O, logHCN, logNH3, logkcl, kapp_tables, lambda_i=i)
       # Incluir Pi4 solo si se desea; aquí incluimos Pi3 y además λ_i normalizado (p.ej. λ_i/μm):
       lam_val = wavelengths[i]  # array escalar o de largo N con valor de λ en μm
       lam_arr = np.full_like(Pi1, lam_val)

       # Matriz X_i (N filas × 4 columnas):
       X_i = np.column_stack([Pi1, Pi2, np.full_like(Pi1, Pi3), lam_arr])
       y_i = F_train[:, i]  # flujo o profundidad para la banda i

       all_X.append(X_i)
       all_y.append(y_i)

   # Concatenar:
   X_all = np.vstack(all_X)  # shape: (13*N, 4)
   y_all = np.hstack(all_y)  # shape: (13*N,)
   ```

2. **Entrenar SymbolicRegressor con $\lambda$ incluido**

   ```python
   model_all = SymbolicRegressor(
       operators=["+", "-", "*", "/", "log", "pow"],
       loss="mse",
       maxsize=16,        # permitir mayor complejidad si conviene
       uncertainties=True,
       n_jobs=-1,
       timeout=7200       # 2 horas, ajustar según recursos
   )
   model_all.fit(X_all, y_all)
   expr_global = model_all.get_best()
   print("Expresión global con λ:", expr_global)
   ```

3. **Evaluación**

   * Repetir en el set de prueba: concatenar bandas en `X_all_test`, `y_all_test`.
   * Calcular MSE global y comparar con la Aproximación A.
   * Analizar si la expresión final usa efectivamente $\lambda$ o si opta por ignorarla.

---

## 10. Formas de comparar con Matchev et al. (2022)

1. **Expresión final y coeficientes**

   * Matchev et al. encontraron (complejidad 9)

     $$
       f_{\rm Matchev}(\Pi_1,\Pi_2) 
       = \ln\bigl(4.4645\,\Pi_1\bigr) - \gamma_E - 2\ln2.
     $$
   * Comparar tu coeficiente simbólico $\alpha_{\rm fit}$ en
     $\ln(\alpha_{\rm fit}\,\Pi_1)$ con $\alpha_{\rm theo} = 4.4645$.
   * Evaluar la diferencia porcentual

     $$
       \frac{|\alpha_{\rm fit} - 4.4645|}{4.4645} \times 100\,\%.
     $$

2. **MSE en profundidad de tránsito**

   * Calcular MSE$_{\rm Matchev}$ al evaluar
     $\Delta F_{\rm Matchev}(\Pi_1,\Pi_2)$ en tu dataset de prueba.
   * Calcular MSE$_{\rm fit}$ al usar $\Delta F_{\rm fit}$ que obtuviste con SymbolFit.
   * Comparar ambos valores; idealmente MSE$_{\rm Matchev}\approx10^{-15}$ (si tus tablas de $\kappa_j$ coinciden) y MSE$_{\rm fit}$≈MSE$_{\rm Matchev}$.

3. **Mapa de calor de residuales (2D)**

   * Graficar el residual $\Delta F_{\rm fit} - \Delta F_{\rm true}$ en un plano $(\Pi_1,\,\Pi_2)$.
   * Matchev et al. muestran un mapa en la Fig. 6: se espera un patrón alrededor de cero sin sesgos.
   * Implementar algo así con `matplotlib.pyplot.imshow()` o similar.

4. **Validación de degeneraciones en 1D**

   * Tomar dos subconjuntos de muestras donde $\Pi_1\approx\text{constante}$ y variar $T,m,g$ (mantener $\Pi_1$ fijo).
   * Verificar que $\Delta F$ se mantenga constante (dentro de la incertidumbre de ruido).
   * Análogamente para $\Pi_2$ constante con niveles de opacidad variables.

5. **Análisis de complejidad vs. error**

   * Graficar la **frontera de Pareto** (`model.plot_pareto()`) con com­ple­jidad en el eje x y MSE en el eje y.
   * Comparar dónde aparece la forma de Matchev (complejidad 9) y su error mínimo.

---

## 11. Resumen de archivos y estructura de carpetas

```
/
├─ README.md        ← este archivo
├─ metadata.json    ← define nombres de columnas, rangos, colores
├─ training.npy     ← datos de entrenamiento (79 999 × 18)
├─ testing.npy      ← datos de prueba
├─ kapp_tables/     ← carpetas/tables con κ_j(λ_i) para H₂O, HCN, NH₃
│    ├─ H2O/        ← archivos binarios o CSV por banda
│    ├─ HCN/
│    └─ NH3/
├─ scripts/
│    ├─ calc_pis.py    ← módulos con funciones para calcular π
│    ├─ fit_band.py    ← script para Aproximación A (única banda)
│    ├─ fit_all.py     ← script para Aproximación B (multi‐banda + λ)
│    ├─ evaluate.py    ← evaluación de MSE, mapas de calor, comparaciones
│    └─ utils.py       ← funciones auxiliares (carga, normalización, etc.)
└─ figures/
     ├─ pareto.png     ← gráfico Pareto (complejidad vs. error)
     ├─ heatmap.png    ← mapa de calor de residuales
     └─ comparisons.png← comparación entre ecuación de Matchev y fit
```

---

## 12. Referencias principales

* **Márquez‐Neila, P., Cubillos, P., Lavie, B., et al. (2018).**
  “Analytical Modeling of Exoplanet Transit Spectroscopy”, *arXiv:1806.03944v1* (PDF).
  → Dataset de 100 000 espectros sintéticos de WASP-12b (13 bandas + 5 parámetros).

* **Matchev, K. et al. (2022).**
  “Rediscovering Analytical Expressions for Transit Spectra with Symbolic Regression”, *Astroph. J.*, 10xx, arXiv:220x.xxxxx.
  → Empleo de PySR para obtener la forma
  $\ln\bigl(4.4645\,\Pi_1\bigr) - \gamma_E - 2\ln2$.

* **Heng, K. & Kitzmann, D. (2017).**
  “Analytical Transmission Spectra for Exoplanet Atmospheres”, *MNRAS*, 470, 2972–2981.
  → Bases de datos de opacidad molecular $\kappa_j(\lambda)$.

---
