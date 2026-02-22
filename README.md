# Computación Científica Avanzada

¡Bienvenido al repositorio central de proyectos! Este espacio global consolida los tres proyectos prácticos desarrollados durante el curso de **Maestría en Física de Computación Científica Avanzada**.

Cada una de las entregas ha sido diseñada para abordar un conjunto específico de habilidades de alto rendimiento computacional aplicadas y validadas con problemas de física computacional y astrofísica.

---

## Proyecto 1: [Montecarlo](./Montecarlo/)

**El objetivo de este módulo es demostrar la capacidad de implementación y uso de técnicas de simulación de Montecarlo.**
A través del histórico modelo dinámico de radiación de _Daisyworld_ (Watson y Lovelock, 1983), se analiza el comportamiento regulador auto-organizado del clima planetario.

- El núcleo del trabajo reside en la formulación e implementación de Cadenas de Markov Monte Carlo (**MCMC**).
- Se utiliza MCMC para realizar muestreos probabilísticos bayesianos extensos en aras de hallar los mejores valores (y la incerteza asociada) dentro de un gran espacio de parámetros continuos. El objetivo es estabilizar el entorno artificial buscando valores objetivo paramétricos.

## Proyecto 2: [Paralelización](./Pararelización/)

**El objetivo de este módulo es exhibir la destreza de optimizar código computacionalmente costoso, enfocado al reparto del procesamiento con estrategias de paralelización.**
Partiendo del elevado coste en tiempo que implica realizar un ajuste MCMC a gran escala sobre ODEs integradas de funciones dinámicas (como el de Daisy World), la carga total es refactorizada y distribuida a través de dos esquemas de cómputos concurrentes y separados:

- **Paralelismo compartiendo memoria local (`Joblib`)**: Se demuestra un _multiprocessing_ sencillo en CPU, repartiendo el cálculo de las verosimilitudes (_log_probability_) en procesos independientes que reducen un tiempo considerable respecto a un recorrido de walkers meramente secuencial.
- **Procesamiento distribuido (`MPI`)**: Implementación del acople de _Message Passing Interface_ paralela. Este es un procedimiento realizado a nivel más bajo de abstracción de red lógica y control subyacente a nivel maestro-esclavos, propicio no solo en simulaciones de un ordenador _host_ multinúcleo sino también en esquemas multi-nodo clúster (_High Performance Computing_).

## Proyecto 3: [Machine Learning](<./Machine\ Learning/>)

**El objetivo fundamental es exponer las inmensas capacidades de usar modelos interpretativos de Machine Learning en sistemas astrofísicos reales.**
El énfasis del proyecto es recrear y aplicar metodologías punteras publicadas; de esta manera, se busca reproducir explícitamente el enfoque propuesto por _Matchev et al. (2022)_ sobre regresión de espectroscopía de tránsito atmosférico.

- Se aborda el modelamiento de _datasets_ de exoplanetas de tipo "hot Jupiters" reducidos a una serie de grupos espaciales adimensionales ($\Pi$).
- Aplica explícitamente el método empírico de la **Regresión Simbólica** (`PySR`/`SymbolFit`): en lugar de usar cajas negras tipo redes neuronales, busca fórmulas analíticas parsimoniosas a base de mutaciones de operadores matemáticos sobre las observables dadas del exoplaneta en estudio, confirmando así el modelo paramétrico predicho analíticamente en su totalidad.

---

### Requisitos Comunes

Los proyectos operan bajo un stack enlazado en **Python (≥ 3.8)**. Un ambiente base se puede construir instalando las dependencias intersecadas requeridas en todos los directorios:

```bash
pip install numpy scipy matplotlib numba emcee corner ipywidgets joblib mpi4py sympy scikit-learn pysr symbolfit
```

_Repositorio organizado para fines académicos de la materia._
