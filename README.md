# Motor de Matchmaking (Machine Learning) - Gaply

Este repositorio contiene el pipeline completo de Ingeniería de Datos y Entrenamiento de Machine Learning Supervisado para el sistema de recomendación de perfiles de **Gaply**, aislado de la aplicación móvil.

## Estructura del Repositorio

```text
gaply-ml-matchmaking/
├── data/
│   ├── raw/
│   │   ├── perfiles_crudos.csv                 # 10,000 perfiles base con distribución homogénea
│   │   └── pares_interacciones_crudos.csv      # 20,000 pares de interacción antes de normalizar
│   └── processed/
│       └── dataset_normalizado_balanceado.csv  # Dataset escalado con MinMaxScaler [0.0 - 1.0]
├── models/
│   ├── modelo_matchmaking_gaply.pkl            # Modelo ganador serializado (persistencia de aprendizaje)
│   └── scaler_normalizador.pkl                 # Escalador guardado para inferencia en nuevos pares
├── notebooks/
│   └── pipeline_entrenamiento_gaply.ipynb      # Notebook estructurado paso a paso para Google Colab
└── README.md
```

---

## Respuestas Técnicas de Ingeniería de Datos (Sustentación)

### 1. ¿Cuál es nuestra variable categórica / objetivo?
* **Variable Objetivo (Target de Clase):** `is_match` (binaria categórica: `1` = Match exitoso, `0` = No Match).
* **Variables Categóricas de Entrada (en perfiles crudos):** `university` (4 categorías), `career` (12 categorías) e `interests` (15 categorías posibles, seleccionando de 4 a 5 por estudiante).

### 2. ¿Cuántas características (features) hay por clase?
Cada par de usuarios evaluado para las clases `0` y `1` cuenta con **4 características numéricas explicativas ($X$)**:
1. `misma_universidad` (Binaria: `0` o `1`)
2. `misma_carrera` (Binaria: `0` o `1`)
3. `intereses_en_comun` (Discreta: `0` a `5`)
4. `similitud_jaccard` (Continua: proporción de intersección sobre unión de intereses, de `0.0` a `1.0`)

### 3. ¿El set de datos es homogéneo?
**Sí, se garantizó homogeneidad en dos niveles:**
* **Homogeneidad en los Perfiles (`perfiles_crudos.csv`):** Exactamente 2,500 estudiantes por cada una de las 4 universidades (25.0% c/u), distribución equitativa en las 12 carreras y muestreo uniforme sin sesgos artificiales en los 15 intereses.
* **Homogeneidad en las Clases (`pares_interacciones_crudos.csv`):** Balanceo simétrico 50/50 (~10,000 casos positivos `is_match = 1` y ~10,000 casos negativos `is_match = 0`), eliminando la Paradoja de la Exactitud (*Accuracy Paradox*).

### 4. ¿Cómo se normalizan los datos para que los lea la máquina?
1. **Codificación Relacional (Feature Engineering):** Los textos de ambos perfiles (`user_a` y `user_b`) se transforman en comparaciones matemáticas mediante igualdad booleana (`misma_universidad`, `misma_carrera`) y teoría de conjuntos (`intereses_en_comun` y coeficiente `similitud_jaccard`).
2. **Escalado (`MinMaxScaler`):** Dado que `intereses_en_comun` varía entre `0` y `5` mientras las demás variables operan entre `0` y `1`, se aplicó `MinMaxScaler` de `scikit-learn`:
   $$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$
   De esta forma, todas las características en `dataset_normalizado_balanceado.csv` están estrictamente acotadas en el rango `[0.0, 1.0]`.

### 5. ¿Qué es el ruido en nuestros datos y qué lo genera?
* **Definición:** El ruido representa la variabilidad estocástica del comportamiento humano (dos estudiantes con alta afinidad teórica que deciden no dar *match*, o dos perfiles distintos que sí conectan por factores externos al perfil).
* **Cómo se genera en el código:** Durante la generación del dataset (`Celda 2`), una vez calculado el puntaje determinista (`score_base >= 2.5`), se aplica una **probabilidad estocástica controlada del 22% (`RUIDO_PROBABILIDAD = 0.22`)** mediante `rng.random() < 0.22` que invierte la etiqueta ideal. Esto evita que los algoritmos memoricen una regla trivial (sobreajuste del 100%) y sitúa el techo de rendimiento real en torno al **78%**.

---

## Resultados del Entrenamiento

| Modelo | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.7827** | **0.7810** | **0.7876** | **0.7843** |
| **Gradient Boosting** | **0.7827** | **0.7810** | **0.7876** | **0.7843** |
| **Logistic Regression** | 0.7622 | 0.7459 | 0.7976 | 0.7709 |
