# Motor de Matchmaking (Machine Learning) - Gaply

Este repositorio contiene el pipeline completo de Ingeniería de Datos y Entrenamiento de Machine Learning Supervisado para el sistema de recomendación de perfiles de **Gaply**, aislado de la aplicación móvil, además de dos simuladores web para probar el modelo en vivo.

## Estructura del Repositorio

```text
gaply-ml-matchmaking/
├── .devcontainer/
│   └── devcontainer.json                       # Contenedor de desarrollo (VS Code / Codespaces)
├── data/
│   ├── raw/
│   │   ├── perfiles_crudos.csv                 # 2,500 perfiles base (4 universidades · 12 carreras · 15 intereses)
│   │   └── pares_interacciones_crudos.csv      # 10,000 pares con sus 4 features en crudo (5,045 / 4,955)
│   └── processed/
│       └── dataset_normalizado_balanceado.csv  # 20,000 filas escaladas con MinMaxScaler a [0.0 - 1.0] (10,028 / 9,972)
├── models/
│   ├── modelo_matchmaking_gaply.pkl            # Modelo ganador serializado (Random Forest, 100 árboles, profundidad ≤ 6)
│   └── scaler_normalizador.pkl                 # MinMaxScaler congelado para inferencia en nuevos pares
├── notebooks/
│   └── pipeline_entrenamiento_gaply.ipynb      # Pipeline en 4 etapas paso a paso para Google Colab
├── scripts/
│   ├── plantilla_index.html                    # Plantilla del simulador HTML con marcadores de datos
│   └── exportar_modelo_web.py                  # Exporta modelo/scaler/perfiles, valida paridad y genera index.html
├── app.py                                      # Simulador web con Streamlit (formulario lateral + inferencia vectorizada)
├── index.html                                  # Simulador autocontenido (~412 KB, doble clic, sin Python ni servidor)
├── requirements.txt                            # Dependencias fijadas de app.py (local o Streamlit Community Cloud)
├── .gitignore
└── README.md
```

## Cómo Ejecutar el Simulador

### Opción A — `index.html` (local, sin instalar nada)

1. Descargue el repositorio en `.zip` y extráigalo.
2. Busque la carpeta y haga doble clic en `index.html` (se abre directamente en el navegador, `file://`).
3. Complete el formulario (nombre, universidad, carrera y 3–5 intereses) y pulsa **🔍 Buscar Matches**.

El modelo `RandomForest` y el `MinMaxScaler` están embebidos en el propio archivo como JSON (100 árboles, profundidad ≤ 6), por lo que las predicciones son **idénticas bit a bit** a las de scikit-learn, sin necesidad de Python, servidor ni conexión a internet.

### Opción B — `app.py` (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```

También puede desplegarse gratis en [Streamlit Community Cloud](https://share.streamlit.io/) apuntando a `app.py`.

### Opción C — GitHub Codespaces / Dev Container

Al abrir el repositorio en [Codespaces](https://github.com/features/codespaces) o en VS Code con la extensión *Dev Containers*, el archivo `.devcontainer/devcontainer.json` instala `requirements.txt`, arranca `streamlit run app.py` y publica el puerto **8501** con vista previa automática.

### Regenerar `index.html`

Si se reentrena el modelo o cambian los datos:

```bash
python scripts/exportar_modelo_web.py
```

El script valida la paridad exacta contra `predict_proba()` de scikit-learn antes de escribir el HTML.

## Datasets

| Archivo | Filas | Contenido |
| :--- | ---: | :--- |
| `data/raw/perfiles_crudos.csv` | 2,500 | `user_id`, `nombre`, `university`, `career`, `interests` (3–5 valores separados por `\|`), `num_interests` |
| `data/raw/pares_interacciones_crudos.csv` | 10,000 | Las 4 features en crudo + `is_match` (5,045 negativos / 4,955 positivos) |
| `data/processed/dataset_normalizado_balanceado.csv` | 20,000 | Las 4 features escaladas a `[0, 1]` + `is_match` (10,028 positivos / 9,972 negativos) |

**Distribución de perfiles:** 4 universidades (625 perfiles cada una), 12 carreras y 15 intereses posibles; cada perfil elige entre 3 y 5 de ellos (820 perfiles con 3, 827 con 4 y 853 con 5).

## Entrenamiento

El notebook `notebooks/pipeline_entrenamiento_gaply.ipynb` ejecuta el flujo completo en **4 etapas**:

1. **Etapa 1 — Datos crudos:** genera `perfiles_crudos.csv` y `pares_interacciones_crudos.csv` con distribución homogénea.
2. **Etapa 2 — Procesamiento:** aplica `MinMaxScaler` a las 4 features, exporta `dataset_normalizado_balanceado.csv` y guarda `scaler_normalizador.pkl`.
3. **Etapa 3 — Entrenamiento:** partición 80/20 estratificada y comparación de tres modelos; el de mayor *accuracy* se serializa en `modelo_matchmaking_gaply.pkl`.
4. **Etapa 4 — Inferencia en vivo:** recarga el `.pkl` desde disco y predice un par nuevo **sin reentrenar**.

**Modelos comparados**

| Modelo | Hiperparámetros |
| :--- | :--- |
| **Random Forest** ← ganador | `n_estimators=100`, `max_depth=6`, `class_weight='balanced'` |
| Gradient Boosting | `n_estimators=100`, `max_depth=4` |
| Logistic Regression | `class_weight='balanced'` |

## Resultados del Entrenamiento

| Modelo | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.7827** | **0.7810** | **0.7876** | **0.7843** |
| **Gradient Boosting** | **0.7827** | **0.7810** | **0.7876** | **0.7843** |
| **Logistic Regression** | 0.7622 | 0.7459 | 0.7976 | 0.7709 |

Random Forest y Gradient Boosting empatan en las cuatro métricas; el modelo final seleccionado fue **Random Forest** (`models/modelo_matchmaking_gaply.pkl`).

## Lógica del Simulador

Ambos simuladores (`app.py` e `index.html`) comparten exactamente la misma lógica:

1. **Features vectorizadas** por perfil: `misma_universidad`, `misma_carrera`, `intereses_en_comun` y `similitud_jaccard` (intersección / unión), sin bucles fila a fila.
2. **`MinMaxScaler`** congelado sobre las 4 columnas.
3. **`predict_proba()`** del Random Forest → probabilidad de match (clase 1).
4. **Filtrado ≥ 65%**: se muestran los perfiles que superan el umbral (15 tarjetas destacadas + tabla completa); si nadie lo alcanza, se muestran los **top 5** absolutos.

**Validaciones de entrada:** nombre obligatorio y entre 3 y 5 intereses; las opciones se cargan dinámicamente desde el CSV de perfiles.

### Paridad navegador ↔ Python

`scripts/exportar_modelo_web.py` reconstruye la inferencia en NumPy desde el JSON embebido y la compara con `predict_proba()` de scikit-learn: **diferencia máxima 0.0** en 6 escenarios × 2,500 perfiles. `index.html` no referencia ningún recurso externo, por lo que funciona sin red.

## Requisitos

- **Opción A (`index.html`):** solo un navegador.
- **Opción B (`app.py`):** Python 3.10+ con `requirements.txt`. Las versiones están fijadas a propósito: `scikit-learn==1.4.0` es la versión con la que se entrenó el modelo y `pyarrow==17.0.0` es la última compatible con `numpy==1.26.4`.
