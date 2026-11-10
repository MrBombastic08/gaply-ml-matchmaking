# Motor de Matchmaking (Machine Learning) - Gaply

Este repositorio contiene el pipeline completo de Ingeniería de Datos y Entrenamiento de Machine Learning Supervisado para el sistema de recomendación de perfiles de **Gaply**, aislado de la aplicación móvil, además de dos simuladores web para probar el modelo en vivo.

## Estructura del Repositorio

```text
gaply-ml-matchmaking/
├── .devcontainer/
│   └── devcontainer.json                       # Configuración del contenedor de desarrollo (VS Code / Codespaces)
├── data/
│   ├── raw/
│   │   ├── perfiles_crudos.csv                 # 2,500 perfiles base con distribución homogénea
│   │   └── pares_interacciones_crudos.csv      # 10,000 pares de interacción antes de normalizar (5,045 / 4,955)
│   └── processed/
│       └── dataset_normalizado_balanceado.csv  # Dataset escalado con MinMaxScaler [0.0 - 1.0]
├── models/
│   ├── modelo_matchmaking_gaply.pkl            # Modelo ganador serializado (RandomForest, 100 árboles)
│   └── scaler_normalizador.pkl                 # Escalador guardado para inferencia en nuevos pares
├── notebooks/
│   └── pipeline_entrenamiento_gaply.ipynb      # Notebook estructurado paso a paso para Google Colab
├── scripts/
│   ├── plantilla_index.html                    # Plantilla del simulador HTML con marcadores de datos
│   └── exportar_modelo_web.py                  # Exporta modelo/scaler/perfiles y genera index.html
├── app.py                                      # Simulador web con Streamlit (sidebar + inferencia vectorizada)
├── index.html                                  # Simulador autocontenido (doble clic, sin Python ni servidor)
├── requirements.txt                            # Dependencias de app.py (local o Streamlit Community Cloud)
└── README.md
```

## Cómo Ejecutar el Simulador

### Opción A — `index.html` (local, sin instalar nada)

1. Descargue el repositorio en .Zip y extráigalo 
3. Busque la carpeta y doble clic en `index.html` (se abre directamente en el navegador, `file://`).
2. Completa el formulario (nombre, universidad, carrera y 3–5 intereses) y pulsa **🔍 Buscar Matches**.

El modelo `RandomForest` y el `MinMaxScaler` están embebidos en el propio archivo como JSON (100 árboles, profundidad ≤ 6), por lo que las predicciones son **idénticas bit a bit** a las de scikit-learn, sin necesidad de Python, servidor ni conexión a internet.

### Opción B — `app.py` (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```

También puede desplegarse gratis en [Streamlit Community Cloud](https://share.streamlit.io/) apuntando a `app.py`.

### Regenerar `index.html`

Si se reentrena el modelo o cambian los datos:

```bash
python scripts/exportar_modelo_web.py
```

El script valida la paridad exacta contra `predict_proba()` de scikit-learn antes de escribir el HTML.

## Lógica del Simulador

Ambos simuladores comparten la misma lógica (definida por el modelo entrenado):

1. **Features vectorizadas** por perfil: `misma_universidad`, `misma_carrera`, `intereses_en_comun` y `similitud_jaccard` (intersección / unión), sin bucles fila a fila.
2. **`MinMaxScaler`** congelado sobre las 4 columnas.
3. **`predict_proba()`** del Random Forest → probabilidad de match (clase 1).
4. **Filtrado ≥ 65%**: se muestran los perfiles que superan el umbral (tarjetas destacadas + tabla completa); si nadie lo alcanza, se muestran los **top 5** absolutos.

## Resultados del Entrenamiento

| Modelo | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.7827** | **0.7810** | **0.7876** | **0.7843** |
| **Gradient Boosting** | **0.7827** | **0.7810** | **0.7876** | **0.7843** |
| **Logistic Regression** | 0.7622 | 0.7459 | 0.7976 | 0.7709 |
