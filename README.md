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

## Resultados del Entrenamiento

| Modelo | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.7827** | **0.7810** | **0.7876** | **0.7843** |
| **Gradient Boosting** | **0.7827** | **0.7810** | **0.7876** | **0.7843** |
| **Logistic Regression** | 0.7622 | 0.7459 | 0.7976 | 0.7709 |
