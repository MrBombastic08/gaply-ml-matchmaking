"""
Simulador Web Avanzado de Matchmaking — Gaply
=============================================

Emula el comportamiento real de la app Gaply sobre el pipeline de ML congelado:

1. Un usuario nuevo ingresa sus datos en la interfaz (sidebar).
2. Sus datos se comparan contra la base estática de 2,500 perfiles generando
   un DataFrame temporal con 4 columnas (totalmente vectorizado, sin bucles
   fila por fila):
       misma_universidad | misma_carrera | intereses_en_comun | similitud_jaccard
3. Esas 4 columnas pasan por el MinMaxScaler congelado y luego por
   modelo.predict_proba() para obtener P(match = 1).
4. Se muestran los perfiles con probabilidad >= 65% ordenados de mayor a
   menor; si ninguno supera el umbral, se muestran los top 5 absolutos.

Ejecutar:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Constantes (idénticas a las del notebook de entrenamiento)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RUTA_MODELO = BASE_DIR / "models" / "modelo_matchmaking_gaply.pkl"
RUTA_ESCALADOR = BASE_DIR / "models" / "scaler_normalizador.pkl"
RUTA_PERFILES = BASE_DIR / "data" / "raw" / "perfiles_crudos.csv"

# Orden exacto de features con el que se entrenó el modelo
FEATURES = ["misma_universidad", "misma_carrera", "intereses_en_comun", "similitud_jaccard"]

UMBRAL_MATCH = 0.65     # 65% mínimo de probabilidad para considerar "Match"
TOP_ABSOLUTO = 5        # Fallback: top 5 absolutos si nadie supera el umbral
MIN_INTERESES = 3       # El usuario debe elegir entre 3 y 5 intereses
MAX_INTERESES = 5
TARJETAS_PRINCIPALES = 15  # Nº de tarjetas destacadas (el resto va a la tabla)

st.set_page_config(
    page_title="Gaply · Simulador de Matchmaking",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Carga de recursos (se ejecuta una sola vez y queda en memoria)
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    """Carga el modelo predictivo congelado (.pkl)."""
    return joblib.load(RUTA_MODELO)


@st.cache_resource
def cargar_escalador():
    """Carga el MinMaxScaler congelado (.pkl)."""
    return joblib.load(RUTA_ESCALADOR)


@st.cache_data
def cargar_perfiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga la base de 2,500 perfiles + su one-hot de intereses.

    El one-hot (2500 x 15) permite contar los intereses en común con una sola
    suma vectorizada sobre las columnas elegidas, sin iterar fila por fila.
    """
    df = pd.read_csv(RUTA_PERFILES)
    df["interests_set"] = df["interests"].str.split("|").apply(frozenset)
    one_hot = df["interests"].str.get_dummies("|").astype("int32")
    return df, one_hot


# ---------------------------------------------------------------------------
# Lógica de comparación (vectorizada)
# ---------------------------------------------------------------------------
def construir_features(
    df: pd.DataFrame,
    one_hot: pd.DataFrame,
    universidad: str,
    carrera: str,
    intereses_usuario: list[str],
) -> pd.DataFrame:
    """Genera el DataFrame temporal con las 4 features del pipeline.

    Todas las operaciones son vectorizadas sobre las 2.500 filas:
      * igualdad booleana  -> Series == escalar
      * intersección       -> suma del one-hot restringido a los intereses
      * unión              |A ∪ B| = |A| + |B| - |A ∩ B|
    """
    # Intereses válidos elegidos por el usuario (las opciones salen del CSV)
    cols_usuario = [c for c in intereses_usuario if c in one_hot.columns]

    # intereses_en_comun: contar, por perfil, cuántos de sus intereses están
    # en el set del usuario nuevo (una suma sobre columnas del one-hot).
    intereses_comunes = (
        one_hot[cols_usuario].sum(axis=1)
        if cols_usuario
        else pd.Series(0, index=df.index, dtype="int32")
    ).astype("int32")

    # Union de intereses: |usuario| + |perfil| - |interseccion|
    tam_usuario = len(cols_usuario)
    tam_perfil = one_hot.sum(axis=1)                     # 4 o 5 intereses
    union = (tam_usuario + tam_perfil - intereses_comunes).clip(lower=1)

    df_features = pd.DataFrame(
        {
            "misma_universidad": (df["university"] == universidad).astype("int32"),
            "misma_carrera": (df["career"] == carrera).astype("int32"),
            "intereses_en_comun": intereses_comunes,
            "similitud_jaccard": (intereses_comunes / union).astype("float64"),
        },
        index=df.index,
    )
    return df_features[FEATURES]


def predecir_probabilidades(
    df_features: pd.DataFrame, scaler, modelo
) -> np.ndarray:
    """Aplica scaler.transform() + modelo.predict_proba() (clase 1 = Match).

    Se envuelve la salida del escalador en un DataFrame con los nombres de
    feature originales para replicar exactamente la inferencia del notebook
    (evita el warning "X does not have valid feature names").
    """
    X_escalado = scaler.transform(df_features)
    X_normalizado = pd.DataFrame(X_escalado, columns=FEATURES, index=df_features.index)

    clases = list(modelo.classes_)
    col_match = clases.index(1) if 1 in clases else -1  # 1 = Match
    probabilidades = modelo.predict_proba(X_normalizado)[:, col_match]
    return probabilidades


def buscar_matches(
    df: pd.DataFrame,
    one_hot: pd.DataFrame,
    scaler,
    modelo,
    universidad: str,
    carrera: str,
    intereses_usuario: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Ejecuta la inferencia completa sobre los 2.500 perfiles.

    Devuelve (perfiles_con_probabilidad, features, sin_matches).
    """
    df_features = construir_features(df, one_hot, universidad, carrera, intereses_usuario)

    df_resultado = df.copy()
    df_resultado["probabilidad"] = predecir_probabilidades(df_features, scaler, modelo)
    df_resultado["intereses_en_comun"] = df_features["intereses_en_comun"]
    df_resultado["similitud_jaccard"] = df_features["similitud_jaccard"]

    df_resultado = df_resultado.sort_values("probabilidad", ascending=False)

    df_matches = df_resultado[df_resultado["probabilidad"] >= UMBRAL_MATCH]
    sin_matches = df_matches.empty
    if sin_matches:
        df_matches = df_resultado.head(TOP_ABSOLUTO)

    return df_matches, df_resultado, sin_matches


# ---------------------------------------------------------------------------
# Interfaz de usuario
# ---------------------------------------------------------------------------
def intereses_en_comun(fila, intereses_usuario: set) -> list[str]:
    """Intersección exacta de intereses entre el perfil y el usuario nuevo."""
    return sorted(fila["interests_set"] & intereses_usuario)


def pintar_tarjeta(fila, intereses_usuario: set) -> None:
    """Dibuja una tarjeta con Nombre, Carrera, % de Match y intereses comunes."""
    prob = fila["probabilidad"]
    comunes = intereses_en_comun(fila, intereses_usuario)

    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"#### {fila['nombre']}")
            st.caption(f"🎓 {fila['university']} · 📚 {fila['career']}")
        with c2:
            st.metric("Match", f"{prob:.0%}")
        st.progress(float(prob), text=f"{prob:.0%} de probabilidad de match")
        st.markdown(
            f"**Intereses en común ({len(comunes)}):** "
            + (", ".join(f"`{i}`" for i in comunes) if comunes else "_ninguno_")
        )
        st.caption(f"Intereses del perfil: {' · '.join(sorted(fila['interests_set']))}")


def pintar_resultados(
    df_matches: pd.DataFrame,
    df_todos: pd.DataFrame,
    intereses_usuario: set,
    sin_matches: bool,
    nombre: str,
) -> None:
    """Muestra los matches en tarjetas (+ tabla completa con todos)."""
    total_evaluados = len(df_todos)
    st.divider()
    st.subheader(f"Matches para {nombre}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Perfiles evaluados", f"{total_evaluados:,}".replace(",", "."))
    col2.metric("Matches ≥ 65%", len(df_matches) if not sin_matches else 0)
    col3.metric("Mejor probabilidad", f"{df_todos['probabilidad'].max():.0%}")
    col4.metric("Umbral", f"{UMBRAL_MATCH:.0%}")

    if sin_matches:
        st.warning(
            f"Ningún perfil superó el umbral del {UMBRAL_MATCH:.0%}. "
            f"A continuación, los {TOP_ABSOLUTO} mejores perfiles en orden de probabilidad."
        )

    # --- Tarjetas (columnas de 2 en 2) ---
    tarjetas = df_matches.head(TARJETAS_PRINCIPALES)
    for i in range(0, len(tarjetas), 2):
        cols = st.columns(2)
        for j, (_, fila) in enumerate(tarjetas.iloc[i : i + 2].iterrows()):
            with cols[j]:
                pintar_tarjeta(fila, intereses_usuario)

    if len(df_matches) > TARJETAS_PRINCIPALES:
        st.info(
            f"Mostrando las {TARJETAS_PRINCIPALES} tarjetas destacadas; "
            f"los {len(df_matches)} matches completos están en la tabla."
        )

    # --- Tabla con todos los matches (los que superan el umbral) ---
    with st.expander(f"📋 Ver los {len(df_matches)} matches en tabla", expanded=False):
        df_tabla = df_matches.copy()
        df_tabla["intereses_comunes"] = [
            ", ".join(sorted(s & intereses_usuario)) for s in df_tabla["interests_set"]
        ]
        st.dataframe(
            df_tabla[
                [
                    "user_id",
                    "nombre",
                    "university",
                    "career",
                    "intereses_comunes",
                    "probabilidad",
                ]
            ].rename(
                columns={
                    "user_id": "ID",
                    "nombre": "Nombre",
                    "university": "Universidad",
                    "career": "Carrera",
                    "intereses_comunes": "Intereses en común",
                    "probabilidad": "Probabilidad",
                }
            ),
            hide_index=True,
        )


def main() -> None:
    modelo = cargar_modelo()
    scaler = cargar_escalador()
    df_perfiles, one_hot = cargar_perfiles()

    # --- Título principal ---
    st.title("Gaply · Simulador de Matchmaking")
    st.caption(
        f"Inferencia en vivo sobre **{len(df_perfiles):,}** perfiles con el modelo "
        "Random Forest / Gradient Boosting congelado (Min-Max + predict_proba)."
    )

    # --- Sidebar: datos del usuario nuevo ---
    with st.sidebar:
        st.header("👤 Usuario nuevo")
        with st.form("formulario_usuario"):
            nombre = st.text_input("Nombre", placeholder="Ej. Valentina Rojas")

            universidades = sorted(df_perfiles["university"].unique())
            universidad = st.selectbox("Universidad", universidades)

            carreras = sorted(df_perfiles["career"].unique())
            carrera = st.selectbox("Carrera", carreras)

            opciones_intereses = sorted(one_hot.columns.tolist())
            intereses = st.multiselect(
                f"Intereses (selecciona {MIN_INTERESES} a {MAX_INTERESES})",
                options=opciones_intereses,
                help="Las opciones se cargan dinámicamente desde el CSV de perfiles.",
            )

            enviado = st.form_submit_button("🔍 Buscar Matches", type="primary")

        st.caption(
            f"Base de datos: `data/raw/perfiles_crudos.csv` · "
            f"Modelo: `models/modelo_matchmaking_gaply.pkl`"
        )

    # --- Validación + inferencia ---
    if enviado:
        if not nombre.strip():
            st.sidebar.warning("Escribe un nombre para continuar.")
        elif not (MIN_INTERESES <= len(intereses) <= MAX_INTERESES):
            st.sidebar.warning(
                f"Debes seleccionar entre {MIN_INTERESES} y {MAX_INTERESES} intereses "
                f"(llevas {len(intereses)})."
            )
        else:
            with st.spinner("Comparando contra 2.500 perfiles y ejecutando el modelo..."):
                df_matches, df_todos, sin_matches = buscar_matches(
                    df_perfiles,
                    one_hot,
                    scaler,
                    modelo,
                    universidad,
                    carrera,
                    intereses,
                )
            st.session_state["resultados"] = {
                "matches": df_matches,
                "todos": df_todos,
                "intereses": set(intereses),
                "sin_matches": sin_matches,
                "nombre": nombre.strip(),
                "universidad": universidad,
                "carrera": carrera,
            }

    # --- Render de resultados (persisten entre reruns) ---
    if "resultados" in st.session_state:
        r = st.session_state["resultados"]
        pintar_resultados(
            r["matches"], r["todos"], r["intereses"], r["sin_matches"], r["nombre"]
        )
    else:
        st.info(
            "Completa tus datos en la lateral y pulsa **🔍 Buscar Matches** "
            "para compararte contra la base de 2.500 perfiles."
        )
        with st.expander("ℹ️ ¿Cómo funciona este simulador?"):
            st.markdown(
                """
            1. **Cargas tus datos** (universidad, carrera y 3–5 intereses).
            2. **Comparación vectorizada**: se generan 4 features por perfil —
               `misma_universidad`, `misma_carrera`, `intereses_en_comun` y
               `similitud_jaccard` (intersección / unión) — sin bucles fila a fila.
            3. **Modelo congelado**: las 4 columnas pasan por el `MinMaxScaler`
               guardado y por `predict_proba()` del modelo entrenado en Colab.
            4. **Filtrado**: se muestran los perfiles con **≥ 65%** de probabilidad
               de match, de mayor a menor. Si nadie llega al umbral, se muestran
               los **top 5 absolutos**.
            """
            )


if __name__ == "__main__":
    main()
