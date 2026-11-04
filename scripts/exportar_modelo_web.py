"""
Genera index.html (autocontenido) a partir del pipeline congelado de Gaply.

Uso:
    python scripts/exportar_modelo_web.py

Pasos:
  1. Carga models/modelo_matchmaking_gaply.pkl, models/scaler_normalizador.pkl
     y data/raw/perfiles_crudos.csv (NUNCA los modifica).
  2. Exporta el RandomForest (100 árboles), los parámetros del MinMaxScaler y
     los perfiles a JSON compacto.
  3. Valida la PARIDAD exacta: inferencia pura NumPy reconstruida desde el JSON
     exportado contra sklearn.predict_proba() — debe ser idéntica bit a bit.
  4. Inyecta los datos en scripts/plantilla_index.html y escribe index.html
     en la raíz del repositorio.

El HTML resultante corre con doble clic (file://), sin Python ni servidor.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]

# Orden exacto de features con el que se entrenó el modelo
FEATURES = ["misma_universidad", "misma_carrera", "intereses_en_comun", "similitud_jaccard"]
UMBRAL = 0.65


# ---------------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------------
def exportar_arboles(modelo) -> list[dict]:
    """Serializa cada árbol del RandomForest a arrays planos de JSON."""
    arboles = []
    for estimador in modelo.estimators_:
        t = estimador.tree_
        n = t.node_count
        hojas = t.children_left == -1
        arboles.append(
            {
                "l": t.children_left.astype(int).tolist(),   # hijo izquierdo
                "r": t.children_right.astype(int).tolist(),  # hijo derecho
                "f": t.feature.astype(int).tolist(),         # feature (-2 = hoja)
                "t": t.threshold.tolist(),                   # umbral float64
                # Las hojas almacenan [P(0), P(1)] tal cual los usa predict_proba
                "v": [t.value[i][0].tolist() if hojas[i] else None for i in range(n)],
            }
        )
    return arboles


def exportar_datos(perfiles: pd.DataFrame) -> dict:
    universidades = sorted(perfiles["university"].unique())
    carreras = sorted(perfiles["career"].unique())
    intereses = sorted({i for s in perfiles["interests"] for i in s.split("|")})
    ui = {v: i for i, v in enumerate(universidades)}
    ci = {v: i for i, v in enumerate(carreras)}
    filas = [
        [r.user_id, r.nombre, ui[r.university], ci[r.career], r.interests]
        for r in perfiles.itertuples()
    ]
    return {
        "universidades": universidades,
        "carreras": carreras,
        "intereses": intereses,
        "perfiles": filas,
    }


# ---------------------------------------------------------------------------
# Inferencia de referencia desde el JSON exportado (misma matemática que JS)
# ---------------------------------------------------------------------------
def inferir_desde_json(arboles: list[dict], scale: np.ndarray, offset: np.ndarray,
                       X_crudo: np.ndarray) -> np.ndarray:
    """Replica scaler.transform() + RandomForest.predict_proba() con NumPy.

    * X * scale + offset  en float64 (orden exacto de MinMaxScaler.transform)
    * cast a float32      (lo que hace sklearn dentro de predict)
    * recorrido de árbol   (float32 <= umbral float64, como en C)
    * promedio de P(hoja)  en orden de estimadores, sin renormalizar
    """
    Xs = X_crudo * scale + offset          # float64, multiplicar y luego sumar
    X32 = Xs.astype(np.float32)
    n = X32.shape[0]
    salida = np.zeros(n, dtype=np.float64)

    for arbol in arboles:
        l = np.asarray(arbol["l"], dtype=np.int64)
        r = np.asarray(arbol["r"], dtype=np.int64)
        f = np.asarray(arbol["f"], dtype=np.int64)
        t = np.asarray(arbol["t"], dtype=np.float64)
        p1 = np.array(
            [np.nan if v is None else v[1] for v in arbol["v"]], dtype=np.float64
        )

        nodo = np.zeros(n, dtype=np.int64)
        while True:
            interno = f[nodo] != -2
            if not interno.any():
                break
            idx = np.flatnonzero(interno)
            act = nodo[idx]
            izq = X32[idx, f[act]] <= t[act]   # float32 vs float64 (promoción igual que en C)
            nodo[idx] = np.where(izq, l[act], r[act])

        salida += p1[nodo]

    return salida / len(arboles)


# ---------------------------------------------------------------------------
# Validación de paridad
# ---------------------------------------------------------------------------
def validar(modelo, escalador, perfiles: pd.DataFrame, arboles, esc, datos) -> None:
    rng = np.random.default_rng(42)
    universidades = datos["universidades"]
    carreras = datos["carreras"]
    intereses = datos["intereses"]

    # 0) El escalador exportado reproduce scaler.transform() exactamente
    X_prueba = pd.DataFrame(rng.random((500, 4)) * np.array([1, 1, 5, 1]), columns=FEATURES)
    ref_esc = escalador.transform(X_prueba)
    mi_esc = X_prueba.to_numpy() * np.array(esc["scale"]) + np.array(esc["min"])
    assert np.array_equal(ref_esc, mi_esc), "El escalador exportado no coincide"
    print("OK escalador exportado: identico a scaler.transform() (bit a bit)")

    # 1) Inferencia JSON vs sklearn en escenarios aleatorios + uno real
    max_diff = 0.0
    for intento in range(6):
        if intento == 5:
            # Escenario ancla: el mismo que la prueba end-to-end de app.py
            uni, car = "Universidad de los Andes", "Derecho"
            sel = ["Videojuegos", "Música", "Deportes", "Viajes", "Ciencia"]
        else:
            uni = universidades[rng.integers(len(universidades))]
            car = carreras[rng.integers(len(carreras))]
            k = int(rng.integers(3, 6))
            sel = list(rng.choice(intereses, size=k, replace=False))

        set_sel = set(sel)
        comunes = perfiles["interests"].str.split("|").apply(
            lambda s: len(set(s) & set_sel)
        ).astype("int32")
        union = len(sel) + perfiles["interests"].str.split("|").apply(len) - comunes
        X_crudo = pd.DataFrame(
            {
                "misma_universidad": (perfiles["university"] == uni).astype("int64"),
                "misma_carrera": (perfiles["career"] == car).astype("int64"),
                "intereses_en_comun": comunes,
                "similitud_jaccard": comunes / union,
            }
        )[FEATURES].to_numpy()

        # Referencia sklearn (mismos envoltorios de DataFrame que app.py)
        Xs = escalador.transform(pd.DataFrame(X_crudo, columns=FEATURES))
        proba_sk = modelo.predict_proba(
            pd.DataFrame(Xs, columns=FEATURES)
        )[:, 1]
        proba_json = inferir_desde_json(
            arboles, np.array(esc["scale"]), np.array(esc["min"]), X_crudo
        )

        diff = float(np.max(np.abs(proba_sk - proba_json)))
        max_diff = max(max_diff, diff)
        coinciden = np.array_equal(proba_sk, proba_json)

        if intento == 5:  # escenario ancla: Andes/Derecho/5 intereses (app.py)
            n_matches = int((proba_sk >= UMBRAL).sum())
            print(f"  ancla: {uni} / {car} / {sel} -> "
                  f"{n_matches} matches >= 65%, max={proba_sk.max():.6f}")
            assert n_matches == 734, f"Se esperaban 734 matches, salieron {n_matches}"

        print(f"  escenario {intento + 1}: {'IDENTICO' if coinciden else f'DIFF {diff:.3e}'} "
              f"(max prob {proba_sk.max():.4f})")
        assert coinciden, f"Paridad fallida (diff={diff:.3e})"

    print(f"OK paridad bit a bit: max diff = {max_diff} en 6 escenarios x 2.500 perfiles")


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main() -> None:
    modelo = joblib.load(RAIZ / "models" / "modelo_matchmaking_gaply.pkl")
    escalador = joblib.load(RAIZ / "models" / "scaler_normalizador.pkl")
    perfiles = pd.read_csv(RAIZ / "data" / "raw" / "perfiles_crudos.csv")

    print(f"Modelo: {type(modelo).__name__} con {len(modelo.estimators_)} arboles")
    print(f"Perfiles: {len(perfiles)} filas")

    arboles = exportar_arboles(modelo)
    esc = {
        "scale": [float(x) for x in escalador.scale_],
        "min": [float(x) for x in escalador.min_],
    }
    datos = exportar_datos(perfiles)

    # --- Validación ANTES de generar el HTML ---
    validar(modelo, escalador, perfiles, arboles, esc, datos)

    # --- Inyección en la plantilla ---
    json_arboles = json.dumps(arboles, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    json_esc = json.dumps(esc, separators=(",", ":"))
    json_datos = json.dumps(datos, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    plantilla = (RAIZ / "scripts" / "plantilla_index.html").read_text(encoding="utf-8")
    html = plantilla
    for marca, payload in (
        ("/*__MODELO__*/null", json_arboles),
        ("/*__ESCALADOR__*/null", json_esc),
        ("/*__DATOS__*/null", json_datos),
    ):
        assert html.count(marca) == 1, f"Marca no encontrada (o duplicada): {marca}"
        html = html.replace(marca, payload)

    salida = RAIZ / "index.html"
    salida.write_text(html, encoding="utf-8", newline="\n")

    kb = salida.stat().st_size / 1024
    print(f"\nOK index.html generado: {kb:.0f} KB "
          f"(arboles {len(json_arboles) / 1024:.0f} KB, datos {len(json_datos) / 1024:.0f} KB)")
    print("Abre index.html con doble clic — no requiere Python ni servidor.")


if __name__ == "__main__":
    main()
