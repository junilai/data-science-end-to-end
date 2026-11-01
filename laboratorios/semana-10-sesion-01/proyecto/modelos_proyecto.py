import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Misma preparacion que el pipeline de la Semana 9, Sesion 2.
datos = pd.read_csv(
    "../../semana-08-sesion-02/obt_emergencias_canton_dia.csv",
    dtype={"cod_canton": str, "cod_provincia": str},
    parse_dates=["fecha"],
)
datos = datos.dropna(subset=["obj_n_emergencias_t7", "lag_28d", "media_prev_28d"])
datos = datos[~datos["es_zona_no_delimitada"]].copy()
fecha_objetivo = datos["fecha"] + pd.Timedelta(days=7)
datos = datos[~fecha_objetivo.between("2024-01-01", "2024-01-31")].copy()

datos["log_media_prev_28d"] = np.log1p(datos["media_prev_28d"])
datos["media_mismo_dia"] = datos[["lag_7d", "lag_14d", "lag_28d"]].mean(axis=1)
datos["log_media_mismo_dia"] = np.log1p(datos["media_mismo_dia"])
datos["log_lag_1d"] = np.log1p(datos["lag_1d"])
datos["es_fin_de_semana"] = datos["es_fin_de_semana"].astype(int)
datos["es_feriado_t7"] = datos["es_feriado_t7"].astype(int)

OBJETIVO = "obj_n_emergencias_t7"
NUMERICAS = ["log_media_prev_28d", "log_media_mismo_dia", "log_lag_1d"]
CATEGORICAS = ["region", "nombre_dia"]
YA_LISTAS = ["es_fin_de_semana", "es_feriado_t7"]

# Misma division por fecha, con 7 dias de separacion.
CORTE = pd.Timestamp("2025-08-25")
ent = datos[datos["fecha"] < CORTE - pd.Timedelta(days=7)]
prueba = datos[datos["fecha"] >= CORTE]
X_ent, y_ent = ent[NUMERICAS + CATEGORICAS + YA_LISTAS], ent[OBJETIVO]
X_prueba, y_prueba = prueba[NUMERICAS + CATEGORICAS + YA_LISTAS], prueba[OBJETIVO]
print("Entrenamiento:", X_ent.shape, "Prueba:", X_prueba.shape)


def preparacion():
    return ColumnTransformer([
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), NUMERICAS),
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAS),
        ("ya_listas", "passthrough", YA_LISTAS),
    ])


resultados = []


def registrar(nombre, pred_ent, pred_prueba, segundos):
    mae_ent = mean_absolute_error(y_ent, pred_ent)
    mae_prueba = mean_absolute_error(y_prueba, pred_prueba)
    resultados.append({
        "modelo": nombre,
        "mae_entrenamiento": round(mae_ent, 4),
        "mae_prueba": round(mae_prueba, 4),
        "brecha": round(mae_prueba - mae_ent, 4),
        "segundos": round(segundos, 3),
    })
    print(f"{nombre:32s} entrenamiento {mae_ent:8.4f}   prueba {mae_prueba:8.4f}")


def evaluar(nombre, modelo):
    # El objetivo se entrena en log (hallazgo 1) y se devuelve a la escala
    # original para medir el error en emergencias.
    pipeline = Pipeline([
        ("preparar", preparacion()),
        ("modelo", TransformedTargetRegressor(regressor=modelo, func=np.log1p, inverse_func=np.expm1)),
    ])
    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    segundos = time.perf_counter() - inicio
    registrar(nombre, pipeline.predict(X_ent), pipeline.predict(X_prueba), segundos)
    return pipeline


# Linea base: regla del dominio "dentro de 7 dias, igual que hoy". El dia
# t+7 es el mismo dia de la semana que t, asi que respeta la estacionalidad
# semanal. No entrena nada.
inicio = time.perf_counter()
registrar("linea base (igual que hoy)", ent["n_emergencias"], prueba["n_emergencias"],
          time.perf_counter() - inicio)

# Referencia: la mediana de todo el pais, sin mirar el canton.
dummy = DummyRegressor(strategy="median").fit(X_ent, y_ent)
registrar("dummy mediana", dummy.predict(X_ent), dummy.predict(X_prueba), 0.0)

evaluar("ridge alpha=1", Ridge(alpha=1.0))
evaluar("ridge alpha=100", Ridge(alpha=100.0))
evaluar("gradient boosting", GradientBoostingRegressor(random_state=42))
evaluar("gradient boosting max_depth=5", GradientBoostingRegressor(max_depth=5, random_state=42))

tabla = pd.DataFrame(resultados)
tabla.to_csv("resultados_proyecto.csv", index=False)
print()
print(tabla.to_string(index=False))

base = tabla.loc[0, "mae_prueba"]
mejor = tabla.loc[tabla["mae_prueba"].idxmin()]
print(f"\nMejor en prueba: {mejor['modelo']} ({mejor['mae_prueba']:.4f}), "
      f"{(1 - mejor['mae_prueba'] / base) * 100:.1f}% menos error que la linea base")
