import os
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Misma preparacion y division que en la Semana 9, Sesion 2 y la Semana 10, Sesion 1.
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
COLUMNAS = NUMERICAS + CATEGORICAS + YA_LISTAS

CORTE = pd.Timestamp("2025-08-25")
ent = datos[datos["fecha"] < CORTE - pd.Timedelta(days=7)]
prueba = datos[datos["fecha"] >= CORTE]
X_ent, y_ent = ent[COLUMNAS], ent[OBJETIVO]
X_prueba, y_prueba = prueba[COLUMNAS], prueba[OBJETIVO]
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


def evaluar(nombre, modelo):
    pipeline = Pipeline([
        ("preparar", preparacion()),
        ("modelo", TransformedTargetRegressor(regressor=modelo, func=np.log1p, inverse_func=np.expm1)),
    ])
    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    segundos = time.perf_counter() - inicio
    mae_prueba = mean_absolute_error(y_prueba, pipeline.predict(X_prueba))
    resultados.append({"modelo": nombre, "mae_prueba": round(mae_prueba, 4),
                       "segundos": round(segundos, 3), "parametros": None})
    print(f"{nombre:32s} prueba {mae_prueba:8.4f}   {segundos:7.2f} s")
    return pipeline


ensamblaje = evaluar("gradient boosting max_depth=5", GradientBoostingRegressor(max_depth=5, random_state=42))
red = evaluar("red neuronal (64)", MLPRegressor(hidden_layer_sizes=(64,), max_iter=200,
                                                early_stopping=True, random_state=42))

mlp = red.named_steps["modelo"].regressor_
parametros_red = sum(m.size for m in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
resultados[-1]["parametros"] = parametros_red
print("Parametros de la red:", parametros_red, "| iteraciones:", mlp.n_iter_)

importancias = pd.Series(
    ensamblaje.named_steps["modelo"].regressor_.feature_importances_,
    index=ensamblaje.named_steps["preparar"].get_feature_names_out(),
).sort_values(ascending=False)
print("Las 3 columnas mas importantes para el ensamblaje:")
print(importancias.head(3).round(4).to_string())

# Prueba del LLM: 10 filas de prueba y 30 de entrenamiento como ejemplos,
# con las columnas que una persona puede leer.
LEGIBLES = ["canton", "nombre_dia", "es_feriado_t7", "n_emergencias", "lag_7d",
            "lag_14d", "lag_28d", "media_prev_7d", "media_prev_28d"]
muestra = prueba.sample(10, random_state=7)
ejemplos = ent.sample(30, random_state=7)

filas_ejemplo = ejemplos[LEGIBLES].round(2).copy()
filas_ejemplo["emergencias_en_7_dias"] = ejemplos[OBJETIVO].astype(int).values
texto = (
    "Estimas cuantas emergencias tendra un canton de Ecuador dentro de 7 dias. "
    "n_emergencias son las de hoy; lag_7d, lag_14d y lag_28d, las de hace 7, 14 y 28 dias; "
    "media_prev_7d y media_prev_28d, el promedio diario de los ultimos 7 y 28 dias. "
    "El dia dentro de 7 dias cae el mismo dia de la semana que hoy (nombre_dia). "
    "Abajo hay 30 filas con el resultado real y 10 filas sin resultado. Responde solo "
    "con 10 numeros, uno por linea, en el mismo orden de las filas sin resultado, sin "
    "texto adicional.\n\n"
    "Filas con resultado:\n"
    + filas_ejemplo.to_csv(index=False)
    + "\nFilas sin resultado:\n"
    + muestra[LEGIBLES].round(2).to_csv(index=False)
)
with open("prompt_llm_proyecto.txt", "w") as archivo:
    archivo.write(texto)
print("\nprompt_llm_proyecto.txt escrito:", len(texto), "caracteres")

y_muestra = muestra[OBJETIVO].values
pred_ensamblaje = ensamblaje.predict(muestra[COLUMNAS])
pred_red = red.predict(muestra[COLUMNAS])
comparacion = muestra[["canton", "fecha", "nombre_dia", "n_emergencias"]].copy()
comparacion["real"] = y_muestra
comparacion["ensamblaje"] = pred_ensamblaje.round(1)
comparacion["red"] = pred_red.round(1)

muestra_10 = {
    "gradient boosting max_depth=5": mean_absolute_error(y_muestra, pred_ensamblaje),
    "red neuronal (64)": mean_absolute_error(y_muestra, pred_red),
}
if os.path.exists("respuestas_llm_proyecto.txt"):
    with open("respuestas_llm_proyecto.txt") as archivo:
        predicciones_llm = [float(linea.strip().replace(",", "")) for linea in archivo if linea.strip()]
    if len(predicciones_llm) == 10:
        comparacion["llm"] = predicciones_llm
        muestra_10["llm"] = mean_absolute_error(y_muestra, predicciones_llm)
        resultados.append({"modelo": "llm", "mae_prueba": None, "segundos": 0.0, "parametros": None})

print()
print(comparacion.to_string(index=False))

tabla = pd.DataFrame(resultados)
tabla["mae_10_filas"] = tabla["modelo"].map(muestra_10).round(4)
tabla.to_csv("comparacion_proyecto.csv", index=False)
print()
print(tabla.to_string(index=False))
