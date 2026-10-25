import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Tabla OBT de emergencias por canton y dia (Semana 8), con las 2 features
# de la sesion pasada agregadas aqui mismo.
datos = pd.read_csv(
    "../../semana-08-sesion-02/obt_emergencias_canton_dia.csv",
    dtype={"cod_canton": str, "cod_provincia": str},
    parse_dates=["fecha"],
)

# Mismos filtros que en la sesion pasada: sin objetivo o sin historia de
# 28 dias la fila no sirve, y las zonas no delimitadas no son cantones.
datos = datos.dropna(subset=["obj_n_emergencias_t7", "lag_28d", "media_prev_28d"])
datos = datos[~datos["es_zona_no_delimitada"]].copy()

# Hallazgo 2 de la Semana 8: el objetivo de enero de 2024 esta deprimido
# (periodo anomalo). Se quitan las filas cuyo dia objetivo (t+7) cae ahi.
fecha_objetivo = datos["fecha"] + pd.Timedelta(days=7)
datos = datos[~fecha_objetivo.between("2024-01-01", "2024-01-31")].copy()

datos["log_media_prev_28d"] = np.log1p(datos["media_prev_28d"])
datos["media_mismo_dia"] = datos[["lag_7d", "lag_14d", "lag_28d"]].mean(axis=1)
datos["log_media_mismo_dia"] = np.log1p(datos["media_mismo_dia"])
datos["log_lag_1d"] = np.log1p(datos["lag_1d"])
datos["es_fin_de_semana"] = datos["es_fin_de_semana"].astype(int)
datos["es_feriado_t7"] = datos["es_feriado_t7"].astype(int)
print("Filas para modelar:", len(datos))

OBJETIVO = "obj_n_emergencias_t7"
NUMERICAS = ["log_media_prev_28d", "log_media_mismo_dia", "log_lag_1d"]
CATEGORICAS = ["region", "nombre_dia"]
YA_LISTAS = ["es_fin_de_semana", "es_feriado_t7"]

X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
y = datos[OBJETIVO]

# El modelo predice el futuro a partir de filas con fecha: se divide por
# fecha. Entrenamiento antes del corte, prueba desde el corte (ultimo anio).
# Se dejan 7 dias de separacion: el objetivo de una fila es t+7, asi que
# las ultimas filas de entrenamiento no pueden tener su objetivo dentro
# del periodo de prueba.
CORTE = pd.Timestamp("2025-08-25")
ent = datos["fecha"] < CORTE - pd.Timedelta(days=7)
prueba = datos["fecha"] >= CORTE
X_ent, X_prueba, y_ent, y_prueba = X[ent], X[prueba], y[ent], y[prueba]
print("Entrenamiento:", datos.loc[ent, "fecha"].min().date(), "a", datos.loc[ent, "fecha"].max().date())
print("Prueba:", datos.loc[prueba, "fecha"].min().date(), "a", datos.loc[prueba, "fecha"].max().date())
print(f"Proporcion de prueba: {prueba.sum() / (ent.sum() + prueba.sum()):.1%}")

preparacion = ColumnTransformer([
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

X_ent_listo = preparacion.fit_transform(X_ent)
X_prueba_listo = preparacion.transform(X_prueba)
print("Entrenamiento:", X_ent_listo.shape, "Prueba:", X_prueba_listo.shape)
print(list(preparacion.get_feature_names_out()))

# Comprobacion: el escalador aprendio la media solo de entrenamiento.
escalador = preparacion.named_transformers_["numericas"].named_steps["escalar"]
print("Media aprendida:", escalador.mean_.round(4))
print("Media de entrenamiento:", X_ent[NUMERICAS].mean().values.round(4))
print("Media de todo el dataset:", X[NUMERICAS].mean().values.round(4))
