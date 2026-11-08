"""Preparacion de los datos de mi proyecto, lista para produccion.

Las features que decidiste en tu notebook de la sesion pasada viven aqui,
dentro de funciones. Desde la Semana 10, tus notebooks de experimentos
importan este archivo en vez de copiar el codigo:

    from pipeline_proyecto import cargar_y_dividir, construir_preparacion

Para comprobar que corre de principio a fin, sin notebook:

    python3 pipeline_proyecto.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Tabla OBT de emergencias por canton y dia (Semana 8). La ruta es relativa
# a este archivo, para que funcione aunque se importe desde otra carpeta.
RUTA_DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "../../semana-08-sesion-02/obt_emergencias_canton_dia.csv")

OBJETIVO = "obj_n_emergencias_t7"
NUMERICAS = ["log_media_prev_28d", "log_media_mismo_dia", "log_lag_1d"]
CATEGORICAS = ["region", "nombre_dia"]
YA_LISTAS = ["es_fin_de_semana", "es_feriado_t7"]

# Division por fecha: prueba desde el corte (ultimo anio) y 7 dias de
# separacion, porque el objetivo de cada fila es t+7.
CORTE = pd.Timestamp("2025-08-25")


def agregar_features(datos):
    datos = datos.copy()
    datos["log_media_prev_28d"] = np.log1p(datos["media_prev_28d"])
    datos["media_mismo_dia"] = datos[["lag_7d", "lag_14d", "lag_28d"]].mean(axis=1)
    datos["log_media_mismo_dia"] = np.log1p(datos["media_mismo_dia"])
    datos["log_lag_1d"] = np.log1p(datos["lag_1d"])
    datos["es_fin_de_semana"] = datos["es_fin_de_semana"].astype(int)
    datos["es_feriado_t7"] = datos["es_feriado_t7"].astype(int)
    return datos


def cargar_datos(ruta=RUTA_DATOS):
    datos = pd.read_csv(ruta, dtype={"cod_canton": str, "cod_provincia": str}, parse_dates=["fecha"])
    # Sin objetivo o sin historia de 28 dias la fila no sirve, y las zonas
    # no delimitadas no son cantones.
    datos = datos.dropna(subset=[OBJETIVO, "lag_28d", "media_prev_28d"])
    datos = datos[~datos["es_zona_no_delimitada"]]
    # Hallazgo 2 de la Semana 8: el objetivo de enero de 2024 esta deprimido.
    fecha_objetivo = datos["fecha"] + pd.Timedelta(days=7)
    datos = datos[~fecha_objetivo.between("2024-01-01", "2024-01-31")]
    return agregar_features(datos)


def cargar_y_dividir(ruta=RUTA_DATOS, columnas_extra=()):
    """columnas_extra se agregan a X pero la preparacion las descarta (por
    ejemplo n_emergencias, que usa la linea base "igual que hoy")."""
    datos = cargar_datos(ruta)
    X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS + list(columnas_extra)]
    y = datos[OBJETIVO]
    ent = datos["fecha"] < CORTE - pd.Timedelta(days=7)
    prueba = datos["fecha"] >= CORTE
    return X[ent], X[prueba], y[ent], y[prueba]


def construir_preparacion():
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


# Esto corre con "python3 pipeline_proyecto.py", pero no cuando un
# notebook hace "from pipeline_proyecto import ...".
if __name__ == "__main__":
    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir()
    print("Filas de entrenamiento + prueba (sin los 7 dias de separacion):", len(X_ent) + len(X_prueba))
    preparacion = construir_preparacion()
    X_ent_listo = preparacion.fit_transform(X_ent)
    X_prueba_listo = preparacion.transform(X_prueba)
    print("Entrenamiento:", X_ent_listo.shape, "Prueba:", X_prueba_listo.shape)
    print(list(preparacion.get_feature_names_out()))

    # Comprobacion: el escalador aprendio la media solo de entrenamiento.
    escalador = preparacion.named_transformers_["numericas"].named_steps["escalar"]
    print("Media aprendida:", escalador.mean_.round(4))
    print("Media de entrenamiento:", X_ent[NUMERICAS].mean().values.round(4))
