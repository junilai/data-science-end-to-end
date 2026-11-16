"""Servicio de prediccion de emergencias por canton a 7 dias.

Quien llama (el lote de cada noche, un tablero, otra aplicacion) manda lo que
sabe de un canton al cierre del dia t: la fecha, la region, sus emergencias
pasadas y si t + 7 es feriado. El servicio calcula las features con las mismas
funciones de modelado/pipeline_proyecto.py con que se entreno el modelo.

Desde psets/proyecto/, despues de python -m servicio.exportar_modelo:

    uvicorn servicio.api:app --port 8000
"""

import os
from datetime import date, timedelta

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from modelado.pipeline_proyecto import (CATEGORICAS, NUMERICAS, YA_LISTAS,
                                        agregar_calendario, agregar_features)

CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.environ.get("RUTA_MODELO", os.path.join(CARPETA, "modelo.joblib"))
RUTA_VERSION = os.environ.get("RUTA_VERSION", os.path.join(CARPETA, "modelo_version.txt"))

# Se carga una sola vez, al arrancar. La version es el numero de la version de
# "emergencias-cantones" en el registro de MLflow, que exportar_modelo.py deja
# junto al modelo: el contenedor la sabe sin llegar al servidor de MLflow.
modelo = None
version_modelo = None
if os.path.exists(RUTA_MODELO):
    modelo = joblib.load(RUTA_MODELO)
if os.path.exists(RUTA_VERSION):
    with open(RUTA_VERSION) as archivo:
        version_modelo = archivo.read().strip()

app = FastAPI(title="Emergencias por canton a 7 dias")


class Canton(BaseModel):
    """Lo que se sabe de un canton al cierre del dia t."""

    fecha: date
    region: str
    lag_1d: float | None = Field(default=None, ge=0)
    lag_7d: float = Field(ge=0)
    lag_14d: float = Field(ge=0)
    lag_28d: float = Field(ge=0)
    media_prev_28d: float = Field(ge=0)
    es_feriado_t7: bool = False


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo_cargado": modelo is not None, "version_modelo": version_modelo}


@app.post("/predecir")
def predecir(canton: Canton):
    # Un lag_1d vacio llega como None; como float queda NaN y lo imputa la
    # preparacion con la mediana que aprendio al entrenar.
    datos = pd.DataFrame([canton.model_dump()]).astype({"lag_1d": "float64"})
    datos = agregar_features(agregar_calendario(datos))
    estimacion = modelo.predict(datos[NUMERICAS + CATEGORICAS + YA_LISTAS])[0]
    return {
        "fecha_objetivo": (canton.fecha + timedelta(days=7)).isoformat(),
        "emergencias_estimadas": round(float(estimacion), 2),
        "version_modelo": version_modelo,
    }
