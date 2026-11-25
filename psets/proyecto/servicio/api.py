"""Servicio de prediccion de emergencias por canton a 7 dias, instrumentado.

Quien llama (el lote de cada noche, un tablero, otra aplicacion) manda lo que
sabe de un canton al cierre del dia t: su codigo, la fecha, la region, sus
emergencias pasadas y si t + 7 es feriado. El servicio calcula las features
con las mismas funciones de modelado/pipeline_proyecto.py con que se entreno
el modelo.

Ademas de responder, mide la latencia y el codigo de cada prediccion (las
metricas del sistema, en /metricas), guarda cada prediccion en
monitoreo/predicciones.jsonl, para unirla despues con las emergencias reales
del dia t + 7 (monitoreo/desempeno.py), y vigila la deriva de datos de las
ultimas predicciones contra el entrenamiento (en /deriva).

Desde psets/proyecto/, despues de python -m servicio.exportar_modelo:

    uvicorn servicio.api:app --port 8000
"""

import json
import os
import time
import uuid
from datetime import date, datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from modelado.pipeline_proyecto import (CATEGORICAS, NUMERICAS, YA_LISTAS,
                                        agregar_calendario, agregar_features)
from servicio.deriva import MonitorDeDeriva

CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.environ.get("RUTA_MODELO", os.path.join(CARPETA, "modelo.joblib"))
RUTA_VERSION = os.environ.get("RUTA_VERSION", os.path.join(CARPETA, "modelo_version.txt"))
RUTA_REGISTRO = os.environ.get("RUTA_REGISTRO",
                               os.path.join(CARPETA, "..", "monitoreo", "predicciones.jsonl"))
RUTA_REFERENCIA = os.environ.get("RUTA_REFERENCIA", os.path.join(CARPETA, "referencia_deriva.json"))

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

# La referencia de deriva la escribe python -m monitoreo.deriva_proyecto.
monitor = None
if os.path.exists(RUTA_REFERENCIA):
    with open(RUTA_REFERENCIA) as archivo:
        monitor = MonitorDeDeriva(json.load(archivo))


class Metricas:
    """Las metricas del sistema, en memoria: una latencia y un codigo por
    cada peticion a /predecir. Se reinician cuando el servicio se reinicia."""

    def __init__(self):
        self.latencias_ms = []
        self.codigos = []

    def registrar(self, latencia_ms, codigo):
        self.latencias_ms.append(latencia_ms)
        self.codigos.append(codigo)

    def resumen(self):
        p50 = p95 = None
        if self.latencias_ms:
            p50 = round(float(np.percentile(self.latencias_ms, 50)), 2)
            p95 = round(float(np.percentile(self.latencias_ms, 95)), 2)
        return {
            "peticiones": len(self.codigos),
            "errores_validacion": sum(codigo == 422 for codigo in self.codigos),
            "errores_servidor": sum(codigo >= 500 for codigo in self.codigos),
            "latencia_p50_ms": p50,
            "latencia_p95_ms": p95,
        }


metricas = Metricas()
app = FastAPI(title="Emergencias por canton a 7 dias")


@app.middleware("http")
async def medir(request: Request, call_next):
    """Mide la latencia y el codigo de cada peticion a /predecir. Si la ruta
    lanza una excepcion, la cuenta como error de servidor (500) y la deja
    seguir, para que FastAPI responda 500."""
    if request.url.path != "/predecir":
        return await call_next(request)
    inicio = time.perf_counter()
    try:
        respuesta = await call_next(request)
    except Exception:
        metricas.registrar((time.perf_counter() - inicio) * 1000, 500)
        raise
    metricas.registrar((time.perf_counter() - inicio) * 1000, respuesta.status_code)
    return respuesta


class Canton(BaseModel):
    """Lo que se sabe de un canton al cierre del dia t."""

    cod_canton: str = Field(min_length=1)
    fecha: date
    region: str
    lag_1d: float | None = Field(default=None, ge=0)
    lag_7d: float = Field(ge=0)
    lag_14d: float = Field(ge=0)
    lag_28d: float = Field(ge=0)
    media_prev_28d: float = Field(ge=0)
    es_feriado_t7: bool = False


def registrar_prediccion(registro):
    """Agrega una linea al registro de predicciones (JSONL)."""
    os.makedirs(os.path.dirname(os.path.abspath(RUTA_REGISTRO)), exist_ok=True)
    with open(RUTA_REGISTRO, "a") as archivo:
        archivo.write(json.dumps(registro) + "\n")


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo_cargado": modelo is not None, "version_modelo": version_modelo}


@app.get("/metricas")
def ver_metricas():
    return metricas.resumen()


@app.get("/deriva")
def ver_deriva():
    if monitor is None:
        return {"referencia_cargada": False}
    return {"referencia_cargada": True, **monitor.estado()}


@app.post("/predecir")
def predecir(canton: Canton):
    # Un lag_1d vacio llega como None; como float queda NaN y lo imputa la
    # preparacion con la mediana que aprendio al entrenar.
    datos = pd.DataFrame([canton.model_dump()]).astype({"lag_1d": "float64"})
    datos = agregar_features(agregar_calendario(datos))
    estimacion = round(float(modelo.predict(datos[NUMERICAS + CATEGORICAS + YA_LISTAS])[0]), 2)
    respuesta = {
        "id_prediccion": uuid.uuid4().hex,
        "cod_canton": canton.cod_canton,
        "fecha_objetivo": (canton.fecha + timedelta(days=7)).isoformat(),
        "emergencias_estimadas": estimacion,
        "version_modelo": version_modelo,
    }
    # cod_canton y fecha_objetivo son la llave para unir la prediccion con
    # las emergencias reales, que se conocen al cierre del dia t + 7.
    registrar_prediccion({
        "id_prediccion": respuesta["id_prediccion"],
        "momento": datetime.now().astimezone().isoformat(),
        "version_modelo": version_modelo,
        **canton.model_dump(mode="json"),
        "fecha_objetivo": respuesta["fecha_objetivo"],
        "emergencias_estimadas": estimacion,
    })
    if monitor is not None:
        monitor.registrar(datos.iloc[0].to_dict())
    return respuesta
