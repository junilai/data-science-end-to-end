"""Simula el lote de cada noche contra el servicio: para cada dia del rango,
manda los cantones de ese dia a /predecir, como hara el lote en produccion.
Cada prediccion queda en monitoreo/predicciones.jsonl (que se borra antes de
empezar).

Desde psets/proyecto/:

    python -m monitoreo.simular_lote 2026-07-27 2026-08-23
    python -m monitoreo.simular_lote 2026-07-27 2026-08-09:1 2026-08-23:2

Con un solo tramo usa el modelo exportado (servicio/modelo.joblib). Con
tramos "hasta:version", cada tramo usa esa version del registro de MLflow
(necesita el servidor levantado): asi se simula un cambio de version a
mitad del periodo. Usa TestClient: las peticiones pasan por el servicio
completo, middleware incluido, sin levantar uvicorn.
"""

import logging
import os
import sys

import pandas as pd
from fastapi.testclient import TestClient

from modelado.pipeline_proyecto import cargar_datos
from servicio import api

logging.getLogger("mlflow").setLevel(logging.ERROR)

CAMPOS = ["cod_canton", "fecha", "region", "lag_1d", "lag_7d", "lag_14d", "lag_28d",
          "media_prev_28d", "es_feriado_t7"]


def peticion(fila):
    datos = {campo: fila[campo] for campo in CAMPOS}
    datos["fecha"] = fila["fecha"].date().isoformat()
    datos["es_feriado_t7"] = bool(datos["es_feriado_t7"])
    return {clave: valor for clave, valor in datos.items() if not pd.isna(valor)}


def cargar_version(version):
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050"))
    return mlflow.sklearn.load_model(f"models:/emergencias-cantones/{version}")


if __name__ == "__main__":
    desde = pd.Timestamp(sys.argv[1])
    tramos = [tramo.split(":") for tramo in sys.argv[2:]]

    if os.path.exists(api.RUTA_REGISTRO):
        os.remove(api.RUTA_REGISTRO)
    datos = cargar_datos()
    cliente = TestClient(api.app)

    inicio = desde
    for tramo in tramos:
        hasta = pd.Timestamp(tramo[0])
        if len(tramo) == 2:
            api.modelo, api.version_modelo = cargar_version(tramo[1]), tramo[1]
        dias = datos[datos["fecha"].between(inicio, hasta)].sort_values(["fecha", "cod_canton"])
        for fila in dias.to_dict("records"):
            cliente.post("/predecir", json=peticion(fila))
        print(f"{inicio.date()} a {hasta.date()}: {len(dias)} predicciones con la version {api.version_modelo}")
        inicio = hasta + pd.Timedelta(days=1)

    # Dos peticiones mal formadas, como las de un cliente desactualizado.
    cliente.post("/predecir", json={"fecha": "2026-08-24", "region": "Sierra"})
    cliente.post("/predecir", json={**peticion(dias.iloc[0]), "lag_7d": -1})

    print("Metricas del sistema:", cliente.get("/metricas").json())
