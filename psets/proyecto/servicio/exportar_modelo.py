"""Exporta la version de "emergencias-cantones" con el alias "produccion" a
servicio/modelo.joblib, y su numero a servicio/modelo_version.txt.

Los dos archivos no se suben al repositorio: se regeneran con este script,
con el servidor de MLflow levantado. Desde psets/proyecto/:

    python -m servicio.exportar_modelo

La imagen del servicio lleva los dos archivos adentro, asi que el contenedor
sabe que version sirve sin llegar al servidor de MLflow.
"""

import logging
import os

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

logging.getLogger("mlflow").setLevel(logging.ERROR)

NOMBRE = "emergencias-cantones"
ALIAS = os.environ.get("ALIAS_MODELO", "produccion")
CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(CARPETA, "modelo.joblib")
RUTA_VERSION = os.path.join(CARPETA, "modelo_version.txt")

if __name__ == "__main__":
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050"))
    cliente = MlflowClient()

    version = cliente.get_model_version_by_alias(NOMBRE, ALIAS)
    modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE}/{version.version}")
    joblib.dump(modelo, RUTA_MODELO)
    with open(RUTA_VERSION, "w") as archivo:
        archivo.write(version.version + "\n")

    corrida = cliente.get_run(version.run_id)
    print(f"{NOMBRE}@{ALIAS} = version {version.version} ({corrida.info.run_name}, "
          f"MAE de prueba {corrida.data.metrics['mae_prueba']:.4f})")
    print(f"Guardados {RUTA_MODELO} y {RUTA_VERSION}")
