"""Pone el alias "produccion" en una version de "emergencias-cantones".

Desde psets/proyecto/, con el servidor de MLflow levantado:

    python -m servicio.promover 2

Sin numero, solo lista las versiones con su metrica, su alias y sus etiquetas.
Para volver a la version anterior basta promoverla de nuevo: la etiqueta
"anterior_en_produccion" de cada version dice cual era. Despues de mover el
alias, hay que volver a exportar el modelo (python -m servicio.exportar_modelo) y
reconstruir la imagen.
"""

import logging
import os
import sys

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

logging.getLogger("mlflow").setLevel(logging.ERROR)

NOMBRE = "emergencias-cantones"
ALIAS = "produccion"


def version_en_produccion(cliente):
    try:
        return cliente.get_model_version_by_alias(NOMBRE, ALIAS).version
    except MlflowException:
        return None


def listar(cliente):
    for version in sorted(cliente.search_model_versions(f"name='{NOMBRE}'"), key=lambda v: int(v.version)):
        corrida = cliente.get_run(version.run_id)
        detalle = cliente.get_model_version(NOMBRE, version.version)
        alias = ", ".join(detalle.aliases) or "-"
        print(f"Version {version.version}: {corrida.info.run_name}, "
              f"MAE {corrida.data.metrics['mae_prueba']:.4f}, "
              f"costo {corrida.data.metrics['costo_asimetrico']:.4f}, alias: {alias}, "
              f"etiquetas: {detalle.tags or '-'}")


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050"))
    cliente = MlflowClient()

    if len(sys.argv) > 1:
        version = sys.argv[1]
        anterior = version_en_produccion(cliente)
        cliente.set_registered_model_alias(NOMBRE, ALIAS, version)
        # Queda escrito en la version que reemplazo, para poder volver atras.
        cliente.set_model_version_tag(NOMBRE, version, "anterior_en_produccion", anterior or "ninguna")
        print(f"{NOMBRE}@{ALIAS}: version {anterior or 'ninguna'} -> version {version}")

    listar(cliente)
