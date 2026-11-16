"""Entrena el modelo final del proyecto, mide su error y lo registra como una
version nueva de "emergencias-cantones" en el registro de MLflow.

Desde psets/proyecto/, con el servidor de MLflow levantado:

    python -m servicio.registrar_modelo 5

El numero es max_depth del gradient boosting (el modelo final de la Semana 11
usa 5). Cada vez que lo corres se crea una version nueva, con su corrida y su
metrica: ninguna reemplaza a otra. Para ponerla en produccion, usa
servicio/promover.py.
"""

import hashlib
import logging
import os
import sys

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from modelado.pipeline_proyecto import CORTE, RUTA_DATOS, cargar_y_dividir, construir_preparacion

logging.getLogger("mlflow").setLevel(logging.ERROR)

NOMBRE = "emergencias-cantones"
EXPERIMENTO = "emergencias-cantones-registro"


def huella(ruta):
    with open(ruta, "rb") as archivo:
        return hashlib.md5(archivo.read()).hexdigest()[:12]


def costo_asimetrico(real, estimado):
    """El costo de la Semana 11: quedarse corto cuesta 3 veces mas que pasarse."""
    error = np.asarray(real) - np.asarray(estimado)
    return float(np.mean(np.where(error > 0, 3 * error, -error)))


if __name__ == "__main__":
    max_depth = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050"))
    mlflow.set_experiment(EXPERIMENTO)

    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir()
    modelo = Pipeline([
        ("preparar", construir_preparacion()),
        ("modelo", TransformedTargetRegressor(
            regressor=GradientBoostingRegressor(max_depth=max_depth, random_state=42),
            func=np.log1p, inverse_func=np.expm1)),
    ])

    with mlflow.start_run(run_name=f"gradient boosting max_depth={max_depth}",
                          tags={"mlflow.source.name": "servicio/registrar_modelo.py"}):
        modelo.fit(X_ent, y_ent)
        estimado = modelo.predict(X_prueba)
        mae_prueba = mean_absolute_error(y_prueba, estimado)
        mlflow.log_params({
            "max_depth": max_depth,
            "corte_division": str(CORTE.date()),
            "separacion_dias": 7,
            "huella_datos": huella(RUTA_DATOS),
            "filas_entrenamiento": len(X_ent),
        })
        mlflow.log_metrics({
            "mae_prueba": mae_prueba,
            "mae_entrenamiento": mean_absolute_error(y_ent, modelo.predict(X_ent)),
            "costo_asimetrico": costo_asimetrico(y_prueba, estimado),
        })
        info = mlflow.sklearn.log_model(modelo, "modelo", registered_model_name=NOMBRE)

    print(f"Version {info.registered_model_version} de {NOMBRE}: gradient boosting "
          f"max_depth={max_depth}, MAE de prueba {mae_prueba:.4f}")
