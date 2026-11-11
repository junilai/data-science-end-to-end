"""Exporta el modelo final de MLflow (Semana 11) a servicio/modelo.joblib.

modelo.joblib no se sube al repositorio: se regenera con este script, con el
servidor de MLflow levantado. Desde psets/proyecto/:

    python servicio/exportar_modelo.py

Por defecto exporta el gradient boosting max_depth=5 marcado como
candidato=final. Para exportar otro run: python servicio/exportar_modelo.py <run_id>
"""

import os
import sys

import joblib
import mlflow

RUN_ID_FINAL = "05dac271ba674e7c967cbc31c5da8989"
RUTA_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo.joblib")

if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else RUN_ID_FINAL
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050"))

    modelo = mlflow.sklearn.load_model(f"runs:/{run_id}/modelo")
    joblib.dump(modelo, RUTA_MODELO)

    corrida = mlflow.get_run(run_id)
    print(f"Run {run_id} ({corrida.info.run_name}), mae_prueba = "
          f"{corrida.data.metrics.get('mae_prueba', float('nan')):.4f}")
    print(f"Modelo guardado en {RUTA_MODELO}")
