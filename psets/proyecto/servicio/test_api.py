"""Pruebas del contrato de servicio/api.py. Desde la raiz del fork:

    pytest psets/proyecto -v

No usan modelo.joblib (en GitHub Actions no existe): reemplazan el modelo por
uno de prueba que siempre estima 12 emergencias.
"""

import pandas as pd
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from modelado.pipeline_proyecto import (CATEGORICAS, NUMERICAS, YA_LISTAS, agregar_calendario,
                                        agregar_features, construir_preparacion)
from servicio import api

CANTON = {
    "fecha": "2026-08-24",
    "region": "Sierra",
    "lag_1d": 9,
    "lag_7d": 8,
    "lag_14d": 10,
    "lag_28d": 12,
    "media_prev_28d": 10.5,
    "es_feriado_t7": False,
}


def modelo_de_prueba():
    datos = agregar_features(agregar_calendario(pd.DataFrame([CANTON, CANTON])))
    modelo = Pipeline([("preparar", construir_preparacion()), ("modelo", DummyRegressor(strategy="mean"))])
    modelo.fit(datos[NUMERICAS + CATEGORICAS + YA_LISTAS], [12.0, 12.0])
    return modelo


api.modelo = modelo_de_prueba()
api.version_modelo = "prueba"
cliente = TestClient(api.app)


def test_salud():
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok", "modelo_cargado": True, "version_modelo": "prueba"}


def test_prediccion_valida():
    respuesta = cliente.post("/predecir", json=CANTON)
    assert respuesta.status_code == 200
    # el objetivo es t + 7
    assert respuesta.json() == {"fecha_objetivo": "2026-08-31", "emergencias_estimadas": 12.0,
                                "version_modelo": "prueba"}


def test_lag_1d_vacio_se_imputa():
    canton = CANTON.copy()
    canton.pop("lag_1d")
    respuesta = cliente.post("/predecir", json=canton)
    assert respuesta.status_code == 200


def test_canton_sin_region_se_rechaza():
    canton = CANTON.copy()
    canton.pop("region")
    respuesta = cliente.post("/predecir", json=canton)
    assert respuesta.status_code == 422
    assert any(error["loc"] == ["body", "region"] for error in respuesta.json()["detail"])


def test_emergencias_negativas_se_rechazan():
    respuesta = cliente.post("/predecir", json={**CANTON, "lag_7d": -3})
    assert respuesta.status_code == 422
    assert any(error["loc"] == ["body", "lag_7d"] for error in respuesta.json()["detail"])
