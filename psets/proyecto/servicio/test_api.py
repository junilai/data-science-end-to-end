"""Pruebas del contrato de servicio/api.py. Desde la raiz del fork:

    pytest psets/proyecto -v

No usan modelo.joblib ni el registro de MLflow (en GitHub Actions no existen):
reemplazan el modelo por uno de prueba que siempre estima 12 emergencias, con
una version inventada, "7". Cada prueba escribe su registro de predicciones
en una carpeta temporal (tmp_path) y empieza con las metricas en cero y con
un monitor de deriva vacio, de ventana 4 y con una referencia inventada.
"""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from modelado.pipeline_proyecto import (CATEGORICAS, NUMERICAS, YA_LISTAS, agregar_calendario,
                                        agregar_features, construir_preparacion)
from servicio import api
from servicio.deriva import MonitorDeDeriva

CANTON = {
    "cod_canton": "0501",
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


# Una referencia inventada: la mitad de las predicciones por debajo de log 2
# (unas 6 emergencias) y la mitad por encima, en las dos features.
REFERENCIA = {
    "ventana": 4,
    "umbral": 0.15,
    "log_lag_1d": {"cortes": [2.0], "proporciones": [0.5, 0.5]},
    "log_media_prev_28d": {"cortes": [2.0], "proporciones": [0.5, 0.5]},
}
# Un canton chico: log1p(1) = 0.69, por debajo del corte. CANTON esta por encima.
CANTON_CHICO = {**CANTON, "lag_1d": 1, "lag_7d": 1, "lag_14d": 1, "lag_28d": 1, "media_prev_28d": 1}


class ModeloRoto:
    """Un modelo que falla al predecir, para provocar un error de servidor."""

    def predict(self, X):
        raise RuntimeError("modelo roto")


@pytest.fixture
def cliente(tmp_path):
    api.modelo = modelo_de_prueba()
    api.version_modelo = "7"
    api.RUTA_REGISTRO = str(tmp_path / "predicciones.jsonl")
    api.metricas = api.Metricas()
    api.monitor = MonitorDeDeriva(REFERENCIA)
    # raise_server_exceptions=False: un error dentro de la ruta llega como
    # respuesta 500, igual que con uvicorn, en vez de romper la prueba.
    return TestClient(api.app, raise_server_exceptions=False)


def leer_registro():
    with open(api.RUTA_REGISTRO) as archivo:
        return [json.loads(linea) for linea in archivo]


def test_salud(cliente):
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok", "modelo_cargado": True, "version_modelo": "7"}


def test_prediccion_valida(cliente):
    respuesta = cliente.post("/predecir", json=CANTON)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    # el objetivo es t + 7
    assert cuerpo["fecha_objetivo"] == "2026-08-31"
    assert cuerpo["emergencias_estimadas"] == 12.0
    assert cuerpo["cod_canton"] == "0501"
    assert len(cuerpo["id_prediccion"]) == 32


def test_cada_prediccion_dice_que_version_la_calculo(cliente):
    # dos cantones distintos, la misma version: la que sirve el servicio
    for canton in (CANTON, {**CANTON, "cod_canton": "0901", "region": "Costa", "fecha": "2026-08-29"}):
        respuesta = cliente.post("/predecir", json=canton)
        assert respuesta.status_code == 200
        assert respuesta.json()["version_modelo"] == "7"


def test_cada_prediccion_queda_registrada(cliente):
    respuesta = cliente.post("/predecir", json=CANTON).json()
    lineas = leer_registro()
    assert len(lineas) == 1
    registro = lineas[0]
    assert registro["id_prediccion"] == respuesta["id_prediccion"]
    assert registro["version_modelo"] == "7"
    # la entrada completa, la salida, y la llave para unirla con la verdad
    assert registro["lag_7d"] == 8 and registro["region"] == "Sierra"
    assert registro["emergencias_estimadas"] == 12.0
    assert (registro["cod_canton"], registro["fecha_objetivo"]) == ("0501", "2026-08-31")
    assert "momento" in registro


def test_una_entrada_invalida_no_queda_registrada(cliente):
    cliente.post("/predecir", json={**CANTON, "lag_7d": -3})
    cliente.post("/predecir", json=CANTON)
    assert len(leer_registro()) == 1


def test_metricas_separan_errores_de_validacion_y_de_servidor(cliente):
    cliente.post("/predecir", json=CANTON)
    cliente.post("/predecir", json={**CANTON, "lag_7d": -3})
    sin_region = {clave: valor for clave, valor in CANTON.items() if clave != "region"}
    cliente.post("/predecir", json=sin_region)
    api.modelo = ModeloRoto()
    assert cliente.post("/predecir", json=CANTON).status_code == 500
    cliente.get("/salud")  # no es una prediccion: no cuenta

    resumen = cliente.get("/metricas").json()
    assert resumen["peticiones"] == 4
    assert resumen["errores_validacion"] == 2
    assert resumen["errores_servidor"] == 1
    assert 0 < resumen["latencia_p50_ms"] <= resumen["latencia_p95_ms"]


def test_metricas_sin_peticiones(cliente):
    assert cliente.get("/metricas").json() == {"peticiones": 0, "errores_validacion": 0, "errores_servidor": 0,
                                               "latencia_p50_ms": None, "latencia_p95_ms": None}


def test_lag_1d_vacio_se_imputa(cliente):
    canton = CANTON.copy()
    canton.pop("lag_1d")
    respuesta = cliente.post("/predecir", json=canton)
    assert respuesta.status_code == 200


def test_canton_sin_region_se_rechaza(cliente):
    canton = CANTON.copy()
    canton.pop("region")
    respuesta = cliente.post("/predecir", json=canton)
    assert respuesta.status_code == 422
    assert any(error["loc"] == ["body", "region"] for error in respuesta.json()["detail"])


def test_canton_sin_codigo_se_rechaza(cliente):
    canton = CANTON.copy()
    canton.pop("cod_canton")
    respuesta = cliente.post("/predecir", json=canton)
    assert respuesta.status_code == 422
    assert any(error["loc"] == ["body", "cod_canton"] for error in respuesta.json()["detail"])


def test_emergencias_negativas_se_rechazan(cliente):
    respuesta = cliente.post("/predecir", json={**CANTON, "lag_7d": -3})
    assert respuesta.status_code == 422
    assert any(error["loc"] == ["body", "lag_7d"] for error in respuesta.json()["detail"])


def test_deriva_sin_alerta_con_una_ventana_como_la_referencia(cliente):
    for canton in (CANTON, CANTON_CHICO, CANTON, CANTON_CHICO):
        cliente.post("/predecir", json=canton)
    estado = cliente.get("/deriva").json()
    assert estado["predicciones_en_ventana"] == 4
    assert estado["psi"] == {"log_lag_1d": 0.0, "log_media_prev_28d": 0.0}
    assert estado["alerta"] is False


def test_deriva_alerta_con_una_ventana_alejada_de_la_referencia(cliente):
    # cuatro cantones grandes: todas las predicciones caen en la casilla de arriba
    for _ in range(4):
        cliente.post("/predecir", json=CANTON)
    estado = cliente.get("/deriva").json()
    assert estado["psi"]["log_lag_1d"] > REFERENCIA["umbral"]
    assert estado["alerta"] is True


def test_deriva_no_alerta_con_la_ventana_incompleta(cliente):
    for _ in range(3):
        cliente.post("/predecir", json=CANTON)
    estado = cliente.get("/deriva").json()
    assert estado["predicciones_en_ventana"] == 3
    assert estado["psi"]["log_lag_1d"] > REFERENCIA["umbral"]
    assert estado["alerta"] is False


def test_deriva_guarda_solo_la_ultima_ventana(cliente):
    # cuatro grandes (alerta) y despues cuatro como la referencia: la alerta se apaga
    for canton in [CANTON] * 4 + [CANTON, CANTON_CHICO, CANTON, CANTON_CHICO]:
        cliente.post("/predecir", json=canton)
    estado = cliente.get("/deriva").json()
    assert estado["predicciones_en_ventana"] == 4
    assert estado["alerta"] is False


def test_un_lag_1d_vacio_no_cuenta_en_la_deriva(cliente):
    sin_lag = {clave: valor for clave, valor in CANTON_CHICO.items() if clave != "lag_1d"}
    for canton in (CANTON, sin_lag):
        cliente.post("/predecir", json=canton)
    estado = cliente.get("/deriva").json()
    # lag_1d: solo cuenta CANTON (todo arriba); media_prev_28d: uno y uno
    assert estado["psi"]["log_lag_1d"] > 0
    assert estado["psi"]["log_media_prev_28d"] == 0.0


def test_deriva_sin_referencia(cliente):
    api.monitor = None
    cliente.post("/predecir", json=CANTON)
    assert cliente.get("/deriva").json() == {"referencia_cargada": False}
