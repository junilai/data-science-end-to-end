"""Pruebas de pipeline_proyecto.py. Desde la raiz del fork:

    pytest psets/proyecto -v

No leen data/ (en GitHub Actions ese archivo no existe): cada prueba arma sus
propias filas a mano y, si necesita un archivo, lo escribe en una carpeta
temporal de pytest.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline_proyecto import (CATEGORICAS, CORTE, NUMERICAS, OBJETIVO, YA_LISTAS,
                               agregar_features, cargar_datos, cargar_y_dividir,
                               construir_preparacion)


def filas_crudas(fechas, **cambios):
    """Una fila por fecha con todas las columnas que lee cargar_datos, de un
    solo canton. cambios reemplaza columnas enteras."""
    n = len(fechas)
    datos = pd.DataFrame({
        "fecha": pd.to_datetime(fechas),
        "cod_canton": ["0101"] * n,
        "cod_provincia": ["01"] * n,
        "region": ["Sierra"] * n,
        "nombre_dia": ["Lunes"] * n,
        "es_fin_de_semana": [False] * n,
        "es_feriado_t7": [False] * n,
        "es_zona_no_delimitada": [False] * n,
        "n_emergencias": [10.0] * n,
        "lag_1d": [9.0] * n,
        "lag_7d": [8.0] * n,
        "lag_14d": [10.0] * n,
        "lag_28d": [12.0] * n,
        "media_prev_28d": [10.0] * n,
        OBJETIVO: [11.0] * n,
    })
    for columna, valores in cambios.items():
        datos[columna] = valores
    return datos


def escribir_csv(datos, tmp_path):
    ruta = tmp_path / "obt_prueba.csv"
    datos.to_csv(ruta, index=False)
    return ruta


def test_features_calculadas_a_mano():
    datos = agregar_features(filas_crudas(["2025-03-03"]))
    fila = datos.iloc[0]
    # media del mismo dia de la semana en las 3 semanas previas: (8 + 10 + 12) / 3
    assert fila["media_mismo_dia"] == pytest.approx(10.0)
    assert fila["log_media_mismo_dia"] == pytest.approx(np.log(11.0))
    assert fila["log_media_prev_28d"] == pytest.approx(np.log(11.0))
    assert fila["log_lag_1d"] == pytest.approx(np.log(10.0))
    # los booleanos llegan al modelo como 0/1
    assert fila["es_fin_de_semana"] == 0


def test_objetivo_no_entra_al_modelo(tmp_path):
    fechas = pd.date_range("2025-07-01", "2025-09-30", freq="D")
    ruta = escribir_csv(filas_crudas(fechas), tmp_path)
    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir(ruta)
    for X in (X_ent, X_prueba):
        assert OBJETIVO not in X.columns
        # n_emergencias del dia tampoco entra, salvo que se pida como columna extra
        assert "n_emergencias" not in X.columns
        assert list(X.columns) == NUMERICAS + CATEGORICAS + YA_LISTAS


def test_division_temporal_con_separacion_de_7_dias(tmp_path):
    fechas = pd.date_range("2025-07-01", "2025-09-30", freq="D")
    datos = filas_crudas(fechas)
    ruta = escribir_csv(datos, tmp_path)
    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir(ruta)
    fechas_ent = datos.loc[X_ent.index, "fecha"]
    fechas_prueba = datos.loc[X_prueba.index, "fecha"]
    # ninguna fila de entrenamiento mira un objetivo (t+7) que cae en la prueba
    assert (fechas_ent + pd.Timedelta(days=7)).max() < CORTE
    assert fechas_prueba.min() == CORTE
    # los 7 dias antes del corte no estan en ningun lado
    assert len(X_ent) + len(X_prueba) == len(datos) - 7


def test_cargar_datos_quita_filas_que_no_sirven(tmp_path):
    fechas = ["2023-12-20", "2023-12-26", "2024-01-15", "2024-02-10", "2024-02-11", "2024-02-12"]
    datos = filas_crudas(
        fechas,
        # 2023-12-26 + 7 cae en enero de 2024 (objetivo deprimido) y 2024-01-15 tambien
        es_zona_no_delimitada=[False, False, False, True, False, False],
        lag_28d=[12.0, 12.0, 12.0, 12.0, np.nan, 12.0],
    )
    datos.loc[5, OBJETIVO] = np.nan
    limpios = cargar_datos(escribir_csv(datos, tmp_path))
    # solo sobrevive 2023-12-20: las demas son enero 2024, zona no delimitada,
    # sin lag_28d o sin objetivo
    assert list(limpios["fecha"]) == [pd.Timestamp("2023-12-20")]


def test_preparacion_acepta_una_categoria_nueva():
    entrenamiento = agregar_features(filas_crudas(["2025-03-03", "2025-03-04"],
                                                  region=["Sierra", "Costa"],
                                                  nombre_dia=["Lunes", "Martes"]))
    nueva = agregar_features(filas_crudas(["2025-03-05"], region=["Galapagos"],
                                          nombre_dia=["Miércoles"]))
    columnas = NUMERICAS + CATEGORICAS + YA_LISTAS
    preparacion = construir_preparacion().fit(entrenamiento[columnas])
    resultado = preparacion.transform(nueva[columnas])
    nombres = list(preparacion.get_feature_names_out())
    one_hot = [i for i, nombre in enumerate(nombres) if nombre.startswith("categoricas__")]
    # no falla, y la categoria desconocida queda en ceros
    assert resultado.shape == (1, len(nombres))
    assert (resultado[0, one_hot] == 0).all()
