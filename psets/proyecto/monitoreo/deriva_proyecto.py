"""Simulacion de deriva del proyecto: cuanto se mueve el PSI sin que nada
cambie, y cuanto se mueven el PSI y el costo con una deriva de datos y una
deriva conceptual. Con esos numeros se elige el umbral de la alerta y se
guarda la referencia que usa el servicio (servicio/referencia_deriva.json).

Desde psets/proyecto/, con servicio/modelo.joblib exportado:

    python -m monitoreo.deriva_proyecto

La ventana es una semana de lote: 7 noches x 222 cantones = 1554
predicciones. Una sola noche se mueve demasiado con el dia de la semana y
los feriados (log_lag_1d es el dia anterior: los lunes miran un domingo).
Las ventanas son las semanas completas del periodo de prueba (desde
2025-08-25), y la referencia es el entrenamiento.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

from modelado.pipeline_proyecto import (CATEGORICAS, CORTE, NUMERICAS, OBJETIVO, YA_LISTAS,
                                        agregar_features, cargar_datos)

CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(CARPETA, "..", "servicio", "modelo.joblib")
RUTA_REFERENCIA = os.path.join(CARPETA, "..", "servicio", "referencia_deriva.json")

# Las dos features con mas peso en el gradient boosting (importancia 0.68 y
# 0.30), y log_lag_1d (0.01), el dato mas reciente: el primero que falla si
# la carga diaria no termino antes del lote.
FEATURES = ["log_media_mismo_dia", "log_media_prev_28d", "log_lag_1d"]
VENTANA = 7 * 222
UMBRAL = 0.03      # elegido con la tabla del paso 4 (ver docs/monitoreo.md, seccion Deriva)
MINIMO = 0.0001
LAGS = ["lag_1d", "lag_7d", "lag_14d", "lag_28d", "media_prev_28d"]


def cortes_de(valores):
    """Los 9 cortes interiores de los deciles: 10 casillas, abiertas en los extremos."""
    return [float(c) for c in np.quantile(valores.dropna(), np.linspace(0, 1, 11))[1:-1]]


def proporciones(valores, cortes):
    valores = pd.Series(valores).dropna()
    conteo, _ = np.histogram(valores, bins=[-np.inf] + list(cortes) + [np.inf])
    return conteo / max(len(valores), 1)


def psi(referencia, actual):
    r = np.clip(np.asarray(referencia, dtype=float), MINIMO, None)
    a = np.clip(np.asarray(actual, dtype=float), MINIMO, None)
    return float(np.sum((a - r) * np.log(a / r)))


def costo_asimetrico(real, estimado):
    error = np.asarray(real) - np.asarray(estimado)
    return float(np.mean(np.where(error > 0, 3 * error, -error)))


def medir_semanas(datos, modelo, referencia):
    """PSI de cada feature, MAE y costo de cada semana completa."""
    datos = datos.assign(estimado=modelo.predict(datos[NUMERICAS + CATEGORICAS + YA_LISTAS]),
                         semana=datos["fecha"].dt.to_period("W").dt.start_time)
    filas = []
    for semana, grupo in datos.groupby("semana"):
        if len(grupo) < VENTANA:
            continue  # semana incompleta (la ultima del periodo)
        fila = {"semana": semana.date()}
        for feature in FEATURES:
            fila["psi_" + feature] = psi(referencia[feature]["proporciones"],
                                         proporciones(grupo[feature], referencia[feature]["cortes"]))
        fila["mae"] = float(np.mean(np.abs(grupo[OBJETIVO] - grupo["estimado"])))
        fila["costo"] = costo_asimetrico(grupo[OBJETIVO], grupo["estimado"])
        filas.append(fila)
    return pd.DataFrame(filas)


def carga_incompleta(prueba, fraccion):
    """Deriva de datos que si dana: el lote corre antes de que termine la carga
    del dia t, y lag_1d llega con solo `fraccion` de las emergencias de ayer."""
    datos = prueba.copy()
    datos["lag_1d"] = datos["lag_1d"] * fraccion
    return agregar_features(datos)


def lluvias_en_la_costa(prueba, factor):
    """Deriva de datos con la misma relacion: temporada de lluvias, los
    cantones de la Costa tienen `factor` veces las emergencias, en su historia
    y en su futuro."""
    datos = prueba.copy()
    costa = datos["region"] == "Costa"
    for columna in LAGS + [OBJETIVO]:
        datos.loc[costa, columna] = datos.loc[costa, columna] * factor
    return agregar_features(datos)


def mas_emergencias_en_el_pais(prueba, factor):
    """Deriva de datos con la misma relacion, en todo el pais: todos los
    cantones tienen `factor` veces las emergencias, en su historia y en su futuro."""
    datos = prueba.copy()
    for columna in LAGS + [OBJETIVO]:
        datos[columna] = datos[columna] * factor
    return agregar_features(datos)


def cambio_de_registro(prueba, factor):
    """Deriva conceptual: desde un dia, el sistema registra `factor` veces las
    emergencias (un protocolo nuevo que cuenta como emergencia lo que antes no).
    La historia que ve el modelo es la de antes; el objetivo, el de despues."""
    datos = prueba.copy()
    datos[OBJETIVO] = datos[OBJETIVO] * factor
    return datos


def resumir(nombre, semanas):
    fila = {"escenario": nombre, "semanas": len(semanas)}
    for columna in ["psi_" + feature for feature in FEATURES] + ["mae", "costo"]:
        fila[columna] = round(semanas[columna].mean(), 4)
    fila["alerta"] = round(float(alerta(semanas, UMBRAL).mean()), 3)
    return fila


def alerta(semanas, umbral):
    return (semanas[["psi_" + feature for feature in FEATURES]] > umbral).any(axis=1)


if __name__ == "__main__":
    modelo = joblib.load(RUTA_MODELO)
    datos = cargar_datos()
    entrenamiento = datos[datos["fecha"] < CORTE - pd.Timedelta(days=7)]
    prueba = datos[datos["fecha"] >= CORTE]

    referencia = {"ventana": VENTANA, "umbral": UMBRAL}
    for feature in FEATURES:
        cortes = cortes_de(entrenamiento[feature])
        referencia[feature] = {"cortes": cortes,
                               "proporciones": [float(p) for p in proporciones(entrenamiento[feature], cortes)]}

    # Paso 1: semanas normales.
    normales = medir_semanas(prueba, modelo, referencia)
    print(f"Paso 1. Semanas normales del periodo de prueba ({len(normales)}):")
    print(normales.drop(columns="semana").describe(percentiles=[0.95]).loc[["mean", "95%", "max"]].round(4).to_string())

    # Pasos 2 y 3: las derivas, por intensidad.
    filas = [resumir("normal", normales)]
    for fraccion in [0.9, 0.75, 0.5]:
        filas.append(resumir(f"datos: carga incompleta, lag_1d x{fraccion}",
                             medir_semanas(carga_incompleta(prueba, fraccion), modelo, referencia)))
    for factor in [1.1, 1.3, 1.5]:
        filas.append(resumir(f"datos: lluvias en la Costa x{factor}",
                             medir_semanas(lluvias_en_la_costa(prueba, factor), modelo, referencia)))
    for factor in [1.2, 1.5]:
        filas.append(resumir(f"datos: mas emergencias en el pais x{factor}",
                             medir_semanas(mas_emergencias_en_el_pais(prueba, factor), modelo, referencia)))
    for factor in [1.1, 1.25]:
        filas.append(resumir(f"conceptual: registro x{factor}",
                             medir_semanas(cambio_de_registro(prueba, factor), modelo, referencia)))
    print(f"\nPasos 2 y 3. Promedio por semana, y fraccion de semanas con alerta (umbral {UMBRAL}):")
    print(pd.DataFrame(filas).to_string(index=False))

    # Paso 4: el umbral. Falsas alarmas en semanas normales contra deteccion
    # de una carga incompleta al 75% y de 20% mas emergencias en el pais.
    incompleta = medir_semanas(carga_incompleta(prueba, 0.75), modelo, referencia)
    pais = medir_semanas(mas_emergencias_en_el_pais(prueba, 1.2), modelo, referencia)
    tabla = pd.DataFrame([{"umbral": u, "falsas_alarmas": round(float(alerta(normales, u).mean()), 3),
                           "deteccion_carga_x0.75": round(float(alerta(incompleta, u).mean()), 3),
                           "deteccion_pais_x1.2": round(float(alerta(pais, u).mean()), 3)}
                          for u in [0.02, 0.03, 0.05, 0.10, 0.15]])
    print("\nPaso 4. El umbral:")
    print(tabla.to_string(index=False))

    with open(RUTA_REFERENCIA, "w") as archivo:
        json.dump(referencia, archivo, indent=2)
    print(f"\nReferencia guardada en servicio/referencia_deriva.json (ventana {VENTANA}, umbral {UMBRAL})")
