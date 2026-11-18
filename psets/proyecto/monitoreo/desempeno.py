"""El desempeno del modelo en produccion: une cada prediccion del registro
(monitoreo/predicciones.jsonl) con las emergencias reales del canton en la
fecha objetivo, y calcula las metricas de la Semana 11 por semana y version.

Desde psets/proyecto/:

    python -m monitoreo.desempeno 2026-09-01

La fecha es "hoy": solo se conoce la verdad de las fechas objetivo
anteriores a hoy (las emergencias de un dia se cargan al cierre de ese dia).
Sin fecha, usa la de hoy.

La verdad tarda 7 dias en llegar (el objetivo es t + 7). Mientras tanto, el
indicador temprano compara lo que estima el modelo con la media de 28 dias
que recibio: se puede calcular la misma noche, sin esperar la verdad.
"""

import os
import sys
from datetime import date

import numpy as np
import pandas as pd

from modelado.pipeline_proyecto import RUTA_DATOS

CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_REGISTRO = os.environ.get("RUTA_REGISTRO", os.path.join(CARPETA, "predicciones.jsonl"))

# Referencias del modelo en produccion (version 2) en el periodo de prueba
# (desde 2025-08-25): con ellas se aprobo, y contra ellas se alerta.
MAE_REFERENCIA = 5.549
COSTO_REFERENCIA = 11.767
# Mediana diaria de emergencias_estimadas / media_prev_28d, por dia de la semana
# (0 = lunes), en los dias de prueba sin feriado en t + 7. En prueba, ningun
# dia normal se alejo mas de 5% de su referencia.
RAZON_REFERENCIA = {0: 0.847, 1: 0.737, 2: 0.753, 3: 0.78, 4: 0.867, 5: 1.03, 6: 1.311}
TOLERANCIA = 0.20          # alerta si una metrica se aleja mas de 20% de su referencia
CANTONES = 222             # predicciones que deja el lote completo de una noche


def costo_asimetrico(real, estimado):
    """El costo de la Semana 11: quedarse corto cuesta 3 veces mas que pasarse."""
    error = np.asarray(real) - np.asarray(estimado)
    return float(np.mean(np.where(error > 0, 3 * error, -error)))


def leer_registro(ruta=RUTA_REGISTRO):
    registro = pd.read_json(ruta, lines=True, dtype={"cod_canton": str, "version_modelo": str})
    registro["fecha"] = pd.to_datetime(registro["fecha"])
    registro["fecha_objetivo"] = pd.to_datetime(registro["fecha_objetivo"])
    registro["semana"] = registro["fecha"].dt.to_period("W").dt.start_time.dt.date
    return registro


def leer_verdad(ruta=RUTA_DATOS):
    """Las emergencias reales de cada canton y dia."""
    verdad = pd.read_csv(ruta, usecols=["cod_canton", "fecha", "n_emergencias"],
                         dtype={"cod_canton": str}, parse_dates=["fecha"])
    return verdad.rename(columns={"fecha": "fecha_objetivo", "n_emergencias": "emergencias_reales"})


def unir_con_verdad(registro, verdad, hoy):
    """Las predicciones cuya verdad ya se conoce hoy, con sus emergencias reales."""
    conocidas = registro[registro["fecha_objetivo"] < pd.Timestamp(hoy)]
    return conocidas.merge(verdad, on=["cod_canton", "fecha_objetivo"], how="inner")


def desempeno_por_semana(unidas):
    def resumir(grupo):
        real, estimado = grupo["emergencias_reales"], grupo["emergencias_estimadas"]
        return pd.Series({
            "predicciones": len(grupo),
            "mae": round(float(np.mean(np.abs(real - estimado))), 4),
            "costo_asimetrico": round(costo_asimetrico(real, estimado), 4),
            "sesgo": round(float(np.mean(estimado - real)), 4),
        })

    tabla = unidas.groupby(["semana", "version_modelo"]).apply(resumir, include_groups=False)
    tabla["predicciones"] = tabla["predicciones"].astype(int)
    tabla["mae_vs_referencia"] = (tabla["mae"] / MAE_REFERENCIA).round(2)
    return tabla


def indicador_temprano(registro):
    """Por dia y version: la mediana de emergencias_estimadas / media_prev_28d,
    dividida por la de prueba para ese dia de la semana (1 = como en prueba).
    Se conoce la misma noche. Si se aleja de 1, el modelo responde distinto a
    como respondia en prueba ante la misma historia. Los dias con feriado en
    t + 7 se marcan aparte: ahi el modelo sube a proposito."""
    con_media = registro[registro["media_prev_28d"] > 0].copy()
    con_media["razon"] = con_media["emergencias_estimadas"] / con_media["media_prev_28d"]
    con_media["dia"] = con_media["fecha"].dt.date
    indicador = con_media.groupby(["dia", "version_modelo"]).agg(
        razon=("razon", "median"), feriado=("es_feriado_t7", "max"))
    dia_semana = pd.to_datetime(indicador.index.get_level_values("dia")).dayofweek
    indicador["relativa"] = (indicador["razon"] / dia_semana.map(RAZON_REFERENCIA).values).round(3)
    return indicador


def alertas(registro, tabla, indicador):
    mensajes = []
    for dia, cantidad in registro.groupby(registro["fecha"].dt.date).size().items():
        if cantidad < CANTONES:
            mensajes.append(f"sistema: el lote del {dia} dejo {cantidad} de {CANTONES} predicciones")
    for (dia, version), fila in indicador.iterrows():
        if not fila["feriado"] and abs(fila["relativa"] - 1) > TOLERANCIA:
            mensajes.append(f"modelo (temprana): el {dia} la version {version} estimo {fila['relativa']} "
                            "veces lo esperado para su historia")
    for (semana, version), fila in tabla.iterrows():
        if fila["costo_asimetrico"] > (1 + TOLERANCIA) * COSTO_REFERENCIA:
            mensajes.append(f"modelo: la semana del {semana} (version {version}) tuvo costo {fila['costo_asimetrico']}")
    return mensajes


if __name__ == "__main__":
    hoy = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    if not os.path.exists(RUTA_REGISTRO):
        sys.exit("No hay monitoreo/predicciones.jsonl: el servicio todavia no guardo predicciones.")

    registro = leer_registro()
    unidas = unir_con_verdad(registro, leer_verdad(), hoy)
    print(f"{len(registro)} predicciones registradas; {len(unidas)} con la verdad conocida al {hoy}")
    print(f"Sin verdad todavia: {len(registro) - len(unidas)} (fecha objetivo desde {hoy})")

    print(f"\nDesempeno por semana y version (referencia: MAE {MAE_REFERENCIA}, costo {COSTO_REFERENCIA}):")
    tabla = desempeno_por_semana(unidas)
    print(tabla.to_string())

    indicador = indicador_temprano(registro)
    print("\nIndicador temprano por semana (estimada / media_prev_28d, relativa a prueba; 1 = igual):")
    por_semana = indicador[~indicador["feriado"]].reset_index()
    por_semana["semana"] = pd.to_datetime(por_semana["dia"]).dt.to_period("W").dt.start_time.dt.date
    print(por_semana.groupby(["semana", "version_modelo"])["relativa"].agg(["min", "max"]).to_string())

    print("\nAlertas:")
    for mensaje in alertas(registro, tabla, indicador) or ["ninguna"]:
        print("-", mensaje)
