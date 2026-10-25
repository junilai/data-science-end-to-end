import numpy as np
import pandas as pd

# La misma tabla OBT de emergencias por canton y dia de la Semana 8.
datos = pd.read_csv(
    "../../semana-08-sesion-02/obt_emergencias_canton_dia.csv",
    dtype={"cod_canton": str, "cod_provincia": str},
    parse_dates=["fecha"],
)

# Emergencias del canton dentro de 7 dias.
OBJETIVO = "obj_n_emergencias_t7"

# Igual que en la exploracion: sin objetivo o sin historia de 28 dias la
# fila no sirve, y las zonas no delimitadas no son cantones reales.
datos = datos.dropna(subset=[OBJETIVO, "lag_28d", "media_prev_28d"])
datos = datos[~datos["es_zona_no_delimitada"]].copy()
print("Filas para modelar:", len(datos))

# Feature 1: nivel del canton en escala log. Sale del hallazgo 1 (Quito y
# Guayaquil aplastan la escala original).
datos["log_media_prev_28d"] = np.log1p(datos["media_prev_28d"])

# Feature 2: promedio de las ultimas 3 semanas en el mismo dia de la
# semana. El dia t+7 cae el mismo dia de la semana que t-7, t-14 y t-28.
datos["media_mismo_dia"] = datos[["lag_7d", "lag_14d", "lag_28d"]].mean(axis=1)
datos["log_media_mismo_dia"] = np.log1p(datos["media_mismo_dia"])

# Evidencia: compara el objetivo segun cada feature nueva.
datos["log_objetivo"] = np.log1p(datos[OBJETIVO])

print("\nCorrelacion con el objetivo (escala original):")
print(datos[["media_prev_28d", "log_media_prev_28d", "media_mismo_dia", OBJETIVO]]
      .corr()[OBJETIVO].round(3))

print("\nCorrelacion con log(1 + objetivo):")
print(datos[["log_media_prev_28d", "log_media_mismo_dia", "log_objetivo"]]
      .corr()["log_objetivo"].round(3))

# Error si se predice el objetivo solo con cada feature (sin modelo), en log.
for col in ["log_media_prev_28d", "log_media_mismo_dia"]:
    error = (datos["log_objetivo"] - datos[col]).abs().mean()
    print(f"Error absoluto medio en log usando {col}: {error:.3f}")

# Por que importa el dia de la semana: el objetivo promedio por dia.
print("\nObjetivo promedio por dia de la semana del dia t (= dia de t+7):")
print(datos.groupby("nombre_dia")[OBJETIVO].mean().sort_values().round(2))

# Donde gana la feature 2: el fin de semana, cuando la media de 28 dias
# subestima.
for col in ["log_media_prev_28d", "log_media_mismo_dia"]:
    datos[f"error_{col}"] = (datos["log_objetivo"] - datos[col])
print("\nError promedio con signo (positivo = la feature subestima):")
print(datos.groupby("es_fin_de_semana")[["error_log_media_prev_28d", "error_log_media_mismo_dia"]]
      .mean().round(3))

print(datos[["canton", "fecha", "media_prev_28d", "log_media_prev_28d",
             "media_mismo_dia", OBJETIVO]].head())
