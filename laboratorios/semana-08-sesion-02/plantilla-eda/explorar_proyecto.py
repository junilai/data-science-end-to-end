import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("figuras", exist_ok=True)

# Tabla OBT del warehouse (emergencias por canton y dia) exportada a CSV.
datos = pd.read_csv(
    "../obt_emergencias_canton_dia.csv",
    dtype={"cod_canton": str, "cod_provincia": str},
    parse_dates=["fecha"],
)

# El modelo predice cuantas emergencias tendra un canton dentro de 7 dias.
OBJETIVO = "obj_n_emergencias_t7"

# Los ultimos 7 dias de cada canton no tienen objetivo (aun no ocurren):
# no sirven para entrenar.
print("Filas sin objetivo:", datos[OBJETIVO].isna().sum())
datos = datos.dropna(subset=[OBJETIVO])

# Paso 1: resumen.
print("Filas y columnas:", datos.shape)
print(datos.dtypes)
print(datos.isna().sum())
print(datos.describe().round(2))

# Paso 2: distribucion de la variable objetivo.
plt.figure()
plt.hist(datos[OBJETIVO], bins=40)
plt.title(OBJETIVO)
plt.savefig("figuras/distribucion_objetivo.png")
plt.close()

# La escala original queda aplastada por Quito y Guayaquil; en log se ve la forma.
plt.figure()
plt.hist(np.log1p(datos[OBJETIVO]), bins=40)
plt.title(f"log(1 + {OBJETIVO})")
plt.savefig("figuras/distribucion_objetivo_log.png")
plt.close()

# Paso 3: atipicos de la variable objetivo con la regla del IQR.
q1 = datos[OBJETIVO].quantile(0.25)
q3 = datos[OBJETIVO].quantile(0.75)
iqr = q3 - q1
fuera = datos[(datos[OBJETIVO] < q1 - 1.5 * iqr) | (datos[OBJETIVO] > q3 + 1.5 * iqr)]
print(f"Atipicos de {OBJETIVO} segun IQR: {len(fuera)}")
print(fuera.head(10))

# Paso 4: relacion entre el objetivo y lag_7d (emergencias del mismo dia
# de la semana anterior, conocidas al momento de predecir).
sub = datos.dropna(subset=["lag_7d"]).sample(20000, random_state=42)
plt.figure()
plt.scatter(sub["lag_7d"] + 1, sub[OBJETIVO] + 1, s=4, alpha=0.3)
plt.xscale("log")
plt.yscale("log")
plt.title("lag_7d vs objetivo (escala log, muestra de 20000)")
plt.xlabel("lag_7d + 1")
plt.ylabel(f"{OBJETIVO} + 1")
plt.savefig("figuras/lag7d_vs_objetivo.png")
plt.close()

por_dia = datos.groupby("nombre_dia")["n_emergencias"].mean().sort_values()
plt.figure()
plt.bar(por_dia.index, por_dia.values)
plt.title("Emergencias promedio por dia de la semana")
plt.savefig("figuras/emergencias_por_dia.png")
plt.close()

# Paso 5: consultas que confirman los hallazgos.

# Hallazgo 1: los atipicos del IQR son cantones grandes, no errores.
print("\nCantones con mas atipicos segun IQR:")
print(fuera["canton"].value_counts().head(10))
print("Porcentaje de las emergencias que aportan Quito y Guayaquil:",
      round(datos[datos["canton"].isin(["QUITO", "GUAYAQUIL"])][OBJETIVO].sum()
            / datos[OBJETIVO].sum() * 100, 1))
print("Promedio del objetivo por region:")
print(datos.groupby("region")[OBJETIVO].agg(["size", "mean", "median"]).round(2))

# Hallazgo 2: enero de 2024 esta marcado como periodo anomalo y hay
# "zonas no delimitadas" que no son cantones reales.
print("\nPeriodo anomalo:")
print(datos.groupby("es_periodo_anomalo")["fecha"].agg(["min", "max", "size"]))
total_dia = datos.groupby("fecha")["n_emergencias"].sum()
print("Total diario promedio dic-2023, ene-2024, feb-2024:")
print(total_dia.groupby(total_dia.index.to_period("M")).mean().loc["2023-12":"2024-02"].round(0))
print("Zonas no delimitadas:")
print(datos.groupby("es_zona_no_delimitada")[OBJETIVO].agg(["size", "mean"]).round(2))

# Hallazgo 3: las columnas n_* del mismo dia suman exactamente
# n_emergencias, y los lags tienen nulos al inicio de cada serie.
tipos = [c for c in datos.columns if c.startswith("n_") and c != "n_emergencias"]
print("\nFilas donde la suma de categorias == n_emergencias:",
      (datos[tipos].sum(axis=1) == datos["n_emergencias"]).mean())
print("Correlacion con el objetivo:")
vars_lag = ["n_emergencias", "lag_1d", "lag_7d", "lag_14d", "lag_28d",
            "media_prev_7d", "media_prev_28d"]
print(datos[vars_lag].corrwith(datos[OBJETIVO]).round(3))
print("Correlacion en escala log:")
print(np.log1p(datos[vars_lag]).corrwith(np.log1p(datos[OBJETIVO])).round(3))
print("Nulos en lags y medias moviles:")
print(datos[vars_lag[1:]].isna().sum())
print("Fecha del primer lag_28d disponible:",
      datos.dropna(subset=["lag_28d"])["fecha"].min())
