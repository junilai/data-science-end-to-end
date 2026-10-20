import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figuras", exist_ok=True)
viajes = pd.read_csv("../starter/viajes_taxi.csv")

# TODO 1: quita solo los errores que el Ejercicio 1 ya confirmo (tarifa
# 999.99 y duracion negativa). No quites los demas atipicos. Imprime
# cuantas filas quedan.
limpios = viajes[(viajes["tarifa"] != 999.99) & (viajes["duracion_min"] >= 0)]
print("Filas originales:", len(viajes))
print("Filas tras quitar errores confirmados:", len(limpios))

# TODO 2: grafico de dispersion de distancia_km (eje x) contra tarifa
# (eje y), guardado en figuras/distancia_vs_tarifa.png. Mira el grafico
# antes de seguir: busca puntos que no sigan la tendencia general y
# cuentalos con un filtro.
plt.figure()
plt.scatter(limpios["distancia_km"], limpios["tarifa"], s=8, alpha=0.5)
plt.title("Distancia vs tarifa")
plt.xlabel("distancia_km")
plt.ylabel("tarifa (USD)")
plt.savefig("figuras/distancia_vs_tarifa.png")
plt.close()

# En el grafico aparece una linea horizontal en tarifa = 70: la tarifa
# fija de los viajes entre Manhattan y el aeropuerto JFK.
viajes_70 = limpios[limpios["tarifa"] == 70]
print("\nViajes con tarifa exactamente 70:", len(viajes_70))
print(viajes_70.groupby(["zona_origen", "zona_destino"]).size())

aeropuerto = limpios[(limpios["zona_origen"] == "JFK Airport") | (limpios["zona_destino"] == "JFK Airport")]
print("Viajes que tocan JFK:", len(aeropuerto))
print("Tarifa de los viajes JFK que no pagan 70:")
print(aeropuerto[aeropuerto["tarifa"] != 70].groupby(["zona_origen", "zona_destino"])["tarifa"].agg(["count", "mean"]).round(2))

# TODO 3: propina promedio y cantidad de viajes con propina 0, por
# metodo de pago (groupby + agg).
propinas = limpios.groupby("metodo_pago", dropna=False).agg(
    viajes=("propina", "size"),
    propina_promedio=("propina", "mean"),
    propina_cero=("propina", lambda x: (x == 0).sum()),
)
print("\nPropina por metodo de pago:")
print(propinas.round(2))

# TODO 4: correlacion de distancia_km, duracion_min, pasajeros y propina
# con la tarifa.
cols = ["distancia_km", "duracion_min", "pasajeros", "propina"]
print("\nCorrelacion con la tarifa:")
print(limpios[cols].corrwith(limpios["tarifa"]).round(3))

# TODO 5: cuantos viajes tienen pasajeros vacio y cuantos tienen 0.
print("\nViajes con pasajeros vacio:", limpios["pasajeros"].isna().sum())
print("Viajes con pasajeros = 0:", (limpios["pasajeros"] == 0).sum())
