import pandas as pd

viajes = pd.read_csv("../starter/viajes_taxi.csv")
viajes = viajes[(viajes["tarifa"] != 999.99) & (viajes["duracion_min"] >= 0)].copy()

ZONAS_MANHATTAN = ["Midtown", "Upper East Side", "Upper West Side", "Lower Manhattan", "Harlem"]

# TODO 1: crea la columna es_tarifa_fija (1 o 0): vale 1 cuando el viaje
# va de JFK Airport a una zona de Manhattan, o de una zona de Manhattan a
# JFK Airport. Imprime cuantos viajes la tienen en 1, y la tarifa minima
# y maxima de esos viajes.
desde_jfk = (viajes["zona_origen"] == "JFK Airport") & viajes["zona_destino"].isin(ZONAS_MANHATTAN)
hacia_jfk = viajes["zona_origen"].isin(ZONAS_MANHATTAN) & (viajes["zona_destino"] == "JFK Airport")
viajes["es_tarifa_fija"] = (desde_jfk | hacia_jfk).astype(int)

fijos = viajes[viajes["es_tarifa_fija"] == 1]
print("Viajes con es_tarifa_fija = 1:", len(fijos))
print("Tarifa minima:", fijos["tarifa"].min(), "| tarifa maxima:", fijos["tarifa"].max())
print("Viajes con tarifa 70 fuera de la regla:",
      ((viajes["tarifa"] == 70) & (viajes["es_tarifa_fija"] == 0)).sum())

# TODO 2: codifica zona_origen con one-hot (pd.get_dummies), con el
# prefijo "origen" y dtype=int. Imprime los nombres de las columnas
# nuevas.
dummies = pd.get_dummies(viajes["zona_origen"], prefix="origen", dtype=int)
viajes = pd.concat([viajes, dummies], axis=1)
print("\nColumnas one-hot:", len(dummies.columns))
print(list(dummies.columns))

# TODO 3: codifica zona_origen de forma ordinal: un numero entero por
# zona, en orden alfabetico, en una columna origen_ordinal.
zonas = sorted(viajes["zona_origen"].unique())
mapa = {zona: i for i, zona in enumerate(zonas)}
viajes["origen_ordinal"] = viajes["zona_origen"].map(mapa)
print("\nColumnas ordinal: 1 (origen_ordinal)")
print(mapa)

# TODO 4: para cada columna original del dataset, decide si existe en el
# momento en que alguien pide el taxi (antes del viaje) e imprime la
# decision.
disponible = {
    "id_viaje": "si (pero es un identificador, no aporta)",
    "fecha_hora_inicio": "si",
    "zona_origen": "si",
    "zona_destino": "si",
    "distancia_km": "si (como distancia estimada de la ruta)",
    "duracion_min": "no, se conoce al terminar el viaje",
    "pasajeros": "si",
    "metodo_pago": "no siempre, se confirma al pagar",
    "tarifa": "no, es el objetivo",
    "propina": "no, se da al terminar el viaje",
}
print()
for col in pd.read_csv("../starter/viajes_taxi.csv", nrows=0).columns:
    print(f"{col}: {disponible[col]}")
