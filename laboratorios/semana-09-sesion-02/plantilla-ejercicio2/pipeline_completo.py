import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

viajes = pd.read_csv("../starter/viajes_features.csv")

NUMERICAS = ["distancia_km", "pasajeros"]
YA_LISTAS = ["hora_sin", "hora_cos", "es_fin_de_semana", "es_hora_pico", "es_tarifa_fija"]
CATEGORICAS = ["zona_origen", "zona_destino", "metodo_pago"]

X = viajes[NUMERICAS + YA_LISTAS + CATEGORICAS]
y = viajes["tarifa"]

# TODO 1: divide X e y en entrenamiento (80%) y prueba (20%), con
# random_state=42.
X_ent, X_prueba, y_ent, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)
print("Entrenamiento:", X_ent.shape, "Prueba:", X_prueba.shape)

# TODO 2: arma un Pipeline para las columnas NUMERICAS (imputar con la
# mediana, despues escalar) y otro para las CATEGORICAS (imputar con el
# valor mas frecuente, despues OneHotEncoder con handle_unknown="ignore" y
# sparse_output=False). Juntalos en un ColumnTransformer llamado
# "preparacion", con las columnas YA_LISTAS pasando sin cambios
# ("passthrough").
numericas = Pipeline([
    ("imputar", SimpleImputer(strategy="median")),
    ("escalar", StandardScaler()),
])
categoricas = Pipeline([
    ("imputar", SimpleImputer(strategy="most_frequent")),
    ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preparacion = ColumnTransformer([
    ("numericas", numericas, NUMERICAS),
    ("categoricas", categoricas, CATEGORICAS),
    ("ya_listas", "passthrough", YA_LISTAS),
])

# TODO 3: aprende la preparacion solo con entrenamiento y transforma
# entrenamiento y prueba. Imprime cuantas columnas quedan.
X_ent_listo = preparacion.fit_transform(X_ent)
X_prueba_listo = preparacion.transform(X_prueba)
print("\nEntrenamiento listo:", X_ent_listo.shape, "Prueba lista:", X_prueba_listo.shape)
print("Columnas despues de preparar:", X_ent_listo.shape[1])
nombres = preparacion.get_feature_names_out()
print(list(nombres))
for grupo in ["numericas", "categoricas", "ya_listas"]:
    print(grupo, sum(n.startswith(grupo + "__") for n in nombres))

# TODO 4: imprime las medianas que aprendio el imputador de numericas
# (atributo statistics_) junto a las medianas de entrenamiento y las de
# todo el dataset.
imputador = preparacion.named_transformers_["numericas"].named_steps["imputar"]
medianas = pd.DataFrame({
    "imputador": imputador.statistics_,
    "entrenamiento": X_ent[NUMERICAS].median().values,
    "todo_el_dataset": X[NUMERICAS].median().values,
}, index=NUMERICAS)
print("\nMedianas:")
print(medianas)

# TODO 5: transforma el viaje nuevo de abajo, que sale de una zona que el
# entrenamiento nunca vio, e imprime cuantas columnas de zona_origen
# quedan en 1.
nuevo = pd.DataFrame([{
    "distancia_km": 9.0, "pasajeros": 1, "hora_sin": 0.5, "hora_cos": -0.866,
    "es_fin_de_semana": 0, "es_hora_pico": 1, "es_tarifa_fija": 0,
    "zona_origen": "Bronx", "zona_destino": "Midtown", "metodo_pago": "tarjeta",
}])
nuevo_listo = preparacion.transform(nuevo)
cols_origen = [i for i, n in enumerate(nombres) if n.startswith("categoricas__zona_origen_")]
print("\nColumnas de zona_origen en 1 para el viaje desde el Bronx:",
      int(nuevo_listo[0, cols_origen].sum()), "de", len(cols_origen))
print("Columnas del viaje nuevo:", nuevo_listo.shape[1])

# Con pd.get_dummies aplicado solo al viaje nuevo:
solo_nuevo = pd.get_dummies(nuevo, columns=CATEGORICAS, dtype=int)
print("Columnas con pd.get_dummies:", solo_nuevo.shape[1])
print(list(solo_nuevo.columns))
