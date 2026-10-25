import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

viajes = pd.read_csv("./starter/viajes_features.csv")

print(viajes.isna().sum()[lambda s: s > 0])

cols = ["distancia_km", "pasajeros"]
y = viajes["tarifa"]

# version con fuga: imputa y escala con todo el dataset y despues divide
X = viajes[cols].copy()
X[cols] = SimpleImputer(strategy="median").fit_transform(X)
X[cols] = StandardScaler().fit_transform(X)

X_ent, X_prueba, y_ent, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = LinearRegression().fit(X_ent, y_ent)
print("MAE con fuga:", round(mean_absolute_error(y_prueba, modelo.predict(X_prueba)), 4))

# version corregida: primero divide y el pipeline aprende solo con entrenamiento
X_ent, X_prueba, y_ent, y_prueba = train_test_split(viajes[cols], y, test_size=0.2, random_state=42)

modelo = Pipeline([
    ("imputar", SimpleImputer(strategy="median")),
    ("escalar", StandardScaler()),
    ("regresion", LinearRegression()),
])
modelo.fit(X_ent, y_ent)
print("MAE corregido:", round(mean_absolute_error(y_prueba, modelo.predict(X_prueba)), 4))

print("Media de distancia con todo el dataset:", round(viajes["distancia_km"].mean(), 4))
print("Media de distancia solo entrenamiento:", round(X_ent["distancia_km"].mean(), 4))
