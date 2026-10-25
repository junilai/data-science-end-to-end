import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder

viajes=pd.read_csv("./starter/viajes_taxi.csv")

viajes_clean=viajes.query('tarifa < 999 and duracion_min > 0')

print(viajes_clean.head())

print(viajes_clean.info())

fecha_inicio=pd.to_datetime(viajes_clean['fecha_hora_inicio'])

viajes_clean['fecha_hora_inicio']= fecha_inicio
viajes_clean['hora']= fecha_inicio.dt.hour
viajes_clean['dia_semana']= fecha_inicio.dt.dayofweek

print(viajes_clean.head())

viajes_clean['hora_sin']=np.sin(2*np.pi*viajes_clean['hora']/24)
viajes_clean['hora_cos']=np.cos(2*np.pi*viajes_clean['hora']/24)

print(viajes_clean[['fecha_hora_inicio', 'hora', 'hora_sin', 'hora_cos']])

viajes_codificado=pd.get_dummies(viajes_clean, columns=['zona_origen', 'zona_destino'])

print(viajes_codificado)

encoder=OrdinalEncoder()

viajes_codificado['metodo_pago_ordinal']=encoder.fit_transform(viajes_codificado[['metodo_pago']]).ravel()

print(viajes_codificado.head())

viajes_clean['es_fin_de_semana']=(viajes_clean['dia_semana']>=5).astype(int)

# hora pico: lunes a viernes de 7 a 9 y de 16 a 19
horas_pico=[7, 8, 9, 16, 17, 18, 19]
viajes_clean['es_hora_pico']=((viajes_clean['hora'].isin(horas_pico)) & (viajes_clean['es_fin_de_semana']==0)).astype(int)

print(viajes_clean[['fecha_hora_inicio', 'hora', 'dia_semana', 'es_fin_de_semana', 'es_hora_pico']].head(10))

# la hora pico aporta algo a la tarifa?
# se dejan fuera los viajes de JFK porque su tarifa fija no depende de la hora,
# y los de distancia 0 porque no se puede calcular la tarifa por km
print("Viajes con distancia 0:", (viajes_clean["distancia_km"]==0).sum())
sin_jfk=viajes_clean.query('zona_origen != "JFK Airport" and zona_destino != "JFK Airport" and distancia_km > 0').copy()
sin_jfk['tarifa_km']=sin_jfk['tarifa']/sin_jfk['distancia_km']
sin_jfk['velocidad_kmh']=sin_jfk['distancia_km']/sin_jfk['duracion_min']*60

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
print(sin_jfk.groupby('es_hora_pico')[['tarifa_km', 'velocidad_kmh', 'tarifa']].agg(['count', 'mean', 'median']).round(2))

print(sin_jfk.groupby('hora')['velocidad_kmh'].median().round(1))

print(sin_jfk[['tarifa', 'distancia_km', 'es_hora_pico', 'hora', 'hora_sin', 'hora_cos']].corr()['tarifa'].round(3))

# regresion con y sin la hora pico para ver si mejora
X1=sin_jfk[['distancia_km']]
X2=sin_jfk[['distancia_km', 'es_hora_pico']]
y=sin_jfk['tarifa']

modelo1=LinearRegression().fit(X1, y)
modelo2=LinearRegression().fit(X2, y)

print('R2 solo distancia:', round(modelo1.score(X1, y), 4))
print('R2 distancia + hora pico:', round(modelo2.score(X2, y), 4))
print('Coeficiente de es_hora_pico:', round(modelo2.coef_[1], 2))
