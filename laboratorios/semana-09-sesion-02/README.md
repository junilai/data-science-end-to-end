# Semana 9, Sesión 2: Pipeline reproducible de features

> Objetivo de la sesión: construir un pipeline de transformación reproducible con scikit-learn (imputación, codificación, escalado) para el dataset de tu proyecto, que evite la fuga de información entre entrenamiento y validación.

**Lectura previa de esta sesión:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

**Referencia de esta sesión:** [Pipelines and composite estimators (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/compose.html).

## Requisitos antes de esta sesión

Instala scikit-learn:

```
pip install scikit-learn
```

Verifica con:

```
python3 -c "import sklearn; print(sklearn.__version__)"
```

Este laboratorio usa `sparse_output=False` en `OneHotEncoder`, disponible desde scikit-learn 1.2. Si tu versión es anterior, actualízala con `pip install --upgrade scikit-learn`.

Ten a mano el script de features de tu proyecto de la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la fuga de información, y por qué un modelo con fuga puede verse excelente en la evaluación y fallar en producción?**

_(tu respuesta)_

**El artículo describe varias formas en que se produce una fuga. ¿Cuál de ellas viste la sesión pasada con la duración del viaje?**

_(tu respuesta)_

## Pregunta de apertura

Para escalar la distancia, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

_(tu respuesta)_

## Ejercicio 1: encontrar y corregir la fuga

Enunciado: [`starter/viajes_features.csv`](./starter/viajes_features.csv) tiene los viajes de taxi con las features de la sesión pasada, sin los errores confirmados, y con la distancia 0 y los 0 pasajeros marcados como faltantes. Un compañero imputa y escala la distancia y los pasajeros con todo el dataset, después divide en entrenamiento y prueba, y entrena una regresión lineal. El código está en la diapositiva.

**Respuesta 1, antes de ver la solución:** ¿en qué línea entra el conjunto de prueba al entrenamiento?

En el `fit` del imputador y del escalador, que se hace **antes** de `train_test_split`. Ahí la mediana que se usa para imputar y la media y desviación estándar que se usan para escalar se calculan con todas las filas, incluidas las que después quedan en prueba. Así el conjunto de prueba ya influyó en cómo se transformó el entrenamiento.

**Respuesta 2:** ¿cuánto esperas que cambie el error absoluto medio en prueba al corregir la fuga? Si casi no cambia, ¿significa que no había fuga?

Muy poco. Con 2975 viajes y una división 80/20, la mediana y la media casi no cambian: la media de la distancia es 8.517 con todo el dataset y 8.572 con solo entrenamiento. El MAE pasa de **2.5375** (con fuga) a **2.5373** (corregido). Que casi no cambie **no significa que no hubiera fuga**: la fuga es un error en el procedimiento, no en el número. Aquí su efecto es chico porque la estadística filtrada es estable, pero con pocos datos, con una división por fecha (donde la distribución cambia con el tiempo) o con transformaciones que usan el objetivo (como el *target encoding*), la misma fuga infla la métrica mucho más.

**La formulación completa, tal como la resuelvo yo:**

Script: `starter/fuga.py` (se corre desde la carpeta de la sesión, con `python3 starter/fuga.py`). Compara la versión con fuga y la corregida.

La corrección es dividir primero y meter la imputación y el escalado en un `Pipeline` junto con el modelo, para que `fit` solo vea entrenamiento:

```python
X_ent, X_prueba, y_ent, y_prueba = train_test_split(
    viajes[["distancia_km", "pasajeros"]], viajes["tarifa"], test_size=0.2, random_state=42)

modelo = Pipeline([
    ("imputar", SimpleImputer(strategy="median")),
    ("escalar", StandardScaler()),
    ("regresion", LinearRegression()),
])
modelo.fit(X_ent, y_ent)
print(mean_absolute_error(y_prueba, modelo.predict(X_prueba)))
```

Resultado: MAE con fuga de 2.5375 y corregido de 2.5373. Faltantes en el dataset: 25 en `distancia_km`, 138 en `pasajeros` y 60 en `metodo_pago`.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el pipeline completo

Solo o en pareja. Parte de la plantilla [`plantilla-ejercicio2/pipeline_completo.py`](./plantilla-ejercicio2/pipeline_completo.py) y córrela desde su carpeta. Completa los 5 `TODO`: división, `ColumnTransformer` con tres grupos de columnas, `fit` solo con entrenamiento, verificación de las medianas, y un viaje desde una zona que el entrenamiento nunca vio.

**Respuesta 1:** ¿cuántas columnas quedan después de preparar los datos? Explica de dónde sale ese número a partir de los tres grupos de columnas.

Quedan **25 columnas** (entrenamiento de 2380 × 25 y prueba de 595 × 25):

- **Numéricas:** 2 (`distancia_km` y `pasajeros`). Se imputan y escalan, pero siguen siendo una columna cada una.
- **Categóricas:** 18. One-hot crea una columna por cada valor visto en entrenamiento: 8 de `zona_origen`, 8 de `zona_destino` y 2 de `metodo_pago`.
- **Ya listas:** 5 (`hora_sin`, `hora_cos`, `es_fin_de_semana`, `es_hora_pico` y `es_tarifa_fija`), que pasan sin cambios.

2 + 18 + 5 = 25.

**Respuesta 2:** ¿las medianas que aprendió el imputador coinciden con las de entrenamiento o con las de todo el dataset? ¿Qué demuestra eso?

Coinciden con las de **entrenamiento**. Para `distancia_km` el imputador aprendió 6.94, que es la mediana de entrenamiento, mientras que la de todo el dataset es 6.93. Para `pasajeros` las tres valen 1. Esto demuestra que `fit_transform(X_ent)` aprendió solo de entrenamiento, y que `transform(X_prueba)` aplica esos mismos valores a prueba sin volver a calcularlos. La prueba no influye en la preparación, así que no hay fuga.

**Respuesta 3:** ¿qué le pasa al viaje desde el Bronx? ¿Qué habría pasado con `pd.get_dummies` aplicado por separado a ese viaje?

Con `handle_unknown="ignore"`, el Bronx no coincide con ninguna zona vista en entrenamiento, así que sus **8 columnas de `zona_origen` quedan en 0** (0 de 8 en 1). El viaje sale con las mismas 25 columnas que espera el modelo y se puede predecir: el modelo lo trata como "una zona que no es ninguna de las conocidas". Con `pd.get_dummies` aplicado solo a ese viaje salen **10 columnas** con otros nombres (`zona_origen_Bronx`, `zona_destino_Midtown`, `metodo_pago_tarjeta`…). Faltarían las otras columnas de zona y sobraría `zona_origen_Bronx`, así que no coincidiría con las columnas del entrenamiento y el modelo fallaría o leería las columnas desalineadas.

## Ejercicio 3: el pipeline de tu proyecto

Usa tu archivo con las 2 features de la sesión pasada y la plantilla [`plantilla-pipeline/pipeline_proyecto.py`](./plantilla-pipeline/pipeline_proyecto.py).

### Paso 1: clasifica tus columnas (5 minutos)

Escribe qué columnas son numéricas, cuáles categóricas y cuáles ya están listas, y cuál es el objetivo. Deja fuera toda columna que no exista al momento de predecir.

Script: `plantilla-pipeline/pipeline_proyecto.py`. Usa `obt_emergencias_canton_dia.csv` y agrega las features de la sesión pasada.

- **Objetivo:** `obj_n_emergencias_t7`.
- **Numéricas:** `log_media_prev_28d` y `log_media_mismo_dia` (las 2 features de la sesión pasada) y `log_lag_1d` (log de las emergencias del día t).
- **Categóricas:** `region` y `nombre_dia`.
- **Ya listas:** `es_fin_de_semana` y `es_feriado_t7` (convertidas a 0/1).
- **Fuera:** `obj_n_emergencias_t3` (es de t+3, futuro), las fechas y los identificadores, `nombre_feriado` (casi todo nulo; ya está resumido en `es_feriado_t7`) y `canton` (222 valores; su nivel ya lo captura `log_media_prev_28d`). También se quitan las filas sin objetivo o sin 28 días de historia, las zonas no delimitadas y las filas cuyo día objetivo cae en enero de 2024 (período anómalo).

### Paso 2: decide cómo dividir (5 minutos)

¿Tu modelo predice el futuro a partir de filas con fecha? Si sí, divide por fecha; si no, al azar. Justifica.

**Sí, se divide por fecha.** El modelo predice las emergencias de dentro de 7 días con datos hasta hoy. Si se divide al azar, el entrenamiento tendría días posteriores a los de prueba, y como los lags de días vecinos son casi iguales (correlación de 0.98 o más), el modelo "vería" la prueba y la métrica saldría inflada.

- **Entrenamiento:** del 2021-07-29 al 2025-08-17.
- **Prueba:** del 2025-08-25 al 2026-08-24 (el último año, 20.1 % de las filas).
- Entre los dos quedan **7 días de separación**: como el objetivo de cada fila es t+7, sin esa separación las últimas filas de entrenamiento tendrían su objetivo dentro del período de prueba.

### Paso 3: construye y corre el pipeline (15 minutos)

Completa la plantilla y córrela. Anota la forma de los conjuntos de entrenamiento y prueba ya preparados, y la lista de columnas que produce.

- Filas para modelar: 404 484. **Entrenamiento: (321 900, 16). Prueba: (81 030, 16).**
- 16 columnas: 3 numéricas + 4 de `region` (Amazonía, Costa, Insular, Sierra) + 7 de `nombre_dia` + 2 ya listas:

```
numericas__log_media_prev_28d, numericas__log_media_mismo_dia, numericas__log_lag_1d,
categoricas__region_Amazonía, categoricas__region_Costa, categoricas__region_Insular, categoricas__region_Sierra,
categoricas__nombre_dia_Domingo, categoricas__nombre_dia_Jueves, categoricas__nombre_dia_Lunes,
categoricas__nombre_dia_Martes, categoricas__nombre_dia_Miércoles, categoricas__nombre_dia_Sábado,
categoricas__nombre_dia_Viernes, ya_listas__es_fin_de_semana, ya_listas__es_feriado_t7
```

- Comprobación sin fuga: la media que aprendió el escalador ([2.4624, 2.4265, 2.3795]) es igual a la de entrenamiento y distinta de la de todo el dataset ([2.4554, 2.4196, 2.3729]).

## Antes de la próxima sesión

**Lectura:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

_(tu respuesta)_
