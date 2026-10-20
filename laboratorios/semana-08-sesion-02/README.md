# Semana 8, Sesión 2: Exploración y visualización de datos

> Objetivo de la sesión: explorar el dataset de tu proyecto con estadística descriptiva y visualizaciones para identificar patrones, outliers y supuestos que romperían un modelo antes de entrenarlo.

**Lectura previa de esta sesión:** ["A Data Scientist's Essential Guide to Exploratory Data Analysis"](https://towardsdatascience.com/a-data-scientists-essential-guide-to-exploratory-data-analysis-25637eee0cf6/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 1 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

pandas ya lo tienes desde la Semana 4. Instala matplotlib:

```
pip install matplotlib
```

Verifica con:

```
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

Los scripts de hoy guardan cada gráfico como imagen en una carpeta `figuras/`. También puedes correr el mismo código en un notebook de Jupyter, donde el gráfico aparece debajo de la celda.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué preguntas responde la exploración de datos antes de entrenar cualquier modelo?**

_(tu respuesta)_

**El artículo distingue la exploración de una sola variable de la exploración de la relación entre dos. ¿Qué herramienta propone para cada una?**

_(tu respuesta)_

## Pregunta de apertura

Entrenas un modelo que predice la tarifa de un viaje en taxi. Sin que lo sepas, 15 de los 3000 viajes tienen una tarifa de 999.99 por un error de captura. ¿Qué le pasa al modelo? ¿Lo notarías mirando solo su resultado final?

_(tu respuesta)_

## Ejercicio 1: resumen, distribución y atípicos

Enunciado: [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv) tiene 3000 viajes en taxi de Nueva York: fecha y hora de inicio, zona de origen y de destino, distancia, duración, pasajeros, método de pago, tarifa y propina. Primera mirada: qué contiene, cómo se distribuye la tarifa y qué valores están fuera de lo esperable.

**Respuesta 1, antes de ver la solución:** en `describe()`, ¿qué dos valores de la tabla te harían sospechar de un error sin necesidad de graficar nada?

_(tu respuesta)_

**Respuesta 2:** la regla del IQR marca más de 300 viajes con tarifa atípica. ¿Esperas que todos sean errores? ¿Cómo lo averiguarías?

_(tu respuesta)_

**La formulación completa, tal como la resuelvo yo:**

_(anota aquí el script y lo que encontró, mientras se revisa en clase)_

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: relaciones entre variables

Solo o en pareja. Parte de la plantilla [`plantilla-ejercicio2/explorar_relaciones.py`](./plantilla-ejercicio2/explorar_relaciones.py) y córrela desde su carpeta (`cd plantilla-ejercicio2`, `python3 explorar_relaciones.py`): lee el mismo `starter/viajes_taxi.csv`. Quita solo los errores que el Ejercicio 1 ya confirmó y completa los 5 `TODO`: dispersión de distancia contra tarifa, propina por método de pago, correlaciones con la tarifa, y pasajeros vacíos o en 0.

**Hallazgo 1: qué muestra el gráfico de distancia contra tarifa, además de la tendencia general.**

Además de la relación casi lineal (correlación 0.96), aparece una línea horizontal en tarifa = 70 sin importar la distancia: son **300 viajes** con tarifa exactamente 70, todos entre JFK Airport y Manhattan/Harlem (Midtown↔JFK 117, JFK→Lower Manhattan 35, JFK→Upper West Side 33, etc.). Es la tarifa fija del aeropuerto, no un error. Los viajes JFK↔Brooklyn/Queens no la tienen (pagan en promedio unos 52 a 56) y hay 9 viajes JFK→JFK con tarifa promedio de 14.29.

**Hallazgo 2: qué pasa con la propina según el método de pago, y si crees que describe a los pasajeros o al sistema que registra los datos.**

Los **888 viajes en efectivo tienen propina 0** (promedio 0.00), mientras que ninguno de los 2027 viajes con tarjeta tiene propina 0 (promedio 4.73). Esto describe al sistema, no a los pasajeros: la propina en efectivo no se registra. Además, 60 viajes no tienen método de pago (propina promedio de 3.35, 16 con propina 0). Para el modelo, propina 0 en efectivo significa "dato no capturado" y no "no dio propina".

**Hallazgo 3: la duración tiene una correlación con la tarifa tan alta como la distancia. Si el modelo tiene que estimar la tarifa en el momento en que alguien pide el taxi, ¿puede usar la duración?**

No. La duración tiene una correlación de 0.956 (la distancia, 0.960), pero solo se conoce cuando el viaje termina. Si el modelo la usa, hay fuga de información (*data leakage*): funciona en el entrenamiento y falla en producción. Lo mismo pasa con la propina (0.633). Sí se pueden usar la distancia estimada de la ruta, las zonas y la hora. Los pasajeros casi no influyen (-0.051) y tienen 119 vacíos y 19 en 0, que también son errores de captura.

## Ejercicio 3: explora el dataset de tu proyecto

Usa el archivo que tu pipeline de la Semana 7 deja listo para modelar, y la plantilla [`plantilla-eda/explorar_proyecto.py`](./plantilla-eda/explorar_proyecto.py).

### Paso 1: resumen y variable objetivo (5 minutos)

Reemplaza la ruta del archivo y el nombre de la columna que tu modelo va a predecir. Corre los pasos 1 a 3.

Archivo: `obt_emergencias_canton_dia.csv` (emergencias por cantón y día, del 2021-07-01 al 2026-08-31, 222 cantones). Objetivo: `obj_n_emergencias_t7`, es decir, las emergencias del cantón dentro de 7 días.

- 422 912 filas y 37 columnas. Se quitan 1568 filas sin objetivo (los últimos 7 días de cada cantón) y quedan **421 344**.
- Faltantes: `nombre_feriado` 405 216 (normal, solo existe en feriados), `lag_1d` 224, `lag_7d` y `media_prev_7d` 1568, `lag_14d` 3136, `lag_28d` y `media_prev_28d` 6272. Las demás columnas no tienen faltantes.
- El objetivo tiene media 41.8, mediana 8 y máximo 3653: está muy sesgado a la derecha.
- El IQR (Q1 = 3, Q3 = 22) marcó **53 230 atípicos** (12.6 %).

### Paso 2: relaciones (10 minutos)

Completa los pasos 4 y 5: la relación entre tu objetivo y la variable que esperas que más influya, y las consultas que confirman lo que encuentres.

- Dispersión `lag_7d` contra objetivo en escala log (`figuras/lag7d_vs_objetivo.png`): la relación es fuerte y lineal en log. La correlación es 0.989 en escala original y 0.920 en log. La más alta en log es `media_prev_28d` (0.948), así que la media reciente del cantón es el mejor predictor.
- Promedio de emergencias por día de la semana (`figuras/emergencias_por_dia.png`): domingo 52.6 y sábado 48.4, contra 35.5 el martes. Hay estacionalidad semanal clara.
- Consultas: atípicos por cantón, promedio por región, total diario alrededor de enero de 2024, zonas no delimitadas, suma de las columnas `n_*` y nulos en los lags.

### Paso 3: tres hallazgos que afectarían el modelado (10 minutos)

Para cada hallazgo, escribe qué encontraste, el número o el gráfico que lo confirma, y qué cambia en el modelo por su causa (una fila que quitas, una columna que no puedes usar, una variable que hay que transformar).

**Hallazgo 1:**

**Los atípicos del IQR no son errores, son los cantones grandes.** Quito, Guayaquil, Ambato, Machala, Riobamba, Ibarra y Loja aparecen como atípicos los 1881 días (todos sus días). Solo Quito y Guayaquil aportan el **39.1 %** de todas las emergencias (promedio diario de 2044 y 1620, contra una mediana general de 8). **Cambio en el modelo:** no se quita ninguna fila. Se transforma el objetivo con `log1p` (la correlación con `media_prev_28d` sube a 0.948 en log) o se predice relativo al nivel del cantón, y se incluye el cantón (o su media histórica) como variable. Si no, el error lo dominan dos cantones.

**Hallazgo 2:**

**Enero de 2024 es un período anómalo y hay "zonas no delimitadas".** En enero de 2024 (`es_periodo_anomalo = true`, 6944 filas) el total diario nacional cae a **6716**, contra 9665 en diciembre y 9545 en febrero (−30 %). Además, las 3762 filas de zona no delimitada tienen un promedio de 0.99 emergencias y no son cantones reales. **Cambio en el modelo:** se excluyen del entrenamiento las filas cuyo objetivo cae en enero de 2024 (o se usa `es_periodo_anomalo` como variable) y se quitan las zonas no delimitadas, porque meten ruido sin representar ningún cantón.

**Hallazgo 3:**

**Fuga de información y nulos estructurales en las variables temporales.** Las columnas `n_seguridad_ciudadana`, `n_gestion_sanitaria`, etc. suman exactamente `n_emergencias` en el **100 %** de las filas. Son el mismo dato desglosado, y para predecir `n_emergencias` del mismo día serían fuga. Para el objetivo t+7 solo se pueden usar si se conocen al cierre del día t. Además, los lags tienen nulos al inicio de cada serie (6272 en `lag_28d`, el primero disponible es el 2021-07-29). **Cambio en el modelo:** se quitan los primeros 28 días de cada cantón (o se imputa con indicador). La partición train/test tiene que ser **temporal**, no aleatoria, porque los lags de días vecinos están casi duplicados (correlación de 0.98 o más) y una partición aleatoria inflaría la métrica.

## Antes de la próxima sesión

**Lectura:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

La tarifa fija de JFK y la hora del día no existen como columnas del dataset de taxis. ¿Cómo le darías esa información a un modelo?

_(tu respuesta)_
