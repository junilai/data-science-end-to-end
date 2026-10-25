# Semana 9, Sesión 1: Ingeniería de características: principios aplicados

> Objetivo de la sesión: explicar por qué una misma variable cruda puede necesitar codificaciones distintas según el modelo que la consuma, y diseñar al menos dos features derivadas para el dataset de tu proyecto.

**Lectura previa de esta sesión:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen (O'Reilly, 2022), Capítulo 5: Feature Engineering.

## Requisitos antes de esta sesión

pandas y matplotlib, de la sesión pasada. numpy se instala junto con pandas; verifica con:

```
python3 -c "import numpy; print(numpy.__version__)"
```

Ten a mano tus 3 hallazgos de la exploración de la Semana 8, Sesión 2: las features de hoy salen de ahí.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué diferencia a una feature que tiene sentido de negocio de una que solo mejora una métrica?**

_(tu respuesta)_

**Elige uno de los ejemplos del artículo: ¿qué conocimiento del negocio traduce a una columna, y cómo?**

_(tu respuesta)_

## Pregunta de apertura

La exploración encontró dos patrones que no existen como columnas: la tarifa fija entre JFK y Manhattan, y viajes que cuestan más por kilómetro a ciertas horas. El modelo solo ve columnas. ¿Cómo le das esa información?

_(tu respuesta)_

## Ejercicio 1: features de fecha y hora

Enunciado: [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv), el mismo dataset de la sesión pasada, sin los errores confirmados. A partir de `fecha_hora_inicio`, construir hora, día de la semana, fin de semana, hora pico y la hora como ciclo, y comprobar si la hora pico aporta algo a la tarifa.

**Respuesta 1, antes de ver la solución:** si dejas la hora como un número del 0 al 23, ¿qué tan lejos queda la hora 23 de la hora 0 para un modelo que mide distancias entre números? ¿Y en el reloj?

Como número quedan a 23 unidades de distancia, la máxima posible. En el reloj están a solo 1 hora. Por eso la hora se codifica como ciclo con `hora_sin = sin(2π·hora/24)` y `hora_cos = cos(2π·hora/24)`: así las 23:00 y las 00:00 quedan juntas en el círculo.

**Respuesta 2:** la duración del viaje no existe al pedir el taxi, pero la hora sí. ¿Qué parte de lo que la duración le diría al modelo puede traer `es_hora_pico`?

El tráfico. La duración le dice al modelo si el viaje fue lento (más tiempo detenido, más cobro). `es_hora_pico` anticipa esa lentitud antes del viaje: en hora pico la velocidad mediana baja a 16.5 km/h, contra 21.4 km/h fuera de ella. Lo que no puede traer es la parte imprevisible de cada viaje, como un accidente o un desvío.

**La formulación completa, tal como la resuelvo yo:**

Script: `starter/mi_primer_feature_eng.py` (se corre desde la carpeta de la sesión). Crea `hora`, `dia_semana`, `hora_sin`, `hora_cos`, `es_fin_de_semana` (sábado y domingo) y `es_hora_pico` (lunes a viernes de 7 a 9 y de 16 a 19, las horas donde la velocidad cae a unos 17 km/h).

Sin los viajes de JFK (tarifa fija) ni los 25 de distancia 0:

| es_hora_pico | viajes | tarifa/km (mediana) | velocidad km/h (mediana) | tarifa promedio |
|---|---|---|---|---|
| 0 | 1795 | 3.23 | 21.4 | 20.06 |
| 1 | 786 | 3.53 | 16.5 | 22.67 |

- La correlación de `es_hora_pico` con la tarifa es 0.121. La de `hora` es 0.030 y las de `hora_sin`/`hora_cos` son cercanas a 0: la hora sola no sirve, lo que importa es si es hora pico.
- En una regresión lineal, el R² pasa de 0.9657 (solo distancia) a 0.9748 (distancia + hora pico). El coeficiente es +2.05 USD por viaje en hora pico.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: zona, codificación y disponibilidad

Solo o en pareja. Parte de la plantilla [`plantilla-ejercicio2/features_de_zona.py`](./plantilla-ejercicio2/features_de_zona.py) y córrela desde su carpeta. Completa los 4 `TODO`.

**Respuesta 1:** ¿cuántos viajes tienen `es_tarifa_fija` en 1, y qué tarifa mínima y máxima tienen? ¿Qué te dice eso sobre lo que un modelo lineal puede aprender con esta columna?

**300 viajes**, todos con tarifa mínima y máxima de **70.0**, y ningún viaje con tarifa 70 queda fuera de la regla. La tarifa de esos viajes no depende de la distancia. Un modelo lineal con solo `distancia_km` intentaría ajustar una recta y se equivocaría en todos ellos. Con `es_tarifa_fija` puede aprender un ajuste propio para esos viajes. Aun así, el modelo seguiría sumando el efecto de la distancia, así que lo ideal es que la tarifa sea 70 cuando la columna vale 1 (una regla, o una interacción `es_tarifa_fija × distancia`). Un árbol de decisión lo aprende directamente.

**Respuesta 2:** ¿cuántas columnas crea one-hot para la zona de origen, y cuántas la codificación ordinal? Si el modelo es una regresión lineal, ¿cuál de las dos usarías, y por qué la otra le da información falsa?

One-hot crea **8 columnas** (una por zona) y la ordinal **1** (`origen_ordinal`, de 0 = Brooklyn a 7 = Upper West Side, en orden alfabético). Para una regresión lineal usaría **one-hot**. La ordinal le dice al modelo que las zonas tienen un orden y distancias entre sí, cosa que no es cierta: Queens (5) no está "entre" Midtown (4) y Upper East Side (6), y JFK (2) no es "el doble" de Harlem (1). Además, la regresión forzaría un efecto que sube o baja en línea recta con ese número. La ordinal sí sirve en modelos de árboles, que solo cortan por umbrales.

**Respuesta 3:** ¿qué columnas del dataset no existen al pedir el taxi? ¿Qué pasaría con el modelo si las usaras para entrenar?

No existen: `duracion_min` y `propina` (se conocen al terminar el viaje), `tarifa` (es el objetivo) y, en general, `metodo_pago` (se confirma al pagar). `distancia_km` solo se puede usar como distancia estimada de la ruta, e `id_viaje` existe pero no aporta nada. Si se entrena con ellas hay fuga de información: el modelo se ve excelente en la validación (la duración tiene una correlación de 0.956 con la tarifa) y luego en producción esas columnas no están o hay que inventarlas, así que las predicciones fallan.

## Ejercicio 3: dos features para tu proyecto

Usa el mismo archivo de la exploración de la Semana 8 y la plantilla [`plantilla-features/features_proyecto.py`](./plantilla-features/features_proyecto.py).

### Paso 1: el momento de la predicción (5 minutos)

Escribe qué predice tu modelo y en qué momento: qué información existe en ese instante y cuál todavía no.

El modelo predice `obj_n_emergencias_t7`, las emergencias que tendrá un cantón **dentro de 7 días**, y la predicción se hace al cierre del día t.

- **Existe:** todo lo del día t y antes (`n_emergencias` y sus categorías, `lag_1d` … `lag_28d`, `media_prev_7d`, `media_prev_28d`), los datos fijos del cantón (cantón, provincia, región) y lo que se sabe del calendario de t+7 (`es_feriado_t7`, el día de la semana, que es el mismo que el de t).
- **No existe:** lo que pase entre t+1 y t+7: las emergencias de esos días, `obj_n_emergencias_t3` (de t+3) y, obviamente, el objetivo.

### Paso 2: construye las 2 features (15 minutos)

Completa la plantilla con 2 columnas nuevas, derivadas de las existentes. Al menos una tiene que salir de uno de tus hallazgos de la exploración.

Script: `plantilla-features/features_proyecto.py`.

```python
# Feature 1: nivel del canton en escala log
datos["log_media_prev_28d"] = np.log1p(datos["media_prev_28d"])

# Feature 2: promedio de las ultimas 3 semanas en el mismo dia de la semana
datos["media_mismo_dia"] = datos[["lag_7d", "lag_14d", "lag_28d"]].mean(axis=1)
```

### Paso 3: justifica cada feature (5 minutos)

Para cada una: de qué hallazgo o conocimiento del dominio sale, qué evidencia muestra que aporta (una correlación, una diferencia de promedios), y por qué existe en el momento de predecir. Si alguna es categórica, di qué codificación usarías según el modelo que planeas entrenar.

**Feature 1:**

`log_media_prev_28d`. **Sale del hallazgo 1 de la Semana 8:** Quito y Guayaquil aportan el 39 % de las emergencias y aplastan la escala, así que el IQR marcaba como atípicos a cantones enteros. En escala original la media de 28 días ya correlaciona 0.982 con el objetivo, pero esa cifra la dominan los cantones grandes. En escala log, la correlación de `log_media_prev_28d` con `log(1 + objetivo)` es **0.947**, y eso vale parejo para cantones chicos y grandes. Sirve como el "nivel" del cantón, para predecir el objetivo en log y luego volver con `expm1`. **Existe al predecir** porque usa solo los 28 días anteriores a t. Es numérica, así que no necesita codificación.

**Feature 2:**

`media_mismo_dia` (promedio de `lag_7d`, `lag_14d` y `lag_28d`). **Sale de la estacionalidad semanal** que se vio en la exploración: el domingo promedia 52.9 emergencias y el martes 35.8. Como t+7 cae el mismo día de la semana que t-7, t-14 y t-28, esta media compara "domingos con domingos". La correlación con el objetivo sube a **0.992** (contra 0.982 de `media_prev_28d`). Sobre todo, quita el sesgo por día: usando la media de 28 días, el error promedio en log es −0.174 entre semana (sobreestima) y +0.135 el fin de semana (subestima). Con `media_mismo_dia` queda en −0.051 y −0.047. **Existe al predecir** porque los tres lags son anteriores a t. Es numérica, sin codificación. Si además se usara `nombre_dia` como categórica, iría con one-hot en una regresión lineal y con ordinal en un modelo de árboles.

## Antes de la próxima sesión

**Lectura:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

Para escalar la distancia antes de entrenar, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

_(tu respuesta)_
