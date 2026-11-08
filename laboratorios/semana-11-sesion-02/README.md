# Semana 11, Sesión 2: Esquemas de validación y evaluación de modelos

> Objetivo de la sesión: seleccionar un esquema de validación adecuado al contexto de los datos de tu proyecto (aleatorio, temporal, por grupo) y elegir la métrica de evaluación correcta según el costo relativo de cada tipo de error.

**Lectura previa de esta sesión:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 6 (desarrollo y evaluación de modelos offline).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn. Ten a mano tu `pipeline_proyecto.py` y el modelo final que elegiste la sesión pasada. Los laboratorios de hoy son notebooks, como los de la Semana 10 y la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la paradoja de la exactitud (accuracy paradox)? ¿Qué métricas recomienda en su lugar?**

_(tu respuesta)_

**El artículo cita la ley de Goodhart: "cuando una medida se convierte en objetivo, deja de ser una buena medida". ¿Qué ejemplo da, y qué tiene que ver con elegir un modelo por una sola métrica?**

_(tu respuesta)_

## Pregunta de apertura

La sesión pasada elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

_(tu respuesta)_

## Ejercicio 1: tres formas de validar el mismo modelo

Notebook: [`starter/ejercicio1_validacion.ipynb`](./starter/ejercicio1_validacion.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Ya trae una función `evaluar` que corre una validación cruzada con el esquema que le pases y guarda el error de cada pliegue. Los 4 `TODO` son tres esquemas de validación y dos métricas. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** con una sola división, el gradient boosting dio 1.2754 y el bosque 1.3016. Con 5 pliegues al azar, ¿esperas que los cinco errores del gradient boosting sean parecidos entre sí? ¿Cuánto esperas que varíen?

Parecidos, pero no iguales: cada pliegue valida con 595 viajes distintos. Salieron **[1.28, 1.24, 1.16, 1.24, 1.23]**, con un promedio de **1.2288** y una desviación de **0.036**. Entre el mejor y el peor pliegue hay 0.12 USD: unas cinco veces lo que separaba al gradient boosting del bosque con una sola división (1.2989 − 1.2754 = 0.02). Una sola división no alcanza para decidir entre dos modelos tan parecidos. *(En mi corrida, el bosque con una división dio 1.2989; el 1.3016 del enunciado probablemente sale de otra versión de scikit-learn.)*

**Respuesta 2 (TODO 2):** en la validación temporal, cada pliegue entrena con los viajes anteriores y valida con los siguientes. ¿Esperas un error mayor o menor que con pliegues al azar? ¿Por qué?

**Mayor**, y con más variación. Salió **1.3099** de promedio con una desviación de **0.069**, contra 1.2288 y 0.036 al azar. Hay dos razones: los primeros pliegues entrenan con muy pocos viajes (el primero, solo con una sexta parte de los datos), y el modelo tiene que predecir viajes posteriores a los que vio, como en la app real, donde no se conoce el futuro. El esquema al azar mezcla viajes de todo el período y es un poco optimista.

**Respuesta 3 (TODO 3):** en la validación por zona, ninguna zona de origen aparece a la vez en entrenamiento y en validación. ¿Qué pliegue crees que va a tener el peor error?

El que deja fuera **JFK Airport**. Confirmado: el pliegue 5 valida con **Brooklyn y JFK Airport** y tiene un MAE de **6.43**. Los otros cuatro pliegues quedan entre 1.19 y 1.31. Si el modelo nunca vio viajes que salen de JFK, no puede aprender sus tarifas: las de JFK → Brooklyn o Queens rondan los 55 USD sin tarifa fija, y la tarifa promedio desde JFK es 65.02, la más alta de todas las zonas. Por eso la desviación de este esquema es enorme (**2.06**).

**Respuesta 4 (TODO 4):** la app muestra una tarifa estimada antes del viaje, y el pasajero reclama si la real sale más de 20% por encima. ¿Qué métrica dice mejor cuántos reclamos va a haber: el MAE, el RMSE, o el porcentaje de viajes subestimados en más de 20%?

El **porcentaje de viajes subestimados en más de 20 %**, porque cuenta exactamente los casos que generan un reclamo: **4.37 %** de los viajes de prueba, unos 26 de 595. El MAE (1.2754) y el RMSE (1.9213) promedian los errores en las dos direcciones. No distinguen si el error es por arriba (el pasajero paga menos de lo estimado y no reclama) o por abajo, ni si supera el 20 %. El RMSE además castiga más los errores grandes, pero sigue sin decir cuántos viajes cruzan el umbral del reclamo.

**Lo que confirmó la tabla, esquema por esquema:**

| esquema | modelo | MAE promedio | desviación | peor pliegue |
|---|---|---|---|---|
| al azar | gradient boosting | 1.2288 | 0.0361 | 1.28 |
| al azar | bosque | 1.2887 | 0.0286 | 1.33 |
| temporal | gradient boosting | 1.3099 | 0.0693 | 1.40 |
| por zona | gradient boosting | 2.2996 | 2.0648 | 6.43 (Brooklyn y JFK Airport) |

Con una sola división, el gradient boosting se equivoca en 1.2754 de MAE y 1.9213 de RMSE, y subestima en más de 20 % el 4.37 % de los viajes. El esquema que más cambia la conclusión es el de zona: el modelo funciona para zonas que ya vio, pero no para una zona nueva.

## Durante los ConcepTests

**ConcepTest 1. Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

**ConcepTest 2. Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

## Ejercicio 2: el 95% que no detecta nada

Solo o en pareja. Notebook: [`starter/ejercicio2_fraude_y_umbral.ipynb`](./starter/ejercicio2_fraude_y_umbral.ipynb), sobre [`starter/pagos_taxi.csv`](./starter/pagos_taxi.csv): 6000 pagos de la app de taxis, de los que unos 300 resultaron fraudulentos. Mismo esqueleto del Ejercicio 1, con un problema de clasificación. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué accuracy tiene la línea base que nunca predice fraude? ¿Y qué recall?

Accuracy de **0.9508** (acierta en el 95 % de los pagos, porque el 95.1 % no son fraude) y recall de **0**: no detecta ni uno de los 295 fraudes. Es la paradoja de la exactitud.

**Respuesta 2:** ¿cuánto mejora la accuracy del gradient boosting sobre la línea base, y cuánto mejora el recall? ¿Cuál de los dos números describe mejor lo que gana la empresa?

La accuracy apenas sube **2 puntos**, de 0.9508 a **0.9712**. El recall pasa de 0 a **0.4610**: el modelo detecta el 46 % de los fraudes, con una precisión de 0.9051. Lo que gana la empresa lo describe el **recall**, porque cada fraude detectado es un contracargo de unos 40 USD que se evita, y la accuracy casi no se mueve porque el 95 % de los pagos son legítimos de todos modos. La regresión logística queda por debajo, con un recall de 0.2576.

**Respuesta 3:** con el umbral 0.5, ¿cuántos fraudes deja pasar el gradient boosting? ¿Qué umbral eligió el paso 5, y cuántos fraudes y falsas alarmas tiene con ese umbral?

Con el umbral 0.5 deja pasar **159 de los 295 fraudes** (detecta 136) y da solo 14 falsas alarmas, con un costo de 6402 USD. El paso 5 eligió el umbral **0.1**: deja pasar **122 fraudes** y da **222 falsas alarmas**, con un costo de **5546 USD**, un 13.4 % menos. Como un fraude sin detectar cuesta 40 USD y una falsa alarma solo 3, conviene bajar el umbral y revisar más pagos aunque la mayoría sean legítimos. Los 5 pasos de la comprobación dan `OK`.

**Si terminas antes:** si la revisión manual de una falsa alarma costara 20 dólares en vez de 3, ¿el umbral elegido sube o baja? Cambia la constante y comprueba.

**Sube, hasta 0.5.** Con las mismas probabilidades y falsas alarmas de 20 USD, los costos quedan: umbral 0.05 → 15 780, 0.1 → 9320, 0.15 → 7820, 0.2 → 7300, 0.3 → 6780 y **0.5 → 6640**. Si revisar un pago legítimo cuesta casi la mitad de lo que cuesta un fraude, ya no conviene marcar tantos: el umbral sube hasta el máximo de la lista y se aceptan más fraudes sin detectar a cambio de muchas menos revisiones.

## Ejercicio 3: la validación y la métrica de tu proyecto

Usa tu `pipeline_proyecto.py` y el modelo final de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `validacion_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que evalúe tu modelo final (y el segundo mejor) con validación cruzada, con dos esquemas: el que corresponde a tus datos (temporal si predices el futuro, por grupo si la misma entidad aparece muchas veces, estratificado si clasificas una clase poco frecuente) y uno al azar, para comparar. El notebook guarda una tabla con el promedio, la desviación y cada pliegue.

### Entregable 2: la métrica del negocio

Escribe qué error es más caro en tu proyecto (sobreestimar o subestimar, un falso positivo o un falso negativo) y cuánto, aunque sea aproximado. Calcula en el mismo notebook una métrica que refleje ese costo, además de la métrica que venías usando.

### Entregable 3: respuestas

**¿Qué esquema de validación corresponde a tu proyecto, y por qué?**

Notebook: `proyecto/validacion_proyecto.ipynb`, que guarda la tabla en `proyecto/validacion_proyecto.csv`. Importa las features de `pipeline_proyecto.py`.

**Temporal.** El modelo predice el futuro: las emergencias de un cantón dentro de 7 días. En producción siempre entrena con el pasado y predice fechas que no vio. Por eso usé 5 pliegues de ventana creciente, cortados **por fecha y no por fila** (cada fecha tiene una fila por cantón, y cortar por fila partiría un mismo día entre entrenamiento y validación). Entre entrenamiento y validación dejé **7 días de separación**, porque el objetivo de cada fila es t+7: sin ella, las últimas filas de entrenamiento tendrían su objetivo dentro del período de validación. Cada pliegue valida unos 10 meses (67 266 filas), del 2022-06 al 2026-08. Un esquema por grupo (por cantón) no corresponde, porque el modelo siempre predice cantones que ya conoce.

**¿Cambió la conclusión entre el esquema al azar y el tuyo? Muestra los números.**

| esquema | modelo | MAE promedio | desviación | pliegues | costo asimétrico |
|---|---|---|---|---|---|
| temporal | línea base (igual que hoy) | 6.7306 | 0.2264 | 6.98, 6.80, 6.43, 6.50, 6.95 | 13.48 |
| temporal | **gradient boosting max_depth=5** | **5.5787** | 0.2917 | 5.99, 5.74, 5.31, 5.19, 5.67 | **11.91** |
| temporal | red neuronal (64) | 5.7096 | 0.4890 | 6.64, 5.62, 5.50, 5.20, 5.58 | 12.58 |
| al azar | línea base (igual que hoy) | 6.8955 | 0.0656 | 6.89, 6.82, 7.01, 6.85, 6.91 | 13.81 |
| al azar | gradient boosting max_depth=5 | 5.6078 | 0.0475 | 5.56, 5.58, 5.69, 5.57, 5.64 | 12.21 |
| al azar | red neuronal (64) | 5.6600 | 0.0627 | 5.68, 5.56, 5.69, 5.74, 5.63 | 12.40 |

El **orden** no cambia: en los dos esquemas el gradient boosting queda primero. Lo que cambia es **la confianza**. Al azar, los pliegues parecen casi idénticos (desviación de 0.05) porque mezclan días vecinos de todo el período. En el temporal la desviación es **6 veces mayor** (0.29 para el gradient boosting y 0.49 para la red), porque cada pliegue es un año distinto con su propio nivel de emergencias. El esquema al azar daría una falsa sensación de precisión. Además, los dos esquemas dan vuelta la conclusión de la sesión pasada: con una sola división la red ganaba (5.50 contra 5.55), pero con validación cruzada el gradient boosting le gana.

**¿La diferencia entre tu modelo final y el segundo es mayor que la variación entre pliegues? ¿Sigues eligiendo el mismo?**

**No es mayor.** En el esquema temporal, el gradient boosting le gana a la red por **0.13** de MAE en promedio (5.58 contra 5.71), menos que la desviación entre pliegues (0.29 y 0.49). Pliegue por pliegue, el gradient boosting gana en 3 de 5 en MAE y en 2 de 5 en costo. Gran parte de la ventaja sale del primer pliegue, entrenado con solo 10 meses, donde la red se equivoca mucho más (6.64 contra 5.99).

**Sí, sigo eligiendo el gradient boosting**, pero no porque sea claramente más preciso. Es **más estable**: su peor pliegue es 5.99, contra 6.64 de la red, y su desviación es casi la mitad. También tiene el menor costo asimétrico promedio (11.91 contra 12.58), se puede explicar con sus importancias y es determinista. Con estos números no se puede afirmar que uno sea más preciso que el otro.

**¿Qué métrica del negocio elegiste, y qué dice que no decía la anterior?**

Elegí el **costo asimétrico**: en emergencias, **subestimar es más caro que sobreestimar**. Si el modelo predice menos emergencias de las que ocurren, el cantón queda con menos personal y unidades de las necesarias, y una emergencia mal atendida cuesta mucho más que una unidad ociosa. Supuse, de forma aproximada, que una emergencia subestimada cuesta **3 veces** lo que una sobreestimada. También calculé el % de cantón-días subestimados en más de 20 %.

**Qué dice que no decía el MAE:** en el esquema temporal, el gradient boosting mejora el MAE de la línea base en un **17.1 %** (6.73 → 5.58), pero el costo asimétrico solo en un **11.6 %** (13.48 → 11.91). Además, subestima en más de 20 % el **34.1 %** de los cantón-días, más que la propia línea base (32.8 %). Los modelos entrenados en log tienden a predecir por debajo de los picos: ganan porque no sobreestiman, pero siguen subestimando demasiado. El MAE pesa igual los dos errores y lo esconde. Con este costo, el siguiente paso sería corregir hacia arriba (por ejemplo, predecir un percentil alto en vez del promedio). En cantones pequeños, ese 20 % es muy sensible: con 2 emergencias reales, predecir 1.5 ya cuenta.

## Antes de la próxima sesión

**Lectura:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

_(tu respuesta)_
