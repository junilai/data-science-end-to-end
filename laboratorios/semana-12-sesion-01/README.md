# Semana 12, Sesión 1: Patrones de despliegue: ¿batch u online?

> Objetivo de la sesión: elegir entre servir el modelo de tu proyecto en modalidad batch o en modalidad online, justificando la decisión según la latencia que exige el caso de uso propuesto, y proteger con pruebas automáticas, que corren en cada cambio, el código que lo sirve.

**Lectura previa de esta sesión:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 7 (despliegue de modelos y servicio de predicción).

## Requisitos antes de esta sesión

Hoy se agregan tres herramientas: joblib (guardar un modelo entrenado en un archivo), pytest (pruebas automáticas) y GitHub Actions (integración continua). Las librerías nuevas ya están en `requirements.txt`, junto con FastAPI para la sesión siguiente. Sincroniza tu fork y reconstruye el contenedor ("Rebuild Container"), como en la Semana 11. Comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Debe mostrar `pytest 8.3.3` y `fastapi 0.115.4`.

Los laboratorios de hoy son scripts, no notebooks: lo que producen corre sin nadie mirando. Cada ejercicio entrena primero su modelo con un script que ya viene completo (`entrenar.py`), que lo guarda en un archivo `.joblib` dentro de la carpeta del ejercicio. Ese archivo no se sube al repositorio (`*.joblib` está en `.gitignore`).

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué distingue un sistema de predicción batch de uno online? Da un ejemplo de cada uno.**

_(tu respuesta)_

**¿Qué limitación tiene el sistema batch que el online resuelve, y qué cuesta resolverla?**

_(tu respuesta)_

## Pregunta de apertura

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

_(tu respuesta)_

## Ejercicio 1: la tarifa de los viajes programados, por lotes

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). La app permite reservar un viaje para el día siguiente. Cada noche, [`predecir_lote.py`](./starter/ejercicio1/predecir_lote.py) estima la tarifa de los 60 viajes reservados en [`viajes_programados.csv`](./starter/ejercicio1/viajes_programados.csv) con la red neuronal que elegiste en la Semana 11, y deja las estimaciones en una tabla que la app lee por la mañana. [`test_predecir_lote.py`](./starter/ejercicio1/test_predecir_lote.py) tiene sus pruebas. Los 5 `TODO`: cargar el modelo, estimar, armar la tabla de salida y dos pruebas.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
python predecir_lote.py
pytest -v
```

**Respuesta 1 (antes de los TODO):** sin resolver, `predecir_lote.py` escribe una tabla con una sola columna, y `pytest` dice `1 failed, 2 passed`. ¿Cómo pueden pasar dos pruebas si el script no estima ninguna tarifa?

Porque una prueba sin `assert` siempre pasa. Las pruebas de los `TODO 4` y `5` solo llaman a `predecir_lote` y no comprueban nada: si la función no lanza un error, pytest las da por buenas. La única que comprueba algo es `test_columnas_de_salida`, que espera tres columnas y recibe una, y es la que falla. Que una prueba pase no significa que el código esté bien. Significa que nada de lo que esa prueba revisa falló.

**Respuesta 2 (TODO 1 y 2):** 4 de los 60 viajes no tienen pasajeros y 2 no tienen método de pago. ¿Esperas que el script falle en esos viajes, que los salte, o que les estime una tarifa? ¿Por qué?

Que les estime una tarifa. El pipeline guardado incluye la preparación: `pasajeros` vacío se imputa con la mediana de entrenamiento, y `metodo_pago` vacío con la categoría más frecuente, con imputadores que se ajustaron al entrenar. El script no falla ni salta esas filas, y los 60 viajes salen con tarifa. Saltarlos sería peor: la app abriría por la mañana un viaje reservado sin tarifa. Además, `pasajeros` casi no influye en la tarifa (correlación −0.05 en la Semana 8), así que imputarlo cambia poco la estimación.

**Respuesta 3 (TODO 3):** todas las filas de la tabla llevan la misma `version_modelo`. ¿Para qué guardarla en cada fila?

Para saber qué modelo produjo cada estimación cuando la tabla ya está lejos del script: la lee la app, se acumula con las de otras noches o se compara con la tarifa real. Si mañana se reentrena el modelo, las filas nuevas llevan otra huella. Así, cuando un pasajero reclama por una tarifa, se puede identificar el modelo exacto y reproducirla, y se pueden comparar los errores de dos versiones sobre los mismos días. Si la versión solo quedara en el log del script, se perdería al mezclar tablas. La huella es el MD5 del `.joblib`, así que cambia con cualquier cambio del archivo, aunque se llame igual.

**Respuesta 4 (TODO 4 y 5):** antes de escribir las dos pruebas, ¿qué cambio en `predecir_lote.py` haría fallar la del `TODO 4`? ¿Y la del `TODO 5`?

- **TODO 4** (una fila por viaje, mismos `id_viaje` y mismo orden): fallaría si el script quitara filas, por ejemplo con un `dropna()` antes de predecir para saltar los viajes incompletos. También si las ordenara, por ejemplo por tarifa, o si armara `id_viaje` con un índice distinto al de las estimaciones.
- **TODO 5** (ningún viaje sin tarifa y todas las filas con la versión que se pasó): fallaría si el script dejara `NaN` en la tarifa de los viajes sin pasajeros, por ejemplo si se predijera fuera del pipeline, sin el imputador, y se rellenara con vacío. También si escribiera una versión fija o vacía en lugar de `version_modelo`.

**Lo que mostró el lote:**

Modelo **`c75c7248502c`** (la huella cambia con cada `python entrenar.py` en otra máquina, porque el `.joblib` no es idéntico byte a byte): **60 viajes leídos, 60 filas** en `tarifas_estimadas.csv`.

- Tarifa más baja: **4.67** (viaje 90014). Tarifa más alta: **70.26** (viaje 90051), la tarifa fija del aeropuerto, que el modelo aprendió. Media 22.87, mediana 17.33.
- Los 4 viajes sin pasajeros sí recibieron tarifa: 90023 → **33.94** (Queens → Upper West Side, 10.6 km), 90024 → **29.88** (Brooklyn → Harlem, 10.2 km), 90025 → **25.80** (Midtown → Harlem, 8.4 km) y 90048 → **17.58** (Harlem → Upper East Side, 4.8 km). Son tarifas acordes a su distancia.
- Los 2 sin método de pago también: 90044 → 60.79 y 90050 → 29.59.
- `pytest -v`: **3 passed**.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: la cola de revisión de fraude

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). Cada madrugada, [`puntuar_pagos.py`](./starter/ejercicio2/puntuar_pagos.py) calcula la probabilidad de fraude de los 300 pagos de ayer ([`pagos_ayer.csv`](./starter/ejercicio2/pagos_ayer.csv)) con el gradient boosting de la Semana 11, y arma la cola de revisión manual con los pagos que alcanzan el umbral de 0.10 que elegiste ese día. Mismo esqueleto del Ejercicio 1: los pasos 1 a 3 tienen `TODO`; el paso 4 (la cola) y el paso 5 (las pruebas de [`test_puntuar_pagos.py`](./starter/ejercicio2/test_puntuar_pagos.py)) solo tienen un comentario que dice qué hacer. Primero `python entrenar_fraude.py`; después resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`.

**Respuesta 1:** ¿cuántos pagos quedaron en la cola de revisión? ¿Cuántos fraudes esperarías encontrar en los 300 pagos, si la proporción es la de la Semana 11?

**26 pagos** en la cola (umbral 0.10), el primero con probabilidad 0.9984. En la Semana 11, 295 de 6000 pagos eran fraude (4.9 %), así que en 300 pagos esperaría unos **15 fraudes** (14.8). La cola tiene casi el doble de pagos que fraudes esperados, y eso es lo buscado: con el umbral bajo de la Semana 11, un fraude que se escapa cuesta mucho más que revisar un pago legítimo. El equipo revisa unos 11 pagos de más para no dejar pasar fraudes. Aun así, algunos fraudes pueden quedar fuera de la cola, porque en la Semana 11 el recall con 0.10 no llegaba al 100 %.

**Respuesta 2:** este modelo corre por lotes, una vez al día. ¿Qué pierde la empresa por no evaluar cada pago en el momento en que ocurre? ¿Qué tendría que pasar con la revisión para que eso importe?

Pierde la posibilidad de **frenar el pago antes de que se complete**. Por lotes, el fraude se detecta hasta un día después: el dinero ya salió, y lo único que queda es reclamar, revertir o bloquear la tarjeta para la próxima vez. En ese día, un mismo estafador puede repetir el fraude varias veces. Eso solo importa si la revisión puede **actuar en el momento**: bloquear el pago, pedir una verificación extra o retener el viaje mientras se decide, en segundos y antes de que el pasajero se suba. Si la revisión es manual y se trabaja durante el día, una probabilidad calculada en milisegundos no cambia nada, porque igual espera a que alguien la mire. Para que valga la pena evaluar en línea, la acción tiene que ser automática e inmediata.

**Respuesta 3:** la prueba de la cola de revisión arma su propia tabla de cuatro pagos con probabilidades escritas a mano, en vez de usar el modelo. ¿Qué gana la prueba con eso?

Gana que prueba **solo la lógica de `a_revisar`** (filtrar por umbral, incluir el límite, ordenar de mayor a menor) con un resultado que se sabe de antemano: 4, 2 y 3. Si usara el modelo, las probabilidades dependerían del entrenamiento, de la semilla y de la versión de scikit-learn, y no se podría escribir el resultado esperado a mano. Además, puede poner un caso borde a propósito: el 0.10, que es exactamente el umbral y tiene que entrar. Con un modelo real sería casi imposible que una probabilidad saliera justo 0.10. La prueba no necesita el `.joblib` ni los datos, así que corre en GitHub Actions. Y si falla, el error está en `a_revisar` y no en el modelo.

**Si terminas antes:** sube `UMBRAL` a 0.50 y vuelve a correr `puntuar_pagos.py`. ¿Cuántos pagos quedan en la cola? ¿Siguen pasando todas las pruebas? ¿Por qué?

Quedan **13 pagos** en la cola (contra 26 con 0.10), menos que los ~15 fraudes esperados: con 0.50 seguro se escapan fraudes. **Las 4 pruebas siguen pasando**, porque ninguna usa la constante `UMBRAL`: `test_cola_de_revision` le pasa su propio umbral (0.10) a `a_revisar`, y las demás no miran el umbral. Las pruebas comprueban que el código hace lo que debe con cualquier umbral, no que el umbral elegido sea el correcto. Esa es una decisión de negocio (el costo de la Semana 11), y una prueba unitaria no la protege. Si se quisiera proteger, haría falta una prueba explícita como `assert UMBRAL == 0.10`, pero eso congela una decisión, no una lógica. Después de probar, dejé `UMBRAL = 0.10`.

## Ejercicio 3: el despliegue y las pruebas de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md). Esta vez no hay plantilla.

### Entregable 1: `docs/despliegue.md`

Una página que diga si tu modelo se sirve por lotes o en línea, y por qué:

- Qué decisión habilita la predicción, quién la usa y cuánto tiempo puede esperar por ella (un número: milisegundos, horas, un día).
- Cuántos casos distintos hay que predecir, y si se conocen de antemano.
- Qué datos necesitan tus features en el momento de predecir, y si están disponibles en ese momento.
- Qué se pierde con el patrón que descartas.

### Entregable 2: las primeras pruebas de tu proyecto

Un archivo `psets/proyecto/modelado/test_pipeline_proyecto.py` con al menos tres pruebas sobre tu `pipeline_proyecto.py`, cada una con al menos un `assert`. Las pruebas arman sus propios datos (un `DataFrame` de pocas filas escrito a mano), no leen tu archivo de `data/`: en GitHub Actions ese archivo no existe. Ideas de qué probar: que la columna objetivo no está entre las que recibe el modelo; que una feature calculada da el valor que sabes calcular a mano; que una feature que mira hacia atrás en el tiempo no usa una fila del futuro; que la preparación acepta una categoría que no vio al entrenar.

Comprueba desde la raíz de tu fork que pasan, con el mismo comando que va a usar GitHub:

```
pytest psets/proyecto -v
```

### Entregable 3: la integración continua de tu fork

El archivo [`.github/workflows/pruebas.yml`](../../.github/workflows/pruebas.yml) le pide a GitHub que, en cada push, cree una máquina limpia, instale `requirements.txt` y corra `pytest psets/proyecto`. En tu fork viene apagado: en la página de tu fork, pestaña **Actions**, presiona "I understand my workflows, go ahead and enable them". Haz push de tus pruebas y espera la marca verde junto a tu commit.

**Pega el enlace a la ejecución de GitHub Actions donde pasan tus pruebas:**

_(tu respuesta)_

**¿Qué patrón elegiste para tu proyecto, y qué número de latencia lo decide?**

**Por lotes**, una vez al día ([`psets/proyecto/docs/despliegue.md`](../../psets/proyecto/docs/despliegue.md)). El modelo predice las emergencias de cada cantón a **7 días**, para planificar recursos y turnos de la semana próxima. Esa decisión puede esperar **horas** (la tabla de la noche está lista antes de la jornada), no milisegundos. Además, los casos son siempre los mismos **222 cantones**, conocidos de antemano, y todas las features (lags, medias de 28 días, calendario y feriados) existen al cierre del día *t*. Estimar los 222 toma unos 4 ms, así que un servicio encendido todo el día no aportaría nada: una consulta a media tarde devolvería lo mismo que la tabla de la noche anterior.

**Rompe a propósito una de tus pruebas (cambia un valor esperado), haz push y mira qué pasa en GitHub. Después arréglala. ¿Qué te dijo GitHub, y dónde lo viste?**

_(tu respuesta)_

## Antes de la próxima sesión

**Lectura:** ["Journey to Full-Stack Data Scientist: Model Deployment"](https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/), Towards Data Science.

Hoy la app lee las tarifas de una tabla calculada de noche. Pero cuando un pasajero escribe un destino que no estaba reservado, no hay ninguna fila que leer. ¿Qué tendría que existir para que la app le pregunte al modelo en ese momento, y qué le mandaría?

_(tu respuesta)_
