# Semana 13, Sesión 1: Despliegue en producción: versionado y registro de modelos

> Objetivo de la sesión: versionar y registrar el modelo desplegado de tu proyecto, explicando qué problema resuelve un registro de modelos frente a sobrescribir el mismo archivo cada vez que se reentrena.

**Lectura previa de esta sesión:** ["What Is Model Governance?"](https://www.ibm.com/think/topics/model-governance), IBM.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 10: Infrastructure and Tooling for MLOps.

## Requisitos antes de esta sesión

Nada nuevo que instalar. El registro de modelos es parte del mismo servidor de MLflow que usas desde la Semana 11: se levanta solo al iniciar el Codespace, en el puerto 5050 (si no responde, `bash .devcontainer/mlflow-server.sh`). Los scripts lo encuentran por la variable `MLFLOW_TRACKING_URI`. Los laboratorios de hoy son scripts.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la gobernanza de modelos, y qué componentes la forman?**

_(tu respuesta)_

**El artículo pide un inventario central de modelos. ¿Qué datos tendría que tener ese inventario sobre cada modelo, y para qué los necesita alguien que no lo entrenó?**

_(tu respuesta)_

## Pregunta de apertura

La sesión pasada, tu servicio cargaba `modelo.joblib`. Si la semana que viene reentrenas y vuelves a exportar, el archivo se reemplaza. Si mañana un usuario reclama por una predicción de hoy, ¿cómo sabrías qué modelo la produjo? ¿Y cómo volverías al modelo anterior si el nuevo resulta peor?

_(tu respuesta)_

## Ejercicio 1: la tarifa, con registro

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). Es el servicio de la tarifa de la sesión pasada, ahora con el modelo en el registro de MLflow en vez de un archivo. [`registrar_modelo.py`](./starter/ejercicio1/registrar_modelo.py) entrena la red neuronal con el número de neuronas que le pases y la registra como una versión nueva del modelo `tarifa-taxis`. [`promover.py`](./starter/ejercicio1/promover.py) pone el alias `produccion` en la versión que le pases. [`versiones.py`](./starter/ejercicio1/versiones.py) lista las versiones con su error y sus alias (no lo modifiques). [`servicio.py`](./starter/ejercicio1/servicio.py) carga la versión que tiene el alias `produccion`. Los 4 `TODO`: registrar, promover, cargar desde el registro y responder qué versión calculó cada tarifa.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python registrar_modelo.py 64
python versiones.py
python promover.py 1
pytest -v
```

**Respuesta 1 (antes de los TODO):** `registrar_modelo.py` dice `Version None`, `versiones.py` no imprime nada y `promover.py` termina con un error. ¿Qué dice el error, y qué te dice sobre lo que hay en el registro?

El error es:

```
mlflow.exceptions.RestException: INVALID_PARAMETER_VALUE: Registered model alias produccion not found.
```

Dice que en el registro no hay nada a lo que ponerle el alias, ni que buscar con él. `registrar_modelo.py` entrenó la red (MAE 1.1695) y guardó la corrida en el experimento, pero sin el `TODO 1` no guarda el modelo ni lo registra. Por eso la versión es `None` y `versiones.py` no encuentra ninguna versión de `tarifa-taxis`: el modelo registrado ni siquiera existe. `promover.py`, sin el `TODO 2`, no pone el alias, y al leerlo con `get_model_version_by_alias` falla. Una corrida en el experimento no es un modelo registrado: el registro solo tiene lo que se registra explícitamente. Además, `pytest` da `1 failed, 1 passed`: falla la prueba que espera `version_modelo` en la predicción.

**Respuesta 2 (TODO 1):** registra dos versiones, una con 64 neuronas y otra con 8. Antes de correr `versiones.py`, ¿cuántas versiones esperas ver, y qué pasó con la primera cuando registraste la segunda?

Dos versiones, y la primera **sigue intacta**. Registrar con el mismo `registered_model_name` no sobrescribe: crea la versión siguiente, cada una apuntando a su propia corrida y a su propio artefacto. Es lo contrario de exportar otra vez `modelo.joblib`.

```
Version 1: red de 64 neuronas, MAE 1.1695, alias: -
Version 2: red de 8 neuronas, MAE 1.1877, alias: -
```

Ninguna tiene alias todavía: registrar no es desplegar. Que una versión exista no significa que alguien la use.

**Respuesta 3 (TODO 2 y 3):** pon `produccion` en la versión 2 y levanta el servicio (`uvicorn servicio:app --port 8000`). ¿Qué versión dice `/salud`? Ahora mueve el alias a la versión 1 sin reiniciar uvicorn: ¿qué versión responde el servicio? ¿Y después de reiniciarlo?

- Con `produccion` en la versión 2: `/salud` responde `{"estado":"ok","modelo_cargado":true,"version_modelo":"2"}`.
- Después de `python promover.py 1`, **sin reiniciar**: sigue respondiendo la **versión 2**, tanto en `/salud` como en las tarifas. El servicio busca el alias y carga el modelo una sola vez, al importarse. Mover el alias cambia el registro, no el proceso que ya está corriendo.
- **Después de reiniciar** uvicorn: responde la **versión 1**.

Por eso mover el alias no basta: hay que reiniciar (o volver a desplegar) el servicio. Y por eso conviene que el servicio diga qué versión tiene cargada, en vez de suponer que es la que dice el registro: durante ese intervalo, las dos no coinciden.

**Respuesta 4 (TODO 4):** las pruebas usan un modelo de prueba y una versión inventada, `"7"`, sin servidor de MLflow. ¿Qué prueba falla antes del `TODO 4`, y qué comprueba que el `/salud` no comprobaba?

Falla **`test_cada_prediccion_dice_que_version_la_calculo`**: la respuesta era `{"tarifa_estimada": 20.0}` y la prueba espera también `"version_modelo": "7"`. `test_salud_dice_que_version_sirve` pasaba, porque `/salud` ya decía la versión.

La diferencia: `/salud` dice qué versión tiene el servicio **ahora**. La predicción dice qué versión **calculó esa tarifa**. Cuando un pasajero reclama por una tarifa de hace una semana, `/salud` ya no sirve, porque el alias pudo moverse y el servicio reiniciarse. La respuesta guardada con su `version_modelo` permite rastrear esa tarifa hasta su corrida en el registro (parámetros, datos, métricas) y reproducirla. La prueba lo comprueba sin MLflow: lo que importa es que la versión cargada viaje en cada respuesta.

**Lo que respondió el servicio:**

Viaje Midtown → JFK Airport (2026-11-05 08:30, 26.4 km, 2 pasajeros, tarjeta):

- **Versión 2** (red de 8 neuronas, MAE 1.1877): `{"tarifa_estimada": 69.98, "version_modelo": "2"}`.
- Alias movido a la 1 sin reiniciar: otra vez `{"tarifa_estimada": 69.98, "version_modelo": "2"}`.
- **Versión 1** (red de 64 neuronas, MAE 1.1695), después de reiniciar: `{"tarifa_estimada": 70.07, "version_modelo": "1"}`.

`pytest -v`: 2 passed. La tarifa cambia poco porque las dos redes aprendieron la tarifa fija del aeropuerto. Sin `version_modelo` no se sabría cuál de las dos dio cada número.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el ciclo de vida del modelo de fraude

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El equipo de fraude entrena modelos candidatos cada tanto, y quiere una regla: una versión llega a `produccion` solo si detecta más fraudes (mayor recall al umbral de 0.10) que la que está ahí. [`ciclo_fraude.py`](./starter/ejercicio2/ciclo_fraude.py) entrena tres candidatos, registra cada uno y decide si lo promueve. Los pasos 1 a 3 tienen `TODO`: registrar una versión con sus métricas, leer la métrica de una versión y averiguar qué versión está en producción. Los pasos 4 (decidir y mover el alias) y 5 (etiquetar la versión con el resultado) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`; después, `python ciclo_fraude.py`.

**Respuesta 1:** ¿qué versión quedó en producción, y por qué no la última que se registró?

Quedó la **versión 2**, el gradient boosting:

```
logistica: version 1, recall 0.644, precision 0.304, promovida: True
gradient-boosting: version 2, recall 0.661, precision 0.429, promovida: True
logistica-muy-regularizada: version 3, recall 0.102, precision 0.667, promovida: False
En produccion: version 2
```

La versión 1 entró porque no había ninguna en producción. La 2 la reemplazó porque detecta más fraudes (recall 0.661 contra 0.644). La 3 es la última registrada, pero su recall es 0.102: detecta 1 de cada 10 fraudes, contra 2 de cada 3 de la versión 2. Registrar una versión no la pone en producción: llega solo si pasa la regla. `comprobar.py`: los 5 pasos `OK`.

**Respuesta 2:** la regresión logística muy regularizada tiene la mayor precisión de las tres. ¿Por qué la regla la rechazó? ¿Qué tendría que cambiar en la regla si la empresa decide que las falsas alarmas cuestan más que los fraudes no detectados?

Porque la regla mira **solo el recall**, y en eso es la peor (0.102 contra 0.661). Su precisión alta (0.667) es engañosa: con `C=0.001` el modelo casi no aprende, casi nunca llega a 0.10 y da muy pocas alertas. Las pocas que da suelen acertar, pero se le escapan casi 9 de cada 10 fraudes. Es la regla de la Semana 11: un fraude que se escapa cuesta mucho más que revisar un pago legítimo.

Si las falsas alarmas costaran más, habría que cambiar la métrica de la regla, no el código del registro. Se puede comparar por **precisión**, o mejor por el **costo total** de la Semana 11 con los costos nuevos: `costo_fn × fraudes no detectados + costo_fp × falsas alarmas`, promoviendo la versión con menor costo. Lo más probable es que también haya que **subir el umbral**: 0.10 se eligió con los costos viejos, y la regla debería evaluar cada candidato en su propio umbral óptimo. Conviene además un piso de recall, para no promover un modelo que casi no alerta, como esta logística.

**Respuesta 3:** abre el modelo `fraude-pagos` en la interfaz de MLflow (puerto 5050, pestaña **Models**). Si dentro de un mes alguien pregunta por qué la versión 3 nunca llegó a producción, ¿dónde está la respuesta?

En las **etiquetas de la versión 3**: `validacion = rechazada` y `comparada_con = 2`. Dicen que se evaluó, que no pasó y contra qué versión. El porqué está en las **métricas de su corrida** (la versión enlaza a la corrida `logistica-muy-regularizada`): recall 0.102 contra 0.661 de la versión 2. La regla misma (recall mayor que el de producción) está en `ciclo_fraude.py`. En la interfaz, en **Models → fraude-pagos → Version 3**, se ven las etiquetas y el enlace a la corrida, y el alias `produccion` aparece en la versión 2. Nadie tiene que acordarse de nada ni buscar un chat: la decisión quedó escrita junto al modelo.

**Si terminas antes:** vuelve a correr `python ciclo_fraude.py`. ¿Cuántas versiones hay ahora, y cuál está en producción? ¿Es lo que esperabas?

Ahora hay **6 versiones**, y en producción sigue la **versión 2**:

```
logistica: version 4, recall 0.644, precision 0.304, promovida: False
gradient-boosting: version 5, recall 0.661, precision 0.429, promovida: False
logistica-muy-regularizada: version 6, recall 0.102, precision 0.667, promovida: False
En produccion: version 2
```

Es lo esperado. Con los mismos datos y `random_state=42`, los candidatos dan exactamente las mismas métricas, y las tres versiones nuevas quedan `rechazada` contra la 2. La 4 y la 6 tienen menos recall. La 5 es el mismo gradient boosting con recall idéntico (0.661), y la regla pide *estrictamente* mayor, así que un empate no mueve el alias. Eso está bien: cambiar a un modelo igual no gana nada y genera un despliegue sin motivo. El costo es que el registro se llena de versiones duplicadas, que siguen ahí con su etiqueta `rechazada`, porque nunca se borran.

## Ejercicio 3: el registro de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el servicio que construiste la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: el modelo, en el registro

- `servicio/registrar_modelo.py`: entrena tu modelo final, guarda su métrica principal en la corrida y lo registra como una versión nueva de un modelo con el nombre de tu proyecto. Cada vez que lo corres crea una versión; ninguna reemplaza a otra.
- `servicio/promover.py`: pone el alias `produccion` en la versión que le pases.
- Al menos dos versiones registradas (por ejemplo, con dos valores de un hiperparámetro), y una de ellas con el alias `produccion`.

Corre los scripts desde `psets/proyecto/` como módulos (`python -m servicio.registrar_modelo`), para que encuentren `modelado/`.

### Entregable 2: el servicio sabe qué versión sirve

- El script que exportaba el modelo (Semana 12) ahora exporta la versión con el alias `produccion`, y guarda su número junto al modelo (por ejemplo, en `servicio/modelo_version.txt`). La imagen lleva los dos archivos adentro: el contenedor no necesita llegar al servidor de MLflow.
- `/salud` y cada predicción responden `version_modelo`.
- Una prueba en `servicio/test_api.py` que compruebe que la predicción incluye la versión, con un modelo de prueba y una versión inventada.

### Entregable 3: `docs/despliegue.md`

Agrega la sección **Versionado y registro**: cómo se llama tu modelo en el registro, qué versión está en producción y con qué métrica, con qué criterio se mueve el alias, y los pasos para volver a la versión anterior.

**¿Con qué criterio decides que una versión nueva de tu modelo pasa a producción? ¿Quién lo decide en tu proyecto, y dónde queda registrada la decisión?**

Mi modelo está en el registro como **`emergencias-cantones`**. Registré dos versiones: la **1**, gradient boosting `max_depth=3` (MAE 5.7118, costo asimétrico 12.3553), y la **2**, `max_depth=5` (MAE **5.5490**, costo **11.7665**). La 2 está en `produccion` y reproduce exactamente el MAE del modelo final de la Semana 11.

**Criterio:** una versión nueva pasa si cumple tres condiciones:
- Se entrenó con la misma división temporal (corte 2025-08-25, 7 días de separación, que quedan como parámetros de la corrida), para que las métricas sean comparables.
- Tiene **menor costo asimétrico** (3 × emergencias faltantes + 1 × sobrantes) que la de producción. Es la métrica de negocio: quedarse corto con los recursos de un cantón es lo caro.
- **No empeora el MAE.** Si el costo baja solo porque el modelo sobreestima en todos lados, no es una mejora.

**Quién lo decide:** yo, como responsable del proyecto. `python -m servicio.promover` lista las versiones con sus dos métricas, y `python -m servicio.promover <versión>` mueve el alias.

**Dónde queda registrada:** en el registro de MLflow. El alias `produccion` dice cuál sirve, cada versión enlaza a su corrida (parámetros, huella de los datos, métricas), y `promover.py` deja en la versión promovida la etiqueta `anterior_en_produccion` (la 2 tiene `anterior_en_produccion = 1`), que dice a qué volver si sale mal. El criterio y los pasos para volver atrás están en `docs/despliegue.md`, sección **Versionado y registro**.

**Si tu servicio corre en un contenedor sin acceso al servidor de MLflow, ¿cómo sabe qué versión está sirviendo?**

Porque la versión viaja **dentro de la imagen**, junto al modelo. `python -m servicio.exportar_modelo` busca en el registro la versión con el alias `produccion`. Guarda su modelo en `servicio/modelo.joblib` y su número en `servicio/modelo_version.txt`. El `Dockerfile` copia los dos archivos, y al arrancar `api.py` carga el modelo y lee el número, sin conectarse a MLflow. `/salud` y cada predicción responden `version_modelo`.

Lo comprobé levantando el contenedor con `--network none`, sin red: `/salud` respondió `{"estado":"ok","modelo_cargado":true,"version_modelo":"2"}`, y Latacunga del 2026-08-24, `{"fecha_objetivo":"2026-08-31","emergencias_estimadas":108.79,"version_modelo":"2"}`.

El número dice la versión que se **exportó**, no la que el registro tiene hoy en `produccion`. Si se mueve el alias y no se exporta ni se reconstruye la imagen, el contenedor sigue sirviendo la versión anterior, y lo dice. `test_cada_prediccion_dice_que_version_la_calculo` comprueba con una versión inventada (`"7"`) que la versión llega en cada predicción. En total, `pytest psets/proyecto`: 12 passed.

## Antes de la próxima sesión

**Lectura:** ["Machine learning model monitoring: Best practices"](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/), Datadog.

Tu servicio responde en unos milisegundos, `/salud` dice que todo está bien, y cada tarifa dice qué versión la calculó. ¿Qué le pasa a un modelo que responde rápido pero cada vez peor? ¿Cómo te enterarías?

_(tu respuesta)_
