# Semana 12, Sesión 2: Despliegue de modelos como servicio de predicción

> Objetivo de la sesión: exponer el modelo entrenado de tu proyecto como un servicio de predicción mediante una API, empaquetado en un contenedor, con pruebas de su contrato de entrada y salida.

**Lectura previa de esta sesión:** ["Journey to Full-Stack Data Scientist: Model Deployment"](https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 5 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

Nada nuevo que instalar: FastAPI y uvicorn ya llegaron con el contenedor reconstruido la sesión pasada. Los laboratorios de hoy son scripts y un `Dockerfile`. El servicio escucha en el puerto 8000: cuando lo levantes, aparece en la pestaña **Ports** como "Servicio de predicción (Semana 12)", y su documentación interactiva queda en la ruta `/docs` de ese puerto.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué pasos separan un modelo entrenado en un notebook de un modelo que otra aplicación puede usar?**

_(tu respuesta)_

**¿Para qué usa el artículo un contenedor, si el servicio ya corre en la máquina de quien lo escribió?**

_(tu respuesta)_

## Pregunta de apertura

La sesión pasada, la app leía las tarifas de una tabla calculada de noche. Cuando un pasajero escribe un destino que no estaba reservado, no hay ninguna fila que leer. ¿Qué tendría que existir para que la app le pregunte al modelo en ese momento, y qué le mandaría?

_(tu respuesta)_

## Ejercicio 1: la tarifa, como servicio

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). [`servicio.py`](./starter/ejercicio1/servicio.py) es una API de FastAPI con dos rutas: `/salud`, que dice si el servicio está vivo y tiene su modelo, y `/predecir`, que recibe un viaje como JSON y devuelve la tarifa estimada. El viaje llega como lo conoce la app (fecha y hora, zonas, distancia, pasajeros, método de pago), y [`features_taxis.py`](./starter/ejercicio1/features_taxis.py) calcula el resto de las features igual que se calcularon para entrenar. [`test_servicio.py`](./starter/ejercicio1/test_servicio.py) prueba la API sin levantarla, y el [`Dockerfile`](./starter/ejercicio1/Dockerfile) la empaqueta. Los 5 `TODO`: el esquema de entrada, la predicción, dos pruebas y el comando del contenedor.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
pytest -v
uvicorn servicio:app --port 8000
```

Con el servicio levantado, abre el puerto 8000 desde la pestaña **Ports**, agrega `/docs` a la dirección y prueba `/predecir` con **Try it out**. También desde otra terminal:

```
curl -X POST localhost:8000/predecir -H "Content-Type: application/json" -d '{"fecha_hora_inicio": "2026-11-05 08:30", "zona_origen": "Midtown", "zona_destino": "JFK Airport", "distancia_km": 26.4, "pasajeros": 2, "metodo_pago": "tarjeta"}'
```

**Respuesta 1 (antes de los TODO):** sin resolver, `/salud` dice que el modelo está cargado, `/predecir` responde 200 y `pytest` dice `3 passed`. ¿Qué de esa salida debería preocuparte?

Que todo se ve sano y el servicio no sirve. `/predecir` responde 200 con `{"tarifa_estimada": 0.0}` para cualquier viaje: el código fija `tarifa = 0.0` y nunca llama al modelo. `/salud` solo comprueba que el archivo se cargó, no que se use. Y los `3 passed` no prueban nada: dos de las tres pruebas no tienen `assert`, así que pasan aunque la respuesta sea absurda. La única con `assert` es `/salud`, justo la que no mira la predicción. Un código 200 dice que la petición se atendió, no que la respuesta esté bien. Una tarifa de 0.0 para un viaje al aeropuerto es un error que ningún monitor de disponibilidad detectaría.

**Respuesta 2 (TODO 1):** con el esquema completo, ¿qué esperas que responda el servicio a un viaje sin `distancia_km`? ¿Y a uno sin `zona_origen`?

- **Sin `distancia_km`:** 200 con una tarifa. El campo es opcional (`None` por defecto) y el pipeline imputa la mediana de entrenamiento. Midtown → JFK sin distancia dio **68.56** (con 26.4 km, 70.39): sigue cerca de la tarifa fija porque `es_tarifa_fija` sale de las zonas, no de la distancia. Midtown → Harlem sin distancia dio 15.99, la tarifa de un viaje "típico".
- **Sin `zona_origen`:** **422** (`"type": "missing"`, `"loc": ["body", "zona_origen"]`, `"msg": "Field required"`). La zona es obligatoria: sin ella no se puede calcular `es_tarifa_fija`, y no se le puede inventar un valor razonable a un viaje.

**Respuesta 3 (TODO 2):** el modelo se entrenó con `hora_sin`, `hora_cos` y `es_hora_pico`, pero la app manda solo la fecha y hora. ¿Qué pasaría si el servicio calculara `es_hora_pico` con otras horas que las del entrenamiento?

El modelo se equivocaría sin dar ningún error. Recibe un 0 o un 1 como siempre, pero ese número ya no significa lo que aprendió: un viaje de las 8:30 marcado como "no pico" se cobra como uno de horas tranquilas. Lo medí sobre los 2975 viajes: invertir `es_hora_pico` cambia la tarifa en **1.92 en promedio** y hasta **4.90**. Midtown → Harlem (8.4 km) a las 8:30 da 28.19 con pico y 25.34 sin él. Es el *training-serving skew*: la respuesta sigue siendo 200 y no aparece ninguna alerta. Por eso el servicio no reescribe las features: importa `agregar_features` de `features_taxis.py`, la misma función con que se construyó `viajes_features.csv`.

**Respuesta 4 (TODO 3 y 4):** el código 422 del viaje sin zona de origen, ¿lo produce tu código o FastAPI? ¿La petición llega al modelo?

Lo produce **FastAPI**, no mi código. Antes de llamar a `predecir`, FastAPI valida el JSON contra el esquema `Viaje` con pydantic. Si falta un campo obligatorio, o la fecha no se puede leer como `datetime` (`"mañana"` también da 422, `datetime_from_date_parsing`), responde 422 con el detalle del error y **no ejecuta `predecir`**: la petición no llega a `agregar_features` ni al modelo. Por eso la prueba solo necesita el `TestClient`. Mi código no tiene ningún `if` para validar: el esquema hace de contrato.

**Respuesta 5 (TODO 5):** construye y corre la imagen con el `CMD` sin resolver (`docker build -t servicio-tarifa .` y `docker run -p 8000:8000 servicio-tarifa`). El log dice que uvicorn está corriendo, pero `curl localhost:8000/salud` falla. ¿Por qué?

Porque uvicorn, sin `--host`, escucha en `127.0.0.1`, y dentro del contenedor esa dirección es el propio contenedor. El log lo dice: `Uvicorn running on http://127.0.0.1:8000`. `-p 8000:8000` reenvía el puerto del host a la interfaz de red del contenedor, no a su `localhost`, así que la conexión llega y nadie la atiende (`curl` termina con error 56, *connection reset*). Desde dentro sí responde: `docker exec` + petición a `127.0.0.1:8000/salud` devolvió `{"estado":"ok","modelo_cargado":true}`. Con `--host 0.0.0.0` uvicorn escucha en todas las interfaces, el log dice `Uvicorn running on http://0.0.0.0:8000` y `curl` responde desde afuera.

**Lo que respondió el servicio:**

- Midtown → JFK Airport (2026-11-05 08:30, 26.4 km, 2 pasajeros, tarjeta): **`{"tarifa_estimada": 70.39}`**, la tarifa fija del aeropuerto. Dio lo mismo con uvicorn local y con el contenedor.
- El mismo viaje sin `zona_origen`, código **422**:

```
{"detail":[{"type":"missing","loc":["body","zona_origen"],"msg":"Field required","input":{"fecha_hora_inicio":"2026-11-05 08:30","zona_destino":"JFK Airport","distancia_km":26.4,"pasajeros":2,"metodo_pago":"tarjeta"}}]}
```

- `pytest -v`: **3 passed**, ahora con `assert` en las tres.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el fraude, en el momento del pago

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). La sesión pasada, los pagos se revisaban al día siguiente. Ahora la empresa quiere decidir en el momento del cobro si un pago va a revisión. [`servicio_fraude.py`](./starter/ejercicio2/servicio_fraude.py) expone el mismo gradient boosting en `/evaluar_pago`, que devuelve la probabilidad de fraude y la decisión. Mismo esqueleto del Ejercicio 1: los pasos 1 a 3 tienen `TODO` (el esquema, la probabilidad y la prueba del pago válido); el paso 4 (la decisión con el umbral) y el paso 5 (la prueba del pago sin monto) solo tienen un comentario. El `Dockerfile` ya está completo. Primero `python entrenar_fraude.py`; después resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`.

**Respuesta 1:** levanta el servicio y evalúa dos pagos: uno de madrugada, por app, con tarjeta nueva, dos intentos fallidos y ningún viaje previo; otro de la tarde, con tarjeta conocida y 22 viajes previos. ¿Qué probabilidad y qué decisión recibió cada uno?

- De madrugada (02:40, app, 41.30, 8.6 km, tarjeta nueva, 2 intentos fallidos, 0 viajes previos): **`{"probabilidad_fraude": 0.9996, "revisar": true}`**.
- De la tarde (14:10, tarjeta, 19.80, 6.3 km, tarjeta conocida, 0 intentos, 22 viajes previos): **`{"probabilidad_fraude": 0.0159, "revisar": false}`**.

El primero junta todas las señales de riesgo y va a revisión. El segundo queda muy por debajo del umbral de 0.10. `comprobar.py`: los 5 pasos `OK`. `pytest -v`: 3 passed.

**Respuesta 2:** el servicio devuelve la decisión (`revisar`) además de la probabilidad. ¿Qué gana la app con eso? ¿Qué tendría que cambiar si la empresa decide mañana que el umbral es 0.20?

La app no tiene que saber nada del modelo ni del umbral: lee `revisar` y actúa (retiene el pago o lo deja pasar). El umbral sale del análisis de costos de la Semana 11, y queda junto al modelo en un solo lugar. Si cada app (Android, iOS, web) comparara la probabilidad por su cuenta, habría que actualizarlas todas para cambiarlo y, mientras tanto, convivirían umbrales distintos. La probabilidad igual se devuelve para registrar y auditar la decisión.

Con un umbral de 0.20 solo cambia **`UMBRAL = 0.10` en `servicio_fraude.py`**, y después se reconstruye y se vuelve a desplegar la imagen. La app no cambia, y el modelo tampoco se reentrena. La prueba `test_pago_valido` sigue pasando (0.25 también alcanza 0.20). Mejor todavía sería leer el umbral de una variable de entorno, como `RUTA_MODELO`, para cambiarlo sin reconstruir la imagen.

**Respuesta 3:** el mismo pago, evaluado por este servicio y por el lote de la sesión pasada, ¿da la misma probabilidad? ¿Qué tiene que ser igual en los dos para que así sea?

Sí. Mandé al servicio los 300 pagos de `pagos_ayer.csv` de la sesión pasada y comparé con `pagos_puntuados.csv` del lote: la **diferencia máxima es 0.0**, y los mismos **26 pagos** superan 0.10 (el pago 70058 da 0.9984 en los dos). Para que eso pase tiene que ser igual:

- **El modelo:** el mismo `entrenar_fraude.py`, con los mismos datos y `random_state=42`, y la misma versión de scikit-learn (1.5.2) para cargar el `.joblib`. Lo seguro es que los dos carguen el mismo archivo, que se identifica por su huella (`version_modelo`).
- **La preparación:** las mismas `agregar_features` y `columnas` de `preparacion_pagos.py`, importadas y no copiadas, con las columnas en el mismo orden. Los dos `preparacion_pagos.py` son idénticos.
- **Los datos de entrada:** los mismos 7 campos, con el mismo significado (por ejemplo, `fecha_hora` en la misma zona horaria, porque de ahí sale `hora`).

**Si terminas antes:** manda a `/evaluar_pago` un pago con `"monto": "cuarenta"`. ¿Qué responde el servicio, y qué línea de `servicio_fraude.py` produce esa respuesta?

Responde **422**:

```
{"detail":[{"type":"float_parsing","loc":["body","monto"],"msg":"Input should be a valid number, unable to parse string as a number","input":"cuarenta"}]}
```

Ninguna línea mía escribe esa respuesta. La produce la declaración **`monto: float`** de la clase `Pago`: FastAPI valida el JSON contra ese esquema antes de entrar a `evaluar_pago`, y como `"cuarenta"` no se puede convertir a número, devuelve el 422 sin llegar al modelo. Un `"monto": "40"` sí pasa, porque pydantic sí lo convierte a 40.0.

## Ejercicio 3: el servicio de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el modelo final que registraste en MLflow (Semana 11). Esta vez no hay plantilla. Aunque tu proyecto se despliegue por lotes, el servicio también se construye: es la forma en que otro sistema (tu propio lote, un tablero, otra aplicación) le pide una predicción a tu modelo.

### Entregable 1: el servicio, en `psets/proyecto/servicio/`

- Un script que exporte tu modelo final desde MLflow a `servicio/modelo.joblib` (`mlflow.sklearn.load_model("runs:/<run_id>/modelo")` y `joblib.dump`). El archivo no se sube: se regenera con ese script.
- `servicio/api.py`, con `/salud` y una ruta de predicción. El esquema de entrada recibe los datos como los conoce quien llama, y el servicio calcula las features con las mismas funciones de tu `modelado/pipeline_proyecto.py`, importadas desde ahí (`from modelado.pipeline_proyecto import ...`), no copiadas.
- `servicio/Dockerfile` y `servicio/requirements.txt`, con la misma versión de scikit-learn con que entrenaste. La imagen se construye desde `psets/proyecto/`, porque necesita `modelado/` además de `servicio/`: `docker build -f servicio/Dockerfile -t <tu-servicio> .`

El servicio corre desde `psets/proyecto/`: `uvicorn servicio.api:app --port 8000`. El archivo [`psets/proyecto/conftest.py`](../../psets/proyecto/conftest.py) hace que las pruebas también encuentren `modelado/` desde ahí.

### Entregable 2: `servicio/test_api.py`

Al menos tres pruebas con `TestClient`: `/salud`, una predicción válida con un modelo de prueba (no tu `modelo.joblib`, que no existe en GitHub Actions) y una entrada inválida que recibe 422. El workflow de la sesión pasada ya corre todo lo que hay en `psets/proyecto`: en tu próximo push, GitHub corre también estas pruebas.

### Entregable 3: respuestas

En `proyecto/README.md`, agrega la sección **Cómo correr el servicio**: cómo se exporta el modelo, cómo se construye y se levanta el contenedor, y un ejemplo de petición con su respuesta.

**¿Qué datos recibe tu servicio, y cuáles de tus features calcula él mismo? ¿Cómo sabes que las calcula igual que en el entrenamiento?**

**Recibe** lo que se sabe de un cantón al cierre del día *t*: `fecha`, `region`, sus emergencias pasadas (`lag_1d`, que puede venir vacío; `lag_7d`, `lag_14d`, `lag_28d`; `media_prev_28d`), todas ≥ 0, y `es_feriado_t7`. Una región o un lag faltante, o una emergencia negativa, se rechazan con 422.

**Calcula** el resto con las funciones de `modelado/pipeline_proyecto.py`, importadas y no copiadas:

- `agregar_calendario`: `nombre_dia` y `es_fin_de_semana` a partir de la fecha. La agregué hoy, porque quien llama conoce la fecha pero no esas columnas de la OBT.
- `agregar_features`: `log_media_prev_28d`, `media_mismo_dia` → `log_media_mismo_dia`, `log_lag_1d` y los booleanos como 0/1.
- La imputación y el escalado los hace el pipeline guardado en el modelo, con lo que aprendió al entrenar.

**Cómo sé que es igual:**

1. `agregar_calendario` reproduce `nombre_dia` y `es_fin_de_semana` en el **100 % de las 422 912 filas** de la OBT, y `test_calendario_desde_la_fecha` lo protege en GitHub.
2. Mandé al servicio, con sus datos crudos, 800 filas reales del período de prueba, y comparé con `modelo.predict` sobre las features de `cargar_datos`, que es lo que usó el entrenamiento: **diferencia máxima 0.0**.
3. `test_features_calculadas_a_mano` fija los valores de `agregar_features`, y la usan tanto el entrenamiento como el servicio. Si alguien la cambia, cambia para los dos y la prueba lo marca.

**Pega una petición a tu servicio corriendo en el contenedor, y su respuesta:**

Imagen construida desde `psets/proyecto/` con `docker build -f servicio/Dockerfile -t servicio-emergencias .` y levantada con `docker run -p 8000:8000 servicio-emergencias`. Latacunga, lunes 2026-08-24 (un día del período de prueba):

```
$ curl -X POST localhost:8000/predecir -H "Content-Type: application/json" \
  -d '{"fecha": "2026-08-24", "region": "Sierra", "lag_1d": 141, "lag_7d": 106, "lag_14d": 116, "lag_28d": 109, "media_prev_28d": 121.25, "es_feriado_t7": false}'
{"fecha_objetivo":"2026-08-31","emergencias_estimadas":108.79,"version_modelo":"961785a60016"}
```

El valor real del 2026-08-31 fue 119. Para Quito el mismo día, el servicio estimó **1604.99** contra 1667 reales. `/salud` responde `{"estado":"ok","modelo_cargado":true,"version_modelo":"961785a60016"}`.

## Antes de la próxima sesión

**Lectura:** ["What Is Model Governance?"](https://www.ibm.com/think/topics/model-governance), IBM.

Tu servicio carga `modelo.joblib`, el archivo que exportaste hoy. Si la semana que viene reentrenas y vuelves a exportar, el archivo se reemplaza. Si mañana un usuario reclama por una predicción de hoy, ¿cómo sabrías qué modelo la produjo? ¿Y cómo volverías al modelo anterior si el nuevo resulta peor?

_(tu respuesta)_
