# Semana 13, Sesión 2: Monitoreo en producción I: métricas de rendimiento

> Objetivo de la sesión: definir las métricas de monitoreo del modelo de tu proyecto en producción, distinguiendo métricas de sistema (latencia, disponibilidad) de métricas de modelo (desempeño).

**Lectura previa de esta sesión:** ["Machine learning model monitoring: Best practices"](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/), Datadog.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 8: Data Distribution Shifts and Monitoring.

## Requisitos antes de esta sesión

Nada nuevo que instalar. Los laboratorios de hoy son scripts. El Ejercicio 1 no necesita levantar uvicorn: `simular_noviembre.py` le manda las peticiones al servicio con `TestClient`, como las pruebas.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿por qué no alcanza con vigilar la salud y el volumen de peticiones del servicio? ¿Qué hay que medir además?**

_(tu respuesta)_

**El artículo dice que la verdad (ground truth) a veces llega con demora, o no llega. ¿Qué propone medir mientras tanto?**

_(tu respuesta)_

## Pregunta de apertura

Tu servicio responde en unos milisegundos, `/salud` dice que todo está bien, y cada tarifa dice qué versión la calculó. ¿Qué le pasa a un modelo que responde rápido pero cada vez peor? ¿Cómo te enterarías?

_(tu respuesta)_

## Ejercicio 1: la tarifa, instrumentada

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). Es el servicio de la tarifa con la versión 1 del modelo (la red de 64 neuronas), ahora con dos agregados. Un middleware (una función que corre alrededor de cada petición) mide cuánto tarda cada predicción y con qué código responde; `/metricas` resume esas mediciones. Y cada predicción queda guardada en `predicciones.jsonl`, con un identificador que la app guarda junto al viaje. [`simular_noviembre.py`](./starter/ejercicio1/simular_noviembre.py) manda al servicio los 1156 viajes de noviembre y escribe `cobros.csv`, lo que el sistema de cobros anotó al terminar cada viaje: el identificador de la predicción y la tarifa cobrada (no lo modifiques). [`desempeno.py`](./starter/ejercicio1/desempeno.py) une las dos cosas y calcula el error por semana. Los 4 `TODO`: medir la latencia, resumir las métricas, guardar cada predicción y calcular el error por semana.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
python simular_noviembre.py
python desempeno.py
pytest -v
```

**Respuesta 1 (antes de los TODO):** los 1156 viajes recibieron su tarifa. ¿Qué dicen `/metricas` y `desempeno.py`? ¿Qué podrías afirmar sobre el servicio en noviembre?

Nada. `/metricas` responde `{}`, `desempeno.py` termina con `No hay predicciones.jsonl: el servicio no esta guardando sus predicciones.`, y `pytest` da `2 failed`. Los 1156 viajes recibieron tarifa, pero no se sabe cuánto tardó cada una, si hubo errores (hubo 3 peticiones mal formadas y nadie las contó) ni si las tarifas se parecían a lo que después se cobró. Lo único que se puede afirmar es que el servicio respondió. No se puede decir si fue rápido, si falló alguna vez ni si acertó. Un servicio sin instrumentar está "funcionando" hasta que alguien se queja.

**Respuesta 2 (TODO 1 y 2):** antes de correr la simulación, ¿qué latencia p95 esperas, y cuántos errores de validación? ¿Esperas que la latencia cambie entre la primera y la última semana del mes?

Esperaba una latencia p95 de unos pocos milisegundos, porque la red de 64 neuronas sobre un viaje es una multiplicación de matrices pequeña, y **3 errores de validación**: las tres peticiones mal formadas que manda `simular_noviembre.py`, sin `fecha_hora_inicio` ni `zona_destino`. Errores de servidor, ninguno. Y la latencia **no debería cambiar** entre semanas: el modelo hace el mismo cálculo para cualquier viaje, sin importar la fecha ni si acierta.

Salió así: `{'peticiones': 1159, 'errores_validacion': 3, 'errores_servidor': 0, 'latencia_p50_ms': 2.57, 'latencia_p95_ms': 3.61}`. Medida por semana, la p95 fue de 2.58, 2.91, 2.90 y 2.93 ms: estable.

**Respuesta 3 (TODO 3):** el servicio devuelve `id_prediccion` y lo guarda en el registro. ¿Para qué lo necesita el sistema de cobros? ¿Qué pasaría si solo se guardaran la fecha y la tarifa estimada?

Es la llave para unir la estimación con la verdad. El sistema de cobros anota la tarifa cobrada junto al `id_prediccion` del viaje, y `desempeno.py` une las dos tablas por ese id: cada tarifa cobrada queda al lado de la estimación exacta que se le mostró a ese pasajero, con su versión de modelo y su entrada.

Con solo la fecha y la tarifa estimada no hay forma confiable de emparejarlas. En noviembre hay varios viajes en el mismo minuto (el primero y el segundo del archivo empiezan los dos el 2026-11-02 a las 00:14), y unir por fecha mezclaría estimaciones de un viaje con cobros de otro. Además, sin el viaje guardado no se podría ver *dónde* se equivoca el modelo (por zona, por distancia), y sin la versión no se sabría qué modelo culpar. Un pasajero que reclama da su id de viaje, y con el `id_prediccion` se encuentra su estimación en el registro.

**Respuesta 4 (TODO 4):** el MAE en prueba de la versión 1 fue 1.17. Antes de correr `desempeno.py`, ¿qué esperas para cada semana de noviembre?

Algo parecido a 1.17 en todas las semanas, con algo de ruido (unos 300 viajes por semana), quizás un poco más alto porque noviembre es más reciente que los datos de entrenamiento. Nada en las métricas del sistema hacía pensar otra cosa: misma latencia, cero errores de servidor.

No fue así: las dos primeras semanas dieron 1.18 y 1.34, pero desde la semana del 16 el MAE salta a **4.29 y 4.30, 3.7 veces la referencia**.

**Lo que mostró el monitoreo:**

`/metricas` después de `simular_noviembre.py`: **1159 peticiones, latencia p50 2.57 ms y p95 3.61 ms, 3 errores de validación, 0 de servidor**. La p95 por semana fue de 2.58, 2.91, 2.90 y 2.93 ms.

`desempeno.py`:

```
            viajes     mae  veces_la_referencia
semana
2026-11-02     291  1.1792                 1.01
2026-11-09     318  1.3403                 1.15
2026-11-16     308  4.2916                 3.67
2026-11-23     239  4.3042                 3.68
```

El cambio es del **16 de noviembre** en adelante: el error medio por día (cobrada − estimada) está cerca de 0 hasta el 15 (−0.34 a 0.58 del 12 al 15) y salta a +3.7 el 16, para quedarse cerca de +4. No es ruido, es un sesgo. En los viajes que no son de tarifa fija, la mediana de cobrada ÷ estimada pasa de 1.00 a **1.20**: se cobra un 20 % más que antes por el mismo viaje. Los viajes de tarifa fija siguen cobrándose 70. Las métricas de sistema no vieron nada: misma latencia y ningún error, mientras el modelo se equivocaba casi 4 veces más. `pytest -v`: 2 passed.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el fraude, cuando la verdad tarda tres semanas

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El servicio de fraude guardó los 5874 pagos que evaluó en noviembre en [`predicciones_fraude.csv`](./starter/ejercicio2/predicciones_fraude.csv): la probabilidad, si mandó el pago a revisión y cuánto tardó. Un fraude se confirma cuando el titular de la tarjeta reclama el cobro, entre 2 y 21 días después; los confirmados están en [`contracargos.csv`](./starter/ejercicio2/contracargos.csv). Hoy es 14 de diciembre. [`monitoreo_fraude.py`](./starter/ejercicio2/monitoreo_fraude.py) arma el reporte del mes. Los pasos 1 a 3 tienen `TODO`: la latencia p95 por día, la tasa de alertas por semana y los fraudes confirmados hasta hoy. Los pasos 4 (el recall de las semanas cuyos fraudes ya se confirmaron) y 5 (las alertas) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`; después, `python monitoreo_fraude.py`.

**Respuesta 1:** el reporte da tres tipos de alerta. ¿Cuáles son de sistema y cuáles de modelo? ¿El día de la latencia alta coincide con alguna semana de recall bajo?

```
- sistema: el 2026-11-10 la latencia p95 fue 78.2 ms
- modelo (indirecta): la semana del 2026-11-16 mando a revision 0.05 de los pagos
- modelo (indirecta): la semana del 2026-11-23 mando a revision 0.046 de los pagos
- modelo: la semana del 2026-11-16 detecto 0.276 de los fraudes
```

- **De sistema:** la latencia. El 10 de noviembre la p95 fue de 78.2 ms, contra unos 6 ms los demás días.
- **De modelo:** el recall, la métrica que le importa al negocio. La semana del 16 detectó el 27.6 % de los fraudes, contra 0.767 y 0.736 de las dos primeras semanas y 0.661 de referencia.
- **De modelo, indirecta:** la tasa de alertas, que bajó a 5.0 % y 4.6 % contra 7.58 % de referencia.

**No coinciden.** El día lento (10 de noviembre) cae en una semana con buen recall (0.736), y la semana de recall bajo (la del 16) tiene latencia normal. Son problemas distintos: uno del servicio, el otro del modelo. Por eso se vigilan por separado: con la latencia sola no se habría visto la caída de detección, y con el recall solo no se habría visto el día lento.

**Respuesta 2:** la tasa de alertas se puede calcular el mismo día; el recall, tres semanas después. ¿Qué semana mostró primero el problema la tasa de alertas, y cuándo lo confirmó el recall? ¿Por qué la tasa de alertas no alcanza para decir que el modelo empeoró?

La tasa de alertas lo mostró en la **semana del 16 de noviembre**: bajó de 0.0706 y 0.0685 a 0.0503, por debajo del 75 % de la referencia. Se pudo ver ya durante esa semana, día a día. El recall lo confirmó recién cuando esa semana estuvo madura, el domingo 22 más 21 días, es decir **el 13 de diciembre**: detectó 21 de 76 fraudes (0.276). Son unas tres semanas entre el primer aviso y la confirmación.

La tasa de alertas no alcanza porque mide cuántos pagos se mandan a revisión, no si eran fraude. Puede bajar por razones buenas: si llegan menos fraudes, o pagos menos riesgosos, que el modelo deje pasar más pagos es correcto. También puede quedar igual mientras el modelo marca los pagos equivocados. Solo comparando contra los fraudes confirmados (los contracargos) se sabe si los fraudes se escaparon. La tasa de alertas sirve para mirar antes, no para concluir.

**Respuesta 3:** ¿por qué el reporte no muestra el recall de la semana del 23 de noviembre, si ya hay fraudes confirmados de esa semana?

Porque todavía no está madura: su domingo es el 29 de noviembre, y 29 + 21 días es el **20 de diciembre**, después de hoy (14 de diciembre). Al 14 hay 66 fraudes confirmados de esa semana, pero en total llegan a haber 82 (el último contracargo se confirma el 20 de diciembre). Con los 66 de hoy, el recall parcial sería 26 de 66 = 0.394. Pero ese número no es comparable, porque los fraudes que faltan no son una muestra al azar: son los que tardan más en reclamarse, y bien podrían ser los que el modelo no marcó. Un recall con la verdad incompleta puede verse mejor o peor de lo que es. Por eso el reporte solo calcula el recall de semanas cuyo plazo máximo de confirmación ya pasó.

**Si terminas antes:** cambia `HOY` a `"2026-11-30"` y vuelve a correr. ¿Qué semanas tienen recall? ¿Desde qué día el reporte puede mostrar la caída del recall de la semana del 16?

Al 30 de noviembre solo tiene recall la **semana del 2 de noviembre** (73 fraudes, 56 detectados, 0.767), porque su domingo 8 + 21 = 29 de noviembre ≤ 30. La del 9 madura el 6 de diciembre, y la del 16, el **13 de diciembre**: desde ese día el reporte muestra su recall de 0.276 y la alerta `modelo: la semana del 2026-11-16 detecto 0.276 de los fraudes`. Lo comprobé recorriendo `HOY` día por día desde el 30 de noviembre: la semana del 16 aparece por primera vez el 13 de diciembre. Al 30 de noviembre, en cambio, el reporte ya mostraba las alertas indirectas de tasa de alertas de las semanas del 16 y del 23: el aviso temprano llega dos semanas antes que la confirmación. (Volví a dejar `HOY = "2026-12-14"`.)

## Ejercicio 3: el monitoreo de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el servicio de las dos sesiones anteriores. Esta vez no hay plantilla.

### Entregable 1: el servicio, instrumentado

- Un middleware que mida la latencia y el código de cada predicción, y una ruta `/metricas` con, al menos, la cantidad de peticiones, los errores de validación, los errores de servidor y la latencia p50 y p95.
- Cada predicción queda en un registro (un archivo JSONL en `monitoreo/`, por ejemplo) con su identificador, el momento, la versión del modelo, la entrada y la salida. Lo que haga falta para unirla después con la verdad tiene que estar en ese registro.
- Pruebas en `servicio/test_api.py` que comprueben que una predicción queda registrada y que `/metricas` separa los errores de validación de los de servidor, con un registro temporal (`tmp_path`).

### Entregable 2: `monitoreo/desempeno.py`

Une el registro con la verdad de tu proyecto y calcula la métrica de la Semana 11 por período (día o semana) y por versión. Si en tu proyecto la verdad no llega, o llega muy tarde, define una métrica que se pueda calcular antes y explica qué te dice y qué no.

### Entregable 3: `docs/monitoreo.md`

Una tabla con cada métrica que vigilas: si es de sistema o de modelo, de dónde sale, cuándo se conoce y con qué umbral alerta.

**¿Cuánto tarda en llegar la verdad en tu proyecto, y qué métrica te avisa mientras tanto?**

**7 días.** El lote predice cada noche las emergencias de cada cantón para *t + 7*, y la verdad son las emergencias de ese día, que se conocen al cierre de *t + 7*, cuando termina la carga diaria. Para una semana completa de predicciones, la verdad está 7 días después de su domingo. Cada predicción queda en `monitoreo/predicciones.jsonl` con `cod_canton` y `fecha_objetivo`, la llave para unirla con `n_emergencias` de ese cantón y ese día. `python -m monitoreo.desempeno <hoy>` calcula el MAE y el costo asimétrico de la Semana 11 por semana y por versión, solo con las predicciones cuya verdad ya se conoce.

**Mientras tanto avisa el indicador temprano:** la mediana de `emergencias_estimadas / media_prev_28d` de la noche, dividida por la de prueba para ese día de la semana. Se calcula la misma noche, sin la verdad. En prueba, ningún día normal se alejó más de 5 % de su referencia, y alerta fuera de ±20 % (los días con feriado en *t + 7* se excluyen, porque ahí el modelo sube a propósito). **Me dice** si el modelo empezó a responder distinto ante la misma historia: una feature mal calculada, un modelo mal exportado, una versión nueva que se comporta raro. **No me dice** si el modelo acierta. Si la realidad cambia y el modelo sigue respondiendo como siempre, el indicador queda en 1 y solo el MAE de 7 días después lo muestra.

En una simulación de 4 semanas (versión 1 y después versión 2, 6216 predicciones), el indicador se quedó entre 0.984 y 1.011. Con la verdad al 1 de septiembre:

| Semana | Versión | MAE | Costo asimétrico |
|---|---|---|---|
| 2026-07-27 | 1 | 5.82 | 14.52 |
| 2026-08-03 | 1 | 5.52 | 13.30 |
| 2026-08-10 | 2 | 5.33 | 11.75 |
| 2026-08-17 | 2 | 4.84 | 11.09 |

Hubo una alerta: la versión 1 en su primera semana, más de 20 % sobre el costo de referencia. La tabla completa está en `docs/monitoreo.md`.

**¿Qué falla de tu modelo no vería ninguna de tus métricas de sistema?**

Que estime mal sin dejar de responder. El servicio puede responder en 4 ms, con 0 errores de validación y de servidor, la versión correcta y los 222 cantones de cada noche, y aun así quedarse corto en las emergencias. Por ejemplo:

- Si cambia cómo se registran las emergencias (un canal nuevo de reporte, una reclasificación de tipos) o si un evento cambia la relación entre la historia de 28 días y la semana siguiente (un período como enero de 2024, la temporada de lluvias en la Costa), el modelo devuelve números con la misma forma de siempre y el sistema lo ve todo bien. Es lo que pasó con la tarifa del Ejercicio 1: misma latencia, cero errores y 3.7 veces más error.
- Si el lote manda datos válidos pero equivocados, como un lag desplazado un día, el esquema no los rechaza.

El primer caso solo lo ven el MAE y el costo asimétrico contra la verdad, 7 días después. El segundo lo puede ver antes el indicador temprano, si cambia la relación entre lo que entra y lo que sale.

## Antes de la próxima sesión

**Lectura:** ["Data Drift vs. Concept Drift and Why Monitoring for Them is Important"](https://medium.com/whylabs/data-drift-vs-concept-drift-and-why-monitoring-for-them-is-important-a68c08b141f0), WhyLabs.

Desde el 16 de noviembre, el modelo de la tarifa se equivoca casi cuatro veces más que en prueba. Los viajes que recibe el servicio son del mismo tipo que los de octubre: las mismas zonas, las mismas distancias, las mismas horas. ¿Qué cambió, entonces? ¿Lo hubieras detectado mirando solo los viajes que llegan?

_(tu respuesta)_
