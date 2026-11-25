# Semana 14, Sesión 1: Monitoreo en producción II: deriva de datos y deriva conceptual

> Objetivo de la sesión: distinguir deriva de datos de deriva conceptual en el modelo de tu proyecto, y diseñar una alerta que detecte cuál de las dos lo está degradando.

**Lectura previa de esta sesión:** ["Data Drift vs. Concept Drift and Why Monitoring for Them is Important"](https://medium.com/whylabs/data-drift-vs-concept-drift-and-why-monitoring-for-them-is-important-a68c08b141f0), WhyLabs.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 9: Continual Learning and Test in Production.

## Requisitos antes de esta sesión

Nada nuevo que instalar. El Ejercicio 1 es un notebook (simula la deriva y elige el umbral); el Ejercicio 2, un script del servicio (la alerta queda ahí, corriendo con cada predicción).

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la deriva de datos y qué es la deriva conceptual? Da un ejemplo de cada una.**

_(tu respuesta)_

**¿Por qué el artículo dice que hay que monitorear las dos, y no solo el desempeño del modelo?**

_(tu respuesta)_

## Pregunta de apertura

Desde el 16 de noviembre, el modelo de la tarifa se equivoca casi cuatro veces más que en prueba. Los viajes que recibe son del mismo tipo que los de octubre: las mismas zonas, las mismas distancias, las mismas horas. ¿Qué cambió, entonces? ¿Lo hubieras detectado mirando solo los viajes que llegan?

_(tu respuesta)_

## Ejercicio 1: ¿deriva de datos o deriva conceptual?

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). [`ejercicio1_deriva.ipynb`](./starter/ejercicio1/ejercicio1_deriva.ipynb) entrena la versión 1 del modelo de la tarifa y simula semanas de 300 viajes: semanas normales, semanas de feriado con más viajes al aeropuerto (deriva de datos) y semanas con un aumento de la tarifa por taxímetro (deriva conceptual). Para cada una mide el PSI de dos features contra los viajes de entrenamiento y el MAE del modelo. Los 4 `TODO`: la fórmula del PSI, medir 200 semanas normales, y las dos derivas simuladas.

Corre todas las celdas antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1 y 2):** antes de medir las semanas normales, ¿qué PSI esperas para una semana en la que nada cambió? Después de medirlas: ¿cuál fue el máximo en 200 semanas?

Esperaba un PSI **cercano a 0, pero no 0**. Con 300 viajes repartidos en 10 casillas, cada una tiene unos 30 viajes y su proporción varía por azar alrededor del 10 %. Calculaba algo entre 0.02 y 0.05. La prueba completa contra entrenamiento, con muchos más viajes, ya da 0.017 en distancia y 0.009 en zona.

En las 200 semanas normales el PSI promedio fue **0.050 en distancia y 0.035 en zona**, el percentil 95 fue 0.092 y 0.080, y el **máximo fue 0.121 en distancia y 0.109 en zona**. Ese máximo es el piso del umbral: una alerta por debajo de 0.12 sonaría en semanas en las que nada cambió. El MAE de esas semanas fue 1.16 en promedio y 1.32 como máximo.

**Respuesta 2 (TODO 3 y 4):** antes de correr, ¿qué esperas que pase con el PSI y con el MAE en la semana de feriado? ¿Y en la semana del aumento?

- **Feriado:** esperaba que el PSI subiera en las dos features, con más viajes desde y hacia JFK (más distancias largas y más JFK como origen), y que el MAE casi no cambiara, porque el modelo ya aprendió la tarifa fija del aeropuerto. Salió así, y el MAE incluso bajó: con peso 4, PSI 0.189 en distancia y 0.196 en zona, y MAE 0.969. Los viajes al aeropuerto con tarifa fija de 70 son los más fáciles de estimar.
- **Aumento:** esperaba lo contrario, un PSI igual al de una semana normal (los viajes son los mismos) y un MAE mucho mayor. Salió así: PSI 0.077 y 0.062, dentro de lo normal, y MAE **4.16**, 3.6 veces el de prueba. Es lo que se vio en noviembre en la sesión pasada.

**Respuesta 3 (Paso 4):** en la semana de feriado con peso 8, el PSI es más de diez veces el de una semana normal. ¿Empeoró el modelo? ¿Por qué?

**No, mejoró.** Con peso 8 el PSI es 0.542 en distancia y 0.419 en zona, unas diez veces el de una semana normal (0.051 y 0.033), y el **MAE baja de 1.19 a 0.94**. El PSI mide cuánto cambió *lo que llega*, no si el modelo se equivoca. En una semana de feriado llegan más viajes al aeropuerto, pero el modelo los vio al entrenar y los estima muy bien: la mayoría tiene tarifa fija de 70 (`es_tarifa_fija`), el caso más fácil. Más viajes fáciles significa menos error promedio. Un PSI alto dice "esta semana no se parece al entrenamiento", y eso vale la pena saberlo, pero no dice que el modelo esté mal.

**Respuesta 4 (Paso 5):** ¿qué umbral de PSI elegirías para la alerta, y con qué dos números de la tabla lo justificas?

**0.15.** Con 0.15 las falsas alarmas son **0 %** en las 200 semanas normales, y la detección de la semana de feriado con peso 4 es **97.5 %**. Con 0.10 la detección es del 100 %, pero hay 4 % de falsas alarmas: una semana normal de cada 25 sonaría sin motivo. Con 0.20 no hay falsas alarmas, pero la detección cae a 71.5 %. El 0.15 queda por encima del máximo normal (0.121) y por debajo de casi todas las semanas de feriado. La alerta no mira el aumento de 20 %, que con PSI de 0.046 está en el rango normal: ningún umbral de PSI lo detectaría.

**Lo que mostró la simulación:**

| Escenario | PSI de distancia | PSI de zona | MAE |
|---|---|---|---|
| Semanas normales (máximo de 200) | 0.1214 | 0.1091 | 1.3164 |
| Feriado, aeropuerto ×4 | 0.2211 | 0.1965 | 1.0441 |
| Aumento de 20% | 0.0456 | 0.0426 | 4.0505 |

Las dos derivas son promedios de 10 semanas (Paso 4). La deriva de datos mueve el PSI y no empeora el modelo. La deriva conceptual empeora el MAE 3.4 veces y no mueve el PSI.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: la alerta, en el servicio

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El servicio de la tarifa de la sesión pasada tiene ahora una ruta `/deriva`. [`deriva.py`](./starter/ejercicio2/deriva.py) trae `MonitorDeDeriva`: guarda los últimos 300 viajes que recibió el servicio y calcula el PSI de `distancia_km` y de `zona_origen` contra la referencia que guarda `entrenar.py` (`referencia_deriva.json`), con el umbral de 0.15. Los pasos 1 a 3 tienen `TODO` (las proporciones numéricas, las categóricas y el PSI); los pasos 4 (guardar solo los últimos viajes) y 5 (el estado, con la alerta) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`. Después:

```
python entrenar.py
python simular_semanas.py
```

[`simular_semanas.py`](./starter/ejercicio2/simular_semanas.py) manda al servicio una semana normal, una de feriado y una del aumento, y muestra `/deriva` después de cada una (no lo modifiques).

**Respuesta 1:** ¿qué dijo `/deriva` después de cada semana? ¿Alguna alerta te sorprendió, o te faltó?

```
Despues de la semana normal: {'viajes_en_ventana': 300, 'psi': {'distancia_km': 0.0534, 'zona_origen': 0.0546}, 'alerta': False}
Despues de la semana feriado: {'viajes_en_ventana': 300, 'psi': {'distancia_km': 0.258, 'zona_origen': 0.2211}, 'alerta': True}
Despues de la semana aumento: {'viajes_en_ventana': 300, 'psi': {'distancia_km': 0.0578, 'zona_origen': 0.0477}, 'alerta': False}
```

Ninguna sorpresa respecto del notebook, pero **la alerta está al revés de lo que importa**. Midiendo el MAE de cada semana con la tarifa cobrada: normal **1.15**, feriado **1.04**, aumento **4.07**. La única alerta fue en la semana en que el modelo anduvo mejor, y no hubo alerta en la semana en que se equivocó casi cuatro veces más. Me faltó la alerta del aumento, aunque sabía que el monitor no podía verla. Como la ventana guarda solo los últimos 300 viajes, después de la semana del aumento ya no quedaba nada de la de feriado: la alerta se apagó sola. `comprobar.py`: los 5 pasos `OK`.

**Respuesta 2:** la alerta del monitor no dijo nada en la semana del aumento, que es la que más empeora el modelo. ¿Qué alerta de la sesión pasada sí la habría visto, y por qué tarda más?

La del **MAE por semana** de `desempeno.py`: une cada predicción con la tarifa cobrada y alerta cuando el error se aleja de la referencia. En la sesión pasada, la semana del 16 de noviembre dio 4.29, 3.67 veces la referencia. Tarda más porque **necesita la verdad**: la tarifa cobrada existe solo cuando el viaje termina y el sistema de cobros la anota, y el MAE de una semana necesita los cobros de toda la semana. El PSI se calcula con lo que llega, en el momento. El MAE, con lo que pasó después. En la tarifa la demora es de horas o un día. En el fraude de la sesión pasada eran tres semanas.

**Respuesta 3:** escribe la regla que usaría el equipo para decidir, con las dos alertas (la de deriva de este ejercicio y la del MAE de la sesión pasada), si lo que pasa es deriva de datos, deriva conceptual o nada. ¿Qué haría en cada caso?

| Alerta de deriva (PSI > 0.15) | Alerta de MAE (> 1.2 × referencia) | Diagnóstico | Qué hace el equipo |
|---|---|---|---|
| No | No | Nada | Nada. |
| **Sí** | No | **Deriva de datos que el modelo absorbe** (como el feriado) | Anota la causa (feriado, evento) y sigue mirando el MAE cuando llegue la verdad. No reentrena. |
| No | **Sí** | **Deriva conceptual** (como el aumento) | Cambió la relación entre los viajes y la tarifa. Averigua qué pasó (una tarifa nueva, un recargo) y reentrena con viajes posteriores al cambio, o corrige la salida si el cambio es conocido y uniforme (×1.20 en las tarifas sin tarifa fija). |
| **Sí** | **Sí** | **Deriva de datos que daña** | Llegan viajes que el modelo no sabe estimar. Revisa qué tipo de viaje cambió, si hay un error en los datos de entrada, y reentrena incluyendo esos viajes. |

Mientras la verdad no llega, la alerta de deriva sola no basta para reentrenar: solo dice que algo cambió y hay que mirar.

**Si terminas antes:** el monitor mira solo dos features. Inventa una semana con deriva de datos que este monitor no vería. ¿Empeoraría el modelo?

**Una semana de madrugadas:** los viajes con hora de inicio entre las 22:00 y las 2:00 (`hora_cos > 0.7`, el 20 % de la prueba) tienen 8 veces más probabilidad de salir, por ejemplo la semana de un festival nocturno. La distancia y la zona de origen siguen iguales, así que el monitor no la ve: en 10 semanas simuladas, PSI de 0.049 en distancia y 0.053 en zona, **0 alertas**. Lo mismo pasa con una semana con 8 veces más pagos en efectivo (0 alertas) o más viajes en hora pico (0 alertas).

¿Empeoraría el modelo? **Poco:** el MAE sube de 1.17 a **1.23** con las madrugadas y a 1.20 con el efectivo. El modelo entrenó con esas horas y esos métodos de pago, así que los sabe estimar. Una deriva que no ve el monitor no es necesariamente un problema, pero tampoco hay forma de saberlo sin la verdad. Para cubrirla habría que agregar `hora_sin`/`hora_cos` y `metodo_pago` al monitor, con el riesgo de más falsas alarmas.

## Ejercicio 3: la deriva de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el servicio instrumentado de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: la simulación, en `monitoreo/`

Un notebook o un script que, con tus datos de prueba:

- mida el PSI de al menos dos de tus features en ventanas normales (del tamaño de lo que recibe tu servicio en un período), para saber cuánto se mueve sin que nada cambie;
- simule una deriva de datos y una deriva conceptual que tengan sentido en tu problema, y mida el PSI y tu métrica de la Semana 11 en cada una;
- elija el umbral de la alerta con esos números, y guarde la referencia que necesita el servicio.

### Entregable 2: la alerta, en el servicio

Un monitor de deriva en tu servicio (puedes adaptar `deriva.py`) con una ruta que diga el PSI de la ventana actual y si hay alerta, y una prueba en `servicio/test_api.py` que compruebe que la alerta se enciende con una ventana alejada de la referencia.

### Entregable 3: `docs/monitoreo.md`

Agrega la sección **Deriva**: la tabla de la simulación, el umbral elegido y por qué, y qué deriva no detecta tu alerta.

**En tu proyecto, ¿qué deriva de datos es la más probable, y empeoraría tu modelo? ¿Y qué deriva conceptual?**

**Deriva de datos más probable:** la **carga incompleta**. El lote corre de noche, y si la carga de emergencias del día *t* no terminó, `lag_1d` llega con menos emergencias de las que hubo. Empeora poco (con `lag_1d` al 75 % el costo pasa de 11.77 a 13.80), porque el modelo casi no usa `lag_1d` (importancia 0.01): se apoya en `log_media_mismo_dia` (0.68) y `log_media_prev_28d` (0.30). La alerta la detecta en el 100 % de las semanas, y sirve más como aviso de un problema de datos que del modelo.

La que **más daña** es otra deriva de datos: **más emergencias en el país**, por crecimiento de la población o una temporada de lluvias que se extiende. Con 20 % más emergencias en todos los cantones, en la historia y en el futuro, la relación es la misma, pero el costo sube de 11.77 a **16.63**: el gradient boosting no extrapola, y los cantones con niveles nunca vistos se quedan en la punta de los árboles. La alerta la detecta en el 98 % de las semanas.

**Deriva conceptual:** un **cambio en cómo se registran las emergencias**. Por ejemplo, el ECU 911 empieza a contar como emergencia un tipo de llamada que antes no contaba. La historia que ve el modelo es la de antes y el objetivo, el de después. Con 25 % más registros el costo casi se triplica (**34.85**), y el PSI no se mueve (0.006 y 0.012, como una semana normal).

La simulación (`monitoreo/deriva_proyecto.py`) usa ventanas de una semana de lote (1554 predicciones) y tres features. El máximo normal en 52 semanas fue 0.038, y el umbral elegido es **0.03**: 1 falsa alarma al año, contra 98 % de detección de la deriva del país. La tabla completa está en `docs/monitoreo.md`, sección **Deriva**. El servicio responde en `GET /deriva`, y 6 pruebas nuevas en `servicio/test_api.py` comprueban que la alerta se enciende con una ventana alejada de la referencia, que no se enciende con la ventana incompleta y que se apaga cuando la ventana vuelve a ser normal. En total, `pytest psets/proyecto` da 23 passed.

**Si tu alerta de deriva se enciende y tu métrica de desempeño no cambia, ¿qué haces?**

No reentreno. Es deriva de datos que el modelo absorbe, y la alerta me dice que mire los datos, no el modelo:

1. **Miro qué feature se movió.** Si es solo `log_lag_1d`, lo más probable es la carga incompleta: reviso si la carga del día terminó antes del lote, porque es un problema de datos aunque todavía no se note en el costo (con 90 % de los datos el costo apenas sube, de 11.77 a 12.41). Si se movieron `log_media_mismo_dia` y `log_media_prev_28d`, cambió el nivel de emergencias, y eso termina dañando.
2. **Espero la verdad de esas semanas** (7 días) y confirmo con `monitoreo/desempeno.py` que el MAE y el costo siguen dentro del 20 % de su referencia.
3. **Si se sostiene**, anoto la causa y, si el cambio es permanente (por ejemplo, la distribución de emergencias del país cambió y el modelo igual acierta), regenero la referencia de deriva con datos recientes, para que la alerta deje de sonar por algo que ya se sabe.

Si en cambio el costo sube después, paso a la regla de deriva de datos que daña: arreglar los datos si estaban mal, o reentrenar con datos recientes si el cambio es real.

## Antes de la próxima sesión

**Lectura:** ["How to Effectively Communicate Results to Stakeholders As a Data Scientist"](https://medium.com/@sahin.samia/how-to-effectively-communicate-results-to-stakeholders-as-a-data-scientist-711e88dd464f), Sahin Samia.

En las últimas cuatro sesiones construiste un servicio, un registro, métricas y alertas, con PSI, MAE, p95 y recall. ¿Qué cambia entre explicarle tu proyecto a otro científico de datos y a quien paga la factura? Escribe la primera oración que le dirías a cada uno.

_(tu respuesta)_
