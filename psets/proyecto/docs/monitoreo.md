# Monitoreo del modelo en producción

El modelo estima, cada noche, las emergencias de los 222 cantones para el día *t + 7*. Se vigila en dos frentes:

- **El sistema:** si el servicio responde, rápido y sin errores. Sale de `/metricas` del servicio y del registro de predicciones.
- **El modelo:** si sus estimaciones siguen siendo buenas. Necesita la verdad, que llega 7 días después, y mientras tanto hay un indicador temprano.

Cada predicción queda en `monitoreo/predicciones.jsonl` (una línea por predicción) con:
- su `id_prediccion` y el `momento` en que se hizo, con zona horaria;
- la `version_modelo`;
- la entrada completa: `cod_canton`, `fecha`, `region`, los lags, `media_prev_28d` y `es_feriado_t7`;
- la salida: `fecha_objetivo` y `emergencias_estimadas`.

La llave para unirla con la verdad es **`cod_canton` + `fecha_objetivo`**. `python -m monitoreo.desempeno <hoy>` hace esa unión y calcula las métricas.

## Las métricas

| Métrica | Tipo | De dónde sale | Cuándo se conoce | Alerta si |
|---|---|---|---|---|
| Peticiones | Sistema | `/metricas` (`peticiones`) | En el momento | La noche termina con 0 peticiones: el lote no corrió |
| Predicciones por noche | Sistema | Registro: predicciones por `fecha` | La misma noche | Menos de **222**: el lote quedó incompleto y hay cantones sin estimación |
| Errores de validación (422) | Sistema | `/metricas` (`errores_validacion`) | En el momento | **Más de 0** en el lote. El lote manda datos de la OBT, así que un 422 significa que cambió la carga de datos (un lag negativo, una columna que falta) |
| Errores de servidor (≥ 500) | Sistema | `/metricas` (`errores_servidor`) | En el momento | **Más de 0**: el modelo o el servicio fallaron con datos válidos |
| Latencia p50 y p95 | Sistema | `/metricas` (`latencia_p50_ms`, `latencia_p95_ms`) | En el momento | p95 **mayor que 100 ms** en una noche completa. Medido: 4.75 ms en 6218 peticiones. La primera petición después de arrancar el contenedor tardó 93 ms (arranque en frío), así que no se alerta con pocas peticiones. Para un lote de 222 cantones la latencia casi no importa, pero un salto indica un problema del contenedor |
| Versión servida | Sistema | `/salud` y cada respuesta (`version_modelo`) | En el momento | Es **distinta** de la versión con el alias `produccion` en el registro: se movió el alias y no se volvió a desplegar |
| Indicador temprano: estimada ÷ `media_prev_28d` | Modelo (indirecta) | Registro, sin verdad | **La misma noche** | La mediana del día, relativa a la de prueba para ese día de la semana, sale de **1 ± 0.20** en un día sin feriado en *t + 7*. En prueba ningún día normal se alejó más de 5 %. Los días con feriado se excluyen, porque ahí el modelo sube a propósito |
| MAE por semana y versión | Modelo | Registro + emergencias reales (`n_emergencias` del cantón en `fecha_objetivo`) | **7 días después** de cada predicción (al cierre de *t + 7*). La semana completa, 7 días después de su domingo | Es **mayor que 1.2 × 5.549** (el MAE de prueba de la versión 2) |
| Costo asimétrico por semana y versión (3 × faltante + 1 × sobrante) | Modelo (la de negocio) | Registro + emergencias reales | Igual que el MAE | Es **mayor que 1.2 × 11.767** (el costo de prueba de la versión 2) |
| Sesgo (estimada − real) | Modelo | Registro + emergencias reales | Igual que el MAE | Sin umbral: se reporta para ver hacia dónde se equivoca. Negativo significa que el modelo se queda corto, que es el error caro |

## Lo que mostró una simulación

`python -m monitoreo.simular_lote 2026-07-27 2026-08-09:1 2026-08-23:2` manda al servicio los 222 cantones de cada noche durante 4 semanas: la versión 1 hasta el 9 de agosto y la versión 2 desde el 10, como si se hubiera promovido ese día. Son 6216 predicciones y 2 peticiones mal formadas.

`/metricas`: 6218 peticiones, 2 errores de validación, 0 de servidor, latencia p50 3.36 ms y p95 4.75 ms.

`python -m monitoreo.desempeno 2026-09-01`:

| Semana | Versión | Predicciones | MAE | Costo asimétrico | Sesgo |
|---|---|---|---|---|---|
| 2026-07-27 | 1 | 1554 | 5.818 | **14.521** ⚠ | −2.884 |
| 2026-08-03 | 1 | 1554 | 5.524 | 13.299 | −2.251 |
| 2026-08-10 | 2 | 1554 | 5.331 | 11.746 | −1.084 |
| 2026-08-17 | 2 | 1554 | 4.840 | 11.094 | −1.414 |

- La única alerta es el costo de la versión 1 en su primera semana, por encima del 20 % de tolerancia sobre el costo con que se aprobó la versión 2. Con la versión 2, el costo y el sesgo bajan: se queda corta menos veces.
- El indicador temprano se mantuvo entre 0.984 y 1.011 todas las semanas.
- Al 2026-08-20, la semana del 10 de agosto solo tiene la verdad de 666 de sus 1554 predicciones, y de sus primeros días: el MAE parcial de una semana sin terminar no se compara con el de una semana completa.

## Lo que no ve ninguna métrica de sistema

El servicio puede responder en 4 ms, con 0 errores y la versión correcta, y aun así estimar mal. Por ejemplo, si cambia la forma en que se registran las emergencias (un canal nuevo de reporte, una reclasificación) o si un evento cambia la relación entre la historia y el futuro. Para eso están las métricas de modelo, y la buena llega 7 días tarde. El indicador temprano solo avisa si el modelo responde distinto a como respondía ante la misma historia (una feature mal calculada, un modelo mal exportado). No avisa si la realidad cambió y el modelo sigue respondiendo igual.

## Deriva

`python -m monitoreo.deriva_proyecto` mide la deriva con los datos de prueba (desde 2025-08-25) y el modelo en producción (versión 2), y guarda la referencia que usa el servicio en `servicio/referencia_deriva.json`.

- **Features vigiladas:** tres, cada una repartida en los deciles del entrenamiento.
  - `log_media_mismo_dia` y `log_media_prev_28d`, las de más peso en el gradient boosting (importancia 0.68 y 0.30).
  - `log_lag_1d`, que casi no pesa (0.01), pero es el dato más reciente: el primero que falla si la carga diaria no terminó antes del lote.
- **Ventana:** una semana de lote, **7 noches × 222 cantones = 1554 predicciones**. Una sola noche se mueve demasiado: `log_lag_1d` del lunes mira un domingo, y el día después de un feriado el PSI de una noche llegó a 0.35 (el 2 de enero), sin que nada estuviera mal. En semanas completas, el PSI normal no pasa de 0.038.

### Lo que mostró la simulación

Promedio de las 52 semanas completas del período de prueba, en cada escenario:

| Escenario | Tipo | PSI `media_mismo_dia` | PSI `media_prev_28d` | PSI `lag_1d` | MAE | Costo asimétrico | Semanas con alerta (0.03) |
|---|---|---|---|---|---|---|---|
| Normal (máximo de 52 semanas: 0.012, 0.038 y 0.019) | — | 0.006 | 0.012 | 0.006 | 5.55 | 11.77 | 1.9 % (1 semana) |
| Carga incompleta: `lag_1d` llega con el 90 % | Datos | 0.006 | 0.012 | 0.071 | 5.60 | 12.41 | 100 % |
| Carga incompleta: `lag_1d` con el 50 % | Datos | 0.006 | 0.012 | 0.261 | 6.32 | 16.10 | 100 % |
| Lluvias en la Costa × 1.3 | Datos | 0.008 | 0.016 | 0.010 | 6.40 | 13.89 | 9.6 % |
| Lluvias en la Costa × 1.5 | Datos | 0.014 | 0.025 | 0.017 | 7.51 | 17.15 | 26.9 % |
| Más emergencias en el país × 1.2 | Datos | 0.025 | 0.033 | 0.046 | 7.21 | 16.63 | 98.1 % |
| Más emergencias en el país × 1.5 | Datos | 0.092 | 0.117 | 0.119 | 11.04 | 27.28 | 100 % |
| Cambio de registro: objetivo × 1.1 | Conceptual | 0.006 | 0.012 | 0.006 | 7.30 | 19.29 | 1.9 % |
| Cambio de registro: objetivo × 1.25 | Conceptual | 0.006 | 0.012 | 0.006 | 12.06 | 34.85 | 1.9 % |

- **Carga incompleta:** el lote corre antes de que termine la carga del día *t*, y `lag_1d` llega con parte de las emergencias de ayer. Es la deriva de datos más probable en un lote nocturno. Daña poco, porque el modelo casi no usa `lag_1d`, pero dice que el dato no está completo.
- **Lluvias en la Costa:** los cantones de la Costa tienen más emergencias, en la historia y en el futuro.
- **Más emergencias en el país:** todos los cantones tienen más emergencias, en la historia y en el futuro.
- **Cambio de registro:** desde un día, el sistema cuenta como emergencia lo que antes no contaba. La historia que ve el modelo es la de antes y el objetivo, el de después. Es deriva conceptual. El 1.9 % de alertas de las dos filas de cambio de registro es la misma semana normal de la falsa alarma: las features son las de siempre.

Las derivas de datos de "más emergencias" mantienen la relación entre historia y futuro, y aun así el costo sube: un cantón con más emergencias que todas las que vio el modelo queda en la punta de los árboles, y el gradient boosting no extrapola. Por eso esa deriva hay que verla, aunque la relación no cambie.

### El umbral: 0.03

| Umbral | Falsas alarmas (52 semanas normales) | Detección: carga incompleta al 75 % | Detección: país × 1.2 |
|---|---|---|---|
| 0.02 | 11.5 % | 100 % | 100 % |
| **0.03** | **1.9 % (1 semana)** | **100 %** | **98.1 %** |
| 0.05 | 0 % | 98.1 % | 46.2 % |
| 0.10 | 0 % | 75.0 % | 0 % |

Lo justifican dos números. Con 0.03 hay **una falsa alarma en un año** (la semana del 18 de mayo de 2026, con PSI 0.038 en `log_media_prev_28d`), y se detecta el **98 %** de las semanas con 20 % más emergencias en el país, una deriva que sube el costo de 11.77 a 16.63. Con 0.05 no hay falsas alarmas, pero esa deriva se escapa en más de la mitad de las semanas (46 %). Revisar una alarma al año cuesta mucho menos que una semana planificada con un modelo que se queda corto.

El servicio responde en `GET /deriva` el PSI de las tres features en las últimas 1554 predicciones, y solo alerta con la ventana llena, porque con pocas predicciones el PSI es ruido: con una sola predicción, el PSI pasó de 8.

### Lo que esta alerta no detecta

- **La deriva conceptual:** con el registro × 1.25 el costo casi se triplica (34.85) y el PSI no se mueve, porque lo que llega al modelo es igual que siempre. Solo la ven el MAE y el costo contra la verdad, 7 días después (`monitoreo/desempeno.py`).
- **La deriva de una parte de los cantones:** con lluvias en la Costa × 1.5 el costo sube a 17.15 y la alerta se enciende solo en 27 % de las semanas. La Costa es una parte de los 222 cantones, y su aumento apenas mueve la distribución del país. Un PSI por región lo vería antes.
- **Las features no vigiladas:** `es_feriado_t7`, la región o el día de la semana. Un feriado mal cargado en el calendario no movería ninguno de los tres PSI.

**La regla:**

- Si la alerta de deriva se enciende y el costo no cambia, es deriva de datos que el modelo absorbe: se revisa la carga de datos (¿terminó antes del lote?) y se espera la verdad.
- Si el costo sube sin alerta de deriva, es deriva conceptual, y hay que reentrenar con datos recientes.
- Si suben los dos, es deriva de datos que daña. Primero se arregla la carga, si ese era el problema. Si los datos están bien (el país de verdad tiene más emergencias), se reentrena con datos recientes para que el modelo vea esos niveles.
