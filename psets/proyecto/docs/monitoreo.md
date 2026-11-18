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
