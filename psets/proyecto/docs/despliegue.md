# Despliegue del modelo: por lotes

**Decisión: el modelo se sirve por lotes (batch), una vez al día.** Cada noche, después de cerrar el día *t*, se estima cuántas emergencias tendrá cada cantón el día *t + 7* y se guarda una tabla con una fila por cantón. Quien planifica la semana siguiente lee esa tabla. Nadie espera a que el modelo responda.

## Qué decisión habilita y cuánto puede esperar

- **Decisión:** cuántos recursos (turnos de personal, unidades de respuesta) asignar a cada cantón para un día de la semana próxima. Subestimar cuesta más que sobreestimar (en la Semana 11 lo medí con un costo de 3 a 1): un cantón con pocos recursos responde tarde.
- **Quién la usa:** el equipo que arma la planificación semanal de turnos por cantón. No la usa un ciudadano ni un operador durante una llamada.
- **Cuánto puede esperar:** la predicción es para dentro de 7 días, y los turnos se arman con días de anticipación. Una tabla que esté lista a la mañana siguiente basta: la latencia aceptable es de **horas (hasta unas 12 horas, de la medianoche al inicio de la jornada)**, no milisegundos.

## Cuántos casos hay y si se conocen de antemano

- **222 cantones por día**, siempre los mismos. El conjunto completo de casos se conoce de antemano: no aparece un cantón nuevo a mitad del día.
- Con el gradient boosting elegido en la Semana 11, estimar los 222 cantones toma unos pocos milisegundos (medido: alrededor de 4 ms). El costo de calcularlos todos cada noche es despreciable, y no hay casos que haya que predecir "al vuelo".

## Qué datos necesitan las features y cuándo existen

Las features de `modelado/pipeline_proyecto.py` usan solo información conocida al cierre del día *t*:

| Feature | De dónde sale | ¿Disponible al predecir? |
|---|---|---|
| `log_lag_1d` | emergencias del día *t − 1* | Sí, desde el cierre del día |
| `log_media_mismo_dia` | media de los lags de 7, 14 y 28 días | Sí |
| `log_media_prev_28d` | media de los 28 días previos | Sí |
| `region`, `nombre_dia` | catálogo de cantones y calendario | Sí, se conocen siempre |
| `es_fin_de_semana`, `es_feriado_t7` | calendario (feriado del día *t + 7*) | Sí, el calendario de feriados se publica con anticipación |

Ninguna feature necesita algo que solo exista en el momento de una consulta. Los lags dependen de que la carga diaria de emergencias haya terminado, por eso el lote corre de noche, después de esa carga.

## Qué se pierde con el patrón descartado (en línea)

- **Lo que daría el servicio en línea:** una predicción actualizada al minuto, por ejemplo con las emergencias de la mañana del mismo día. Pero el modelo predice a 7 días y sus features son diarias. Una consulta a las 15:00 devolvería lo mismo que la tabla de la noche anterior.
- **Lo que se pierde con el lote:** si ocurre algo extraordinario durante el día (un desastre, un evento masivo), la tabla no lo refleja hasta la noche siguiente. Para eso no sirve este modelo: esos casos se atienden con la operación en tiempo real, no con la planificación de la semana próxima.
- **Lo que se ahorra:** no hay que mantener un servicio encendido todo el día, con su monitoreo de latencia y disponibilidad. Basta un trabajo programado que corra una vez y deje la tabla con la versión del modelo en cada fila, como en el laboratorio de la Semana 12.

**El número que decide:** la planificación puede esperar **horas** por la predicción, y los **222 casos** se conocen cada noche. Por lotes alcanza.
