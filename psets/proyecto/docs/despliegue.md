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

## Versionado y registro

El modelo vive en el registro de MLflow como **`emergencias-cantones`**. Cada vez que se reentrena (`python -m servicio.registrar_modelo <max_depth>`, desde `psets/proyecto/`) se crea una versión nueva con su corrida: los parámetros, la huella del archivo de datos, el corte de la división y sus métricas. Ninguna versión reemplaza a otra.

| Versión | Modelo | MAE de prueba | Costo asimétrico (3 × faltante + 1 × sobrante) | Alias |
|---|---|---|---|---|
| 1 | gradient boosting `max_depth=3` | 5.7118 | 12.3553 | |
| 2 | gradient boosting `max_depth=5` | **5.5490** | **11.7665** | `produccion` |

**En producción: versión 2**, el mismo modelo final de la Semana 11 (MAE de prueba 5.5490, igual al de la corrida `05dac271…`).

### Criterio para mover el alias

Una versión nueva pasa a `produccion` solo si cumple las tres condiciones:

1. Se entrenó con la misma división temporal (`corte_division` y `separacion_dias` en su corrida), para que las métricas sean comparables.
2. Tiene **menor costo asimétrico** que la versión en producción. Es la métrica de negocio: quedarse corto con los recursos de un cantón cuesta más que sobrar.
3. Su **MAE de prueba no empeora**. Si el costo baja porque el modelo sobreestima en todos lados, no es una mejora.

La decisión la toma quien mantiene el proyecto (yo), con `python -m servicio.promover` sin argumentos, que lista las versiones con sus dos métricas. La decisión queda registrada en el propio registro: el alias dice qué versión sirve, y la etiqueta `anterior_en_produccion` de cada versión promovida dice a cuál reemplazó.

### Cómo sabe el servicio qué versión sirve

`python -m servicio.exportar_modelo` exporta la versión con el alias `produccion` a `servicio/modelo.joblib` y escribe su número en `servicio/modelo_version.txt`. La imagen lleva los dos archivos, así que el contenedor no necesita llegar al servidor de MLflow. `/salud` y cada predicción responden `version_modelo`, y una predicción guardada en la tabla del lote se puede rastrear hasta su corrida.

### Volver a la versión anterior

1. Ver cuál era: `python -m servicio.promover` (la etiqueta `anterior_en_produccion` de la versión actual).
2. Mover el alias: `python -m servicio.promover <versión anterior>`. No se reentrena nada: el modelo de esa versión sigue en el registro.
3. Exportar de nuevo: `python -m servicio.exportar_modelo`.
4. Reconstruir y levantar la imagen: `docker build -f servicio/Dockerfile -t servicio-emergencias .` y `docker run -p 8000:8000 servicio-emergencias`.
5. Comprobar con `/salud` que `version_modelo` es la anterior.

Mover el alias no cambia un servicio que ya está corriendo: el modelo se carga al arrancar. Hasta reconstruir y reiniciar, sigue sirviendo la versión de antes, y `version_modelo` lo dice.
