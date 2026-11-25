# [Nombre de tu proyecto]

> Este es el README de tu proyecto: reemplaza cada sección entre corchetes a medida que avanzas. Escríbelo para alguien que clona tu fork sin haber estado en clase y necesita entender qué hace el proyecto y correrlo, no como una nota para ti.

[Un párrafo: qué problema de negocio resuelve, para quién, y qué predice el modelo. Sale de tu propuesta del PSet 1, que está completa en `docs/`.]

**Métrica de éxito:** [la métrica de negocio y el umbral de decisión del PSet 1]

## Datos

| Fuente | Qué contiene | Cómo se obtiene | Dónde queda |
|---|---|---|---|
| [nombre de la fuente] | [qué filas y columnas trae, desde qué fecha] | [la URL, la API o el procedimiento exacto] | `data/raw/[archivo]` |

Los datos no están en el repositorio. Para regenerarlos desde cero: [los comandos, en orden]. Se completa con el PSet 1.

## Cómo correr el pipeline de datos

[Los comandos, en orden y desde qué carpeta, para ir de la fuente al warehouse: levantar la infraestructura, construir la imagen, subir el flow a Kestra, limpiar, cargar y correr dbt. Cada paso dice qué archivo deja y dónde. Se completa con el PSet 2.]

## Cómo entrenar y evaluar el modelo

[Qué notebook corre cada etapa del modelado, en qué orden, y qué necesita levantado (MLflow). Se completa con el PSet 3.]

## Modelo final

[El modelo elegido, su `run_id` en MLflow y la métrica que lo sostiene. El razonamiento completo está en `docs/decision_modelo.md`. Se completa con el PSet 3.]

## Cómo correr el servicio

El modelo se sirve **por lotes** (ver [`docs/despliegue.md`](docs/despliegue.md)), pero también como API: es la forma en que el lote de cada noche, un tablero u otra aplicación le piden una predicción. El servicio estima las emergencias de un cantón para el día *t + 7* a partir de lo que se sabe al cierre del día *t*. Todos los comandos se corren desde `psets/proyecto/`.

### 1. Registrar y exportar el modelo desde MLflow

El modelo está en el registro de MLflow como `emergencias-cantones`. La versión que sirve es la que tiene el alias `produccion`: hoy, la versión 2 (gradient boosting `max_depth=5`, MAE de prueba 5.549). El criterio para mover el alias y cómo volver atrás están en [`docs/despliegue.md`](docs/despliegue.md#versionado-y-registro). Con el servidor de MLflow levantado (`MLFLOW_TRACKING_URI`, por defecto `http://127.0.0.1:5050`):

```
python -m servicio.registrar_modelo 5   # entrena y registra una version nueva
python -m servicio.promover             # lista las versiones con sus metricas
python -m servicio.promover 2           # pone el alias produccion en la version 2
python -m servicio.exportar_modelo      # exporta la version en produccion
```

El último comando deja `servicio/modelo.joblib` y `servicio/modelo_version.txt` (el número de la versión). Ninguno de los dos está en el repositorio: se regeneran con este comando.

### 2. Construir y levantar el contenedor

La imagen se construye desde `psets/proyecto/`, porque copia `modelado/pipeline_proyecto.py` además de `servicio/`:

```
docker build -f servicio/Dockerfile -t servicio-emergencias .
docker run -p 8000:8000 -v "$(pwd)/monitoreo:/app/monitoreo" servicio-emergencias
```

El volumen `-v` deja el registro de predicciones (`monitoreo/predicciones.jsonl`) fuera del contenedor, para que sobreviva a un reinicio y lo lea `monitoreo/desempeno.py`. Sin Docker: `uvicorn servicio.api:app --port 8000`. La documentación interactiva queda en `http://localhost:8000/docs`.

### 3. Rutas

- `GET /salud`: si el servicio está vivo, si tiene modelo y qué versión del registro sirve. La lee de `modelo_version.txt`, que va dentro de la imagen, así que el contenedor no necesita llegar a MLflow.
- `GET /metricas`: las métricas del sistema desde que arrancó el servicio: peticiones a `/predecir`, errores de validación (422), errores de servidor (≥ 500) y latencia p50 y p95 en milisegundos.
- `GET /deriva`: el PSI de `log_media_mismo_dia`, `log_media_prev_28d` y `log_lag_1d` en las últimas 1554 predicciones (una semana de lote) contra el entrenamiento, y si hay alerta (algún PSI mayor que 0.03 con la ventana llena). La referencia está en `servicio/referencia_deriva.json` y se regenera con `python -m monitoreo.deriva_proyecto`.
- `POST /predecir`: cada respuesta trae su `id_prediccion` y dice qué versión la calculó. Recibe `cod_canton`, `fecha`, `region`, `lag_1d` (opcional), `lag_7d`, `lag_14d`, `lag_28d`, `media_prev_28d` (todas ≥ 0) y `es_feriado_t7`. Calcula `nombre_dia`, `es_fin_de_semana` y las features en log con `agregar_calendario` y `agregar_features` de `modelado/pipeline_proyecto.py`, las mismas del entrenamiento. Un campo obligatorio faltante o una emergencia negativa reciben 422. Cada predicción válida queda en `monitoreo/predicciones.jsonl`, con su entrada, su salida y la versión.

### Ejemplo

Latacunga, lunes 2026-08-24:

```
$ curl -X POST localhost:8000/predecir -H "Content-Type: application/json" \
  -d '{"cod_canton": "0501", "fecha": "2026-08-24", "region": "Sierra", "lag_1d": 141, "lag_7d": 106, "lag_14d": 116, "lag_28d": 109, "media_prev_28d": 121.25, "es_feriado_t7": false}'
{"id_prediccion":"9d6fa49b40e44e4abeef656e54847aae","cod_canton":"0501","fecha_objetivo":"2026-08-31","emergencias_estimadas":108.79,"version_modelo":"2"}
```

(El valor real del 2026-08-31 fue 119.)

### Monitoreo

Las métricas que se vigilan, cuándo se conocen y con qué umbral alertan están en [`docs/monitoreo.md`](docs/monitoreo.md). Desde `psets/proyecto/`:

```
python -m monitoreo.simular_lote 2026-07-27 2026-08-23   # simula 4 semanas del lote contra el servicio
python -m monitoreo.desempeno 2026-09-01                 # MAE y costo por semana y version, con la verdad conocida a esa fecha
python -m monitoreo.deriva_proyecto                      # simula derivas, elige el umbral y guarda la referencia del servicio
```

### Pruebas

```
pytest psets/proyecto -v
```

Desde la raíz del fork. `servicio/test_api.py` prueba el contrato de la API, el registro de cada predicción y las métricas, con un modelo de prueba, una versión inventada y un registro temporal (no necesita `modelo.joblib` ni MLflow), y GitHub Actions la corre en cada push.
