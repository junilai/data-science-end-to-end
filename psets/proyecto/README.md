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

### 1. Exportar el modelo desde MLflow

El servicio carga `servicio/modelo.joblib`, que no está en el repositorio: se regenera desde el registro de MLflow de la Semana 11. Con el servidor de MLflow levantado (`MLFLOW_TRACKING_URI`, por defecto `http://127.0.0.1:5050`):

```
python servicio/exportar_modelo.py
```

Exporta el run `05dac271ba674e7c967cbc31c5da8989` (gradient boosting `max_depth=5`, tag `candidato=final`, MAE de prueba 5.549). Para exportar otro run: `python servicio/exportar_modelo.py <run_id>`.

### 2. Construir y levantar el contenedor

La imagen se construye desde `psets/proyecto/`, porque copia `modelado/pipeline_proyecto.py` además de `servicio/`:

```
docker build -f servicio/Dockerfile -t servicio-emergencias .
docker run -p 8000:8000 servicio-emergencias
```

Sin Docker: `uvicorn servicio.api:app --port 8000`. La documentación interactiva queda en `http://localhost:8000/docs`.

### 3. Rutas

- `GET /salud`: si el servicio está vivo, si tiene modelo y su versión (los primeros 12 caracteres del MD5 de `modelo.joblib`).
- `POST /predecir`: recibe `fecha`, `region`, `lag_1d` (opcional), `lag_7d`, `lag_14d`, `lag_28d`, `media_prev_28d` (todas ≥ 0) y `es_feriado_t7`. Calcula `nombre_dia`, `es_fin_de_semana` y las features en log con `agregar_calendario` y `agregar_features` de `modelado/pipeline_proyecto.py`, las mismas del entrenamiento. Un campo obligatorio faltante o una emergencia negativa reciben 422.

### Ejemplo

Latacunga, lunes 2026-08-24:

```
$ curl -X POST localhost:8000/predecir -H "Content-Type: application/json" \
  -d '{"fecha": "2026-08-24", "region": "Sierra", "lag_1d": 141, "lag_7d": 106, "lag_14d": 116, "lag_28d": 109, "media_prev_28d": 121.25, "es_feriado_t7": false}'
{"fecha_objetivo":"2026-08-31","emergencias_estimadas":108.79,"version_modelo":"961785a60016"}
```

(El valor real del 2026-08-31 fue 119.)

### Pruebas

```
pytest psets/proyecto -v
```

Desde la raíz del fork. `servicio/test_api.py` prueba el contrato de la API con un modelo de prueba (no necesita `modelo.joblib`), y GitHub Actions la corre en cada push.
