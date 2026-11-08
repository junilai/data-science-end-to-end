# Semana 11, Sesión 1: Gestión y trazabilidad de experimentos: eligiendo el modelo ganador con evidencia

> Objetivo de la sesión: registrar de forma trazable los parámetros, métricas y artefactos de las corridas de entrenamiento de las dos sesiones anteriores, y elegir el modelo final de tu proyecto con esa evidencia.

**Lectura previa de esta sesión:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

**Referencia de esta sesión:** [ML Experiment Tracking (documentación oficial de MLflow)](https://mlflow.org/docs/latest/ml/tracking/).

## Requisitos antes de esta sesión

Hoy se agrega una herramienta: MLflow, que registra cada corrida de entrenamiento. Como en un equipo de datos, el registro vive en un servidor de seguimiento (tracking server), no en tu notebook: el Codespace levanta uno en el puerto 5050 cada vez que inicia, y los notebooks lo encuentran por la variable de entorno `MLFLOW_TRACKING_URI`.

Si tu Codespace se creó antes de esta sesión, sincroniza tu fork y reconstruye el contenedor (`Ctrl+Shift+P` o `Cmd+Shift+P` → "Codespaces: Rebuild Container"; en tu laptop, "Dev Containers: Rebuild Container"). Reconstruir instala MLflow y configura el servidor; tus archivos no se pierden. Al terminar, comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Las últimas líneas deben decir `mlflow, version 2.17.2` y `servidor de MLflow: responde en http://127.0.0.1:5050`. Si el servidor no responde (por ejemplo, después de detener y volver a abrir el Codespace sin que arranque), levántalo a mano con `bash .devcontainer/mlflow-server.sh`.

La interfaz de MLflow se abre desde la pestaña **Ports** de VS Code: el puerto 5050, "MLflow (Semana 11)", con el ícono del globo. Ahí ves cada experimento, sus corridas, y puedes compararlas lado a lado.

Los laboratorios de hoy son notebooks, como los de la Semana 10, y la preparación se importa de `preparacion_taxis.py`. El registro (la base de datos del servidor y los modelos guardados) queda en la carpeta `.mlflow/` de la raíz del repositorio, que no se sube (está en `.gitignore`): lo que entregas es el notebook y la tabla de corridas que exporta.

**Si trabajas en tu laptop sin el contenedor (Opción B del README):** instala con `pip install -r requirements.txt` y, en una terminal aparte desde la raíz de tu fork, levanta el mismo servidor con `mlflow server --backend-store-uri sqlite:///.mlflow/mlflow.db --artifacts-destination .mlflow/artefactos --port 5050`. Antes de abrir los notebooks, define la variable `MLFLOW_TRACKING_URI=http://127.0.0.1:5050`. Sin esa variable, MLflow guarda el registro en una carpeta `mlruns` al lado de cada notebook, sin servidor: los notebooks también funcionan, pero cada carpeta tiene su propio registro.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué tres cosas hay que versionar en un proyecto de machine learning, y con qué herramienta versiona el autor cada una?**

_(tu respuesta)_

**¿Por qué no alcanza con Git para versionar los datos?**

_(tu respuesta)_

## Pregunta de apertura

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

_(tu respuesta)_

## Ejercicio 1: registrar cada corrida con MLflow

Notebook: [`starter/ejercicio1_registrar_corridas.ipynb`](./starter/ejercicio1_registrar_corridas.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Entrena el bosque y el gradient boosting de la Semana 10, y después la corrida de un compañero. Los 4 `TODO` son el registro. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (antes de los TODO):** sin resolver, el notebook entrena los tres modelos e imprime sus errores. ¿Cuál de los dos gradient boosting tiene razón? ¿Qué parte de la salida te permite saberlo, y qué parte no?

**Los dos**, cada uno sobre su propia división. El mío da **1.2754** en prueba (división con semilla 42) y el del compañero **1.1867** (semilla 7). Son el mismo modelo con los mismos hiperparámetros, pero evaluados sobre 595 viajes de prueba distintos, así que los números no son comparables entre sí y ninguno está "mal". De la salida, la columna `division` deja ver que las semillas son distintas. Lo que la salida **no** dice es todo lo demás que podría explicar la diferencia: con qué hiperparámetros se entrenó, con qué versión de los datos, con cuántas filas y con qué versión de las librerías. Tampoco queda el modelo para volver a probarlo.

**Respuesta 2 (TODO 2 y 3):** de todo lo que se puede registrar de una corrida, ¿qué crees que explica la diferencia entre los dos gradient boosting? ¿Qué parámetro esperas ver distinto en la tabla del paso 5?

La **división de los datos**: los dos gradient boosting tienen los mismos hiperparámetros (`n_estimators` = 100 y `random_state` = 42), la misma huella de datos (`4800204d50c0`) y las mismas 2380 filas de entrenamiento. Lo único que cambia en la tabla es `params.semilla_division`: **42 contra 7**. Con 595 viajes de prueba, cambiar qué viajes caen en prueba mueve el error en unos 0.09 USD.

**Respuesta 3 (TODO 4):** el registro ya tiene los parámetros y las métricas. ¿Para qué guardar además el modelo entrenado?

Porque con los parámetros y las métricas se puede **describir** el modelo, pero no **usarlo**. Guardar el pipeline entrenado (preparación + modelo) permite:

- Predecir viajes nuevos o servirlo en la app sin reentrenar.
- Comprobar que el número registrado sale de ese modelo exacto. Volver a entrenar no garantiza el mismo resultado si cambian las librerías, los datos o la semilla.
- Comparar o auditar después qué predijo exactamente.

MLflow guarda además las versiones de las librerías con las que se entrenó (`requirements.txt` del artefacto: scikit-learn 1.5.2, numpy 1.26.4 y pandas 2.2.2).

**Lo que mostró la tabla del paso 5:**

| corrida | semilla de división | huella de los datos | n_estimators | MAE prueba | segundos |
|---|---|---|---|---|---|
| gradient boosting (compañero) | 7 | 4800204d50c0 | 100 | 1.1867 | 0.12 |
| gradient boosting | 42 | 4800204d50c0 | 100 | 1.2754 | 0.15 |
| bosque aleatorio | 42 | 4800204d50c0 | 200 | 1.2989 | 0.75 |

La única diferencia registrada entre los dos gradient boosting es la semilla de la división. La huella de los datos es la misma en las tres corridas.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: comparar con evidencia y elegir

Solo o en pareja. Notebook: [`starter/ejercicio2_comparar_y_elegir.ipynb`](./starter/ejercicio2_comparar_y_elegir.ipynb), sobre el mismo `viajes_features.csv`. Mismo esqueleto del Ejercicio 1, con los cinco modelos de la Semana 10 en un experimento propio. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué modelo quedó primero en el registro, y por cuánto le gana al segundo?

La **red neuronal**, con un MAE en prueba de **1.1695**. Le gana al gradient boosting (1.2754) por **0.106 USD por viaje, un 8.3 % menos de error**. El orden completo es: red 1.1695, gradient boosting 1.2754, bosque 1.2989, ridge 2.5565 y línea base 12.5308. Los 5 pasos de la comprobación dan `OK`.

**Respuesta 2:** el paso 5 carga el modelo ganador desde el registro y vuelve a calcular su error. ¿Qué te garantiza que coincida con el número registrado? ¿Qué no te garantiza sobre ese modelo?

Que coincida (1.169505229635501 registrado y reproducido) garantiza que el artefacto guardado **es el mismo modelo** que produjo esa métrica, con los mismos pesos y la misma preparación, y que se puede cargar y usar sin reentrenar. **No garantiza** que sea el mejor modelo en general ni que ese error se mantenga con datos nuevos: el número sale de una sola división de 595 viajes. Con otra semilla, el mismo gradient boosting pasó de 1.28 a 1.19 en el ejercicio 1, una diferencia parecida a la que separa al primero del segundo. Tampoco garantiza que funcione con viajes distintos a los de entrenamiento, como zonas nuevas o cambios de tarifa.

**Respuesta 3:** si mañana alguien cambia una fila de `viajes_features.csv` y vuelve a entrenar, ¿cómo lo notarías mirando solo el registro?

Por el parámetro **`huella_datos`**: es el hash MD5 del archivo, y si cambia un solo byte del CSV, cambia. Las corridas de hoy tienen todas `4800204d50c0`. Una corrida nueva con otra huella indica que se entrenó con otros datos, aunque `filas_entrenamiento` siga en 2380 y los hiperparámetros sean iguales. Así se sabe que sus métricas no se pueden comparar directamente con las de hoy.

**Si terminas antes:** abre la interfaz de MLflow (pestaña **Ports**, puerto 5050), entra al experimento `tarifa-taxis-comparacion`, selecciona las cinco corridas y compáralas lado a lado con **Compare**. Además del error, ¿qué te dice el registro de la red neuronal que no te diría una tabla escrita a mano?

El registro guarda **los 26 parámetros** de la red, incluidos los que nunca escribí porque son valores por defecto: `activation` = relu, `solver` = adam, `learning_rate_init` = 0.001, `alpha` = 0.0001, `batch_size` = auto, `early_stopping` = False, `tol` = 0.0001, `validation_fraction` = 0.1, etc. Además guarda la huella de los datos y la semilla, la duración y el usuario de la corrida, y el **modelo entrenado** con las versiones exactas de las librerías (scikit-learn 1.5.2, numpy 1.26.4 y pandas 2.2.2). Una tabla escrita a mano tendría el error y, con suerte, `hidden_layer_sizes` y `max_iter`. No permitiría reproducir la red si cambia un valor por defecto en otra versión de scikit-learn.

## Ejercicio 3: el registro de tu proyecto

Usa tu `pipeline_proyecto.py` y los modelos que entrenaste en las dos sesiones de la Semana 10. Esta vez no hay plantilla.

### Entregable 1: `experimentos_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que importe de ahí la preparación y la división y registre en MLflow, en un experimento con el nombre de tu proyecto, una corrida por cada modelo de la Semana 10 (línea base, modelo lineal, ensamblaje, red neuronal). Cada corrida registra:

- Los hiperparámetros del modelo.
- Cómo se dividieron los datos (la semilla, o la fecha de corte si divides por fecha) y la huella del archivo de datos.
- La métrica de tu proyecto en entrenamiento y en prueba, y los segundos de entrenamiento.
- El pipeline entrenado, como artefacto.

Al final, el notebook marca la corrida que eliges como final (`candidato` = `final`), carga su modelo desde el registro y comprueba que reproduce la métrica registrada.

### Entregable 2: `corridas_proyecto.csv`

La tabla de corridas de tu experimento, exportada con `mlflow.search_runs(...).to_csv("corridas_proyecto.csv", index=False)` desde el mismo notebook. Antes de hacer commit, borra las corridas de prueba del experimento de tu proyecto (en la interfaz de MLflow: selecciónalas y **Delete**), reinicia el kernel y corre todo: la tabla tiene que salir de una corrida completa.

### Entregable 3: la decisión

**¿Qué modelo elegiste como final? Escribe su `run_id` y los números del registro que sostienen la decisión.**

Notebook: `proyecto/experimentos_proyecto.ipynb`. Importa la preparación y la división de `pipeline_proyecto.py` (que sigue en `semana-09-sesion-02/plantilla-pipeline/`) y la línea base de `proyecto/linea_base.py`. Registra 4 corridas en el experimento `emergencias-cantones`, todas con la misma huella de datos (`7624e6f357a7`) y la misma división (corte el 2025-08-25, 7 días de separación). La tabla exportada está en `proyecto/corridas_proyecto.csv`.

| corrida | MAE entrenamiento | MAE prueba | brecha | segundos |
|---|---|---|---|---|
| red neuronal (64) | 5.6595 | **5.5009** | −0.159 | 10.1 |
| **gradient boosting max_depth=5** (`candidato` = `final`) | 5.5520 | 5.5490 | −0.003 | 23.5 |
| ridge alpha=100 | 6.2442 | 6.0395 | −0.205 | 0.3 |
| línea base (igual que hoy) | 6.9318 | 6.7778 | −0.154 | 0.0 |

Elegí el **gradient boosting con max_depth=5**, `run_id` **`05dac271ba674e7c967cbc31c5da8989`**. La red queda primera por **0.048 emergencias por cantón y día (0.9 %)**, una diferencia menor a la que puede mover un cambio de división (en los taxis, la semilla movió el error un 7 %). Ante ese empate, el gradient boosting se puede explicar con sus importancias, es determinista y no se sobreajusta (brecha de −0.003). Mejora a la línea base en un **18.1 %** (5.549 contra 6.778). Al cargarlo desde el registro, el MAE reproducido es **5.548994361235862**, idéntico al registrado.

**Si otra persona clona tu repositorio, ¿qué necesita para obtener exactamente el mismo número? ¿Qué de eso quedó registrado en MLflow y qué no?**

Necesita el mismo código, los mismos datos, la misma división, los mismos hiperparámetros con la misma semilla y las mismas versiones de las librerías.

- **Quedó en MLflow:** los hiperparámetros (incluido `random_state` = 42), la fecha de corte y la separación de 7 días, la huella del CSV (`7624e6f357a7`), las filas de entrenamiento, las métricas, el modelo entrenado y las versiones de Python y de las librerías con las que se entrenó (`python_env.yaml` y `requirements.txt` del artefacto).
- **No quedó en MLflow:**
  - **El código** (`pipeline_proyecto.py`, `linea_base.py` y el notebook): las reglas de limpieza, como quitar las zonas no delimitadas y el objetivo de enero de 2024, solo están ahí, y la corrida no registra un commit de git.
  - **Los datos en sí:** solo su huella. El CSV está en el repo, pero si alguien lo cambia, la huella solo avisa que es distinto, no permite recuperar el original.
  - **El propio registro:** `.mlflow/` está en `.gitignore`, así que quien clona el repo no recibe las corridas; tiene que volver a correr el notebook.

## Antes de la próxima sesión

**Lectura:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

Hoy elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

_(tu respuesta)_
