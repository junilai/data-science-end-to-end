# Semana 10, Sesión 1: Entrenamiento y comparación de modelos I: línea base y ensamblajes

> Objetivo de la sesión: entrenar un modelo de línea base y un modelo de ensamblaje sobre los datos preparados de tu proyecto, y aplicar regularización cuando el modelo se sobreajusta, sin derivar la matemática del descenso de gradiente.

**Lectura previa de esta sesión:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

**Referencia de esta sesión:** [Ensemble methods (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn, que ya usaste la sesión pasada. Ten a mano el script `pipeline_proyecto.py` de tu proyecto, de la Semana 9, Sesión 2.

Los laboratorios de hoy son notebooks: entrenar y comparar modelos es experimentar, y en un notebook vuelves a correr solo el modelo que cambiaste, sin reentrenar los demás. La preparación no se copia en el notebook: se importa del script, para que todos los modelos usen exactamente la misma.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es un ensamblaje, y en qué se diferencian bagging y boosting en la forma de combinar sus modelos?**

_(tu respuesta)_

**¿Por qué un promedio de muchos árboles puede equivocarse menos que el mejor de esos árboles por separado?**

_(tu respuesta)_

## Pregunta de apertura

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

_(tu respuesta)_

## Ejercicio 1: línea base, un árbol y un bosque

Notebook: [`starter/ejercicio1_linea_base_y_ensamblaje.ipynb`](./starter/ejercicio1_linea_base_y_ensamblaje.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), el mismo de la sesión pasada. La preparación de la Semana 9 está en [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py) y el notebook la importa. Ya trae una función `evaluar` que entrena cada modelo, mide el error absoluto medio en entrenamiento y en prueba, y lo registra en una tabla. Los 4 `TODO` son los modelos, cada uno en su celda. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** la línea base predice siempre la mediana de la tarifa. ¿Cuánto esperas que se equivoque en promedio? ¿Para qué sirve un modelo que no usa ninguna columna?

Mucho: unos **12.5 USD** por viaje en prueba (13.2 en entrenamiento), porque le da la misma tarifa a un viaje de 2 km y a uno a JFK. Sirve como **piso de comparación**: cualquier modelo que use columnas tiene que equivocarse bastante menos que eso. Si no lo hace, no está aprendiendo nada útil. También ayuda a detectar errores: un modelo "bueno" que empata con la línea base suele tener un bug en los datos o en el pipeline.

**Respuesta 2 (TODO 2):** un árbol sin límite puede seguir partiendo los datos hasta dejar un viaje por hoja. ¿Qué error esperas en entrenamiento, y qué esperas en prueba?

En entrenamiento, casi **0**: cada viaje termina en su propia hoja y el árbol devuelve su tarifa exacta (salió **0.0029**). En prueba, bastante más, porque memorizó el ruido de cada viaje en vez del patrón: salió **1.7017**. Es el caso típico de **sobreajuste**: una brecha de 1.70 entre un error casi nulo y el de prueba.

**Respuesta 3 (TODO 3 y 4):** ¿qué esperas que pase con la brecha entre entrenamiento y prueba al limitar la profundidad? ¿Y con 200 árboles promediados?

Al limitar la profundidad, la brecha **se achica**: el árbol ya no puede memorizar, así que sube el error en entrenamiento (1.0098) pero baja el de prueba (1.2728). La brecha pasa de 1.70 a **0.26**. Con 200 árboles promediados, cada árbol se sobreajusta a su propia muestra, pero sus errores se compensan al promediar, y el error en prueba baja respecto al árbol sin límite (1.2989 contra 1.7017). La brecha sigue siendo grande (**0.82**), porque cada árbol del bosque sigue sin límite y en entrenamiento el bosque "recuerda" los viajes que ya vio.

**Lo que confirmó la tabla, TODO por TODO:**

| modelo | MAE entrenamiento | MAE prueba | brecha |
|---|---|---|---|
| línea base (mediana) | 13.1998 | 12.5308 | −0.6690 |
| árbol sin límite | 0.0029 | 1.7017 | 1.6988 |
| árbol max_depth=8 | 1.0098 | 1.2728 | 0.2630 |
| bosque aleatorio (200) | 0.4743 | 1.2989 | 0.8245 |

El mejor en prueba es el **árbol con max_depth=8** (1.2728), apenas por delante del bosque (1.2989), y con la brecha más chica de los modelos que aprenden (0.26). Los dos le ganan por mucho a la línea base: un 90 % menos de error. La tabla queda en `starter/resultados_ejercicio1.csv`.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: regularizar una regresión que se sobreajusta

Solo o en pareja. Notebook: [`starter/ejercicio2_regularizacion.ipynb`](./starter/ejercicio2_regularizacion.ipynb), sobre el mismo `viajes_features.csv` y con la misma preparación importada. Mismo esqueleto del Ejercicio 1; `evaluar` acepta además `grado_interacciones=3`, que agrega como columnas todos los productos entre columnas hasta grado 3. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿cuántas columnas recibe la regresión con interacciones de grado 3, y cuántas filas de entrenamiento hay? ¿Qué tiene que ver esa comparación con el error en prueba que obtuviste?

Recibe **3275 columnas** y solo hay **2380 filas** de entrenamiento: hay más coeficientes que viajes para estimarlos. Con eso la regresión puede ajustar el entrenamiento casi a la perfección (MAE de 0.92), pero los coeficientes salen enormes e inestables, y en prueba explota: un MAE de **~2 930 millones** (brecha de 2.9 × 10⁹). Es sobreajuste extremo: con más columnas que filas, el modelo tiene libertad para memorizar cualquier cosa.

*Nota:* este resultado sale con las versiones del curso (`requirements.txt`: Python 3.11, scikit-learn 1.5.2 y numpy 1.26.4). Con versiones más nuevas, el error en prueba es distinto (22.99) porque el sistema no tiene una solución única y cada versión elige otra. La conclusión es la misma.

**Respuesta 2:** ¿qué cambió entre `lineal grado 3` y `ridge grado 3` en el error de entrenamiento, y qué cambió en el de prueba? ¿Por qué Ridge empeora un poco en entrenamiento?

En entrenamiento, Ridge **empeora un poco**: pasa de 0.9179 a **1.0272**. En prueba **mejora muchísimo**: de ~2 930 millones a **1.3209**, con una brecha de solo 0.29. Ridge (alpha=10) penaliza los coeficientes grandes, así que el modelo ya no puede usar las 3275 columnas con total libertad para calzar cada viaje de entrenamiento. Pierde un poco de ajuste a lo que ya vio a cambio de generalizar.

**Respuesta 3:** ¿qué modelo quedó mejor en prueba? ¿Cuánto mejor que la línea base?

**Gradient boosting**, con un MAE de **1.2754** en prueba y la brecha más chica de los modelos que aprenden (0.20). La línea base tiene 12.5308, así que el error baja **11.26 USD por viaje: un 89.8 % menos**. La tabla queda en `starter/resultados_ejercicio2.csv`.

**Si terminas antes:** la brecha de la línea base es negativa (se equivoca menos en prueba que en entrenamiento). ¿Cómo puede pasar eso con un modelo que no aprende nada?

Por azar de la división. La línea base predice la misma mediana para todos, así que su error depende solo de qué tan dispersas son las tarifas de cada conjunto. Por casualidad, el 20 % de prueba tiene menos viajes caros que el entrenamiento: un 8.9 % de viajes con tarifa fija de JFK contra un 10.4 %, y una desviación estándar de la tarifa de 17.32 contra 18.17. Por eso el error promedio es menor ahí. No significa que el modelo generalice mejor, sino que la prueba le tocó más fácil.

## Ejercicio 3: los primeros modelos de tu proyecto

Usa tu pipeline de la Semana 9, Sesión 2, con la misma división de entrenamiento y prueba. Esta vez no hay plantilla: escribe el notebook tú.

### Entregable 1: `modelos_proyecto.ipynb`

Un notebook, en la misma carpeta que tu `pipeline_proyecto.py` (`psets/proyecto/modelado/`), que importe de ahí la preparación y la división (`from pipeline_proyecto import cargar_y_dividir, construir_preparacion`) en vez de copiarlas, y que entrene:

- Una línea base. Elige la que corresponde a tu problema y justifícala abajo: `DummyRegressor` o `DummyClassifier`, o una regla del dominio (por ejemplo, "mañana igual que hoy" si predices una serie en el tiempo).
- Un modelo lineal (`Ridge` si es regresión, `LogisticRegression` si es clasificación), con la regularización que haga falta.
- Un ensamblaje (`RandomForestRegressor`, `GradientBoostingRegressor` o sus versiones `Classifier`).

### Entregable 2: `resultados_proyecto.csv`

Una fila por modelo, con la métrica de tu proyecto en entrenamiento y en prueba, la brecha y los segundos de entrenamiento, escrita por el notebook. Antes de hacer commit, reinicia el kernel y corre todo: la tabla tiene que salir de una corrida completa, no de celdas corridas en momentos distintos.

### Entregable 3: respuestas

**¿Cuál es tu línea base, y por qué esa y no otra?**

Script: `proyecto/modelos_proyecto.py`, que genera `proyecto/resultados_proyecto.csv`. Usa la misma preparación y la misma división por fecha de la Semana 9, Sesión 2: entrenamiento de 321 900 filas, prueba de 81 030 y 7 días de separación. La métrica es el **MAE en emergencias por cantón y día**. Los modelos se entrenan con el objetivo en log (`log1p`), por el hallazgo 1, y se mide el error ya devuelto a la escala original.

La línea base es la regla del dominio **"dentro de 7 días, igual que hoy"**: predecir `obj_n_emergencias_t7` con `n_emergencias` del día t. Como t+7 cae el mismo día de la semana que t, la regla ya respeta el nivel de cada cantón y la estacionalidad semanal, y no entrena nada. No usé `DummyRegressor`, porque la mediana del país ignora el cantón: su MAE en prueba es **36.88**, contra **6.78** de la regla. Le ganaría cualquier cosa y no diría si el modelo aporta algo frente a lo obvio.

**¿Alguno de tus modelos se sobreajusta? Muestra la evidencia con números de tu tabla.**

No.

| modelo | MAE entrenamiento | MAE prueba | brecha | segundos |
|---|---|---|---|---|
| línea base (igual que hoy) | 6.9318 | 6.7778 | −0.1540 | 0.0 |
| dummy mediana (referencia) | 39.4111 | 36.8794 | −2.5317 | 0.0 |
| ridge alpha=1 | 6.2643 | 6.0555 | −0.2088 | 0.3 |
| ridge alpha=100 | 6.2442 | 6.0395 | −0.2047 | 0.3 |
| gradient boosting | 5.8061 | 5.7118 | −0.0942 | 26.6 |
| gradient boosting max_depth=5 | 5.5520 | 5.5490 | −0.0030 | 41.7 |

Todas las brechas son **negativas o casi cero**. Con 321 900 filas y solo 16 columnas, los modelos no tienen con qué memorizar. Que la prueba salga un poco mejor se explica porque el último año tiene un poco menos de emergencias en promedio (40.2 por cantón y día en prueba, contra 42.8 en entrenamiento), no porque los modelos generalicen "de más".

**¿Qué regularización aplicaste (o por qué no hizo falta), y qué cambió en la tabla?**

Casi no hizo falta. En Ridge probé alpha=1 y alpha=100, y el error en prueba casi no se mueve (6.0555 contra 6.0395) porque hay muchísimas más filas que columnas. En gradient boosting fui en la dirección contraria: como no había sobreajuste, **le di más capacidad** (`max_depth=5` en vez de 3). El error en prueba bajó de 5.7118 a **5.5490** y la brecha siguió en ~0 (−0.003). Si la brecha empezara a crecer, ahí recién convendría limitar la profundidad o bajar `learning_rate`.

**¿Cuánto mejora tu mejor modelo a la línea base?**

El mejor es **gradient boosting con max_depth=5**: MAE de **5.549** en prueba, contra **6.778** de la línea base. Son **1.23 emergencias menos por cantón y día, un 18.1 % menos de error**. El costo es el tiempo: unos 42 s de entrenamiento, contra 0.3 s de Ridge, que ya mejora un 10.9 % a la línea base.

## Antes de la próxima sesión

**Lectura:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

_(tu respuesta)_
