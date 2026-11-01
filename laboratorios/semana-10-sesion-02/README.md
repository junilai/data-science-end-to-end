# Semana 10, Sesión 2: Entrenamiento y comparación de modelos II: ¿ensamblaje, red neuronal simple o un LLM?

> Objetivo de la sesión: comparar, sobre el problema de tu proyecto, un modelo de ensamblaje, una red neuronal simple entrenada con una librería de alto nivel y un LLM usado vía prompting, y decidir cuál conviene según desempeño, costo y explicabilidad.

**Lectura previa de esta sesión:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

**Referencia de esta sesión:** [Neural network models, supervised (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

## Requisitos antes de esta sesión

- Nada nuevo que instalar: la red neuronal es `MLPRegressor`, de scikit-learn.
- Acceso a un LLM de chat (ChatGPT, Claude, Gemini u otro; la versión gratuita alcanza), para el Ejercicio 2.
- La tabla de resultados de tu proyecto de la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿para qué tipo de datos y de problema conviene el aprendizaje automático clásico, y para cuál un LLM?**

_(tu respuesta)_

**El artículo compara las dos opciones en datos, costo y explicabilidad. ¿Cuál de esas tres diferencias pesa más para tu proyecto?**

_(tu respuesta)_

## Pregunta de apertura

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

_(tu respuesta)_

## Ejercicio 1: una red neuronal frente al ensamblaje

Archivo: [`starter/ejercicio1_red_neuronal.py`](./starter/ejercicio1_red_neuronal.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv). El script ya entrena el gradient boosting de la sesión pasada con la misma preparación y la misma división. Los 3 `TODO` son la red neuronal, cuántos parámetros aprendió y qué columnas usa más el ensamblaje. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-10-sesion-02/starter
python3 ejercicio1_red_neuronal.py
```

**Respuesta 1 (TODO 1):** sin resolver, la red sale peor que el ensamblaje y aparece una advertencia `ConvergenceWarning`. ¿Qué crees que significa? ¿Qué esperas al dejarla entrenar hasta 1000 iteraciones?

Significa que la red llegó al límite de iteraciones (200 por defecto) **sin terminar de aprender**: el error todavía estaba bajando cuando se cortó el entrenamiento. Con 200 iteraciones da un MAE de **1.6854** en prueba, peor que el ensamblaje (1.2754). Con hasta 1000 iteraciones, la red converge sola en **692** iteraciones, la advertencia desaparece y el error en prueba baja a **1.1695**, mejor que el ensamblaje.

**Respuesta 2 (TODO 2):** una capa oculta de 64 neuronas recibe las 25 columnas preparadas y entrega una tarifa. ¿Cuántos números tiene que aprender?

**1729 parámetros**: 25 × 64 = 1600 pesos de entrada a la capa oculta, 64 sesgos de la capa oculta, 64 × 1 = 64 pesos de la capa oculta a la salida y 1 sesgo de salida. 1600 + 64 + 64 + 1 = 1729, igual a lo que imprime el script.

**Respuesta 3 (TODO 3):** ¿qué columna esperas que el ensamblaje use más? Si la app le muestra al conductor por qué la tarifa estimada es 70 dólares, ¿cuál de los dos modelos te permite responder?

La **distancia**, y por mucho: `distancia_km` tiene una importancia de **0.9824**. Le siguen `es_tarifa_fija` con 0.0120 y `es_hora_pico` con 0.0028. Para explicar una tarifa de 70 dólares, el **ensamblaje** permite responder: sus importancias y sus árboles muestran que la decisión sale de la distancia y de la marca de tarifa fija de JFK. La red reparte la decisión entre 1729 números sin significado individual, así que no hay una forma directa de decir "es 70 porque…".

**Lo que confirmó la salida, TODO por TODO:**

| modelo | MAE entrenamiento | MAE prueba | segundos |
|---|---|---|---|
| gradient boosting | 1.0705 | 1.2754 | 0.27 |
| red neuronal (200 iteraciones, sin resolver) | 1.6009 | 1.6854 | 0.85 |
| red neuronal (hasta 1000, converge en 692) | 1.0882 | **1.1695** | 5.44 |

- **Parámetros de la red:** 1729.
- **Las 3 columnas más importantes del ensamblaje:** `distancia_km` (0.9824), `es_tarifa_fija` (0.0120) y `es_hora_pico` (0.0028).
- La red gana en error (1.17 contra 1.28), pero tarda unas **20 veces más** en entrenar (5.44 s contra 0.27 s) y no se puede explicar.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: tres soluciones sobre los mismos 20 viajes

Solo o en pareja. Archivo: [`starter/ejercicio2_tres_soluciones.py`](./starter/ejercicio2_tres_soluciones.py), sobre el mismo `viajes_features.csv`. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación.

1. Completa los `TODO` 1 a 3 y corre el script. Además de entrenar, escribe `prompt_llm.txt`: 30 viajes con su tarifa como ejemplos y 20 viajes sin tarifa.
2. Copia `prompt_llm.txt` completo en un LLM de chat. Guarda los 20 números que responda, uno por línea, en `respuestas_llm.txt`, en la carpeta `starter`.
3. Completa los pasos 4 y 5 y vuelve a correr el script hasta que todas las comprobaciones digan `OK`. Para el paso 5, busca el precio por millón de tokens de entrada del modelo que usaste en la página de precios de su proveedor.

```
python3 ejercicio2_tres_soluciones.py
```

**Respuesta 1:** ¿qué LLM usaste? Completa la tabla con los números de tu corrida.

Usé **Claude Opus 5.5** (esfuerzo medio), en el chat de claude.ai, en una conversación nueva y sin archivos. El costo del LLM sale de su precio en la API de Anthropic, **4.00 USD por millón de tokens de entrada**: unos 726 tokens por prompt, 500 prompts de 20 viajes al mes, 1.451 USD. En el chat se paga por suscripción, no por token.

| | Error en los 20 viajes | Costo de entrenar | Costo de 10 000 viajes al mes | ¿Puedes explicar una predicción? |
|---|---|---|---|---|
| Gradient boosting | **0.896** | 0.3 s en la laptop (gratis) | ~0 USD (milisegundos en un servidor propio) | Sí: importancias (distancia 0.98, tarifa fija 0.012) y árboles |
| Red neuronal | 0.964 | 5.4 s en la laptop (gratis) | ~0 USD (milisegundos en un servidor propio) | Difícil: 1729 parámetros sin significado individual |
| LLM (Claude Opus 5.5) | 1.107 (corrida 1) / 1.063 (corrida 2) | 0 (no se entrena) | ~1.45 USD al mes solo en tokens de entrada | No: puede escribir una justificación, pero no es su cálculo real y cambia entre corridas |

**Respuesta 2:** corre el mismo prompt otra vez, en una conversación nueva. ¿Te dio los mismos 20 números? ¿Qué problema le trae eso a una app que le cobra al pasajero?

**No.** Solo 4 de los 20 números coinciden (los tres viajes de 70 de JFK y el primero, de 14.26). Los demás cambian hasta **0.81 USD** en el mismo viaje (en promedio, 0.26), y el MAE pasa de 1.107 a 1.063. Para una app que cobra eso es un problema: **el mismo viaje tendría precios distintos** según cuándo se pida, no se podría reproducir ni auditar una tarifa cobrada, y un pasajero que pide dos veces podría ver dos precios. Los modelos entrenados, en cambio, dan siempre el mismo número para la misma entrada.

**Respuesta 3:** busca en tus 20 respuestas la peor predicción del LLM. ¿Qué tiene de particular ese viaje? ¿Lo habría aprendido el ensamblaje con las 2380 filas de entrenamiento?

La peor es el viaje **JFK Airport → JFK Airport** de 1.01 km, en efectivo, a las 18 h. La tarifa real es **11.46** y el LLM dijo **6.78** (un error de 4.68). Es un viaje raro: sale y vuelve al aeropuerto, y cuesta bastante más de lo que su distancia sugiere. Entre los 30 ejemplos del prompt **no había ningún** viaje JFK → JFK, así que el LLM aplicó la tarifa por km de un viaje corto en Manhattan. En las 2380 filas de entrenamiento **sí hay 7** viajes JFK → JFK, y el ensamblaje predijo **12.74** (un error de 1.28): con todos los datos de entrenamiento aprende cosas que 30 ejemplos no alcanzan a mostrar.

**Si terminas antes:** el error en los 20 viajes de muestra no es el mismo que en los 595 de prueba. ¿Por qué no se mide el LLM sobre los 595, y qué tan confiable es una comparación sobre 20?

El LLM no se mide sobre los 595 por costo y logística: habría que copiar y pegar unos 30 prompts a mano (o pagar la API), y cada corrida da números distintos. La comparación sobre 20 es **poco confiable**: un solo viaje raro pesa mucho, porque el JFK → JFK aporta 0.23 de los 1.107 del LLM. La prueba está en el propio ensamblaje y la red: en los 20 viajes el ensamblaje gana (0.896 contra 0.964), pero en los 595 gana la red (1.1695 contra 1.2754). **El orden se invierte.** Con 20 filas solo se distinguen diferencias grandes.

## Ejercicio 3: ¿qué solución conviene para tu proyecto?

Usa tu pipeline y la tabla de resultados de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `comparar_proyecto.py`

Un script que entrene, con tu pipeline y tu división, el mejor ensamblaje de la sesión pasada y una red neuronal simple (`MLPRegressor` o `MLPClassifier`), y registre para cada uno la métrica de prueba, los segundos de entrenamiento y, para la red, la cantidad de parámetros.

### Entregable 2: la prueba del LLM

Elige 10 filas de tu conjunto de prueba y escribe un prompt con algunas filas de entrenamiento como ejemplos, como el de `prompt_llm.txt`. Pásalo a un LLM y registra su métrica sobre esas 10 filas, junto con la de tus dos modelos sobre las mismas 10. Si tu problema no se puede plantear así (por ejemplo, porque cada predicción necesita una historia larga), explica por qué; eso también es una respuesta.

### Entregable 3: la decisión

Una tabla con las tres opciones (métrica, costo, explicabilidad) y un párrafo que diga cuál usarías en tu proyecto y por qué, con los números de tu tabla.

Script: `proyecto/comparar_proyecto.py`. Entrena el mejor ensamblaje de la sesión pasada y una red con la misma preparación y la misma división por fecha, escribe `prompt_llm_proyecto.txt` (30 filas de entrenamiento como ejemplos y 10 de prueba) y guarda `comparacion_proyecto.csv`. El prompt se pasó a Claude Opus 5.5 en una conversación nueva, y sus 10 respuestas están en `proyecto/respuestas_llm_proyecto.txt`. La métrica es el MAE en emergencias por cantón y día.

| | MAE en prueba (81 030 filas) | MAE en las 10 filas | Entrenamiento | Costo de usarlo (220 cantones al día) | Explicabilidad |
|---|---|---|---|---|---|
| Gradient boosting max_depth=5 | 5.549 | 6.061 | 42 s | ~0 USD | Sí: `log_media_mismo_dia` 0.68, `log_media_prev_28d` 0.30, `log_lag_1d` 0.01 |
| Red neuronal (64) | **5.501** | 6.655 | 26 s (para en 24 iteraciones, con *early stopping*) | ~0 USD | No: 1153 parámetros |
| LLM (Claude Opus 5.5) | sin medir | **4.600** | 0 | ~1.77 USD al mes (6600 predicciones, 668 tokens por cada 10) | No, y cambia entre corridas |

Línea base de la sesión pasada ("igual que hoy"): 6.778 en prueba.

**Decisión: usaría el gradient boosting.** Sobre las 81 030 filas de prueba, la red y el ensamblaje quedan prácticamente empatados (5.501 contra 5.549: 0.05 emergencias por cantón y día, un 0.9 %). Los dos mejoran a la línea base en unos 18 %. Ante ese empate, el ensamblaje gana en lo que importa para un servicio de emergencias: se puede explicar (la predicción sale en un 98 % del promedio reciente del cantón en el mismo día de la semana y del promedio de 28 días), es determinista y se reentrena en menos de un minuto. El LLM ganó en las 10 filas (4.60 contra 6.06), pero 10 filas no alcanzan para concluir nada. En la taxi, el orden de dos modelos se invirtió entre 20 filas y 595. Además, el LLM no da el mismo número dos veces, no se puede medir sobre todo el conjunto de prueba sin pagar la API y cada predicción depende de copiar la historia del cantón en el prompt. No lo descartaría como apoyo para revisar casos raros, pero no como el modelo del pipeline.

## Antes de la próxima sesión

**Lectura:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

_(tu respuesta)_
