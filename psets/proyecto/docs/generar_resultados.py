"""Genera las dos diapositivas de resultados del proyecto, en PDF:
docs/resultados_tecnico.pdf (para el equipo de datos) y
docs/resultados_ejecutivo.pdf (para quien decide si se usa el modelo).

Desde psets/proyecto/, con servicio/modelo.joblib exportado:

    python docs/generar_resultados.py

Los numeros salen del periodo de prueba (52 semanas completas desde
2025-08-25, 222 cantones por dia): el modelo en produccion (version 2)
contra la linea base "igual que hoy" (planificar con las emergencias del dia).
Los de validacion cruzada y monitoreo son los de la Semana 11 y la 13-14.
"""

import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CARPETA, ".."))

from modelado.pipeline_proyecto import CATEGORICAS, CORTE, NUMERICAS, OBJETIVO, YA_LISTAS, cargar_datos  # noqa: E402

AZUL, GRIS, ROJO, TINTA, SUAVE = "#2F5D8A", "#A6A6A6", "#B5443B", "#1F1F1F", "#5A5A5A"


def miles(numero):
    """12345.6 -> "12 346": el separador de miles con espacio."""
    return f"{numero:,.0f}".replace(",", " ")


def costo_asimetrico(error):
    return float(np.mean(np.where(error > 0, 3 * error, -error)))


def calcular():
    modelo = joblib.load(os.path.join(CARPETA, "..", "servicio", "modelo.joblib"))
    datos = cargar_datos()
    prueba = datos[datos["fecha"] >= CORTE].copy()
    prueba["modelo"] = modelo.predict(prueba[NUMERICAS + CATEGORICAS + YA_LISTAS])
    prueba["igual_que_hoy"] = prueba["n_emergencias"]
    prueba["semana"] = prueba["fecha"].dt.to_period("W").dt.start_time
    prueba = prueba[prueba.groupby("semana")["fecha"].transform("size") == 7 * 222]

    resultado = {"semanas": prueba["semana"].nunique()}
    for columna in ["modelo", "igual_que_hoy"]:
        error = prueba[OBJETIVO] - prueba[columna]
        por_semana = pd.DataFrame({"faltan": error.clip(lower=0), "sobran": (-error).clip(lower=0),
                                   "reales": prueba[OBJETIVO], "semana": prueba["semana"]}).groupby("semana").sum()
        resultado[columna] = {
            "mae": float(np.mean(np.abs(error))),
            "costo": costo_asimetrico(error),
            "sesgo": float(-error.mean()),
            "faltan_semana": float(por_semana["faltan"].mean()),
            "sobran_semana": float(por_semana["sobran"].mean()),
            "reales_semana": float(por_semana["reales"].mean()),
        }
    error_abs = (prueba[OBJETIVO] - prueba["modelo"]).abs()
    por_canton = error_abs.groupby(prueba["canton"]).sum()
    resultado["parte_quito_guayaquil"] = float(por_canton[["QUITO", "GUAYAQUIL"]].sum() / por_canton.sum())
    return resultado


def lienzo():
    figura = plt.figure(figsize=(13.33, 7.5))
    figura.patch.set_facecolor("white")
    return figura


def texto(figura, x, y, contenido, **opciones):
    opciones = {"color": TINTA, "va": "top", "ha": "left", **opciones}
    figura.text(x, y, contenido, transform=figura.transFigure, **opciones)


def tecnica(r):
    m, b = r["modelo"], r["igual_que_hoy"]
    figura = lienzo()
    texto(figura, 0.04, 0.95, "El gradient boosting baja 18% el MAE y 13% el costo asimétrico frente a la\n"
          "línea base, pero se queda corto y no ve la deriva conceptual", fontsize=19, weight="bold")
    texto(figura, 0.04, 0.82, "Emergencias por cantón a t + 7 · 222 cantones · prueba: 52 semanas desde 2025-08-25 · versión 2 en el registro",
          fontsize=11, color=SUAVE)

    eje = figura.add_axes([0.06, 0.14, 0.34, 0.56])
    nombres = ["MAE", "Costo asimétrico\n(3 × faltante + 1 × sobrante)"]
    x = np.arange(2)
    for desplazamiento, valores, color, etiqueta in [(-0.19, [b["mae"], b["costo"]], GRIS, "Línea base: igual que hoy"),
                                                     (0.19, [m["mae"], m["costo"]], AZUL, "Gradient boosting max_depth=5")]:
        barras = eje.bar(x + desplazamiento, valores, width=0.36, color=color, label=etiqueta)
        for barra, valor in zip(barras, valores):
            eje.text(barra.get_x() + barra.get_width() / 2, valor + 0.2, f"{valor:.2f}", ha="center", fontsize=11)
    eje.set_xticks(x, nombres, fontsize=11)
    eje.set_ylabel("Emergencias por cantón y día", fontsize=11)
    eje.set_title("Prueba (80 808 cantón-días)", fontsize=12, loc="left")
    eje.legend(frameon=False, fontsize=10, loc="upper left")
    eje.set_ylim(0, b["costo"] * 1.25)
    for lado in ["top", "right"]:
        eje.spines[lado].set_visible(False)

    bloques = [
        ("Modelo y validación", [
            "GradientBoostingRegressor(max_depth=5), objetivo en log1p",
            "Features: medias en log (mismo día, 28 días), lag_1d, región, calendario",
            "Validación temporal, 5 cortes por fecha y 7 días de separación:",
            "MAE 5.58 ± 0.29 (base 6.73) · costo 11.91 (base 13.48)",
        ]),
        ("Monitoreo (Semanas 13 y 14)", [
            "Latencia p95 4.75 ms en 6218 peticiones · 0 errores de servidor",
            "MAE semanal de 4.84 a 5.82 en 4 semanas simuladas (verdad a 7 días)",
            "PSI semanal de 3 features: normal ≤ 0.038 en 52 semanas, alerta en 0.03",
        ]),
        ("Limitaciones", [
            f"Sesgo {m['sesgo']:.2f}: se queda corto, el error caro (base {b['sesgo']:.2f})",
            f"Quito y Guayaquil concentran el {100 * r['parte_quito_guayaquil']:.0f}% del error absoluto",
            "No extrapola: con 20% más emergencias, el costo sube de 11.8 a 16.6",
            "La deriva conceptual no mueve el PSI: solo la ve el MAE, 7 días tarde",
        ]),
    ]
    y = 0.72
    for titulo, lineas in bloques:
        texto(figura, 0.46, y, titulo, fontsize=13, weight="bold", color=AZUL)
        y -= 0.045
        for linea in lineas:
            texto(figura, 0.46, y, "•  " + linea, fontsize=11)
            y -= 0.04
        y -= 0.025
    texto(figura, 0.04, 0.05, "Fuente: psets/proyecto · modelado/pipeline_proyecto.py, MLflow emergencias-cantones v2, "
          "monitoreo/desempeno.py y monitoreo/deriva_proyecto.py", fontsize=9, color=SUAVE)
    return figura


def ejecutiva(r):
    m, b = r["modelo"], r["igual_que_hoy"]
    menos_faltan = b["faltan_semana"] - m["faltan_semana"]
    menos_sobran = b["sobran_semana"] - m["sobran_semana"]
    figura = lienzo()
    texto(figura, 0.04, 0.95, f"Planificar con el modelo deja {miles(menos_faltan)} emergencias menos por semana sin recurso\n"
          f"previsto y libera {miles(menos_sobran)} asignaciones que hoy sobran", fontsize=20, weight="bold")
    texto(figura, 0.04, 0.82, f"Lo que habría pasado en el último año ({r['semanas']} semanas, unas {miles(m['reales_semana'])} emergencias "
          "por semana en los 222 cantones) si cada cantón hubiera planificado con el modelo",
          fontsize=11, color=SUAVE)

    eje = figura.add_axes([0.05, 0.14, 0.45, 0.56])
    grupos = ["Emergencias sin recurso\nprevisto, por semana", "Asignaciones de sobra,\npor semana"]
    x = np.arange(2)
    for desplazamiento, valores, color, etiqueta in [(-0.19, [b["faltan_semana"], b["sobran_semana"]], GRIS, "Como hoy: con las emergencias del día"),
                                                     (0.19, [m["faltan_semana"], m["sobran_semana"]], AZUL, "Con el modelo")]:
        barras = eje.bar(x + desplazamiento, valores, width=0.36, color=color, label=etiqueta)
        for barra, valor in zip(barras, valores):
            eje.text(barra.get_x() + barra.get_width() / 2, valor + 80, miles(valor), ha="center", fontsize=12)
    eje.set_xticks(x, grupos, fontsize=12)
    eje.set_yticks([])
    eje.legend(frameon=False, fontsize=11, loc="upper right")
    eje.set_ylim(0, b["faltan_semana"] * 1.25)
    for lado in ["top", "right", "left"]:
        eje.spines[lado].set_visible(False)

    bloques = [
        ("Recomendamos", ["•  Usar la estimación de cada noche para armar los turnos",
                          "    de la semana próxima, cantón por cantón"]),
        ("Necesitamos", ["•  Que la carga de emergencias del día termine antes de la",
                         "    medianoche",
                         "•  Una persona de planificación que revise el aviso semanal"]),
        ("Lo que el modelo no hace", ["•  No anticipa desastres ni eventos extraordinarios: eso",
                                     "    sigue en la operación del día",
                                     "•  En Quito y Guayaquil se equivoca más: dejar un margen",
                                     "•  Si cambia cómo se registran las emergencias, nos",
                                     "    enteramos una semana después"]),
    ]
    y = 0.72
    for titulo, lineas in bloques:
        texto(figura, 0.56, y, titulo, fontsize=14, weight="bold", color=AZUL)
        y -= 0.048
        for linea in lineas:
            texto(figura, 0.56, y, linea, fontsize=12)
            y -= 0.04
        y -= 0.03
    texto(figura, 0.04, 0.05, "Comparación con datos reales de agosto de 2025 a agosto de 2026. "
          "Una emergencia sin recurso previsto es una emergencia por encima de lo planificado para ese cantón y día.",
          fontsize=9, color=SUAVE)
    return figura


if __name__ == "__main__":
    resultados = calcular()
    for nombre, dibujar in [("resultados_tecnico", tecnica), ("resultados_ejecutivo", ejecutiva)]:
        figura = dibujar(resultados)
        figura.savefig(os.path.join(CARPETA, nombre + ".pdf"))
        plt.close(figura)
        print("Guardado docs/" + nombre + ".pdf")
    m, b = resultados["modelo"], resultados["igual_que_hoy"]
    print(f"Prueba ({resultados['semanas']} semanas): MAE {m['mae']:.3f} contra {b['mae']:.3f}, "
          f"costo {m['costo']:.3f} contra {b['costo']:.3f}")
    print(f"Por semana: faltan {m['faltan_semana']:.0f} contra {b['faltan_semana']:.0f}, "
          f"sobran {m['sobran_semana']:.0f} contra {b['sobran_semana']:.0f}, de {m['reales_semana']:.0f} emergencias")
