"""El monitor de deriva de datos del servicio.

Guarda las features de las ultimas `ventana` predicciones y compara su
distribucion con la de entrenamiento, con el PSI. La referencia (los cortes
de los deciles y sus proporciones, la ventana y el umbral) la escribe
python -m monitoreo.deriva_proyecto en servicio/referencia_deriva.json.
"""

import math

import numpy as np

MINIMO = 0.0001


def proporciones(valores, cortes):
    """Proporcion de los valores en cada casilla (len(cortes) + 1 casillas,
    abiertas en los extremos). Los valores vacios (None o NaN) no cuentan."""
    presentes = [valor for valor in valores if valor is not None and not math.isnan(valor)]
    if not presentes:
        return np.zeros(len(cortes) + 1)
    conteo, _ = np.histogram(presentes, bins=[-np.inf] + list(cortes) + [np.inf])
    return conteo / len(presentes)


def psi(referencia, actual):
    r = np.clip(np.asarray(referencia, dtype=float), MINIMO, None)
    a = np.clip(np.asarray(actual, dtype=float), MINIMO, None)
    return float(np.sum((a - r) * np.log(a / r)))


class MonitorDeDeriva:
    def __init__(self, referencia):
        self.referencia = referencia
        self.ventana = referencia["ventana"]
        self.umbral = referencia["umbral"]
        self.features = [clave for clave in referencia if clave not in ("ventana", "umbral")]
        self.valores = {feature: [] for feature in self.features}

    def registrar(self, fila):
        """Agrega las features de una prediccion (un diccionario) y deja solo
        las ultimas `ventana`."""
        for feature in self.features:
            self.valores[feature].append(fila.get(feature))
            self.valores[feature] = self.valores[feature][-self.ventana:]

    def estado(self):
        """El PSI de cada feature en la ventana actual, y "alerta": True solo
        si la ventana esta llena y algun PSI pasa el umbral."""
        cantidad = len(self.valores[self.features[0]])
        resultado = {"predicciones_en_ventana": cantidad, "ventana": self.ventana,
                     "umbral": self.umbral, "psi": {}, "alerta": False}
        if cantidad == 0:
            return resultado
        for feature in self.features:
            ref = self.referencia[feature]
            resultado["psi"][feature] = round(psi(ref["proporciones"],
                                                  proporciones(self.valores[feature], ref["cortes"])), 4)
        resultado["alerta"] = cantidad >= self.ventana and any(
            valor > self.umbral for valor in resultado["psi"].values())
        return resultado
