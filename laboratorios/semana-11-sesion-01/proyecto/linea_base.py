"""Linea base del proyecto como modelo de scikit-learn, para registrarla en
MLflow igual que los demas: "dentro de 7 dias, igual que hoy".

Predice obj_n_emergencias_t7 con n_emergencias del dia t. No aprende nada.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class IgualQueHoy(BaseEstimator, RegressorMixin):
    def __init__(self, columna="n_emergencias"):
        self.columna = columna

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X[self.columna], dtype=float)
