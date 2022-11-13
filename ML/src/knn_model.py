from collections import Counter
import numpy as np

def euc_dis(x1, x2):
    #Se aplica distancia euclidiana con pitagoras
    return np.sqrt(np.sum((x1 - x2) ** 2))


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        #Calcula distancias entre x y todos los ejemplos en el dataset
        dist = [euc_dis(x, x_train) for x_train in self.X_train]
        #ordena por distancia y regresa los indices del primer k neighbors
        k_idx = np.argsort(dist)[: self.k]
        #extrae las etiquetas de K nearest neighbor de las muestras de entrenamiento
        k_nei_lbl = [self.y_train[i] for i in k_idx]
        #Regresa la etiqueta de clase que apareció más veces
        frec = Counter(k_nei_lbl).frec(1)
        return frec[0][0]


