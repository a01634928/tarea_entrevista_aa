from collections import Counter
import numpy as np

def euc_dis(x1, x2):
    #Se aplica distancia euclidiana con pitagoras
    return np.sqrt(np.sum((x1 - x2) ** 2))


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def train_test_split(X, y):
    np.random.seed(0) # se establece la semilla para utilizar la misma semilla cada vez que se reinicie el kernel
    arr_rand = np.random.rand(X.shape[0])
    split = arr_rand < np.percentile(arr_rand, 80) # se distribuye en una relacion 80/20 train/test respectivamente de acuerdo las indicaciones del ejercicio

    X_train = X[split]
    y_train = y[split]
    X_test =  X[~split]
    y_test = y[~split]
    return X_train, X_test, y_train, y_test


def shuffle_d(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def cross_validation(X, y):
    X, y = shuffle_d(X, y)

    X1, y1 = X[0:29], y[0:29]  
    X2, y2 = X[30:59], y[30:59] 
    X3, y3 = X[60:89], y[60:89]  
    X4, y4 = X[90:119], y[90:119] 
    X5, y5 = X[120:149], y[120:149]
    X_train1, X_test1 = np.concatenate((X1, X2, X3, X4), axis=0), X5
    X_train2, X_test2 = np.concatenate((X1, X2, X3, X5), axis=0), X4
    X_train3, X_test3 = np.concatenate((X1, X2, X4, X5), axis=0), X3
    X_train4, X_test4 = np.concatenate((X1, X3, X4, X5), axis=0), X2
    X_train5, X_test5 = np.concatenate((X2, X3, X4, X5), axis=0), X1

    y_train1, y_test1 = np.concatenate((y1, y2, y3, y4), axis=0), y5
    y_train2, y_test2 = np.concatenate((y1, y2, y3, y5), axis=0), y4
    y_train3, y_test3 = np.concatenate((y1, y2, y4, y5), axis=0), y3
    y_train4, y_test4 = np.concatenate((y1, y3, y4, y5), axis=0), y2
    y_train5, y_test5 = np.concatenate((y2, y3, y4, y5), axis=0), y1

    X_train = {1:X_train1, 2:X_train2, 3:X_train3, 4:X_train4, 5:X_train5}
    X_test = {1:X_test1, 2:X_test2, 3:X_test3, 4:X_test4, 5:X_test5}
    y_train = {1:y_train1, 2:y_train2, 3:y_train3, 4:y_train4, 5:y_train5}
    y_test = {1:y_test1, 2:y_test2, 3:y_test3, 4:y_test4, 5:y_test5}
    return X_train, X_test, y_train, y_test


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
        frec = Counter(k_nei_lbl).most_common(1)
        return frec[0][0]


