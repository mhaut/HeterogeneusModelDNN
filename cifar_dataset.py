import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler


class cifar_dset():
    def __init__(self):
        self.scaler = None
        self.x_train = None; self.y_train = None
        self.x_test = None; self.y_test = None

    def train(self, directory):
        if self.x_train is None:
            (x_train_noesc, self.y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
            self.scaler = StandardScaler()
            for a in x_train_noesc: self.scaler.fit(a.reshape(-1,3))
            self.x_train = []
            for a in x_train_noesc: self.x_train.append(self.scaler.transform(a.reshape(-1,3)).reshape(32,32,3))
            self.x_train = np.array(self.x_train).astype("float32")
            del x_train_noesc
        return tf.data.Dataset.from_tensor_slices((self.x_train.astype("float32"), self.y_train))

    def test(self, directory):
        if self.x_test is None:
            (_, _), (x_test_noesc, self.y_test) = tf.keras.datasets.cifar10.load_data()
            self.x_test = []
            for a in x_test_noesc: self.x_test.append(self.scaler.transform(a.reshape(-1,3)).reshape(32,32,3))
            self.x_test = np.array(self.x_test).astype("float32")
            del x_test_noesc
        return tf.data.Dataset.from_tensor_slices((self.x_test.astype("float32"), self.y_test))
