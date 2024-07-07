#load keras tfds mint
import tensorflow as tf
import keras
from keras.losses import SparseCategoricalCrossentropy
from keras.regularizers import L1L2
from keras.activations import softmax

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_val, y_val = x_test[:3000], y_test[:3000]
x_test, y_test = x_test[3000:], y_test[3000:]
len(x_train), len(y_val), len(x_test), x_train[0].shape, y_train

x = keras.layers.Input(shape=(28, 28, 1))
h = keras.layers.Conv2D(32, 3, activation='relu', kernel_regularizer=L1L2(l1=0.007, l2=0.007))(x)
h = keras.layers.MaxPooling2D()(h)
h = keras.layers.Conv2D(64, 3, activation='relu', kernel_regularizer=L1L2(l1=0.007, l2=0.007))(h)
h = keras.layers.MaxPooling2D()(h)
h = keras.layers.Conv2D(128, 3, activation='relu', kernel_regularizer=L1L2(l1=0.007, l2=0.007))(h) 
f = keras.layers.Flatten()(h)
f = keras.layers.Dense(64, activation='relu',  kernel_regularizer=L1L2(l1=0.007, l2=0.007))(f)
f = keras.layers.Dense(10, kernel_regularizer=L1L2(l1=0.007, l2=0.007))(f)
model = keras.models.Model(inputs=x, outputs=f)
model.summary()

model.compile(
    optimizer="adamW",
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_val, y_val),
    epochs=100,
)