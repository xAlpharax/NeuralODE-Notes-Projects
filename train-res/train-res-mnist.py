####################################################################

import tensorflow as tf
from tensorflow.keras.datasets import mnist

import sys
sys.path.append('./')

####################################################################

batch_size = 256
num_classes = 10
epochs = 5
image_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape((-1,) + image_shape)
x_test = x_test.reshape((-1,) + image_shape)

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test  = tf.keras.utils.to_categorical(y_test, num_classes)

####################################################################

from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from resblocktensorflow import ResBlock

#lighter drcnn than on cifar
def drcnn(input_shape, num_classes):
    x = Input(input_shape)
    y = Conv2D(32, (3, 3), activation='relu')(x)
    y = MaxPooling2D((2,2))(y)
    #y = Dropout(0.1)(y)

    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = MaxPooling2D((2,2))(y)
    #y = Dropout(0.1)(y)

    y = ResBlock(64, (3, 3))(y)
    y = Flatten()(y)
    y = Dense(num_classes, activation='softmax')(y)
    return Model(x,y)

drcnn = drcnn(image_shape, num_classes)

drcnn.compile(loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adadelta(3e-1),
            metrics=['accuracy'])

####################################################################

h = drcnn.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

drcnn.save_weights('weights/resweights/DRCNN-MNIST-weights.h5')

#drcnn.summary()

####################################################################

from utils.visualization import visualize

visualize(h, 'MNIST-DRCNN')

####################################################################

import numpy as np

#Correct y_test
print("Correct testdata:")
print(y_test[0:5])

#Predictions
print("\nPredicting on the test dataset...")
prediction = np.around(drcnn.predict(x_test[0:5]), decimals=2)
print(prediction)

####################################################################