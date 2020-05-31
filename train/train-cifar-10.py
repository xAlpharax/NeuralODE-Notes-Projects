####################################################################

import tensorflow as tf
from tensorflow.keras.datasets import cifar10

import sys
sys.path.append('./')

####################################################################

batch_size = 256
num_classes = 10
epochs = 1
image_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test  = tf.keras.utils.to_categorical(y_test, num_classes)

####################################################################

from dcodnn import dcodnn

dcodnn = dcodnn(image_shape, num_classes)

dcodnn.compile(loss=tf.keras.losses.categorical_crossentropy,
							optimizer=tf.keras.optimizers.Adadelta(2e-1),
							metrics=['accuracy'])

####################################################################

h = dcodnn.fit(x_train, y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=1,
							validation_data=(x_test, y_test))

dcodnn.save_weights('weights/DCODNN-30-CIFAR10-weights.h5')

#dcodnn.summary()

####################################################################

from utils.visualization import visualize

visualize(h)

####################################################################

import numpy as np

#Correct y_test
print("Correct testdata:")
print(y_test[0:5])

#Predictions
print("\nPredicting on the test dataset...")
prediction = np.around(dcodnn.predict(x_test[0:5]), decimals=2)
print(prediction)

####################################################################