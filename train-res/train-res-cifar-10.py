####################################################################

import tensorflow as tf
from tensorflow.keras.datasets import cifar10

import sys
sys.path.append('./')

####################################################################

batch_size = 256
num_classes = 10
epochs = 30
image_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test  = tf.keras.utils.to_categorical(y_test, num_classes)

####################################################################

from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Input, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from resblocktensorflow import ResBlock

#lighter drcnn than on cifar
def drcnn(input_shape, num_classes):
	x = Input(input_shape)
	y = Conv2D(32, (3,3), activation='relu')(x)
	y = BatchNormalization(axis=-1)(y)
	y = MaxPooling2D((2,2))(y)
	y = Dropout(0.1)(y)

	y = Conv2D(64, (3,3), activation='relu')(y)
	y = BatchNormalization(axis=-1)(y)
	y = Conv2D(64, (3,3), activation='relu')(y)
	y = BatchNormalization(axis=-1)(y)
	y = MaxPooling2D((2,2))(y)
	y = Dropout(0.1)(y)

	y = ResBlock(64, (3,3))(y)
	y = BatchNormalization(axis=-1)(y)
	y = Flatten()(y)
	y = Dense(256, activation='relu')(y)
	y = Dense(num_classes, activation='softmax')(y)
	return Model(x,y)

drcnn = drcnn(image_shape, num_classes)

drcnn.compile(loss=tf.keras.losses.categorical_crossentropy,
			optimizer=tf.keras.optimizers.Adadelta(2e-1),
			metrics=['accuracy'])

####################################################################

h = drcnn.fit(x_train, y_train,
			batch_size=batch_size,
			epochs=epochs,
			verbose=1,
			validation_data=(x_test, y_test))

drcnn.save_weights('weights/resweights/DRCNN-30-CIFAR10-weights.h5')

#drcnn.summary()

####################################################################

from utils.visualization import visualize

visualize(h, 'CIFAR10-DRCNN')

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