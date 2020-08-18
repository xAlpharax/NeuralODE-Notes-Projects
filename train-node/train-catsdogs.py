# #Get CatsAndDogs Dataset
# import sys
# sys.path.append('./')

# from utils.CatsAndDogsDataset import CatsAndDogs

# #Train Dataset
# train = CatsAndDogs(mode = 'train').load()

# #Test Dataset
# test = CatsAndDogs(mode = 'test').load()

# print("Train Data: {}".format(train[0].shape))
# print("Test Data: {}".format(test[0].shape))

#mounting drive into dir
from google.colab import drive
drive.mount('/content/NeuralODE-Notes-Projects/drive')

import numpy as np
x_train = np.load('drive/My Drive/CatsAndDogsArrays/CatsAndDogs-images.npy')
x_test = np.load('drive/My Drive/CatsAndDogsArrays/CatsAndDogs-images-test.npy')

y_train = np.load('drive/My Drive/CatsAndDogsArrays/CatsAndDogs-labels.npy')
y_test = np.load('drive/My Drive/CatsAndDogsArrays/CatsAndDogs-labels-test.npy')

print("Train Data: {}".format(x_train.shape))
print("Test Data: {}".format(x_test.shape))

#############################################################

import sys
sys.path.append('./')

###################################################################

#imports
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from odeblocktensorflow import ODEBlock

######################################################

#largest DCODNN network
def DCODNN(input_shape, num_classes):
  x = Input(input_shape)
  y = Conv2D(16, (3,3), activation='relu')(x)
  y = BatchNormalization(axis=-1)(y)
  y = MaxPooling2D(2,2)(y)
  y = Dropout(0.3)(y)

  y = Conv2D(32, (3,3), activation='relu')(x)
  y = BatchNormalization(axis=-1)(y)
  y = MaxPooling2D(2,2)(y)
  y = Dropout(0.3)(y)

  y = Conv2D(128, (3,3), activation='relu')(y)
  y = Conv2D(128, (3,3), activation='relu')(y)
  y = BatchNormalization(axis=-1)(y)
  y = MaxPooling2D(2,2)(y)
  y = Dropout(0.2)(y)

  y = Conv2D(256, (1,1), activation='relu')(y)
  y = ODEBlock(256, (3,3))(y)
  y = BatchNormalization(axis=-1)(y)
  y = MaxPooling2D(2,2)(y)
  y = Dropout(0.2)(y)

  y = Flatten()(y)

  y = Dense(1024, activation='relu')(y)
  y = Dense(512, activation='sigmoid')(y)
  y = Dropout(0.1)(y)
  y = Dense(num_classes, activation='softmax')(y)
  return Model(x,y)

#############################################################

DCODNN = DCODNN((128, 80, 3), 2)

DCODNN.summary()

import matplotlib as plt
plt.imshow(x_train[654])

batch_size = 256
test_batch = 256
epochs = 10

training_loss, testing_loss = np.array([[]]), np.array([[]])
training_acc, testing_acc = np.array([[]]), np.array([[]])

x_test[:test_batch] = (x_test[:test_batch] / 127.5) - 1
x_test = np.float32(x_test)

total_size = len(x_train)
total_test_size = len(x_test)

######################################################

import tensorflow as tf

optimizer = tf.keras.optimizers.Adadelta(5e-3) # Adadelta optimizer
loss_fn = tf.keras.losses.CategoricalCrossentropy() # Categorical Loss for categorical labels
metric = tf.keras.metrics.CategoricalAccuracy() # Categorical Accuracy

@tf.function
def trainfn(model, inputs, labels):
  with tf.GradientTape() as tape:
    # Computing Losses from Model Prediction
    loss = loss_fn(labels, model(inputs))

  gradients = tape.gradient(loss, model.trainable_variables) # Gradients for Trainable Variables with Obtained Losses
  optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # Updated weights

#############################################################################################

import time

### TRAINING CUSTOM DCODNN ###

for epoch in range(epochs):
  start_epoch_time = time.time()

  for index in range(0, total_size, batch_size):
    end_index = total_size if index + batch_size > total_size else index + batch_size

    inputs = x_train[index:end_index] # Slicing operation
    labels = y_train[index:end_index] # Slicing operation
    #print(inputs.shape)

    # normalize data between -1 and 1
    inputs = (inputs / 127.5) - 1
    inputs = np.float32(inputs)

    trainfn(DCODNN, inputs, labels)
    print("Finished Batch")

  _ = metric.update_state(labels, DCODNN(inputs).numpy())
  acc_at_epoch = metric.result().numpy()
  loss_at_epoch = np.mean(loss_fn(labels, DCODNN(inputs).numpy()))

  testing_loss_at_epoch = np.mean(loss_fn(y_test[:test_batch], DCODNN(x_test[:test_batch]).numpy()))
  _ = metric.update_state(y_test[:test_batch], DCODNN(x_test[:test_batch]).numpy())
  testing_acc_at_epoch = metric.result().numpy()

  epoch_time = int(time.time() - start_epoch_time)

  training_loss, testing_loss = np.append(training_loss, loss_at_epoch), np.append(testing_loss, testing_loss_at_epoch)
  training_acc, testing_acc = np.append(training_acc, acc_at_epoch), np.append(testing_acc, testing_acc_at_epoch)
  print("Finished epoch: {:02d} with loss: {:.10f} acc: {:.4f} val_acc: {:.4f} and time taken: {:03d}s".format(epoch+1, loss_at_epoch, acc_at_epoch, testing_acc_at_epoch, epoch_time))

#############################################################################################

from utils.visualization import customvis

customvis('CATSDOGS-DCODNN', training_acc, testing_acc, training_loss, testing_loss)

#############################################################

DCODNN.save_weights('weights/DCODNN-CATSDOGS-weights.h5')

########################################################################