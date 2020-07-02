#Get CelebA Dataset
import sys
sys.path.append('./')
from utils.CelebADataset import CelebA

celeb = CelebA()

train = celeb.load()
test = celeb.testSet()

total_size = len(train)

print("Train Data: {}".format(train[0].shape))
print("Test Data: {}".format(test[0].shape))

#############################################################

#imports
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from odeblocktensorflow import ODEBlock

######################################################

def DCODNN(input_shape, num_classes):
  x = Input(input_shape)
  y = Conv2D(64, (5,5), strides=2, activation='relu')(x)
  y = BatchNormalization(axis=-1)(y)
  y = Dropout(0.3)(y)
  
  y = Conv2D(128, (5,5), strides=2, activation='relu')(y)
  y = Conv2D(128, (5,5), activation='relu')(y)
  y = BatchNormalization(axis=-1)(y)
  y = Dropout(0.2)(y)
  
  y = ODEBlock(128, (3,3))(y)
  y = BatchNormalization(axis=-1)(y)
  y = MaxPooling2D(2,2)(y)
  y = Dropout(0.1)(y)
  
  y = Flatten()(y)
  y = Dense(512, activation='relu')(y)
  y = Dense(num_classes, activation='softmax')(y)
  return Model(x,y)

#############################################################

DCODNN = DCODNN((110, 90, 3), 2)

batch_size = 512
epochs = 50

import numpy as np
training_loss, testing_loss = np.array([[]]), np.array([[]])
training_acc, testing_acc = np.array([[]]), np.array([[]])

x_train = train[0]
x_test = test[0]

x_test = (x_test / 127.5) - 1
x_test = np.float32(x_test)

y_train = train[1]
y_test = test[1]

######################################################

import tensorflow as tf

optimizer = tf.keras.optimizers.Adadelta(3e-2) # Adadelta optimizer
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

    _ = metric.update_state(labels, DCODNN(inputs).numpy())
    acc_at_epoch = metric.result().numpy()
  
  epoch_time = int(time.time() - start_epoch_time)
  loss_at_epoch = loss_fn(labels, DCODNN(inputs).numpy())
  testing_loss_at_epoch = loss_fn(y_test[:3], DCODNN(x_test[:3]).numpy())

  _ = metric.update_state(y_test[:3], DCODNN(x_test[:3]).numpy())
  testing_acc_at_epoch = metric.result().numpy()

  training_loss, testing_loss = np.append(training_loss, loss_at_epoch), np.append(testing_loss, testing_loss_at_epoch)
  training_acc, testing_acc = np.append(training_acc, acc_at_epoch), np.append(testing_acc, testing_acc_at_epoch)
  print("Finished epoch: {:02d} with loss: {:.10f} and time taken: {:03d}s".format(epoch+1, float(loss_at_epoch.numpy()), epoch_time))

#############################################################################################

# rand = np.random.randint(10000) #85

# inp = np.expand_dims(x_train[rand], axis = 0)
# #print(inp.shape)

# print("\n")
# print(DCODNN(inp).numpy())
# print(y_train[rand])

######################################################

from utils.visualization import customvis

customvis('CELEBA-DCODNN', training_acc, testing_acc, training_loss, testing_loss)

#############################################################

DCODNN.save_weights('weights/DCODNN-CELEBA-weights.h5')

########################################################################