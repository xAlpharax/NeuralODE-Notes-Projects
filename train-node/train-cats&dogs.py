#Get CatsAndDogs Dataset
import sys
sys.path.append('./')
from utils.CatsAndDogsDataset import CatsAndDogs

#Train Dataset
CatsAndDogs = CatsAndDogs()
train = CatsAndDogs.load()

#Test Dataset
CatsAndDogs = CatsAndDogs(mode = 'test')
test = CatsAndDogs.load()

total_size = len(train[0])

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
  y = Conv2D(64, (5,5), activation='relu')(x)
  y = BatchNormalization(axis=-1)(y)
  y = MaxPooling2D(2,2)(y)
  y = Dropout(0.3)(y)
  
  y = Conv2D(128, (5,5), activation='relu')(y)
  y = Conv2D(256, (5,5), activation='relu')(y)
  y = BatchNormalization(axis=-1)(y)
  y = MaxPooling2D(2,2)(y)
  y = Dropout(0.2)(y)
  
  y = ODEBlock(256, (3,3))(y)
  y = BatchNormalization(axis=-1)(y)
  y = MaxPooling2D(2,2)(y)
  y = Dropout(0.1)(y)
  
  y = Flatten()(y)
  y = Dense(1024, activation='sigmoid')(y)
  y = Dense(num_classes, activation='softmax')(y)
  return Model(x,y)

#############################################################