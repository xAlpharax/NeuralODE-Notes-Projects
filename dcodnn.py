#imports
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Input, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from odeblocktensorflow import ODEBlock

######################################################
#####            large DCODNN network            #####
######################################################

def DCODNN(input_shape, num_classes):
  x = Input(input_shape)
  y = Conv2D(32, (3,3), activation='relu')(x)
  y = BatchNormalization(axis=-1)(y)
  y = MaxPooling2D(2,2)(y)
  y = Dropout(0.3)(y)

  y = Conv2D(128, (3,3), activation='relu')(y)
  # y = BatchNormalization(axis=-1)(y)
  y = Conv2D(128, (3,3), activation='relu')(y)
  y = BatchNormalization(axis=-1)(y)
  y = MaxPooling2D(2,2)(y)
  y = Dropout(0.3)(y)
  
  y = ODEBlock(128, (3,3))(y)
  y = BatchNormalization(axis=-1)(y)
  y = MaxPooling2D(2,2)(y)
  y = Dropout(0.2)(y)
  
  y = Flatten()(y)
  y = Dense(512, activation='relu')(y)
  y = Dense(256, activation='sigmoid')(y)
  y = Dropout(0.1)(y)
  y = Dense(num_classes, activation='softmax')(y)
  return Model(x,y)

######################################################