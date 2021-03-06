########################################################

#mounting drive folder into dir
from google.colab import drive
drive.mount('/content/NeuralODE-Notes-Projects/drive')

############################################

from PIL import Image
import numpy as np
import glob
import os

################################################################################

class CatsAndDogs:
    def __init__(self, location = './drive/My Drive/Data/CatsAndDogs', mode = 'train'):
        self.location = os.path.join(location, 'training_set') if mode == 'train' else os.path.join(location, 'test_set')
        
    def load(self):
        
        catsimgs, dogsimgs = [], []
        catslabels, dogslabels = [], []

        print("Loading cat dataset...")
        for image in glob.glob(os.path.join(self.location, 'cats/*')):
            
            with open(image, 'rb') as img:
                img_arr = Image.open(img)
                img_arr = img_arr.resize((128, 80)) # opening and resizing the image to 16/10   #(256, 160)
                img_arr = np.asarray(img_arr)
                    
                catsimgs.append(img_arr) # appending said cat image respective list
                
                new_lbl = [0.]*2
                new_lbl[0] = 1.
                catslabels.append(new_lbl) # assigning labels for cat imgages 

        print("Loading dog dataset...")
        for image in glob.glob(os.path.join(self.location, 'dogs/*')):
            
            with open(image, 'rb') as img:
                img_arr = Image.open(img)
                img_arr = img_arr.resize((128, 80)) # opening and resizing the image to 16/10
                img_arr = np.asarray(img_arr)
                    
                dogsimgs.append(img_arr) # appending said dog image respective list
                
                new_lbl = [0.]*2
                new_lbl[1] = 1.
                dogslabels.append(new_lbl) # assigning labels for dog imgages
        
        catsimgs, dogsimgs = np.array(catsimgs), np.array(dogsimgs) # convert images to numpy arrays
        catslabels, dogslabels = np.array(catslabels), np.array(dogslabels)

        all_images = np.concatenate((catsimgs, dogsimgs)) # merge arrays
        all_labels = np.concatenate((catslabels, dogslabels))

        shuff = np.arange(all_images.shape[0])
        np.random.shuffle(shuff) # shuffle mask
        
        #shuffle complete dataset with set shuffle mask
        all_images, all_labels = all_images[shuff], all_labels[shuff]
        
        return all_images, all_labels

############################################################