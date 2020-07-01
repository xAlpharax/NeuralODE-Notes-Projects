##########################################################################################################

#mounting drive folder into dir
from google.colab import drive
drive.mount('/content/drive')

########################################################################################

#imports
import zipfile as zipf
from PIL import Image
import numpy as np

##########################################################################################################

class CelebA:
    def __init__(self, zipLocation= './drive/My Drive/', one_hot=True, flatten=False, start_index = 1):
        self.zipLocation = zipLocation
        self.zip = zipf.ZipFile(self.zipLocation+'img_align_celeba.zip','r')
        self.filelist = self.zip.namelist()
        self.filelist.sort()
        self.validation_index = int(0.8*np.array(self.filelist).size)
        self.test_index = int(0.9*np.array(self.filelist).size)
        self.label = np.loadtxt(self.zipLocation+"list_attr_celeba.txt", skiprows =  2,converters = {0: id}, usecols = 3)
        self.next_batch_index = start_index
        self.one_hot = one_hot
        self.flatten = flatten
        
    def load(self, count = 100000, start_index=1, mode = 'L'):
        
        celeb_img = []
        celeb_label = []
        end_index = start_index + count

        print("Loading dataset...")             
        
        for index, file in enumerate(self.filelist[start_index : end_index], start= start_index):
            
            with self.zip.open(file, 'r') as img:
                img_arr = Image.open(img) #misc.imread(img,mode='L')
                img_arr = img_arr.resize((90, 110)) #misc.imresize(img_arr, (180, 220))
                img_arr = np.asarray(img_arr)
                
                #flatten image to give flat vector instead of 28*28 matrix
                if self.flatten == True:
                    img_arr = img_arr.flatten()
                    
                celeb_img.append(img_arr)
                

        print("Loading labels...\n")             
        for index in range(start_index - 1, end_index - 1):
            new_lbl = [0]*2
            if self.one_hot == True:
                if self.label[index] == 1:
                    new_lbl[0] = 1
                else:
                    new_lbl[1] = 1
                celeb_label.append(new_lbl)
            else:
                celeb_label = self.label[start_index-1:end_index]
        
        
        celeb_img = np.array(celeb_img) #dtype='float32'
        celeb_label = np.array(celeb_label, dtype='float32') # dtype='uint32'
            
        return celeb_img, celeb_label

    
    # def validationSet(self):
    #     return self.load(count = 1000, start_index = self.validation_index)
    
    
    def testSet(self):
        return self.load(count = 1000, start_index = self.test_index)

########################################################################################