from PIL import Image
import numpy as np

################################
#################################################
##########################################

def preprocessceleb(img_path):
    img_arr = Image.open(img_path)
    img_arr = img_arr.convert('RGB')
    img_arr = img_arr.resize((90, 110))

    # Preprocessing the image
    img_arr = np.asarray(img_arr)
    img_arr = (img_arr / 127.5) - 1

    # Make 4D tensor out of the image
    img_arr = np.expand_dims(img_arr, axis=0)

    return img_arr

###################################
##########################################
############################

def preprocessCatsAndDogs(img_path):
    img_arr = Image.open(img_path)
    img_arr = img_arr.convert('RGB')
    img_arr = img_arr.resize((128, 80)) # (256, 160)

    # Preprocessing the image
    img_arr = np.asarray(img_arr)
    img_arr = (img_arr / 127.5) - 1

    # Make 4D tensor out of the image
    img_arr = np.expand_dims(img_arr, axis=0)

    return img_arr