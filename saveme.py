from PIL import Image
import os
from utils.mypath import MyPath
from glob import glob 
import numpy as np 
import pandas as pd
from numpy import asarray
from tqdm import tqdm # progress bar

filename = 'rico_20/savefile.bin'
image_folder = "/bsuhome/hkiesecker/scratch/imageClassification/US/rico_images/"
mainBin = np.array([], dtype=np.uint8)
print("starting")
# for each image in rico
count = 0
if count < 2 :
    count = count + 1
    for imageName in tqdm(os.listdir(image_folder)):
        
        path = image_folder + imageName # path of image
        # load the image and convert into numpy array
        with open(path, 'rb') as f: # given permissions  
            img = Image.open(path).convert('RGB') # convert to color channel
        img = img.resize((256,256)) # resize properly
        numpydata = asarray(img) # convert to array of 3 channel
        np.append(mainBin, numpydata)

# data
mainBin.astype('uint8').tofile(filename) # save bin file in rico_20
print("done.")
