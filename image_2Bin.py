from PIL import Image
import os
from utils.mypath import MyPath
from glob import glob 
import numpy as np 
import pandas as pd
from numpy import asarray
from tqdm import tqdm # progress bar
import csv

"""
Author: Hailee Kiesecker

file takes in image data and labels and converts them into binary files
to be used in the first step of SCAN

"""


filename = 'rico_20/rico_binary/rico_unlabeled_RESIZED.bin'
en_filename_X = 'rico_20/rico_binary/test_X_RESIZED.bin'
en_filename_y = 'rico_20/rico_binary/test_Y_RESIZED.bin'

rico_en = '/bsuhome/hkiesecker/scratch/imageClassification/classTrain_ui.csv' 
image_folder = "/bsuhome/hkiesecker/scratch/imageClassification/US/rico_image/"
bb = []
en_bb_l = []
en_bb = []
max_unlabeled = 66177 #- 66256 
max_labeled = 1281 #1336

print("starting")

def en_conv(dataframe):
    
    with open(dataframe, "r") as f:
        count = 0
        reader = csv.reader(f, delimiter=",")
        for i, row in enumerate(reader): # index, vals
            count = count +1 
            if count == max_labeled:
                break
            else:
                label, imgNumber, path = row    # image number not important
                label_number = label_mod(label)     # convert string to binary 
                img_arr = image_mod(path)
            
                en_bb.append(img_arr)   # append image array 
                en_bb_l.append(label_number)# append label array
        
    #### save iamge array as binary test_X.bin ###
    img_data = np.asarray(en_bb, dtype = 'uint8')   # convert list to array       
    img_data.tofile(en_filename_X)   #save binary 

    ###save label array as binary test_y.bin ###
    label_data = np.asarray(en_bb_l, dtype = 'uint8')
    label_data.tofile(en_filename_y)
    

# change labeled data to numbers 
def label_mod(l_name):
    val = ['tutorial','list','login','form','modal','other','menu','mediaplayer',
        'terms','settings','maps','news','search','chat','bare','gallery','editor','profile',
        'camera','calculator'].index(l_name)
    label_data = np.asarray(val)
    return label_data    # returns array of number value lables arr

def image_mod(image_path):
    with open(image_path, 'rb') as f: # given permissions  
        img = Image.open(image_path).convert('RGB') # convert to color channel
            
    img = img.resize((96,96)) # resize properly
    numpydata = np.asarray(img) # convert to array of 3 channel

    return numpydata# image as array 


# used for unlabeled data full RICO set 
def full_set_conv():
    # for each image in rico
    count = 0
    for imageName in tqdm(os.listdir(image_folder)):
        count = count +1
        if count == max_unlabeled:
            break
        else:
            path = image_folder + imageName # path of image
            # load the image and convert into numpy array
            with open(path, 'rb') as f: # given permissions  
                img = Image.open(path).convert('RGB') # convert to color channel
            
            img = img.resize((96,96)) # resize properly
            width, height = img.size
            numpydata = np.asarray(img) # convert to array of 3 channel
            #np.append(bb, numpydata)
            bb.append(numpydata)

    # convert list to array       
    data = np.asarray(bb, dtype = 'uint8')
    # save binary 
    data.tofile(filename)

full_set_conv() # runs through rico data set converts to arrays then binary
en_conv(rico_en) # should run through each labeled image

print("done.")
