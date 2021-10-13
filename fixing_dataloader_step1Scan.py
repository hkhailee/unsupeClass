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

Removing excess data from rico_20 unlabeled test_x test_y due to dataloading 
error. Assumption: 
        dataset needs to be divisible by the number of processes. 
        we want 8 processes. therefore the testing dataset needts to be of size,
        1336 test_x_y
        6256 unlabeled

THIS IS A CPU JOB
"""

filename = 'rico_20/rico_binary/rico_unlabeled.bin'
en_filename_X = "rico_20/rico_binary/test_X.bin"
en_filename_y = "rico_20/rico_binary/test_y.bin"

rico_en = '/bsuhome/hkiesecker/scratch/imageClassification/classTrain_ui.csv' 
image_folder = "/bsuhome/hkiesecker/scratch/imageClassification/US/rico_images/"

max_unlabeled = 6256
max_labeled = 1336

## reading unlabeled data bin and cutting out a few ending files. // might (will) affect results later but thats a future hailee issue.
with open(filename,'rb') as f1:
    count = 0 
    unlabeled = np.fromfile(f1, dtype=np.uint8)
    UNLABELEDLIST = []
    for obj in tqdm(unlabeled):
        if count == max_unlabeled:
            break
        else:
            UNLABELEDLIST.append(obj)

# save new amounts
# convert list to array       
data = np.asarray(UNLABELEDLIST, dtype = 'uint8')
# save binary 
data.tofile('rico_20/rico_binary/rico_unlabeled_RESIZED.bin')

## reading and removing X (desc)
with open(en_filename_X,'rb') as f2:
    count = 0 
    x = np.fromfile(f2, dtype=np.uint8)
    XLIST = []
    for obj in tqdm(x):
        if count == max_labeled:
            break
        else:
            XLIST.append(obj)

# convert list to array       
data1 = np.asarray(XLIST, dtype = 'uint8')
# save binary 
data1.tofile('rico_20/rico_binary/test_X_RESIZED.bin')

## reading and removing Y (desc)
with open(en_filename_y,'rb') as f3:
    count = 0 
    y = np.fromfile(f3, dtype=np.uint8)
    YLIST = []
    for obj in tqdm(y):
        if count == max_labeled:
            break
        else:
            YLIST.append(obj)
# convert list to array       
data2 = np.asarray(YLIST, dtype = 'uint8')
# save binary 
data2.tofile('rico_20/rico_binary/test_Y_RESIZED.bin')


### evaluate sizes of each list 
print("#######################################################")
print("#######################################################")
print("#######################################################")
print("#######################################################")
print("#######################################################")
print("should be for X 1336: ", len(XLIST))
print("should be for y 1336: ", len(YLIST))
print("should be for unlabeled 6256: ", len(UNLABELEDLIST))
print("#######################################################")
print("#######################################################")
print("#######################################################")
print("#######################################################")
print("#######################################################")





