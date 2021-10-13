from PIL import Image
import os
from utils.mypath import MyPath
from glob import glob 
import numpy as np 
import pandas as pd
from numpy import asarray
from tqdm import tqdm # progress bar
import csv

en_filename_y = "rico_20/rico_binary/test_y.bin"
#with open(en_filename_y,'rb') as f3:
##    y = np.fromfile(f3, dtype=np.uint8)
#    for obj in (y):
#        print(obj)

with open('/bsuhome/hkiesecker/scratch/imageClassification/US/rico_20/rico_binary/class_names_verbal.txt') as f:
    classes = f.read().splitlines()

value = -1
if value == -1:
    print('yesy')
print(len(classes))
print(classes[2])
        