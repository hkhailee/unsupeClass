import os 
import pandas as pd 
import numpy as np 
import tqdm
"""
Author : Hailee Kiesecker
based on the evaluation of the two sample files our creation of a bin file 
has the same format as that of the stl, only it is 256*256 and not 96*96 
having this difference could lead to issues in classification. likely will
not know until after pretext task 

notes after pretext step:


"""

from PIL import Image
import os.path
num =0 
while num != 100:
    filename = os.path.join('/bsuhome/hkiesecker/scratch/imageClassification/US/rico_image/'+ str(num)+'.jpg')
    img = Image.open(filename)
    #print (img.size)

# paths
test = '/bsuhome/hkiesecker/scratch/imageClassification/US/rico_20/rico_binary/test.bin'
binFile = "/bsuhome/hkiesecker/scratch/imageClassification/US/rico_20/rico_binary/rico_unlabeled_RESIZED.bin"
stlFile = "/bsuhome/hkiesecker/scratch/imageClassification/US/stl-10/stl10_binary/train_X.bin"

#test_y's
create = "/bsuhome/hkiesecker/scratch/imageClassification/US/rico_20/rico_binary/test_y_RESIZED.bin"
give = "/bsuhome/hkiesecker/scratch/imageClassification/US/stl-10/stl10_binary/test_y.bin"

#test_x's
created_X = "/bsuhome/hkiesecker/scratch/imageClassification/US/rico_20/rico_binary/test_X_RESIZED.bin"
given_X = "/bsuhome/hkiesecker/scratch/imageClassification/US/stl-10/stl10_binary/test_X.bin"

#visual 
test_y_rico  = "/bsuhome/hkiesecker/scratch/imageClassification/US/rico_20/rico_binary/test_y_RESIZED.bin"
test_y_golden = "/bsuhome/hkiesecker/scratch/imageClassification/US/stl-10/stl10_binary/test_y.bin"
"""
with open(test_y_rico) as classL:
    viewClassL = np.fromfile(classL, dtype=np.uint8)
    for obj in viewClassL:
        print(obj)
"""

#getting values in bin file
# a = np.fromfile(binFile, dtype=np.uint8) # large file

# script to confirm that the label binarys are the same 
def create_label(created, given):
    with open(created,'rb') as f1:
        e_c = np.fromfile(f1, dtype=np.uint8)
        PRINTLIST = []
        for obj in e_c:
            PRINTLIST.append(obj)
        print("label evaluation OURS : ", PRINTLIST)

    with open(given,'rb') as f2:
        e_g = np.fromfile(f2, dtype=np.uint8)
        PRINTLIST = []
        for obj in e_g:
            PRINTLIST.append(obj)
        print("label evaluation THEIRS : ", PRINTLIST)
    
    print("created shape: ", e_c.shape) 
    print("given shape : ", e_g.shape)


# # loading binary file 
def loadfile(data_file, shape ):
    labels = None
    path_to_data = data_file
    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, shape, shape)) #96
        images = np.transpose(images, (0, 1, 3, 2))

    return images, labels
### evaluation --------------------------------------------------------
print("evaluation for labeles")
print("-------------------------------------")
create_label(create, give)

print("evaluation for images with labels")
print("-------------------------------------")
created_data, labels = loadfile(created_X, 256)
given_data, labels = loadfile(given_X, 96)
print("created Shape : ", created_data.shape)
print("given Shape : ", given_data.shape)

print("evaluation of unlabeled")
print("-------------------------------------")
data, labels = loadfile(binFile, 256)
print("created unlabled : ", data.shape)
data_given , labels = loadfile(stlFile, 96)
print("given labeled (unlabeled was empty): ", given_data.shape)

#for unlabeled #########################################
#data, labels = loadfile(binFile)
#img, target = data[0], 255 # 255 is an ignore index
#data = loadfile(test) # laoding binary data from rico
#lables = np.asarray[-1] * data.shape[0] # assining no lables to the data
#print(data.shape)