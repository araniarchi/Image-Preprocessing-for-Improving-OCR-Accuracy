import warnings
warnings.filterwarnings("ignore")
import os
from pathlib import Path
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
from keras.preprocessing.image import load_img, array_to_img, img_to_array
from keras.models import load_model
from PIL import Image
# autoencoder1 = load_model('model/618/autoencoder1relu.h5',compile=False)
autoencoder2 = load_model('model/1500/autoencoder1binary.h5',compile=False)
# autoencoder3 = load_model('model/618/autoencoder2mse.h5',compile=False)
# autoencoder4 = load_model('model/618/autoencoder2binary.h5',compile=False)
# autoencoder5 = load_model('model/618/my_model.h5',compile=False)

height = 1400
width = 900
patch_height = 20
patch_width = 20
v1 = width - patch_width + 1
v2 = height - patch_height +1
v3 = int((height / patch_height) * (width / patch_width))
v4 = int(height / patch_height)
test_image_name = "cam1.jpg"

import pygame
T_image = cv2.imread('input/test/'+test_image_name,0)
T_height, T_width = T_image.shape[:2]
#T_height = T_image.get_height()
#T_width = T_image.get_width()
print("height......Height")
print(T_height)

input_dir  = Path('input')
test = input_dir / 'test'
test_images = sorted(os.listdir(test))
print("Total number of samples in the test set: ", len(test_images))
k=test_images.index(test_image_name)
print("index no:",k)

testin = load_img(test/ test_images[k], grayscale=True, target_size=(width,height))
testin = img_to_array(testin).astype('float32')/255
testin=testin.reshape((width,height))
# imshow(testin)


#create patch of size 28*30 
#total no of patch from one image=270
def patch(x):
    xd=[]
    for i in range(0,v1,patch_width):
        for j in range(0,v2,patch_height):
            xd.append(x[i:i+patch_width,j:j+patch_height])
    xd=np.array(xd)
    return xd

def patch_restore(b):
    
    res=np.empty((0,height), np.float32)
    cur=b[0]
    for i in range(1,v3):
        if i%v4==0:
            res=np.append(res,cur,axis=0)
            cur=b[i]
        else:
            cur=np.append(cur,b[i],axis=1)
    res=np.append(res,cur,axis=0)
    return res  
testpatch=patch(testin)
p=np.expand_dims(testpatch, axis=3)
print('shape of patch ',p.shape)

# pred1=autoencoder1.predict(p)
pred2=autoencoder2.predict(p)
# pred3=autoencoder3.predict(p)
# pred4=autoencoder4.predict(p)
# pred5=autoencoder5.predict(p)
# pred1=np.squeeze(pred1)
pred2=np.squeeze(pred2)
# pred3=np.squeeze(pred3)
# pred4=np.squeeze(pred4)
# pred5=np.squeeze(pred5)
# pred1=patch_restore(pred1)
pred2=patch_restore(pred2)
# pred3=patch_restore(pred3)
# pred4=patch_restore(pred4)
# pred5=patch_restore(pred5)
print(pred2.shape)


f, ax = plt.subplots(5,2, figsize=(10,8))
ax[0,0].imshow(testin, cmap='gray')
# ax[0,1].imshow(pred1, cmap='gray')
# ax[1,0].imshow(testin, cmap='gray')
ax[1,1].imshow(pred2, cmap='gray')
# ax[2,0].imshow(testin, cmap='gray')
# ax[2,1].imshow(pred3, cmap='gray')
# ax[3,0].imshow(testin, cmap='gray')
# ax[3,1].imshow(pred4, cmap='gray')
# ax[4,0].imshow(pred5, cmap='gray')
# plt.show()
#dim = (T_height,T_width)
# pred1 = cv2.resize(pred1, (T_width, T_height), interpolation = cv2.INTER_AREA) 
pred2 = cv2.resize(pred2, (T_width, T_height), interpolation = cv2.INTER_AREA)
# pred3 = cv2.resize(pred1, (T_width, T_height), interpolation = cv2.INTER_AREA)
# pred4 = cv2.resize(pred1, (T_width, T_height), interpolation = cv2.INTER_AREA)
# pred5 = cv2.resize(pred1, (T_width, T_height), interpolation = cv2.INTER_AREA)


from matplotlib import image

# image.imsave('output/withoutpoolin1mse.png',pred1,cmap='gray')
image.imsave('output/'+test_image_name,pred2,cmap='gray')
# image.imsave('output/withpooling2mse.png',pred3,cmap='gray')
# image.imsave('output/withpooling2binary.png',pred4,cmap='gray')
# image.imsave('output/mse.png',pred5,cmap='gray')
