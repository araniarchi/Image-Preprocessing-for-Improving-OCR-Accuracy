import warnings
warnings.filterwarnings("ignore")
import os
from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
from keras.preprocessing.image import load_img, array_to_img, img_to_array
from tensorflow import keras


input_dir  = Path('input')
train = input_dir / 'train'
train_cleaned = input_dir / 'train_cleaned'
test = input_dir / 'test'

height = 1400
width = 900
patch_height = 20
patch_width = 20
numbers_of_image = 1513
v1 = width - patch_width + 1
v2 = height - patch_height +1
v3 = int((height / patch_height) * (width / patch_width))
v4 = int(height / patch_height)
train_image_index = 1
train_image_name = "2.jpg"
test_image_index = 1
test_image_name = "4.jpg"
numbers_of_epochs = 5
numbers_of_batch_size = 50


train_images = sorted(os.listdir(train))
train_labels = sorted(os.listdir(train_cleaned))
test_images = sorted(os.listdir(test))
print("Total number of images in the training set: ", len(train_images))
print("Total number of cleaned images found: ", len(train_labels))
print("Total number of samples in the test set: ", len(test_images))

samples = train_images[:3] + train_labels[:3]

f, ax = plt.subplots(2, 3, figsize=(20,10))
for i, img in enumerate(samples):
    if i>2:
        img = imread(train_cleaned/img)
    else:
        img = imread(train/img)
    ax[i//3, i%3].imshow(img, cmap='gray')
    ax[i//3, i%3].axis('off')
# plt.show() 


X = []
Y = []

for img in train_images:
    img = load_img(train / img, grayscale=True,target_size=(width,height))
    img = img_to_array(img).astype('float32')/255.
    X.append(img)

for img in train_labels:
    img = load_img(train_cleaned / img, grayscale=True,target_size=(width,height))
    img = img_to_array(img).astype('float32')/255.
    Y.append(img)


X = np.array(X)
Y = np.array(Y)

print("Size of X : ", X.shape)
print("Size of Y : ", Y.shape)


X=X.reshape((numbers_of_image, width, height))
Y=Y.reshape((numbers_of_image, width, height))
print("Size of X : ", X.shape)
print("Size of Y : ", Y.shape)


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


f, ax = plt.subplots(1, 2, figsize=(20,10))
ax[0].imshow(X[train_image_index], cmap='gray')
ax[0].axis('off')
#crete patch and restore back to its original form
s=patch(X[train_image_index])
p=patch_restore(s)
ax[1].imshow(p, cmap='gray')
ax[1].axis('off')
# plt.show() 


Xd=patch(X[0])
for i in range(1,numbers_of_image):
    Xd=np.append(Xd,patch(X[i]),axis=0)

Yd=patch(Y[0])
for i in range(1,numbers_of_image):
    Yd=np.append(Yd,patch(Y[i]),axis=0)

print("Size of Xd : ", Xd.shape)
print("Size of Yd : ", Yd.shape)
div = Xd.shape[0]
print(div)


Xd=Xd.reshape((div,patch_width,patch_height,1))
Yd=Yd.reshape((div,patch_width,patch_height,1))
print("Size of Xd : ", Xd.shape)
print("Size of Yd : ", Yd.shape)


f, ax = plt.subplots(1, 2, figsize=(5,5))
ax[0].imshow(Xd[3000].reshape((patch_width,patch_height)),cmap='gray')
ax[0].axis('on')
ax[1].imshow(Yd[3000].reshape((patch_width,patch_height)),cmap='gray')
ax[1].axis('on')
# plt.show() 

# Split the dataset into training and validation.
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(Xd, Yd, test_size=0.1, random_state=111)
print("Total number of training samples: ", X_train.shape)
print("Total number of validation samples: ", X_valid.shape)


from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Input,Conv3D
from keras.optimizers import SGD, Adam, Adadelta, Adagrad


def build_autoenocder():
    input_img = Input(shape=(patch_width,patch_height,1), name='image_input')
    
    x = Conv2D(64, (3,3), activation='relu', padding='same',name='Conv1')(input_img)
    x = Conv2D(32, (3,3), activation='relu', padding='same', name='Conv2')(x)
    x = Conv2D(32, (3,3), activation='relu', padding='same', name='Conv3')(x)
    x = Conv2D(16, (4,4), activation='relu', padding='same', name='Conv4')(x)
    x = Conv2D(1, (4,4), activation='sigmoid', padding='same', name='Conv5')(x)
    
    #model
    autoencoder = Model(inputs=input_img, outputs=x)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

autoencoder = build_autoenocder()
autoencoder.summary()

# autoencoder.compile(loss='sparse_categorical_crossentropy',
#               optimizer=keras.optimizers.RMSprop(),
#               metrics=['accuracy'])
autoencoder.compile(loss='mse', optimizer='adam',metrics=['accuracy'])


history = autoencoder.fit(X_train, y_train, epochs= numbers_of_epochs, batch_size= numbers_of_batch_size, validation_data=(X_valid, y_valid))
autoencoder.save('model/autoencoder1binary.h5')

test_scores = autoencoder.evaluate(X_valid, y_valid, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])
#graph draw
import matplotlib.pyplot as plt 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('autoencoder1binary1 model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("graph_auto1_binary1.jpg")