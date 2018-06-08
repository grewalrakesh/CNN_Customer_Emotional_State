from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator,NumpyArrayIterator
from keras.applications import VGG16
from keras import models, layers, optimizers
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt



def create_newimage(image_tensor, aug_tech, attrib, fpath):

	if aug_tech=='rotation':
		idg=ImageDataGenerator(rotation_range=attrib,)
	elif aug_tech=='wsr':
		idg=ImageDataGenerator(width_shift_range=attrib)
	elif aug_tech=='hsr':
		idg=ImageDataGenerator(horizontal_flip=attrib)
	elif aug_tech=='sr':
		idg=ImageDataGenerator(shear_range=attrib)
	elif aug_tech=='hf':
		idg=ImageDataGenerator(horizontal_flip=attrib)
	elif idg=='zr':
		idg=ImageDataGenerator(zoom_range=attrib)


	x_gen=idg.flow(image_tensor, batch_size=1, shuffle=False, save_to_dir=fpath);

	return x_gen[0][0]

 

img=image.load_img('/Users/nex03343/Desktop/CS231N/project/finalreport/cnn-predicting-customers/lfw1.jpg')
img_tensor=image.img_to_array(img)
img_tensor=img_tensor.astype('float32')
img_tensor=np.expand_dims(img_tensor, axis=0)

_,img_rows, img_cols, channels=img_tensor.shape
# define data preparation

# fit parameters from data

fpath='/Users/nex03343/Desktop/'
augmented_img=create_newimage(img_tensor, 'rotation', 90,fpath)


plt.imshow(image.array_to_img(augmented_img))
	# show the plotj
plt.show()






