
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator,NumpyArrayIterator
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import models, layers, optimizers, losses
import numpy as np
from  keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import csv, sys
import random
import keras.backend as K


def extract_features(conv_base, datagen ,sample_count):
	features=np.zeros(shape=(sample_count, 10,10,512))
	labels=np.zeros(shape=(sample_count,num_classes))


	i=0
	features=conv_base.predict(datagen[0])
	labels=datagen[1]
	# for inputs_batch, labels_batch in datagen:
	# 	print('Extraction {}th label'.format(i))
	# 	features_batch=conv_base.predict(inputs_batch)
	# 	features[i*batch_size:(i+1)*batch_size]=features_batch
	# 	labels[i*batch_size: (i+1)*batch_size]=labels_batch
	# 	i+=1
	# 	if i*batch_size>=sample_count:
	# 		break
	#print(features.shape, labels.shape)
	return features, labels



def multitask_loss(y_true, y_pred):
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))


x_dict={}
img_rows, img_cols=350, 350
channels=3

train_full=False
train_sample_size=10000

batch_size=20
num_classes=2
epochs=20

#img_rows, img_cols=250,250



counter=0
image_files=os.listdir('/Users/nex03343/data/facial_expressions/images')
for img_file in image_files:
	img_path=os.path.join('/Users/nex03343/data/facial_expressions/images',img_file)
	# print('converting image {} into an np array'.format(img_path))
	img=image.load_img(img_path, target_size=(img_rows, img_cols, channels))
	img_array=image.img_to_array(img)
	x_dict[img_file]=img_array
	if counter==train_sample_size+10 and train_full==False:
		break
	counter+=1


print('Image files Loaded')

y_dict={}


y_dict={}

with open('/Users/nex03343/Desktop/CS231N/project/legend.csv', 'rt', encoding='utf8') as f:
	reader=csv.reader(f)
	for row in reader:
		cid=row[-1]
		emotion=row[-2]

		if cid.strip()=='1' and emotion.strip().upper()=='HAPPINESS':
			y_dict[row[1]]=[1,1]
		elif cid.strip()=='0' and emotion.strip().upper()=='HAPPINESS':
			y_dict[row[1]]=[0,1]
		elif cid.strip()=='1' and emotion.strip().upper()!='HAPPINESS':
			y_dict[row[1]]=[1,0]
		else:
			y_dict[row[1]]=[0,0]

		# emotion=emotion.upper()
		# if emotion=='HAPPINESS':
		# 	#y_data[i]=[1,0,0,0,0]
		# 	#y_dict[row[1]]=[1,0,0,0,0,0,0,0]
		# 	y_dict[row[1]]
		# elif emotion=='SADNESS':
		# 	#y_data[i]=[0,1,0,0,0]
		# 	y_dict[row[1]]=[0,1,0,0,0,0,0,0]
		# elif emotion=='ANGER':
		# 	#y_data[i]=[0,0,1,0,0]
		# 	y_dict[row[1]]=[0,0,1,0,0,0,0,0]
		# elif emotion=='CONTEMPT':
		# 	#y_data[i]=[0,0,0,1,0]
		# 	y_dict[row[1]]=[0,0,0,1,0,0,0,0]
		# elif emotion=='DISGUST':
		# 	y_dict[row[1]]=[0,0,0,0,1,0,0,0]
		# elif emotion=='FEAR':
		# 	y_dict[row[1]]=[0,0,0,0,0,1,0,0]
		# elif emotion=='SURPRISE':
		# 	y_dict[row[1]]=[0,0,0,0,0,0,1,0]
		# else:
		# 	y_dict[row[1]]=[0,0,0,0,0,0,0,1]					

		# print(i)

print('csv file loaded')


s_xkeys=set(x_dict.keys())
s_ykeys=set(y_dict.keys())

i_xy_keys=s_xkeys & s_ykeys

x_data_dict={}
y_data_dict={}


if train_full==True:
	train_sample_size=len(i_xy_keys)
for key in list(i_xy_keys)[:train_sample_size]:
	x_data_dict[key]=x_dict[key]
	y_data_dict[key]=y_dict[key]


x_data=[]
y_data=[]

x_data=list(x_data_dict.values())
y_data=list(y_data_dict.values())

del_indices=[]

for i in range(len(x_data)):
	if x_data[i].shape!=(img_rows,img_cols,channels):
		del_indices.append(i)


del_indices.sort(reverse=True)

for i in del_indices:
	x_data.pop(i)
	y_data.pop(i)




val_count=len(x_data)//5
x_val=[]
y_val=[]


val_indices=random.sample(range(1,len(x_data)), val_count)
val_indices.sort(reverse=True)
for idx in val_indices:
	x_val.append(x_data.pop(idx))
	y_val.append(y_data.pop(idx))



x_train=np.zeros((len(x_data),350,350,3))
for i in range(1,len(x_train)):

	x_train[i]=np.array(x_data[i])
	
y_train=np.array(y_data)


x_val_np=np.zeros((len(x_val),350,350,3))
for i in range(1,len(x_val)):
	x_val_np[i]=x_val[i]


x_val=x_val_np
y_val=np.array(y_val)


print('data processing finished x_train size:{}  y_train:{} x_val:{} y_val{} '.format(x_train.shape, y_train.shape, x_val.shape, y_val.shape))

x_train/=255.
x_val/=255.


# train_datagen=ImageDataGenerator(rotation_range=40,
# 	width_shift_range=0.2,
# 	height_shift_range=0.2,
# 	rescale=1./255,
# 	shear_range=0.2,
# 	zoom_range=0.2,
# 	horizontal_flip=True,
# 	fill_mode='nearest')

# train_datagen=ImageDataGenerator()

# datagen_batch=10
# val_datagen=ImageDataGenerator()

# train_gen=train_datagen.flow(x_train,y_train,batch_size=datagen_batch,  shuffle=False)
# val_gen=val_datagen.flow(x_val,y_val,batch_size=10, shuffle=False)


#Model


print('Training begins...')



# model=Sequential()
# model.add(Conv2D(256, kernel_size=(3,3), input_shape=(img_rows,img_cols,channels))) 
# model.add(Activation('relu'))
# model.add(Conv2D(128,(3,3))) 
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.5))

# # model.add(Conv2D(128,(3, 3), padding='same'))
# # model.add(Activation('relu'))
# # # model.add(LeakyReLU(alpha=0.3))

# model.add(Conv2D(64, (3, 3)))
# # model.add(BatchNormalization())
# model.add(Activation('relu'))
# # # model.add(LeakyReLU(alpha=0.3))

# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))


# model.add(Flatten())
# model.add(Dense(512))
# # model.add(BatchNormalization())
# model.add(Activation('relu'))
# # model.add(LeakyReLU(alpha=0.3))
# model.add(Dense(256))
# model.add(Activation('relu'))

# model.add(Dense(num_classes))
# model.add(Activation('softmax'))



###Alternate model


print('Creation of top layers ...')

model=models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=10*10*512))
model.add(layers.Dropout(0.5))

# model.add(BatchNormalization())
# model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(num_classes ,activation='sigmoid'))



conv_base=VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))
print('Feature extraction begins ....')
train_features, train_labels=extract_features(conv_base, (x_train,y_train), len(x_train))
val_features, val_labels=extract_features(conv_base, (x_val,y_val), len(x_val))

train_features=np.reshape(train_features, (len(x_train),10*10*512))
val_features=np.reshape(val_features, (len(x_val),10*10*512))


model.compile(optimizer=optimizers.adam(lr=1e-4),
	loss=losses.binary_crossentropy,
	metrics=['accuracy'])



print ('Training top layers...')
checkpoint = ModelCheckpoint(filepath='/Users/nex03343/Desktop/CS231N/project/model_mt_vgg16.hdf5', monitor='val_acc',\
							verbose=1, save_best_only=True, mode='max')
csv_logger = CSVLogger('/Users/nex03343/Desktop/CS231N/project/training_mt_vgg16.log')

# filepath='/Users/nex03343/Desktop/CS231N/project/modelcheckpoint.hdf5'
#checkpoint=ModelCheckpoint(filepath, monitor='val_acc',verbose=1, save_best_only=True, mode='max')

history=model.fit(train_features, train_labels,
	epochs=epochs,
	batch_size=batch_size,
	verbose=1,
	shuffle=True,
	#steps_per_epoch=200,
	validation_data=(val_features, val_labels),
	callbacks=[csv_logger, checkpoint])








# history=model.fit_generator(
# 	train_gen,
# 	epochs=epochs,
# 	verbose=1,
# 	steps_per_epoch=spe,
# 	validation_data=val_gen,
# 	use_multiprocessing=False)






## Save Model
# model_json = model.to_json()
# with open("/Users/grewalrakesh/data/model_mt.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("/Users/grewalrakesh/data/model_mt.h5")



acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



