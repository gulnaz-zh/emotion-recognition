import numpy
import os
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras import backend as K
from preprocess import load_labels, load_data_for_cnn3d

K.set_image_data_format('channels_last')

image_rows, image_columns, image_depth = 64, 64, 18

# Preprocess data
load_labels()
load_data_for_cnn3d()

if not os.path.exists('cnn3d_finetuning/'):
    os.makedirs('cnn3d_finetuning/')

# Load training images and labels that are stored in numpy array
training_set = numpy.load('results/cnn3d_images.npy')
traininglabels = numpy.load('results/labels.npy')

# MicroExpSTCNN Model
model = Sequential()
model.add(Convolution3D(32, (3, 3, 15), input_shape=(image_depth, image_rows, image_columns, 1), activation='relu'))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, kernel_initializer='random_uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, kernel_initializer='random_uniform'))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'SGD', metrics = ['accuracy'])

model.summary()

filepath="cnn3d_finetuning/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Load pre-trained weights
model.load_weights('models/pretrained3DCNN.hdf5')

# Spliting the dataset into training and validation sets
train_images, validation_images, train_labels, validation_labels =  train_test_split(training_set, traininglabels, test_size=0.2, random_state=4)

# Save validation set in a numpy array
# numpy.save('numpy_validation_dataset/microexpstcnn_val_images.npy', validation_images)
# numpy.save('numpy_validation_dataset/microexpstcnn_val_labels.npy', validation_labels)

# Load validation set from numpy array

# validation_images = numpy.load('numpy_validation_datasets/microexpstcnn_val_images.npy')
# validation_labels = numpy.load('numpy_validation_dataset/microexpstcnn_val_labels.npy')

# Training the model
hist = model.fit(train_images, train_labels, validation_data = (validation_images, validation_labels), callbacks=callbacks_list, batch_size = 16, epochs = 100, shuffle=True)

# Finding Confusion Matrix using pretrained weights

predictions = model.predict(validation_images)
predictions_labels = numpy.argmax(predictions, axis=1)
validation_labels = numpy.argmax(validation_labels, axis=1)
cfm = confusion_matrix(validation_labels, predictions_labels)
print(cfm)
