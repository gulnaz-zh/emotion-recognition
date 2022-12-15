import numpy as np
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
import keras.models as km
import os,cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
        

def visualize_results(history):
    # Plot the accuracy and loss curves
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig("accuracy.png")
    plt.figure()
 
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
 
    plt.savefig("loss.png")

if not os.path.exists('vgg16_finetuning/'):
    os.makedirs('vgg16_finetuning/')

#dimensions of the images
img_width, img_height = 48, 48

#image directories
epochs = 64
batch_size = 128

#channels ordering
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

input_tensor = Input(shape=input_shape)
base_model = km.load_model("models/pretrainedVGG16.hdf5")
# base_model.summary()
print('Model loaded.')

for layer in base_model.layers[:-1]:
	layer.trainable = False

# Create the model
model = Sequential()
for layer in base_model.layers[:-1]: # go through until last layer
	model.add(layer)

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax', name='dense_4'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=['acc'])

# Load training images and labels that are stored in numpy array
training_set = np.load('results/vgg16_images.npy')
traininglabels = np.load('results/labels.npy')

# Spliting the dataset into training and validation sets
train_images, validation_images, train_labels, validation_labels =  train_test_split(training_set, traininglabels, test_size=0.2, random_state=4)

# Training the model
history = model.fit(train_images, train_labels, validation_data = (validation_images, validation_labels), batch_size = 16, epochs = 100, shuffle=True)

visualize_results(history)

(loss, acc) = model.evaluate(
	validation_images, validation_labels,
	steps=validation_images.samples/batch_size,
	max_queue_size=validation_images.samples/batch_size * 2)

print("[INFO] loss: {:.2f}".format(loss))
print("[INFO] accuracy: {:.2f}".format(acc * 100))

#Model Save
model.save_weights('vgg16_finetuning/fine_tuned_weights.hdf5')
model.save('vgg16_finetuning/fine_tuned_model.hdf5')

