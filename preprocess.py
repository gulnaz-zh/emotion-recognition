import pandas as pd
import numpy as np
import os
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import keras.models as models
from keras.preprocessing.image import img_to_array
from keras import backend as K

K.set_image_data_format('channels_last')

path = "dataset/"
emotion_count = 6
if not os.path.exists('results/'):
    os.makedirs('results/')

# Load models for emotion classification
vgg = models.load_model('models/pretrainedVGG16.hdf5')
##vgg.summary()
cnn3d = models.load_model('models/pretrained3DCNN.hdf5')
##smic.summary()
cnn3d_ft = models.load_model('models/pretrained3DCNN_fine_tuned.hdf5')
##football.summary()

def load_labels():
    labels_range = [0]
    label = 0
    labels_cnt = 0
    directorylisting = os.listdir(path)
    directorylisting.sort()

    for emotion in directorylisting:
        emotionpath = path + emotion + "/"
        imagelisting = os.listdir(emotionpath)
        imagelisting.sort()
        for image in imagelisting:
            labels_cnt+=1
        labels_range.append(labels_cnt)
        label += 1
    traininglabels = np.zeros((len(labels_cnt), ), dtype = int)
    for label in range(len(labels_range)-1):
        traininglabels[labels_range[label]:labels_range[label+1]] = label
    traininglabels = np.asarray(traininglabels)

    np.save('results/labels.npy', traininglabels)

def load_data_for_vgg():
    training_list = []
    directorylisting = os.listdir(path)
    directorylisting.sort()

    for emotion in directorylisting:
        emotionpath = path + emotion + "/"
        imagelisting = os.listdir(emotionpath)
        imagelisting.sort()
        print("Loading images from: " + emotionpath)
        for image in imagelisting:
            imagepath = emotionpath + image
            img = cv2.imread(imagepath)
            grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grayimage = cv2.resize(grayimage, (48, 48))
            grayimage = grayimage.astype("float") / 255.0
            grayimage = img_to_array(grayimage)
            grayimage = np.expand_dims(grayimage, axis=0)
            training_list.append(grayimage)
    print("Total # of images: " + str(len(training_list)))
    training_list = np.asarray(training_list)
    np.save('results/vgg16_images.npy', training_list)

def run_vgg():
    load_data_for_vgg()
    vgg_training_list = np.load('results/vgg16_images.npy')
    vgg_training_labels = np.load('results/labels.npy')
    predictions = []
    predictions_labels = []
    for i in range(len(vgg_training_list)):
        pred = vgg.predict(vgg_training_list[i])
        predictions.append(pred)
        predictions_labels.append(np.argmax(pred))
    predictions = np.array(predictions).reshape(len(predictions), emotion_count)
    cfm = confusion_matrix(vgg_training_labels, predictions_labels)
    print("VGG16 emotion classification results:")
    print(cfm)
    print(classification_report(vgg_training_labels, predictions_labels, digits=4))
    np.save('results/vgg16_predictions.npy', predictions)

def load_data_for_cnn3d():
    training_list = []
    directorylisting = os.listdir(path)
    for emotion in directorylisting:
        emotionpath = path + emotion + "/"
        print("Loading images from: " + emotionpath)
        imagelisting = os.listdir(emotionpath)
        for image in imagelisting:
            imagepath = emotionpath + image
            framerange = [x for x in range(18)]
            frames = []
            for frame in framerange:
                   image = cv2.imread(imagepath)
                   imageresize = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
                   grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
                   frames.append(grayimage)
            frames = np.asarray(frames)
            videoarray = np.rollaxis(np.rollaxis(frames, 2, 0), 2, 0)
            training_list.append(videoarray)
        print(str(len(imagelisting)) + " images are downloaded")
    training_list = np.asarray(training_list)
    trainingsamples = len(training_list)
    print("Total # of images: " + str(trainingsamples))

    training_s = np.zeros((trainingsamples, 1, 64, 64, 18))

    for h in range(trainingsamples):
        training_s[h][0][:][:][:] = training_list[h, :, :, :]

    training_s = training_s.astype('float')
    training_s -= np.mean(training_s)
    training_s /= np.max(training_s)

    training_s = np.rollaxis(training_s, 4, 1)
    training_s = np.rollaxis(training_s, 4, 2)
    training_set = np.rollaxis(training_s, 4, 3)

    np.save('results/cnn3d_images.npy', training_set)

def run_cnn3d():
    load_data_for_cnn3d()
    training_list = np.load('results/cnn3d_images.npy')
    training_labels = np.load('results/labels.npy')

    training_labels[training_labels == 1] = 0
    training_labels[training_labels == 2] = 1
    training_labels[training_labels == 3] = 0
    training_labels[training_labels == 4] = 2
    training_labels[training_labels == 5] = 0

    predictions = cnn3d.predict(training_list)
    predictions_labels = np.argmax(predictions, axis=1)
    
    print("CNN3D emotion classification results:")
    cfm = confusion_matrix(training_labels, predictions_labels)
    print(cfm)
    print(classification_report(training_labels,predictions_labels, digits=4))
    np.save('results/cnn3d_predictions.npy', predictions)

def run_cnn3d_ft():
    training_list = np.load('results/cnn3d_images.npy')
    training_labels = np.load('results/labels.npy')

    training_labels[training_labels == 1] = 0
    training_labels[training_labels == 2] = 1
    training_labels[training_labels == 3] = 0
    training_labels[training_labels == 4] = 2
    training_labels[training_labels == 5] = 3

    predictions = cnn3d_ft.predict(training_list)
    predictions_labels = np.argmax(predictions, axis=1)

    print("Fine-tuned CNN3D emotion classification results:")
    cfm = confusion_matrix(training_labels, predictions_labels)
    print(cfm)
    print(classification_report(training_labels,predictions_labels, digits=4))
    np.save('results/cnn3d_ft_predictions.npy', predictions)

load_labels()
run_vgg()
run_cnn3d()
run_cnn3d_ft()