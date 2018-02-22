import sys 
import settings

import cv2
import numpy as np

from keras.models import load_model
from keras.preprocessing import image  

import tensorflow as tf


if settings.PRETRAINED_MODEL == 'VGG16':
    from keras.applications.vgg16 import VGG16, preprocess_input
    pre_model = VGG16(weights = 'imagenet', include_top = False)
    detect_dog_model = VGG16(weights = 'imagenet')
elif  settings.PRETRAINED_MODEL == 'VGG19':
    from keras.applications.vgg19 import VGG19, preprocess_input
    pre_model = VGG19(weights = 'imagenet', include_top = False)
    detect_dog_model = VGG19(weights = 'imagenet')
elif  settings.PRETRAINED_MODEL == 'InceptionV3':
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    pre_model = InceptionV3(weights = 'imagenet', include_top = False)
    detect_dog_model = InceptionV3(weights = 'imagenet')
elif  settings.PRETRAINED_MODEL == 'Resnet50':
    from keras.applications.resnet50 import ResNet50, preprocess_input
    pre_model = ResNet50(weights = 'imagenet', include_top = False)
    detect_dog_model = ResNet50(weights = 'imagenet')
elif  settings.PRETRAINED_MODEL == 'Xception':
    from keras.applications.xception import Xception, preprocess_input
    pre_model = Xception(weights = 'imagenet', include_top = False)
    detect_dog_model = Xception(weights = 'imagenet')

pre_model.summary()
model = load_model(settings.MODEL_TOP_LAYER)
model.summary()

graph = tf.get_default_graph()

def predict_img(img_path):

    with graph.as_default():

        tensor = path_to_tensor(img_path)
        pre_processed_img = preprocess_input(tensor)
        bottleneck_feature = pre_model.predict(tensor)
        predicted_vector = model.predict(bottleneck_feature)
        return settings.DOG_NAMES[np.argmax(predicted_vector)]


def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)	


face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def dog_detector(img_path):
    with graph.as_default():
        img = preprocess_input(path_to_tensor(img_path))
        prediction = np.argmax(detect_dog_model.predict(img))
        print(prediction)
        return ((prediction <= 268) & (prediction >= 151)) 




