import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import ZeroPadding2D, \
    Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Dense, \
    Dropout, Flatten, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import tensorflow.keras.backend as K
import cv2
import dlib
import os
import matplotlib.pyplot as plt


# Define VGG_FACE_MODEL architecture
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

# Load VGG Face model weights
model.load_weights('vgg_face_weights.h5')

# Remove Last Softmax layer and get the image embedding
vgg_face = Model(inputs=model.layers[0].input,
                 outputs=model.layers[-2].output)

# Prepare Training Data
x_train = []
y_train = []
person_folders = os.listdir('faces')
person_rep = dict()
for i, person in enumerate(person_folders):
  person_rep[i] = person
  image_names = os.listdir('faces/'+person+'/')
  for image_name in image_names:
    # resize the image to 224*224
    img = load_img('faces/'+person+'/'+image_name, target_size=(224,224))
    # add channels: img.shape = (224, 224, 3) for RGB
    img = img_to_array(img)
    # add the number of images: img.shape = (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    # convert the image to the format the model requires
    img = preprocess_input(img)
    # get the embedding of the image img_encode.shape = (1,2622)
    img_encode = vgg_face(img)
    # remove the single dimensional np.squeeze(K.eval(img_encode)).shape = (2622,)
    x_train.append(np.squeeze(K.eval(img_encode)).tolist())
    y_train.append(i)
x_train = np.array(x_train)
y_train = np.array(y_train)

# Prepare Validation Data
x_val = []
y_val = []
person_folders = os.listdir('val_faces/')
for i, person in enumerate(person_folders):
    image_names = os.listdir('val_faces/' + person + '/')
    for image_name in image_names:
        img = load_img('val_faces/' + person + '/' + image_name, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        img_encode = vgg_face(img)
        x_val.append(np.squeeze(K.eval(img_encode)).tolist())
        y_val.append(i)

x_val = np.array(x_val)
y_val = np.array(y_val)


# Define softmax classifier,
classifier_model = Sequential()
classifier_model.add(Dense(units=100, input_dim=x_train.shape[1]))
classifier_model.add(Activation('tanh'))
classifier_model.add(Dropout(0.3))
classifier_model.add(Dense(units=10))
classifier_model.add(Activation('tanh'))
classifier_model.add(Dropout(0.2))
classifier_model.add(Dense(units=2))
classifier_model.add(Activation('softmax'))
classifier_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                         optimizer='sgd', metrics=['accuracy'])
classifier_model.fit(x_train, y_train, epochs=100,
                     validation_data=(x_val, y_val))


dnnFaceDetector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")


def plot(img):
    plt.figure(figsize=(8, 4))
    plt.imshow(img[:, :, ::-1])
    plt.show()


# Label names for class numbers
person_rep = {0: 'Kobe Bryant', 1: 'Dwyane Wade'}


for img_name in os.listdir('images/'):
    if img_name == 'crop_img.jpg':
        continue
    # Load Image
    img = cv2.imread('images/' + img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect Faces
    rects = dnnFaceDetector(gray, 1)
    left, top, right, bottom = 0, 0, 0, 0
    for (i, rect) in enumerate(rects):
        # Extract Each Face
        left = rect.rect.left()  # x1
        top = rect.rect.top()  # y1
        right = rect.rect.right()  # x2
        bottom = rect.rect.bottom()  # y2
        width = right - left
        height = bottom - top
        img_crop = img[top:top + height, left:left + width]
        cv2.imwrite('images/crop_img.jpg', img_crop)

        # Get Embeddings
        crop_img = load_img('images/crop_img.jpg', target_size=(224, 224))
        crop_img = img_to_array(crop_img)
        crop_img = np.expand_dims(crop_img, axis=0)
        crop_img = preprocess_input(crop_img)
        img_encode = vgg_face(crop_img)

        # Make Predictions
        embed = K.eval(img_encode)
        person = classifier_model.predict(embed)
        name = person_rep[np.argmax(person)]
        os.remove('images/crop_img.jpg')
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        img = cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_COMPLEX ,
                          1, (255, 255, 255), 2, cv2.LINE_AA)
        img = cv2.putText(img, str(np.max(person)), (right, bottom + 10),
                          cv2.FONT_HERSHEY_COMPLEX , 0.5, (255, 255, 255), 1,
                          cv2.LINE_AA)
    # Save images with bounding box,name and accuracy
    cv2.imwrite('Predictions/' + img_name, img)
    plot(img)
print("Finished")
