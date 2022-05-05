import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
class_names = ['Chen Qiyao', 'Zhang Zy','Ma Dh','Han Yl','unknow']
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)

model = keras.Sequential([
    keras.layers.Conv2D(16, kernel_size=(5,5), activation='relu', input_shape=(128, 128,1)),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Conv2D(32, kernel_size=(5,5), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(50, activation='relu'),
    #keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])
model.load_weights('./weigths.h5')
model.compile(
                optimizer='adam',
            #optimizer= tf.keras.optimizers.SGD(lr = 0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print('model loaded')
a=0
while True:
    pre=0
    sucess, img = cap.read()

     # 转为灰度图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+w), (255, 0, 0))
        imgcut=gray[y: y + h, x: x + w]
        imgcut=cv2.resize(imgcut,(128,128))
        IMAG=np.array([imgcut/255])
    if faces!=():
        cv2.imshow('facecut',imgcut)
        predictions = model.predict(IMAG)
        a=np.argmax(predictions)
        if predictions[0][a] <0.9:
            a=-1
        pre=predictions[0][a]
    cv2.putText(img, class_names[a]+str('%.2f' % (pre*100))+'%', (50,50), font, 1, (0, 0, 255), 3)
    cv2.imshow('face',img)
    k = cv2.waitKey(1)
    if k == 27:    #按 'ESC' to quit
        break
        