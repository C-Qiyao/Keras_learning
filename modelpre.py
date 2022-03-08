from PIL import Image
import numpy as np
import os
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
class_names = ['Chen Qiyao', 'Zhang Zy','Ma Dh','Han Yl']
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'yellow'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(4))
  plt.yticks([])
  thisplot = plt.bar(range(4), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
def loadImage(path):
    faceimg=[]
    idimg=[]
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)] 
    print('reading data')
    for imagePath in imagePaths:
 # 读取图片
        PIL_img = Image.open(imagePath).convert('L')   # convert it to grayscale
        img = PIL_img.resize((128, 128),Image.ANTIALIAS)
        img_numpy = np.array(img, 'float')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        #print('ID:'+str(id)+'   size'+str(img.size)+'  raw'+str(PIL_img.size))
        faceimg.append(img_numpy/255.00)
        idimg.append(id)
    return np.array(faceimg),np.array(idimg)

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=(128, 128,1)),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Conv2D(64, kernel_size=(5,5), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    #keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])

testface,testid=loadImage('Facedata_test')
print('data complete'+'len of testid:'+str(len(testid)))
model.load_weights('./weigths.h5')
model.compile(
                optimizer='adam',
            #optimizer= tf.keras.optimizers.SGD(lr = 0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

predictions = model.predict([1,testface[0]])
predictions
