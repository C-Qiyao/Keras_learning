from PIL import Image
import numpy as np
import os
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
class_names = ['Chen', 'ZHang','Ma','Han']
#print(imagePaths)
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
    keras.layers.Conv2D(16, kernel_size=(5,5), activation='relu', input_shape=(128, 128,1)),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Conv2D(32, kernel_size=(5,5), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Flatten(),
    #keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])

facelist,idlist=loadImage('Facedata')
testface,testid=loadImage('Facedata_test')
'''
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(facelist[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[idlist[i]])
plt.show()
print('data complete'+'len of id:'+str(len(idlist)))
print(type(facelist))
print(tf.__version__)
'''
model.summary()

model.compile(
                optimizer='adam',
            #optimizer= tf.keras.optimizers.SGD(lr = 0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history=model.fit(facelist, idlist, epochs=10)
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Loss', 'accuracy'], loc='upper right')
plt.show()

model.save_weights('./weigths.h5')
print('weight saved')
test_loss, test_acc = model.evaluate(testface,  testid, verbose=2)

print('\nTest accuracy:', test_acc)
print('\nTest loss', test_loss)

predictions = model.predict(testface)