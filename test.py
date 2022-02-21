import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('digit.model')

# for x in range(1,10):
ipn = int(input("select a pic(0-8):"))
while 0 <= ipn < 10:
    img = cv.imread(f'{ipn}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'The figure should be: {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
    ipn = int(input("select a pic(0-8):"))
