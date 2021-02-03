import random
import numpy as np
from keras.datasets import mnist  # our data set
import matplotlib.pyplot as plt
from keras.models import Sequential  # CNN architecture
from keras.layers import Dense, Conv2D, Flatten  # the layers in the CNN
from keras.utils import to_categorical
import cv2 as cv
from keras.utils import plot_model
import tensorflow as tf

# Load the data and split it into train set and test set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data to fit the model
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# One-Hot Encoding
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Build the CNN model
model = Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
hist = model.fit(x_train, y_train_one_hot, validation_data=(x_test, y_test_one_hot), epochs=5)


# Visualize the model accuracy
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train','Val'],loc='upper left')
# plt.show()


# predict on the random 5 test images
for i in range(0, 5):
    n = random.randint(0, 9999)
    predictions = model.predict(x_test[n:n + 1])
    print(f'The prediction of this image is:{np.argmax(predictions)}')
    print(y_test[n])
    image = x_test[n]
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels)
    plt.show()

# predict 10 handwritting numbers
for x in range(0, 10):
    img = cv.imread(f'{x}.png')[:, :, 0]
    img = img.reshape(1,28, 28, 1)
    prediction = model.predict(img)
    print(f'The prediction of this image is:{np.argmax(prediction)}')
    img = cv.imread(f'{x}.png')[:, :, 0]
    img = np.invert(np.array([img]))
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()

