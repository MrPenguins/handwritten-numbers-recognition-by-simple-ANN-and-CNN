import random
import numpy as np
import mnist  # our data set
import matplotlib.pyplot as plt  # graph
from keras.models import Sequential  # ANN architecture
from keras.layers import Dense  # the layers in the ANN
from keras.utils import to_categorical
import cv2 as cv
import tensorflow as tf
from keras.utils import plot_model

# Load the data set
train_images = mnist.train_images()  # training data images
train_labels = mnist.train_labels()  # training data labels
test_images = mnist.test_images()  # test data images
test_labels = mnist.test_labels()  # test data labels

# Normalize the data set
# normalize the pixel values from[0,255] to [-0.5,0.5], to make
# our network easier to train
train_images = tf.keras.utils.normalize(train_images, axis=1)
test_images = tf.keras.utils.normalize(test_images, axis=1)

# Build the model
# 4 layers
# 1 flatten layer ,flatten each 28*28 image into a 784 dimensional vector to pass into the neural network
# 2 layers with 128 neurons and the relu function
# 1 layers with 10 neurons and softmax function
model = Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
# the lost function measures how well the model did on training
# and then tries to improve on it by using the optimizer
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # (for classes that are greater than 2)
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_images,
    to_categorical(train_labels),  # for example,2 is expected to be [0,0,1,0,0,0,0,0,0,0]
    epochs=5,  # the number of iterations over the entire dataset to train on
    batch_size=16  # the number of samples per gradient update for training
)

# Evaluate the model
model.evaluate(
    test_images,
    to_categorical(test_labels)
)



# predict on the random 5 test images
for i in range(0, 5):
    n=random.randint(0,9999)
    predictions = model.predict(test_images[n:n+1])
    print(f'The prediction of this image is:{np.argmax(predictions)}')
    print(test_labels[n])
    image = test_images[n]
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels)
    plt.show()

# predict 10 handwritting numbers
for x in range(0, 10):
    img = cv.imread(f'{x}.png')[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    #print(prediction)
    print(f'The prediction of this image is:{np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
