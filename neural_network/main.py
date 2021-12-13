import numpy as np
import random
import pygame
import tensorflow as tf
from keras.models import load_model
from enum import Enum
from PIL import Image
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

from const import Button, Colors

dbname = 'mnist.keras'
try:
    f = open(dbname)
    f.close()
    print('reading db from file')
    model_sequential = load_model(dbname)
except IOError:
    print('creating db')
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model_sequential = Sequential()
    model_sequential.add(Flatten())

    model_sequential.add(Dense(256, activation=tf.nn.relu))
    model_sequential.add(Dense(256, activation=tf.nn.relu))
    model_sequential.add(Dense(10, activation=tf.nn.softmax))

    model_sequential.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model_sequential.fit(x=x_train, y=y_train, epochs=16, batch_size=32)

    model_sequential.save(dbname)

print(model_sequential.summary())


def predict(screen):
    path = 'digit.png'
    pygame.image.save(screen, path)
    digit_image = Image.open(path)
    digit_image = digit_image.resize((28, 28))
    digit_image = digit_image.convert('L')
    digit_image = np.array(digit_image)
    digit_image = digit_image.reshape(1, 28, 28, 1)
    digit_image = digit_image / 255.0
    prediction = model_sequential.predict([digit_image])[0]
    res = prediction.argsort()[::-1]
    print(res)


screen = pygame.display.set_mode((280, 280))
line_start = None

while True:
    for i in pygame.event.get():
        if i.type == pygame.QUIT:
            pygame.quit()
            break
        if i.type == pygame.MOUSEBUTTONDOWN:
            if i.button == Button.Right.value:
                predict(screen)
        elif i.type == pygame.KEYDOWN and i.key == pygame.K_SPACE:
            screen.fill(Colors.BLACK)
    position = pygame.mouse.get_pos()
    pressed = pygame.mouse.get_pressed()
    if pressed[0]:
        pygame.draw.circle(screen, Colors.WHITE,
                           (position[0] + random.randrange(2), position[1] + random.randrange(2)),
                           random.randrange(1, 5))
    pygame.display.update()
