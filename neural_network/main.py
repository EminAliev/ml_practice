import numpy as np
import keras
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random

dbname = 'mnist.keras'

num_classes = 10
input_shape = (28, 28, 1)
try:
    f = open(dbname)
    f.close()
    print('reading db from file')
    model_sequential = load_model(dbname)
except IOError:
    print('creating db')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)


def program_run(input_image):
    digit_images = np.split(input_image, input_image.shape[1] / 28, axis=1)
    digit_images = np.array(digit_images)


    prediction_result = model.predict(digit_images)


    ans = ""
    for case in np.split(prediction_result, len(digit_images)):
        predicted = np.argmax(case)
        ans += str(predicted)

    print(ans)


def generate_input_image(possible_digits=x_test,
                         input_length=4):
    selected_digit_images = [possible_digits[random.randint(0, len(possible_digits))] for _ in range(input_length)]
    concatenated = np.concatenate(selected_digit_images, axis=1)

    plt.imshow(concatenated, cmap='gray')
    plt.show()

    return concatenated


input_image = generate_input_image(input_length=4)

program_run(input_image)

