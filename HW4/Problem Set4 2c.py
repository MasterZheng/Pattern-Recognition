import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator


def splite(X, Y):
    list_x = []
    list_y = []
    counter = [0 for i in range(10)]
    for i in range(len(Y)):
        label = np.where(Y[i] == 1)[0][0]
        if counter[label] < 100:
            list_x.append(X[i])
            list_y.append(Y[i])
            counter[label] += 1
    return np.array(list_x), np.array(list_y)


def load_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    number_of_classes = 10

    Y_train = np_utils.to_categorical(y_train, number_of_classes)
    Y_test = np_utils.to_categorical(y_test, number_of_classes)

    X_train, Y_train = splite(X_train, Y_train)
    X_test, Y_test = splite(X_test, Y_test)

    # normalization
    X_train /= 255
    X_test /= 255

    return X_train, Y_train, X_test, Y_test


def CNN(X_train, Y_train, X_test, Y_test):
    model = Sequential()

    model.add(Conv2D(32, 3, 3, input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    BatchNormalization(axis=1)
    model.add(Conv2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    BatchNormalization(axis=1)
    model.add(Conv2D(64, 3, 3))
    model.add(Activation('relu'))
    BatchNormalization(axis=1)
    model.add(Conv2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    # Fully connected layer

    BatchNormalization()
    model.add(Dense(512))
    model.add(Activation('relu'))
    BatchNormalization()
    model.add(Dropout(0.5))
    model.add(Dense(10))

    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    gen = ImageDataGenerator(rotation_range=3, width_shift_range=3, height_shift_range=3)

    test_gen = ImageDataGenerator()
    train_generator = gen.flow(X_train, Y_train, batch_size=64)
    test_generator = test_gen.flow(X_test, Y_test, batch_size=64)

    model.fit_generator(train_generator, steps_per_epoch=60000 // 64, epochs=5,
                        validation_data=test_generator, validation_steps=10000 // 64)
    score = model.evaluate(X_test, Y_test)
    print()
    print('Test accuracy: ', score[1])


def main():
    X_train, Y_train, X_test, Y_test = load_dataset()
    CNN(X_train, Y_train, X_test, Y_test)


main()
