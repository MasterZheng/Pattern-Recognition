import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import PIL.Image as Image

SLNN_W1_SIZE = 30
BATCH_SIZE = 10

lamda = 5


def add_layer(inputs, in_size, out_size, activation_function=None, ):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    return outputs


def splite(set):
    X = []
    Y = []
    y = []
    counter = [0 for i in range(10)]
    breakCounter = 0
    for i in range(len(set.labels)):
        one_hot = [0 for i in range(10)]
        label = set.labels[i]
        if counter[label] < 100:
            one_hot[label] = 1
            X.append(set.images[i])
            Y.append(one_hot)
            y.append(label)
            counter[label] += 1
        else:
            counter[label] += 1
            if counter[label] == 100:
                breakCounter += 1
        if breakCounter == 10:
            break
    return np.array(X), np.array(Y), np.array(y)


def single_nn_1(set_train_x, set_train_y, set_test_x, set_test_y):
    train_error = []
    test_error = []
    w1 = []
    w2 = []
    x = tf.placeholder(tf.float32, [None, 784])
    labels = tf.placeholder(tf.float32, [None, 10])
    # z1 = add_layer(x, 784, SLNN_W1_SIZE, activation_function=tf.nn.sigmoid)
    # Y = add_layer(z1, SLNN_W1_SIZE, 10, activation_function=tf.nn.softmax)

    set_w1 = tf.Variable(tf.random_normal([784, SLNN_W1_SIZE]))
    set_b1 = tf.Variable(tf.zeros([1, SLNN_W1_SIZE]) + 0.1, )
    set_w2 = tf.Variable(tf.random_normal([SLNN_W1_SIZE, 10]))
    set_b2 = tf.Variable(tf.zeros([1, 10]) + 0.1, )

    z1 = tf.nn.sigmoid(tf.matmul(x, set_w1) + set_b1)
    Y = tf.nn.softmax(tf.matmul(z1, set_w2) + set_b2)

    cross_entropy = -tf.reduce_sum(labels * tf.log(Y))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    l1 = tf.contrib.layers.l1_regularizer(0.1)(set_w1)
    l2 = tf.contrib.layers.l1_regularizer(0.1)(set_w2)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(30):
        index = [i for i in range(len(set_test_x))]
        np.random.shuffle(index)
        set_train_x = set_train_x[index]
        set_train_y = set_train_y[index]
        train_error.append(1 - sess.run(accuracy, feed_dict={x: set_train_x, labels: set_train_y}))
        test_error.append(1 - sess.run(accuracy, feed_dict={x: set_test_x, labels: set_test_y}))

        for batch in range(100):
            batch_xs = set_train_x[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE, :]
            batch_ys = set_train_y[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE, :]
            sess.run(train_step, feed_dict={x: batch_xs, labels: batch_ys})
            w1.append(10 * sess.run(l1))
            w2.append(10 * sess.run(l2))

    accuracy_value = sess.run(accuracy, feed_dict={x: set_test_x, labels: set_test_y})
    print("Single hidden layer NN classification with L2 regularization error rate: ")
    print(1 - accuracy_value)
    return train_error, test_error, w1, w2


def rotationAndShift(setX, setY):
    datagen = image.ImageDataGenerator(rotation_range=30, horizontal_flip=0.5)
    datagen.fit(setX)
    i = 0
    # for i in range(len(setY)):
    for img_batch in datagen.flow(setX, setY, batch_size=9):
        for img in img_batch:
            for i in range(9):
                img[i] = np.reshape(img, [784, ])
                plt.imshow(img[i].reshape(28, 28), cmap='gray')
                plt.show()
                i = i + 1
            if i >= 9:
                break
    return


def main():
    mnist = input_data.read_data_sets("MNIST_data/")

    set_train_x, set_train_y, set_train_label = splite(mnist.train)
    set_test_x, set_test_y, set_test_label = splite(mnist.test)

    train_error, test_error, w1, w2 = single_nn_1(set_train_x, set_train_y, set_test_x, set_test_y)
    plt.plot(train_error, label='train_error')
    plt.plot(test_error, label='test_error')
    plt.legend(loc=0)  # 图例位置自动

    plt.ylabel("Error")
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.show()

    label = []
    color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
             'tab:olive', 'tab:cyan']
    fig, ax = plt.subplots()
    for i in range(10):
        for j in range(len(set_train_label)):
            if i == set_train_label[j]:
                label.append(set_train_x[j])
        pca = PCA(n_components=2)
        label_pca = pca.fit_transform(label)
        ax.scatter(label_pca[:, 0], label_pca[:, 1], c=color[i], label=i, marker=i)
        label = []
    plt.title("train set")
    plt.show()

    for i in range(10):
        for j in range(len(set_test_label)):
            if i == set_test_label[j]:
                label.append(set_test_x[j])
        label_pca = PCA().fit_transform(label)
        plt.scatter(label_pca[:, 0], label_pca[:, 1], c=color[i], label=i, marker=i)
        label = []
    plt.title("test set")
    plt.show()

    w1speed = []
    w2speed = []
    for i in range(1, len(w1)):
        w1speed.append((w1[i] - w1[i - 1]) / w1[i - 1])
        w2speed.append((w2[i] - w2[i - 1]) / w2[i - 1])

    plt.plot(np.abs(w1speed), label='hidden layer W')
    plt.legend(loc=0)  # 图例位置自动
    plt.ylabel("learning speed")
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.show()

    plt.plot(np.abs(w2speed), label='output layer W')
    plt.legend(loc=0)  # 图例位置自动
    plt.ylabel("learning speed")
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.show()


main()
