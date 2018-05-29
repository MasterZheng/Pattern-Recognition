import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

SLNN_W1_SIZE = 30
BATCH_SIZE = 10
lamda = 5

def splite(set,begin):
    X = []
    Y = []
    counter = [0 for i in range(10)]
    for i in range(begin,len(set.labels)):
        one_hot = [0 for i in range(10)]
        label = set.labels[i]
        if counter[label]<100:
            one_hot[label] = 1
            X.append(set.images[i])
            Y.append(one_hot)
            counter[label] +=1
    return np.array(X),np.array(Y)

def add_layer(inputs, in_size, out_size, activation_function=None,):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

def shuffle(x,y):
    index = [i for i in range(len(x))]
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    return x,y

def single_nn_2(set_train_x,set_train_y,set_test_x,set_test_y):
    x = tf.placeholder(tf.float32, [None, 784])
    labels = tf.placeholder(tf.float32, [None, 10])
    set_w1 = tf.Variable(tf.random_normal([784, SLNN_W1_SIZE]))
    set_b1 = tf.Variable(tf.zeros([1, SLNN_W1_SIZE]) + 0.1,)
    set_w2 = tf.Variable(tf.random_normal([SLNN_W1_SIZE, 10]))
    set_b2 = tf.Variable(tf.zeros([1, 10]) + 0.1,)

    z1 = tf.nn.sigmoid(tf.matmul(x, set_w1) + set_b1)
    Y = tf.nn.softmax(tf.matmul(z1, set_w2) + set_b2)
    L2 = tf.nn.l2_loss(set_w1)+tf.nn.l2_loss(set_w2)+tf.nn.l2_loss(set_b1)+tf.nn.l2_loss(set_b2)
    cross_entropy = -tf.reduce_sum(labels*tf.log(Y))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy+lamda*L2/60000)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(30):
        set_train_x,set_train_y = shuffle(set_train_x,set_train_y)
        for batch in range(100):
            batch_xs = set_train_x[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE, :]
            batch_ys = set_train_y[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE, :]
            sess.run(train_step, feed_dict={x: batch_xs, labels: batch_ys})
    accuracy_value = sess.run(accuracy, feed_dict={x: set_test_x, labels: set_test_y})
    print("Single hidden layer NN classification with L2 regularization error rate: ")
    print(1 - accuracy_value)


def two_nn_1(set_train_x,set_train_y,set_test_x,set_test_y):
    x = tf.placeholder(tf.float32, [None, 784])
    labels = tf.placeholder(tf.float32, [None, 10])

    z1 = add_layer(x, 784, SLNN_W1_SIZE, activation_function=tf.nn.sigmoid)
    z2 = add_layer(z1, SLNN_W1_SIZE, SLNN_W1_SIZE, activation_function=tf.nn.sigmoid)
    Y = add_layer(z2, SLNN_W1_SIZE, 10, activation_function=tf.nn.softmax)

    cross_entropy = -tf.reduce_sum(labels*tf.log(Y))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(30):
        set_train_x,set_train_y = shuffle(set_train_x,set_train_y)
        for batch in range(100):
            batch_xs = set_train_x[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE, :]
            batch_ys = set_train_y[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE, :]
            sess.run(train_step, feed_dict={x: batch_xs, labels: batch_ys})
    accuracy_value = sess.run(accuracy, feed_dict={x: set_test_x, labels: set_test_y})
    print("two hidden layer2 NN classification error rate: ")
    print(1 - accuracy_value)


def two_nn_2(set_train_x,set_train_y,set_test_x,set_test_y):
    x = tf.placeholder(tf.float32, [None, 784])
    labels = tf.placeholder(tf.float32, [None, 10])
    set_w1 = tf.Variable(tf.random_normal([784, SLNN_W1_SIZE], stddev=0.1))
    set_b1 = tf.Variable(tf.zeros([1, SLNN_W1_SIZE]) + 0.1,)
    set_w2 = tf.Variable(tf.random_normal([SLNN_W1_SIZE, SLNN_W1_SIZE], stddev=0.1))
    set_b2 = tf.Variable(tf.zeros([1, SLNN_W1_SIZE]) + 0.1,)
    set_w3 = tf.Variable(tf.random_normal([SLNN_W1_SIZE, 10], stddev=0.1))
    set_b3 = tf.Variable(tf.zeros([1, 10]) + 0.1,)

    Set_a1 = tf.nn.sigmoid(tf.matmul(x, set_w1) + set_b1)
    Set_a2 = tf.nn.sigmoid(tf.matmul(Set_a1, set_w2) + set_b2)
    Y = tf.nn.softmax(tf.matmul(Set_a2, set_w3) + set_b3)
    L2 = tf.nn.l2_loss(set_w1)+tf.nn.l2_loss(set_w2)+tf.nn.l2_loss(set_w3)\
         +tf.nn.l2_loss(set_b1) + tf.nn.l2_loss(set_b2) + tf.nn.l2_loss(set_b3)

    cross_entropy = -tf.reduce_sum(labels*tf.log(Y))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy+lamda*L2/60000)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(30):
        set_train_x,set_train_y = shuffle(set_train_x,set_train_y)
        for batch in range(100):
            batch_xs = set_train_x[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE, :]
            batch_ys = set_train_y[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE, :]
            sess.run(train_step, feed_dict={x: batch_xs, labels: batch_ys})

    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_value = sess.run(accuracy, feed_dict={x: set_test_x, labels: set_test_y})
    print("Two hidden layers NN with L2 regularization classification error rate: ")
    print(1 - accuracy_value)


def three_nn_1(set_train_x,set_train_y,set_test_x,set_test_y):
    x = tf.placeholder(tf.float32, [None, 784])
    labels = tf.placeholder(tf.float32, [None, 10])
    z1 = add_layer(x, 784, SLNN_W1_SIZE, activation_function=tf.nn.sigmoid)
    z2 = add_layer(z1, SLNN_W1_SIZE, SLNN_W1_SIZE, activation_function=tf.nn.sigmoid)
    z3 = add_layer(z2, SLNN_W1_SIZE, SLNN_W1_SIZE, activation_function=tf.nn.sigmoid)
    Y = add_layer(z3, SLNN_W1_SIZE, 10, activation_function=tf.nn.softmax)
    cross_entropy = -tf.reduce_sum(labels*tf.log(Y))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(30):
        set_train_x,set_train_y = shuffle(set_train_x,set_train_y)
        for batch in range(100):
            batch_xs = set_train_x[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE, :]
            batch_ys = set_train_y[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE, :]
            sess.run(train_step, feed_dict={x: batch_xs, labels: batch_ys})

    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_value = sess.run(accuracy, feed_dict={x: set_test_x, labels: set_test_y})
    print("Three hidden layers NN without L2 regularization classification error rate: ")
    print(1 - accuracy_value)


def three_nn_2(set_train_x,set_train_y,set_test_x,set_test_y):
    x = tf.placeholder(tf.float32, [None, 784])
    labels = tf.placeholder(tf.float32, [None, 10])
    set_w1 = tf.Variable(tf.random_normal([784, SLNN_W1_SIZE], stddev=0.1))
    set_b1 = tf.Variable(tf.zeros([1, SLNN_W1_SIZE]) + 0.1,)
    set_w2 = tf.Variable(tf.random_normal([SLNN_W1_SIZE, SLNN_W1_SIZE], stddev=0.1))
    set_b2 = tf.Variable(tf.zeros([1, SLNN_W1_SIZE]) + 0.1,)
    set_w3 = tf.Variable(tf.random_normal([SLNN_W1_SIZE, SLNN_W1_SIZE], stddev=0.1))
    set_b3 = tf.Variable(tf.zeros([1, SLNN_W1_SIZE]) + 0.1,)
    set_w4 = tf.Variable(tf.random_normal([SLNN_W1_SIZE, 10], stddev=0.1))
    set_b4 = tf.Variable(tf.zeros([1, 10]) + 0.1,)

    L2 = tf.nn.l2_loss(set_w1)+tf.nn.l2_loss(set_w2)+tf.nn.l2_loss(set_w3)+tf.nn.l2_loss(set_w4)\
         +tf.nn.l2_loss(set_b1) + tf.nn.l2_loss(set_b2) + tf.nn.l2_loss(set_b3) + tf.nn.l2_loss(set_b4)

    Set_a1 = tf.nn.sigmoid(tf.matmul(x, set_w1) + set_b1)
    Set_a2 = tf.nn.sigmoid(tf.matmul(Set_a1, set_w2) + set_b2)
    Set_a3 = tf.nn.sigmoid(tf.matmul(Set_a2, set_w3) + set_b3)
    Y = tf.nn.softmax(tf.matmul(Set_a3, set_w4) + set_b4)

    cross_entropy = -tf.reduce_sum(labels*tf.log(Y))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy+lamda*L2/60000)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(30):
        set_train_x,set_train_y = shuffle(set_train_x,set_train_y)
        for batch in range(100):
            batch_xs = set_train_x[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE, :]
            batch_ys = set_train_y[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE, :]
            sess.run(train_step, feed_dict={x: batch_xs, labels: batch_ys})

    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_value = sess.run(accuracy, feed_dict={x: set_test_x, labels: set_test_y})
    print("Three hidden layers NN with L2 regularization classification error rate: ")
    print(1 - accuracy_value)

def main():
    mnist = input_data.read_data_sets("MNIST_data/")

    set_train_x,set_train_y=splite(mnist.train,10000)
    set_test_x,set_test_y=splite(mnist.test,2000)

    # set_train_x = set_train_x.reshape([-1, 28, 28, 1])
    # rotationAndShift(set_train_x, set_test_y)
    single_nn_2(set_train_x,set_train_y,set_test_x,set_test_y)
    two_nn_1(set_train_x,set_train_y,set_test_x,set_test_y)
    two_nn_2(set_train_x,set_train_y,set_test_x,set_test_y)
    three_nn_1(set_train_x,set_train_y,set_test_x,set_test_y)
    three_nn_2(set_train_x,set_train_y,set_test_x,set_test_y)

    # ''' CNN '''
    #
    # CNN(set_train_x,set_train_y,set_test_x,set_test_y)

main()