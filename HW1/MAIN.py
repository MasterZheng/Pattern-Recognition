from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

NUM_CLASS = 10


def test(set_pred_y, set_y):
    error = 0
    for i in range(set_y.shape[0]):
        if set_pred_y[i] != set_y[i]:
            error += 1
    return error / set_y.shape[0]


def separate(mnist):
    classes = [[] for i in range(NUM_CLASS)]
    for i in range(mnist.train.labels.shape[0]):
        classes[mnist.train.labels[i]].append(mnist.train.images[i])
    mean = []
    std = []
    var = []
    prior = []
    for i in range(NUM_CLASS):
        classes[i] = np.array(classes[i])
        mean.append(np.mean(classes[i], axis=0))
        std.append(np.std(classes[i], axis=0))
        var.append(np.var(classes[i], axis=0))
        prior.append(classes[i].shape[0] / mnist.train.labels.shape[0])
    return mean, std, var, prior


def discriminant(mnist, mean, var, prior):
    var[:, :] += 1e-2

    joint_log_likelihood = []
    for i in range(NUM_CLASS):
        log_likelihood = - 0.5 * np.sum(np.log(2. * np.pi * var[i, :]))
        log_likelihood -= 0.5 * np.sum(((mnist.test.images - mean[i, :]) ** 2) /
                                       (var[i, :]), axis=1)
        joint_log_likelihood.append(log_likelihood + prior[i])
    result = np.argmax(joint_log_likelihood, axis=0)
    return result


def draw(mean, std):
    for i in range(NUM_CLASS):
        plt.matshow(mean[i].reshape(28, 28), cmap=cm.coolwarm)
        plt.matshow(std[i].reshape(28, 28), cmap=cm.coolwarm)


def main():
    mnist = input_data.read_data_sets("MNIST_data/")
    mean, std, var, prior = separate(mnist)
    draw(mean,std)
    pdf_set = discriminant(mnist, np.array(mean), np.array(var), np.array(prior))
    error = test(pdf_set, mnist.test.labels)
    print("classification error rate: ")
    print(error)


main()
