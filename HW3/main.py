from tensorflow.examples.tutorials.mnist import input_data
from sklearn import svm
from sklearn import cross_validation
from sklearn.model_selection import train_test_split
import numpy as np


def dot_product_kernel(X, Y):
    return X.dot(Y.T)


mnist = input_data.read_data_sets("MNIST_data/")
X = mnist.test.images
Y = mnist.test.labels

train_image, test_image, train_label, test_label = train_test_split(X, Y, test_size=0.1, random_state=0)

svc = svm.SVC(kernel='linear', C=2)
svc.fit(train_image, train_label)
scores = cross_validation.cross_val_score(svc, train_image, train_label, cv=5)
print(scores)
print('Accuracy: %.3f (+/- %.3f)' % (np.mean(scores), np.std(scores) * 2))
print('Test Accuracy: %.3f'%(svc.score(test_image,test_label)))
