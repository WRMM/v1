#coding:utf8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification
from timeit import default_timer as timer
from perceptron import perceptron, softmax
from hingeloss import hingeloss, hingelossWR, \
    reducedHingelossWR, hingelossWR_reduced
import matplotlib
from matplotlib import pylab
matplotlib.use("PS")
from matplotlib.pyplot import *
fig = figure(1, figsize=(13, 7))
fineness = 0.01
lr = 0.025
input_dim = 2
output_dim = 4
n_iter = 30
gap = 0.3
samples = 200
size = 42


OXs, OYs = make_classification(n_samples=samples, n_features=input_dim, n_redundant=0, n_informative=2, n_classes=output_dim, n_clusters_per_class=1, random_state=3, shuffle=True)
np.random.seed(101)
np.random.seed(108)
OXs = np.random.randn(samples, input_dim)

def get_sub(OXs, OYs, cate):
    a, b = [], []
    for i in range(len(OYs)):
        if OYs[i] == cate:
            a.append(OXs[i])
            b.append(OYs[i])
    return np.asarray(a), np.asarray(b)


plt.subplot(2, 4, 1)
OX0, OY0 = get_sub(OXs, OYs, 0)
OX1, OY1 = get_sub(OXs, OYs, 1)
OX2, OY2 = get_sub(OXs, OYs, 2)
OX3, OY3 = get_sub(OXs, OYs, 3)
plt.scatter(OX0[:, 0], OX0[:, 1], s=size, c='#a6cee3', marker='o', label='music')
plt.scatter(OX1[:, 0], OX1[:, 1], s=size, c='#b89c75', marker='o', label='books')
plt.scatter(OX2[:, 0], OX2[:, 1], s=size, c='#ed9047', marker='o', label='film')
plt.scatter(OX3[:, 0], OX3[:, 1], s=size, c='#b15928', marker='o', label='opera')




plt.legend(bbox_to_anchor= (0.75,-0.25), scatterpoints=1)

# plt.subplot(2, 4, 1)
# plt.scatter(OXs[:, 0], OXs[:, 1], c=OYs, s=size, marker='o', cmap=plt.cm.Paired)

plt.title('A. Initial Word Vectors')
frame = pylab.gca()
pylab.ylim([-3.5,3.5])
pylab.xlim([-3.5,3.5])
frame.axes.get_yaxis().set_ticks([])
frame.axes.get_xaxis().set_ticks([])
# plt.legend(['fruit', 'animal', 'tool', 'movie'],['fruit', 'animal', 'tool', 'movie'], bbox_to_anchor=(1, 0, 0, 0), scatterpoints=1)


OX2 = np.copy(OXs)
OX3 = np.copy(OXs)
OX1 = OXs 





W = np.ones((input_dim, output_dim))
W, Xs = perceptron(OX1, OYs, W, lr, n_iter)
x_min, x_max = Xs[:, 0].min() - gap, Xs[:, 0].max() + gap
y_min, y_max = Xs[:, 1].min() - gap, Xs[:, 1].max() + gap
x_min, x_max, y_min, y_max = -3.5, 3.5, -3.5, 3.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, fineness),
                     np.arange(y_min, y_max, fineness))
Z = np.c_[xx.ravel(), yy.ravel()]
predicate = np.argmax(np.dot(Z, W), 1)
predicate = predicate.reshape(xx.shape)
plt.subplot(2, 4, 2)
plt.contourf(xx, yy, predicate, alpha=0.4, cmap=plt.cm.Paired)
plt.scatter(Xs[:, 0], Xs[:, 1], c=OYs, s=size, label="test", marker='o', cmap=plt.cm.Paired)




plt.title('B. Perceptron (iter=30)')
frame = pylab.gca()
pylab.ylim([-3.5,3.5])
pylab.xlim([-3.5,3.5])
frame.axes.get_yaxis().set_ticks([])
frame.axes.get_xaxis().set_ticks([])






W = np.ones((input_dim, output_dim))
W, Xs = softmax(OX3, OYs, W, lr, n_iter)
x_min, x_max = Xs[:, 0].min() - gap, Xs[:, 0].max() + gap
y_min, y_max = Xs[:, 1].min() - gap, Xs[:, 1].max() + gap
x_min, x_max, y_min, y_max = -3.5, 3.5, -3.5, 3.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, fineness),
                     np.arange(y_min, y_max, fineness))
Z = np.c_[xx.ravel(), yy.ravel()]
predicate = np.argmax(np.dot(Z, W), 1)
predicate = predicate.reshape(xx.shape)
plt.subplot(2, 4, 3)
plt.contourf(xx, yy, predicate, alpha=0.4, cmap=plt.cm.Paired)
plt.scatter(Xs[:, 0], Xs[:, 1], c=OYs, s=size, marker='o', cmap=plt.cm.Paired)
plt.title('C. Softmax (iter=30)')
frame = pylab.gca()
pylab.ylim([-3.5,3.5])
pylab.xlim([-3.5,3.5])
frame.axes.get_yaxis().set_ticks([])
frame.axes.get_xaxis().set_ticks([])


W = np.ones((input_dim, output_dim))
W, Xs = hingelossWR(OX2, OYs, W, lr, 1)
x_min, x_max = Xs[:, 0].min() - gap, Xs[:, 0].max() + gap
y_min, y_max = Xs[:, 1].min() - gap, Xs[:, 1].max() + gap
x_min, x_max, y_min, y_max = -3.5, 3.5, -3.5, 3.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, fineness),
                     np.arange(y_min, y_max, fineness))
Z = np.c_[xx.ravel(), yy.ravel()]
predicate = np.argmax(np.dot(Z, W), 1)
predicate = predicate.reshape(xx.shape)
plt.subplot(2, 4, 4)
plt.contourf(xx, yy, predicate, alpha=0.4, cmap=plt.cm.Paired)
plt.scatter(Xs[:, 0], Xs[:, 1], c=OYs, s=size, marker='o', cmap=plt.cm.Paired)
plt.title('D. Maximum-Margin (iter=1)')
frame = pylab.gca()
pylab.ylim([-3.5,3.5])
pylab.xlim([-3.5,3.5])
frame.axes.get_yaxis().set_ticks([])
frame.axes.get_xaxis().set_ticks([])





W = np.ones((input_dim, output_dim))
W, Xs = perceptron(OX1, OYs, W, lr, 200)
x_min, x_max = Xs[:, 0].min() - gap, Xs[:, 0].max() + gap
y_min, y_max = Xs[:, 1].min() - gap, Xs[:, 1].max() + gap
x_min, x_max, y_min, y_max = -3.5, 3.5, -3.5, 3.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, fineness),
                     np.arange(y_min, y_max, fineness))
Z = np.c_[xx.ravel(), yy.ravel()]
predicate = np.argmax(np.dot(Z, W), 1)
predicate = predicate.reshape(xx.shape)
plt.subplot(2, 4, 6)
plt.contourf(xx, yy, predicate, alpha=0.4, cmap=plt.cm.Paired)
plt.scatter(Xs[:, 0], Xs[:, 1], c=OYs, s=size, marker='o', cmap=plt.cm.Paired)
plt.title('E. Perceptron (iter=200)')
frame = pylab.gca()
pylab.ylim([-3.5,3.5])
pylab.xlim([-3.5,3.5])
frame.axes.get_yaxis().set_ticks([])
frame.axes.get_xaxis().set_ticks([])

W = np.ones((input_dim, output_dim))
W, Xs = softmax(OX3, OYs, W, lr, 70)
x_min, x_max = Xs[:, 0].min() - gap, Xs[:, 0].max() + gap
y_min, y_max = Xs[:, 1].min() - gap, Xs[:, 1].max() + gap
x_min, x_max, y_min, y_max = -3.5, 3.5, -3.5, 3.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, fineness),
                     np.arange(y_min, y_max, fineness))
Z = np.c_[xx.ravel(), yy.ravel()]
predicate = np.argmax(np.dot(Z, W), 1)
predicate = predicate.reshape(xx.shape)
plt.subplot(2, 4, 7)
plt.contourf(xx, yy, predicate, alpha=0.4, cmap=plt.cm.Paired)
plt.scatter(Xs[:, 0], Xs[:, 1], c=OYs, s=size, marker='o', cmap=plt.cm.Paired)
plt.title('F. Softmax (iter=70)')
frame = pylab.gca()
pylab.ylim([-3.5,3.5])
pylab.xlim([-3.5,3.5])
frame.axes.get_yaxis().set_ticks([])
frame.axes.get_xaxis().set_ticks([])



W = np.ones((input_dim, output_dim))
W, Xs = hingelossWR(OX2, OYs, W, lr, 20)
x_min, x_max = Xs[:, 0].min() - gap, Xs[:, 0].max() + gap
y_min, y_max = Xs[:, 1].min() - gap, Xs[:, 1].max() + gap
x_min, x_max, y_min, y_max = -3.5, 3.5, -3.5, 3.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, fineness),
                     np.arange(y_min, y_max, fineness))
Z = np.c_[xx.ravel(), yy.ravel()]
predicate = np.argmax(np.dot(Z, W), 1)
predicate = predicate.reshape(xx.shape)
plt.subplot(2, 4, 8)
plt.contourf(xx, yy, predicate, alpha=0.4, cmap=plt.cm.Paired)
plt.scatter(Xs[:, 0], Xs[:, 1], c=OYs, s=size, marker='o', cmap=plt.cm.Paired)
plt.title('G. Maximum-Margin (iter=20)')
frame = pylab.gca()
pylab.ylim([-3.5,3.5])
pylab.xlim([-3.5,3.5])
frame.axes.get_yaxis().set_ticks([])
frame.axes.get_xaxis().set_ticks([])

fig = matplotlib.pyplot.gcf()
plt.subplots_adjust(wspace=0.05)
plt.subplots_adjust(hspace=0.1)
plt.subplots_adjust(left=0.01)
plt.subplots_adjust(right=0.99)
plt.subplots_adjust(top=0.96)
plt.subplots_adjust(bottom=0.01)




plt.show()
