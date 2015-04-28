import numpy as np
# def perceptron(Xs, Ys, W, lr, n_epoch):
#     m = len(Ys)
#     Y = np.zeros([Xs.shape[0], W.shape[1]])
#     indices = range(m)
#     Y[indices, Ys] = 1
#     for epoch in range(n_epoch):
#         for i in range(len(Ys)):
#             x = Xs[i]
#             y = Ys[i]
#             j = np.argmax(np.dot(x, W))
#             print j
#             if j != y:
#                 update = W[:, y] - W[:, j]
#                 W[:, y] += lr * x
#                 W[:, j] -= lr * x
#                 Xs[i] += lr * update
#     return W, Xs

def perceptron(Xs, Ys, W, lr, n_epoch):
    m = len(Ys)
    Y = np.zeros([Xs.shape[0], W.shape[1]])
    indices = range(m)
    Y[indices, Ys] = 1
    
    for epoch in range(n_epoch):
        wx = np.dot(Xs, W)
        f = np.argmax(wx, 1)
        F = np.zeros([Xs.shape[0], W.shape[1]])
        F[indices, f] = 1
        update = np.dot(Y - F, np.transpose(W)) 
        W += lr * np.dot(np.transpose(Xs), Y - F) 
        Xs += lr * update
    return W, Xs




def softmax(Xs, Ys, W, lr, n_epoch):
    m = len(Ys)
    Y = np.zeros([Xs.shape[0], W.shape[1]])
    indices = range(m)
    Y[indices, Ys] = 1
    for epoch in range(n_epoch):
        ewx = np.exp(np.dot(Xs, W))
        P = ewx / np.sum(ewx, 1)[:, None]
        update = np.dot(Y - P, np.transpose(W)) 
        W += lr * np.dot(np.transpose(Xs), Y - P) 
        Xs += lr * update
    return W, Xs

# def softmax(Xs, Ys, W, lr, n_epoch):
#     m = len(Ys)
#     Y = np.zeros([Xs.shape[0], W.shape[1]])
#     indices = range(m)
#     Y[indices, Ys] = 1
#     for epoch in range(n_epoch):
#         for i in range(len(Ys)):
#             x = Xs[i]
#             y = Y[i]
#             ewx = np.exp(np.dot(x, W))
#             P = ewx / np.sum(ewx)
#             update = np.dot(y - P, np.transpose(W)) 
#             W += lr * (np.matrix(x).T * (y - P))
#             Xs[i] += lr * update
#     return W, Xs


def perceptronDual(Xs, Ys, alpha, lr):
    l = len(Ys)
    
    gram = np.dot(Xs, np.transpose(Xs))
    for epoch in range(100): 
        print alpha
        print np.transpose(gram * (Ys * alpha))
        a = np.transpose(gram * (Ys * alpha)) * Ys < 0
        alpha = alpha + lr * a
    
    return np.dot(np.transpose(Xs), alpha * Ys), Xs
#     for epoch in range(100000):
#         W*Ys*
#     return W, Xs


        
if __name__ == '__main__':
    a = np.array([[ 5., 1. , 3.], [ 1., 1. , 1.]])
    b = np.array([1, 2])
#     print np.transpose(a) * b
    print np.exp(a)
    print np.sum(a, 1)
    print a / np.sum(a, 1)[:, None]
    
