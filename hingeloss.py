import numpy as np
def hingeloss(Xs, Ys, W, lr):
    l = len(Ys)
    for epoch in range(100000):
        a = Ys * ((Ys * np.dot(Xs, W)) < 1)
        W = W + lr * np.dot(a, Xs)
        a.shape = (l, 1)
        Xs = Xs + lr * (a * W)
    return W, Xs


# def hingelossWR(Xs, Ys, W, lr):
#     l = len(Ys)
#     for epoch in range(100000):
#         a = Ys * ((np.dot(Xs, W) * Ys) < 1)
# #         print a
#         W = W + lr * np.dot(np.transpose(Xs), a)
# #         a.shape = (l, 1)
#         Xs = Xs + lr * np.dot(a, np.transpose(W))
#     return W, Xs


def hingelossWR(Xs, Ys, W, lr, n_epoch):
    l = len(Ys)
    indices = range(l)
    Y = np.zeros([Xs.shape[0], W.shape[1]])
    indices = range(l)
    Y[indices, Ys] = 1
    for epoch in range(10000):
        wx = np.dot(Xs, W)
        wx_t = wx[indices, Ys]
        wx[indices, Ys] = np.NINF
        Js = np.argmax(wx, axis=1)
        wx_j = wx[indices, Js]
        condition = 1 + wx_j - wx_t > 0
        
        F = np.zeros_like(wx)
        F[indices, Js] = 1
#         W += lr * np.transpose(np.dot(np.transpose(M) * condition, Xs))
#         Xs -= lr * np.transpose(np.transpose(WT[Js] - WT[Ys]) * condition)
        
        aa = ((Y - F).T * condition).T
        update = np.dot(aa, np.transpose(W)) 
        W += lr * np.dot(np.transpose(Xs), aa) 
        Xs += lr * update
    return W, Xs

def hingelossWR_reduced(Xs, Ys, W, lr, n_epoch):
    l = len(Ys)
    indices = range(l)
    Y = np.zeros([Xs.shape[0], W.shape[1]])
    indices = range(l)
    Y[indices, Ys] = 1
    for epoch in range(10000):
        wx = np.dot(Xs, W)
        wx_t = wx[indices, Ys]
        condition = 1  - wx_t > 0
#         W += lr * np.transpose(np.dot(np.transpose(M) * condition, Xs))
#         Xs -= lr * np.transpose(np.transpose(WT[Js] - WT[Ys]) * condition)
        
        aa = (Y.T * condition).T
        update = np.dot(aa, np.transpose(W)) 
        W += lr * np.dot(np.transpose(Xs), aa) 
        Xs += lr * update
    return W, Xs


def reducedHingelossWR(Xs, Ys, W, lr, n_epoch):
    for epoch in range(n_epoch):
        for i in range(len(Ys)):
            x = Xs[i]
            y = Ys[i]
            w = W[:, y]
            if np.dot(w, x) < 1:
                W[:, y] += lr * (x - 1.0 * W[:, y])
                Xs[i] += lr * w
    return W, Xs


if __name__ == '__main__':
    a = np.array([[0., 1, 2], [0, 1, 2]])
    a[[0, 1], [0, 2]] = np.NAN
    print a
    
    x = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    y = np.array([1, 0, 1, 0])
    z = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    print  np.dot(np.transpose(x) * y, z)
    
    
    w = np.array([[1, 1], [2, 2], [3, 3]])
    z = np.array([1, 0, 1])
    indices = np.array([0, 1, 2, 2, 1, 0])
    
    print w
    print np.transpose(w)
    print w
    print w[indices]
    print np.transpose(np.transpose(w) * z)
    
    print int(np.sqrt(101 / 2))
    
    
