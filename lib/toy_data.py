import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
import random


# -------- Toy Data Loader --------

def load_toy(data_name, batch_size):
    if data_name == '25gaussians':
        dataset = []
        for i in range(100000 // 25):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.03
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        while True:
            for i in range(int(len(dataset) / batch_size)):
                yield dataset[i * batch_size:(i + 1) * batch_size] * 1.414

    elif data_name == '8gaussians':
        scale = 2.
        centers = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        while True:
            dataset = []
            for i in range(batch_size):
                point = np.random.randn(2) * 0.02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            yield dataset * 2.828

    elif data_name == 'swissroll':
        while True:
            data = sklearn.datasets.make_swiss_roll(n_samples=batch_size,noise=0.2)[0]
            data = data.astype('float32')[:, [0, 2]]
            yield data / 2.1

    elif data_name == "2spirals":
        while True:
            n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
            d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.1
            d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.1
            x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))
            x += np.random.randn(*x.shape) * 0.1
            yield x / 1.414

    elif data_name == "circles":
        while True:
            data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.01)[0]
            data = data.astype("float32")
            yield data * 5.656

    elif data_name =="2sines":
        while True:
            x = (np.random.rand(batch_size) -0.5) * 2 * np.pi
            u = (np.random.binomial(1,0.5,batch_size) - 0.5) * 2
            y = u * np.sin(x) * 2.5 + np.random.randn(*x.shape) * 0.05
            yield np.stack((x, y), 1) * 2.1

    elif data_name =="checkerboard":
        while True:
            x = np.random.uniform(-(5//2)*np.pi, (5//2)*np.pi, size=(3 * batch_size, 2))
            mask = np.logical_or(np.logical_and(np.sin(x[:,0]) > 0.0, np.sin(x[:,1]) > 0.0), \
            np.logical_and(np.sin(x[:,0]) < 0.0, np.sin(x[:,1]) < 0.0))
            y = np.eye(2)[1*mask]
            x0, x1 = x[:,0] * y[:,0], x[:,1] * y[:,0]
            sample = np.concatenate([x0[...,None],x1[...,None]],axis=-1)
            sqr = np.sum(np.square(sample),axis=-1)
            idxs = np.where(sqr==0)
            sample = np.delete(sample,idxs,axis=0)
            
            yield sample[0:batch_size,:]

    elif data_name == "moon":
        while True:
            x = np.linspace(0, np.pi, batch_size // 2)
            u = np.stack([np.cos(x) + .5, -np.sin(x) + .2], axis=1) * 10.
            u += 0.25*np.random.normal(size=u.shape)
            v = np.stack([np.cos(x) - .5, np.sin(x) - .2], axis=1) * 10.
            v += 0.25*np.random.normal(size=v.shape)
            x = np.concatenate([u, v], axis=0)
            yield x / 2.828

    elif data_name =="target":
        while True:
            shapes = np.random.randint(7, size=batch_size)
            mask = []
            for i in range(7):
                mask.append((shapes==i)*1.)

            theta = np.linspace(0, 2 * np.pi, batch_size, endpoint=False)
            x = (mask[0] + mask[1] + mask[2]) * (np.random.rand(batch_size) -0.5) * 4 +\
            (-mask[3] + mask[4]*0.0 + mask[5]) * 2 * np.ones(batch_size) +\
            mask[6] * np.cos(theta) 
            x += np.random.randn(*x.shape) * 0.05

            y = (mask[3] + mask[4] + mask[5]) * (np.random.rand(batch_size) -0.5) * 4 +\
            (-mask[0] + mask[1]*0.0 + mask[2]) * 2 * np.ones(batch_size) +\
            mask[6] * np.sin(theta)
            y += np.random.randn(*y.shape) * 0.05

            yield np.stack((x, y), 1) / 1.414

def load_noise(batch_size):
    while True:
        yield np.random.randn(batch_size, 2)


# -------- Utils for Toy plotting --------

def plot_toy(data, path):
    plt.clf()
    plt.xticks([])
    plt.yticks([])
    plt.scatter(data[:, 0], data[:, 1], c='cornflowerblue', marker='X')
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.savefig(path)