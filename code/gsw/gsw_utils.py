import numpy as np
import torch
import ot
from sklearn.datasets import make_swiss_roll, make_moons, make_circles


def w2(X,Y):
    M=ot.dist(X,Y)
    a=np.ones((X.shape[0],))/X.shape[0]
    b=np.ones((Y.shape[0],))/Y.shape[0]
    return ot.emd2(a,b,M)


def load_data(name='swiss_roll', n_samples=1000):
    N=n_samples
    if name == 'swiss_roll':
        temp=make_swiss_roll(n_samples=N)[0][:,(0,2)]
        temp/=abs(temp).max()
    elif name == 'half_moons':
        temp=make_moons(n_samples=N)[0]
        temp/=abs(temp).max()
    elif name == '8gaussians':
        # Inspired from https://github.com/caogang/wgan-gp
        scale = 2.
        centers = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        temp = []
        for i in range(N):
            point = np.random.randn(2) * .02
            center = centers[np.random.choice(np.arange(len(centers)))]
            point[0] += center[0]
            point[1] += center[1]
            temp.append(point)
        temp = np.array(temp, dtype='float32')
        temp /= 1.414  # stdev
    elif name == '25gaussians':
        # Inspired from https://github.com/caogang/wgan-gp
        temp = []
        for i in range(int(N / 25)):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    temp.append(point)
        temp = np.array(temp, dtype='float32')
        np.random.shuffle(temp)
        temp /= 2.828  # stdev
    elif name == 'circle':
        temp,y=make_circles(n_samples=2*N)
        temp=temp[np.argwhere(y==0).squeeze(),:]
    else:
        raise Exception("Dataset not found: name must be 'swiss_roll', 'half_moons', 'circle', '8gaussians' or '25gaussians'.")
    X=torch.from_numpy(temp).float()
    return X
