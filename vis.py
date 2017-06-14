import numpy as np
import matplotlib.pyplot as plt

#gt is in the shape of [ny, nx, k]
#k must be onehot
def calcGtImg(gt):
    (ny, nx, k) = gt.shape
    #Handle only 5 clusters for now
    assert(k == 4)
    outImg = np.zeros((ny, nx, 3))
    kColors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]

    for kk in range(k):
        labelIdx = np.nonzero(gt[:, :, kk] == 1)
        outImg[labelIdx] = kColors[kk]

    return outImg

def visualizeGt(gt, name):
    outImg = calcGtImg(gt)
    plt.imshow(outImg)
    plt.title(name)
