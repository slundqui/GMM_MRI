import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

#inputData must be 3 dimensional: [numImages, ny, nx]
#patchSize is (ny, nx)
def tfExtractPatches(inputData, patchSize):
    #Get shapes
    (numImages, ny, nx) = inputData.shape

    #Use tensorflow to get training samples from image
    #Build tensorflow graph
    sess = tf.InteractiveSession()
    tf_xImage = tf.placeholder(tf.float64, shape=[None, ny, nx, 1])

    #Extract patches from image
    tf_xData = tf.extract_image_patches(tf_xImage, [1, patchSize[0], patchSize[1], 1], [1, 1, 1, 1], [1, 1, 1, 1], "SAME")
    #Linearize both data and gt in batch, x, y dimension
    tf_xData = tf.reshape(tf_xData, [-1, patchSize[0]*patchSize[1]])

    #Get data patches
    xData = sess.run(tf_xData, feed_dict={tf_xImage: inputData[:, :, :, np.newaxis]})
    #Close session and reset graph
    sess.close()
    tf.reset_default_graph()
    return xData
