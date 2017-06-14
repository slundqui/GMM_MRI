import numpy as np
from scipy.ndimage import imread
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from data import getImages
from vis import visualizeGt
import tensorflow as tf
import pdb

inputList = "data/data.txt"
gtList = "data/gt.txt"
clusterInName = "data/kmeans_cluster.npy"

np_cluster_means = np.load(clusterInName)

#y by x
patchSize = (5, 5)
numIterations = 25

(trainData, trainGt, testData, testGt) = getImages(inputList, gtList)
(numTrain, ny, nx, drop) = trainGt.shape

trainData = np.concatenate((trainData, testData), axis=0)
(numTest, ny, nx, drop) = testGt.shape

numTotal = numTrain + numTest

unsupGtData = np.zeros((numTest, ny, nx, 5))
unsupGtData[:, :, :, 4] = 1
trainGt = np.concatenate((trainGt, unsupGtData), axis=0)

#Use only first image for now
#trainData = trainData[0, :, :]
#trainData = trainData[np.newaxis, :, :]
#trainGt = trainGt[0, :, :, :]
#trainGt = trainGt[np.newaxis, :, :, :]

#Use tensorflow to get training samples from image
#Build tensorflow graph
sess = tf.InteractiveSession()
tf_xImage = tf.placeholder(tf.float64, shape=[None, ny, nx, 1])

#Extract patches from image
tf_xData = tf.extract_image_patches(tf_xImage, [1, patchSize[0], patchSize[1], 1], [1, 1, 1, 1], [1, 1, 1, 1], "SAME")
#Linearize both data and gt in batch, x, y dimension
tf_xData = tf.reshape(tf_xData, [-1, patchSize[0]*patchSize[1]])

#Initialize clusters
#clusterMeans is [numClusters, numFeatures]
clusterMeans = np_cluster_means.astype(np.float64)
#clusterStds is [numClusters, numFeatures, numFeatures]
#clusterStds = np.tile(np.eye(25), [4, 1, 1]).astype(np.float64)
clusterStds = np.zeros((4, 25, 25))
for k in range(4):
    clusterStds[k] = np.diag(np.abs(-(clusterMeans[k]*clusterMeans[k].transpose())))
#clusterPrior is [numClusters]
clusterPrior = np.array([.25, .25, .25, .25]).astype(np.float64)

#Get data patches
xData = sess.run(tf_xData, feed_dict={tf_xImage: trainData[:, :, :, np.newaxis]})
gtData = np.reshape(trainGt, [-1, 5])

#Grab supervised data
posIdxs = np.nonzero(gtData[:, 4] == 0)
sup_gtData = gtData[posIdxs][:, :-1] #Drop distractor class
sup_xData = xData[posIdxs]

#We use all data for unsupervised portion of gmm


#Run EM
for iteration in range(numIterations):
    #E step for all data
    #mvn_data is [numData, numClusters]
    mvn_data = np.transpose(np.array([mvn.pdf(xData, clusterMeans[k], clusterStds[k]) for k in range(4)]))
    respNum = clusterPrior[np.newaxis, :] * mvn_data
    respDen = np.sum(respNum, axis=1)
    #resp is [numData, numClusters]
    resp = respNum/respDen[:, np.newaxis]
    #respNorm is [numClusters]
    respNorm = np.sum(resp, axis=0)

    gtNorm = np.sum(sup_gtData, axis=0)

    #M step on supervise data
    sup_clusterPrior = np.mean(sup_gtData, axis=0)
    sup_clusterMeans = np.sum(sup_gtData[:, :, np.newaxis] * sup_xData[:, np.newaxis, :], axis=0) / gtNorm[:, np.newaxis]
    xData_xDataT = sup_xData[:, :, tf.newaxis] * sup_xData[:, tf.newaxis, :]
    mu_muT = sup_clusterMeans[:, :, tf.newaxis] * sup_clusterMeans[:, tf.newaxis, :]
    sup_clusterStds = np.sum(sup_gtData[:, :, np.newaxis, np.newaxis] * xData_xDataT[:, np.newaxis, :, :], axis=0) / gtNorm[:, np.newaxis, np.newaxis]
    sup_clusterStds = sup_clusterStds - mu_muT

    #M step on unsupervised data
    unsup_clusterPrior = np.mean(resp, axis=0)
    unsup_clusterMeans = np.sum(resp[:, :, np.newaxis] * xData[:, np.newaxis, :], axis=0) / respNorm[:, np.newaxis]
    xData_xDataT = xData[:, :, tf.newaxis] * xData[:, tf.newaxis, :]
    mu_muT = unsup_clusterMeans[:, :, tf.newaxis] * unsup_clusterMeans[:, tf.newaxis, :]
    unsup_clusterStds = np.sum(resp[:, :, np.newaxis, np.newaxis] * xData_xDataT[:, np.newaxis, :, :], axis=0) / respNorm[:, np.newaxis, np.newaxis]
    unsup_clusterStds = unsup_clusterStds - mu_muT

    new_clusterPrior = (sup_clusterPrior + unsup_clusterPrior)/2.0
    new_clusterMeans = (sup_clusterMeans + unsup_clusterMeans)/2.0
    new_clusterStds = (sup_clusterStds + unsup_clusterStds)/2

    #Calculate sup LL
    prior_ll = np.sum(sup_gtData * np.log(new_clusterPrior[np.newaxis, :]))
    mvn_logdata = np.transpose(np.array([mvn.logpdf(sup_xData, new_clusterMeans[k], new_clusterStds[k]) for k in range(4)]))
    param_ll = np.sum(sup_gtData * mvn_logdata)
    sup_ll = prior_ll + param_ll

    #Calculate unsup LL
    prior_ll = np.sum(resp * np.log(new_clusterPrior[np.newaxis, :]))
    mvn_logdata = np.transpose(np.array([mvn.logpdf(xData, new_clusterMeans[k], new_clusterStds[k]) for k in range(4)]))
    param_ll = np.sum(resp * mvn_logdata)
    unsup_ll = prior_ll + param_ll

    ll = sup_ll + unsup_ll

    print("Iteration", iteration, "  ll:", ll)
    #Set new params
    clusterMeans = new_clusterMeans
    clusterStds = new_clusterStds
    clusterPrior = new_clusterPrior

resp_argmax = np.argmax(resp, axis=1)
resp_onehot = np.zeros(resp.shape)
resp_onehot[np.arange(len(resp_argmax)), resp_argmax] = 1

estImage = np.reshape(resp_onehot, [numTotal, ny, nx, 4])

estTestImg = estImage[7:]
testGtPosIdx = np.nonzero(testGt[:, :, :, 4] == 0)
posEst = estTestImg[testGtPosIdx]
posGt = testGt[testGtPosIdx][:, :-1]

accuracy = np.mean(np.equal(posEst, posGt).astype(np.float32))
print("Final accurcy:", accuracy)

plt.figure()
visualizeGt(estImage[-1, :, :, :], "estImage")

plt.figure()
visualizeGt(testGt[-1, :, :, :4], "gtImage")

plt.show()

pdb.set_trace()






