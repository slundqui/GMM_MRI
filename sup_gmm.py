import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import data
import util
import pdb

#Parameters
inputList = "data/data.txt"
gtList = "data/gt.txt"
clusterInName = "data/kmeans_cluster.npy"
#y by x
patchSize = (5, 5)
#No EM, just one pass through is enough
numIterations = 1

#Load clusters from kmeans
np_cluster_means = np.load(clusterInName)

#Get data
(trainData, trainGt, testData, testGt) = data.getImages(inputList, gtList)
(numTrain, ny, nx, drop) = trainGt.shape
trainData = np.concatenate((trainData, testData), axis=0)
(numTest, ny, nx, drop) = testGt.shape

numTotal = numTrain + numTest

#Make test data all distractor class
unsupGtData = np.zeros((numTest, ny, nx, 5))
unsupGtData[:, :, :, 4] = 1
trainGt = np.concatenate((trainGt, unsupGtData), axis=0)

#Get data
xData = util.tfExtractPatches(trainData, patchSize)
gtData = np.reshape(trainGt, [-1, 5])

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

#Grab supervised data
posIdxs = np.nonzero(gtData[:, 4] == 0)
sup_gtData = gtData[posIdxs][:, :-1] #Drop distractor class
sup_xData = xData[posIdxs]
(numSup, drop) = sup_xData.shape

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

    #M step on supervise ddata
    new_clusterPrior = np.mean(sup_gtData, axis=0)
    new_clusterMeans = np.sum(sup_gtData[:, :, np.newaxis] * sup_xData[:, np.newaxis, :], axis=0) / gtNorm[:, np.newaxis]
    cData = sup_xData[:, np.newaxis, :] - new_clusterMeans[np.newaxis, :, :]
    cData_cDataT = cData[:, :, :, np.newaxis] * cData[:, :, np.newaxis, :]
    new_clusterStds = np.sum(sup_gtData[:, :, np.newaxis, np.newaxis] * cData_cDataT, axis=0) / gtNorm[:, np.newaxis, np.newaxis]

    #Calculate LL
    prior_ll = np.sum(sup_gtData * np.log(new_clusterPrior[np.newaxis, :]), axis=1)
    mvn_logdata = np.transpose(np.array([mvn.logpdf(sup_xData, new_clusterMeans[k], new_clusterStds[k]) for k in range(4)]))
    param_ll = np.sum(sup_gtData * mvn_logdata, axis=1)
    ll = np.mean(prior_ll + param_ll)

    print("Iteration", iteration, "  ll:", ll)
    #Set new params
    clusterMeans = new_clusterMeans
    clusterStds = new_clusterStds
    clusterPrior = new_clusterPrior

#Visualize
resp_argmax = np.argmax(resp, axis=1)
resp_onehot = np.zeros(resp.shape)
resp_onehot[np.arange(len(resp_argmax)), resp_argmax] = 1
estImage = np.reshape(resp_onehot, [numTotal, ny, nx, 4])

plt.figure()
util.visualizeGt(estImage[-1, :, :, :], "estImage")
plt.figure()
util.visualizeGt(testGt[-1, :, :, :4], "gtImage")

plt.show()
