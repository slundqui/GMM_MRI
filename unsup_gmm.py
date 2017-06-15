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
numIterations = 50

#Load clusters from kmeans
np_cluster_means = np.load(clusterInName)

#Get data
(trainData, trainGt, testData, testGt) = data.getImages(inputList, gtList)
(numTrain, ny, nx, drop) = trainGt.shape

#Run on all avaliable data
trainData = np.concatenate((trainData, testData), axis=0)
(numTotal, drop, drop) = trainData.shape

#Extract patches
xData = util.tfExtractPatches(trainData, patchSize)

#Initialize clusters
#clusterMeans is [numClusters, numFeatures]
clusterMeans = np_cluster_means.astype(np.float64)
#clusterStds is [numClusters, numFeatures, numFeatures]
clusterStds = np.zeros((4, 25, 25))
for k in range(4):
    clusterStds[k] = np.diag(np.abs(-(clusterMeans[k]*clusterMeans[k].transpose())))
#clusterPrior is [numClusters]
clusterPrior = np.array([.25, .25, .25, .25]).astype(np.float64)

#Run EM
for iteration in range(numIterations):
    #E step
    #mvn_data is [numData, numClusters]
    mvn_data = np.transpose(np.array([mvn.pdf(xData, clusterMeans[k], clusterStds[k]) for k in range(4)]))
    respNum = clusterPrior[np.newaxis, :] * mvn_data
    respDen = np.sum(respNum, axis=1)
    #resp is [numData, numClusters]
    resp = respNum/respDen[:, np.newaxis]
    #respNorm is [numClusters]
    respNorm = np.sum(resp, axis=0)

    #M step
    new_clusterPrior = np.mean(resp, axis=0)
    new_clusterMeans = np.sum(resp[:, :, np.newaxis] * xData[:, np.newaxis, :], axis=0) / respNorm[:, np.newaxis]
    cData = xData[:, np.newaxis, :] - new_clusterMeans[np.newaxis, :, :]
    cData_cDataT = cData[:, :, :, np.newaxis] * cData[:, :, np.newaxis, :]
    new_clusterStds = np.sum(resp[:, :, np.newaxis, np.newaxis] * cData_cDataT, axis=0) / respNorm[:, np.newaxis, np.newaxis]

    #Calculate LL
    prior_ll = np.sum(resp * np.log(new_clusterPrior[np.newaxis, :]), axis=1)
    mvn_logdata = np.transpose(np.array([mvn.logpdf(xData, new_clusterMeans[k], new_clusterStds[k]) for k in range(4)]))
    param_ll = np.sum(resp * mvn_logdata, axis=1)
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

