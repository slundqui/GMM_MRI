import numpy as np
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import data
import util
import tensorflow as tf
import pdb

#Parameters
inputList = "data/data.txt"
gtList = "data/gt.txt"
clusterOutName = "data/kmeans_cluster.npy"
#y by x
patchSize = (5, 5)
k = 4

#Get images
(trainData, trainGt, testData, testGt) = data.getImages(inputList, gtList)
#Run on all avaliable data
trainData = np.concatenate((trainData, testData), axis=0)

#Get shapes
(numTotal, ny, nx) = trainData.shape

#Build tensorflow graph
sess = tf.InteractiveSession()
#Input placeholders
xImage = tf.placeholder(tf.float32, shape=[None, ny, nx, 1])
gtImage = tf.placeholder(tf.float32, shape=[None, ny, nx, 5])

#Extract patches from image
xData = tf.extract_image_patches(xImage, [1, patchSize[0], patchSize[1], 1], [1, 1, 1, 1], [1, 1, 1, 1], "SAME")
#Linearize both data and gt in batch, x, y dimension
xData = tf.reshape(xData, [-1, patchSize[0]*patchSize[1]])
gtData = tf.reshape(gtImage, [-1, k+1])

#Remove distractor class when training
#I.e., apply only to datapoints with ground truth
dataIdxs = tf.where(tf.not_equal(gtData[:, -1], 0))
valid_xData = tf.gather(xData, dataIdxs)[:, 0, :]
valid_gtData = tf.gather(gtData, dataIdxs)[:, 0, :]

#Initialize cluster centers
cluster = tf.Variable(tf.random_normal([k, patchSize[0]*patchSize[1]], .5, 1e-3))

#E step (cluster assignment based on l2 distance)
assignments = tf.reduce_sum(tf.square(valid_xData[:, tf.newaxis, :] - cluster[tf.newaxis, :, :]), axis=2)
#Hard assign
assignIdx = tf.argmin(assignments, axis=1)
#One hot vector based on argmin
responsibility = tf.one_hot(assignIdx, k)

#Applies E step to all data for visualization
full_assignments = tf.reduce_sum(tf.square(xData[:, tf.newaxis, :] - cluster[tf.newaxis, :, :]), axis=2)
#Hard assign
full_assignIdx = tf.argmin(full_assignments, axis=1)
#One hot vector based on assignIdx
full_responsibility = tf.one_hot(full_assignIdx, k)

#M step (update cluster)
weightedAvg = responsibility[:, :, tf.newaxis] * valid_xData[:, tf.newaxis, :]
normVals = tf.reduce_sum(responsibility, axis=0)
new_cluster = tf.reduce_sum(weightedAvg, axis=0)/normVals[:, tf.newaxis]

#Assignment of new clusters
stepEM = tf.assign(cluster, new_cluster)

#Initialize variables
sess.run(tf.global_variables_initializer())

#Build np input data structure for running graph
feed_dict = {xImage: trainData[:, :, :, np.newaxis], gtImage: trainGt}

#Run
#Get old assignment for stopping criteria
oldAssignment = sess.run(assignIdx, feed_dict=feed_dict)
loop = True
iteration = 0
while loop:
    print("Iteration" + str(iteration))
    iteration += 1
    #Run EM
    sess.run(stepEM, feed_dict=feed_dict)
    #Get new assignment
    newAssignment = sess.run(assignIdx, feed_dict=feed_dict)
    #Check for stopping condition, i.e., no new assignments
    if(np.sum(np.abs(newAssignment - oldAssignment)) == 0):
        loop = False
    oldAssignment = newAssignment

#Evaluate tf nodes for cluster assignments into numpy array
np_clusters = sess.run(cluster)
#Save cluster as .npy file
np.save(clusterOutName, np_clusters)

#Visualize
plt.figure()
util.visualizeGt(testGt[-1, :, :, :-1], "gtImage")

estImage = sess.run(full_responsibility, feed_dict=feed_dict)
estImage = np.reshape(estImage, [numTotal, ny, nx, k])
plt.figure()
util.visualizeGt(estImage[-1, :, :, :], "estImage")

plt.show()
