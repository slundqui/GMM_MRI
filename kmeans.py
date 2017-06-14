import numpy as np
from scipy.ndimage import imread
import matplotlib.pyplot as plt
from data import getImages
from vis import visualizeGt
import tensorflow as tf
import pdb

inputList = "data/data.txt"
gtList = "data/gt.txt"
clusterOutName = "data/kmeans_cluster.npy"

#y by x
patchSize = (5, 5)

(trainData, trainGt, testData, testGt) = getImages(inputList, gtList)
(numTrain, ny, nx, drop) = trainGt.shape
k = 4


#Build tensorflow graph
sess = tf.InteractiveSession()
xImage = tf.placeholder(tf.float32, shape=[None, ny, nx, 1])
gtImage = tf.placeholder(tf.float32, shape=[None, ny, nx, 5])

#Extract patches from image
xData = tf.extract_image_patches(xImage, [1, patchSize[0], patchSize[1], 1], [1, 1, 1, 1], [1, 1, 1, 1], "SAME")
#Linearize both data and gt in batch, x, y dimension
xData = tf.reshape(xData, [-1, patchSize[0]*patchSize[1]])
gtData = tf.reshape(gtImage, [-1, k+1])

#Remove distractor class when training
dataIdxs = tf.where(tf.not_equal(gtData[:, -1], 0))
valid_xData = tf.gather(xData, dataIdxs)[:, 0, :]
valid_gtData = tf.gather(gtData, dataIdxs)[:, 0, :]

#Initialize cluster centers
cluster = tf.Variable(tf.random_normal([k, patchSize[0]*patchSize[1]], .5, 1e-3))

#E step (cluster assignment based on l2 norm)
assignments = tf.reduce_sum(tf.square(valid_xData[:, tf.newaxis, :] - cluster[tf.newaxis, :, :]), axis=2)
#Hard assign
assignIdx = tf.argmin(assignments, axis=1)
#One hot vector based on assignIdx
responsibility = tf.one_hot(assignIdx, k)

#Applies E step to all data
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

feed_dict = {xImage: trainData[:, :, :, np.newaxis], gtImage: trainGt}


loop = True
iteration = 0

#Get old assignment
oldAssignment = sess.run(assignIdx, feed_dict=feed_dict)

while loop:
    print("Iteration" + str(iteration))
    iteration += 1
    #Run em
    sess.run(stepEM, feed_dict=feed_dict)
    #Get new assignment
    newAssignment = sess.run(assignIdx, feed_dict=feed_dict)
    if(np.sum(np.abs(newAssignment - oldAssignment)) == 0):
        loop = False
    oldAssignment = newAssignment

#Save cluster assignments into numpy array
np_clusters = sess.run(cluster)
np.save(clusterOutName, np_clusters)


#Visualize
plt.figure()
visualizeGt(trainGt[0, :, :, :-1], "gtImage")

estImage = sess.run(full_responsibility, feed_dict=feed_dict)
estImage = np.reshape(estImage, [numTrain, ny, nx, k])
plt.figure()
visualizeGt(estImage[0, :, :, :], "estImage")

plt.show()

pdb.set_trace()







