import numpy as np
from scipy.ndimage import imread
import matplotlib.pyplot as plt
from data import getImages
import pdb

#Build tensorflow graph
sess = tf.InteractiveSession()
#xImage = tf.placeholder(tf.float32, shape=[None,



inputList = "data/data.txt"
gtList = "data/gt.txt"
#y by x
patchSize = (5, 5)

(trainData, trainGt, testData, testGt) = getImages(inputList, gtList)
pdb.set_trace()

#Initialize means of each class with k-means, where k is number of classes
#TODO is this what I'm supposed to do for initialization? What are we clusting over?
means = np.zeros(patchSize)
learningRate = .1

halfPatchSize = (np.floor(patchSize[0]/2), np.floor(patchSize[0]/2))

#For every class
for c in range(5):
    #Get indices of class
    (imgIdx, yIdx, xIdx) = np.nonzero(trainGt[:, :, :, c])

    ymin = yIdx - halfPatchSize[0]
    ymax = yIdx + halfPatchSize[0]+1
    xmin = xIdx - halfPatchSize[1]
    xMax = xIdx + halfPatchSize[1]+1




    pdb.set_trace()
    #Get the patch from the image

    classPixVals = trainImg[classIdx]
    #Calculate mean/std and store
    means[c] = np.mean(classPixVals)

pdb.set_trace()









