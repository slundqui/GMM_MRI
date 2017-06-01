import numpy as np
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import pdb

#Takes a filename containing a list of filenames and returns a list of filenames
def readList(inList):
    inputFile = open(inList, 'r')
    out = inputFile.readlines()
    inputFile.close()
    #Remove newline character
    out = [o[:-1] for o in out]
    return out

#Takes a list of filenames and returns a 4d numpy array in the shape of [img, y, x, f]
#All images in list must be of the same dimensions
def loadImages(listOfFn):
    outList = []
    for fn in listOfFn:
        outList.append(imread(fn))
    #Change into multidimension np array
    outArr = np.array(outList).astype(np.float32)
    #Change range to be between 0 and 1
    outArr = outArr / 255.0
    return outArr

#Element-wise Logical and for 3 matrices
def threeWayAnd(in1, in2, in3):
    return np.logical_and(np.logical_and(in1, in2), in3)

inputList = "data/data.txt"
gtList = "data/gt.txt"

inputFn = readList(inputList)
gtFn = readList(gtList)

inputImgs = loadImages(inputFn)
gtImgs = loadImages(gtFn)

#Images are of shape [img, y, x, f]
#Take only one channel of inputImgs since image is in grayscale
inputImgs = inputImgs[:, :, :, 0]

#Transform ground truth images into [img, y, x, 5], where last dimension is one-hot
#based on class. Classes in gt are [red, green, blue, yellow, distractor]
[numImg, ny, nx, nf] = gtImgs.shape
gt = np.zeros([numImg, ny, nx, 5])

#Define threshold variable since r, g, b, and y might not be exactly 1
epsilon = .5

#Find red channel
redIdx = np.nonzero(threeWayAnd((gtImgs[:, :, :, 0] > 1-epsilon), #r channel high
                                (gtImgs[:, :, :, 1] < epsilon),   #g channel low
                                (gtImgs[:, :, :, 2] < epsilon)))  #b channel low
gt[redIdx] = [1, 0, 0, 0, 0]

#Find green channel
greenIdx = np.nonzero(threeWayAnd((gtImgs[:, :, :, 0] < epsilon),   #r channel low
                                  (gtImgs[:, :, :, 1] > 1-epsilon), #g channel high
                                  (gtImgs[:, :, :, 2] < epsilon)))  #b channel low
gt[greenIdx] = [0, 1, 0, 0, 0]

#Find blue channel
blueIdx = np.nonzero(threeWayAnd((gtImgs[:, :, :, 0] < epsilon),    #r channel low
                                 (gtImgs[:, :, :, 1] < epsilon),    #g channel low
                                 (gtImgs[:, :, :, 2] > 1-epsilon))) #b channel high
gt[blueIdx] = [0, 0, 1, 0, 0]

#Find yellow channel
yellowIdx = np.nonzero(threeWayAnd((gtImgs[:, :, :, 0] > 1-epsilon),    #r channel high
                                   (gtImgs[:, :, :, 1] > 1-epsilon),    #g channel high
                                   (gtImgs[:, :, :, 2] < epsilon)))     #b channel low
gt[yellowIdx] = [0, 0, 0, 1, 0]

#Find distractor channel, i.e., pixel is distractor iff it's not one of above classes
gt[:, :, :, 4] = 1 - np.sum(gt[:, :, :, 0:4], axis=3)

##Visualization sanity check
#plt.figure()
#plt.imshow(gtImgs[0, :, :, :])
#plt.figure()
#plt.imshow(gt[0, :, :, 4])
#plt.show()

#Sanity check, last index should be onehot
assert(np.sum(gt) == numImg*ny*nx)

#Change gt matrix into binary
gt = gt.astype(np.bool)

#Initialize means of each class with k-means, where k is number of classes
#TODO is this what I'm supposed to do for initialization? What are we clusting over?
means = np.zeros([5])
stds = np.zeros([5])
for c in range(5):
    #Get indices of class
    classIdx = np.nonzero(gt[:, :, :, c])
    #Get pixel values from images
    classPixVals = inputImgs[classIdx]
    #Calculate mean/std and store
    means[c] = np.mean(classPixVals)
    stds[c] = np.std(classPixVals)

pdb.set_trace()









