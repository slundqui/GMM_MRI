import numpy as np
from scipy.ndimage import imread
import pdb

def readList(inList):
    inputFile = open(inList, 'r')
    out = inputFile.readlines()
    inputFile.close()
    #Remove newline character
    out = [o[:-1] for o in out]
    return out

def loadImages(listOfFn):
    outList = []
    for fn in listOfFn:
        outList.append(imread(fn))
    #Change into multidimension np array
    return np.array(outList)

inputList = "data/data.txt"
gtList = "data/gt.txt"

inputFn = readList(inputList)
gtFn = readList(gtList)

inputImgs = loadImages(inputFn)
gtImgs = loadImages(gtFn)

#Images are of shape [img, y, x, f]



pdb.set_trace()




