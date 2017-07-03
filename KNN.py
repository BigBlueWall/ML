from numpy import *
import operator

def classify(inX, dataSet, label, k):
	
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distance = sqDistances**0.5
	sortedDistIndicies = distance.argsort()
	classCount = {}
	for i in range(k):
		voteIlable = label[sortedDistIndicies[i]]
		classCount[voteIlable] = classCount.get(voteIlable, 0) + 1

	sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)

	return sortedClassCount[0][0]

