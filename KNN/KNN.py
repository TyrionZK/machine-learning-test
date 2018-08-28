from numpy import *
import operator

import matplotlib
import matplotlib.pyplot as plt

from os import listdir

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group,labels

def classify0(inX,dataSet,labels,k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX,(dataSetSize,1))-dataSet
	sqDiffMat = diffMat**2
	distances = sqDiffMat.sum(axis=1)
	sortedDistanceDices = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistanceDices[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]
	
	
def file2matrix(filename):
	fr = open(filename)
	arrayLines = fr.readlines()
	numberOfLines = len(arrayLines)
	returnMat = zeros((numberOfLines,3))
	classLabelVector = []
	index = 0
	for line in arrayLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat,classLabelVector

def autoNorm(dataSet):
	minvals = dataSet.min(0)
	maxvals = dataSet.max(0)
	ranges = maxvals-minvals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minvals,(m,1))
	normDataSet = normDataSet/tile(ranges,(m,1))
	return normDataSet,ranges,minvals

def display(filename):
	datingDataMat,datingLabels = file2matrix(filename)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15*array(datingLabels),15*array(datingLabels))
	plt.show()

def datingClassTest():
	hoRatio = 0.1
	datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
	normDataSet,ranges,minvals = autoNorm(datingDataMat)
	m = normDataSet.shape[0]
	numTestVecs = int(m*hoRatio)
	error_count = 0
	for i in range(numTestVecs):
		classifierResult = classify0(normDataSet[i,:],normDataSet[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
		print(" the classifier come back with %d, the real answer is %d\n"%(classifierResult,datingLabels[i]))
		if(classifierResult!=datingLabels[i]): error_count = error_count + 1
	print("the total error rate is: %f" % (error_count/float(numTestVecs)))
	
def img2vector(filename):
	returnVector = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVector[0,32*i+j] = int(lineStr[j])
	return returnVector

def handwritingClassTest():
	hwLabels = []
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNum = int(fileStr.split('_')[0])
		hwLabels.append(classNum)
		trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
	testFileList = listdir('testDigits')
	error_count = 0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNum = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
		print("the classifier come back with %d, the real answer is %d\n" %(classifierResult,classNum))
		if(classifierResult!=classNum):
			error_count = error_count+1
	print(" the total error rate is %f\n" % (error_count/float(mTest)))