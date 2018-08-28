from numpy import *

def loadDataSet():
	postingList = [['my','dog','has','flea','problems','help','please'],
					['maybe','not','take','hime','to','dog','park','stupid'],
					['my','dalmation','is','so','cute','I','love','him'],
					['stop','posting','stupid','worthless','garbage'],
					['mr','licks','ate','my','steak','how','to','stop','him'],
					['quit','buying','worthless','dog','food','stupid']]
	classVec = [0,1,0,1,0,1]
	return postingList,classVec

def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)
	
def setOfWords2Vec(vocabList,inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print("the word: %s is not in the Vocabulary" % word)
	return returnVec
	
def trainNB0(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	p0Num = ones(numWords); p1Num = ones(numWords)
	p0Denom = 2.0; p1Denom = 2.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vec = log(p1Num/p1Denom)
	p0Vec = log(p0Num/p0Denom)
	return p0Vec,p1Vec,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
	p1 = sum(vec2Classify*p1Vec)+log(pClass1)
	p0 = sum(vec2Classify*p0Vec)+log(1-pClass1)
	if p1>p0:
		return 1
	else:
		return 0
		
def testingNB():
	listOfPosts,listClasses = loadDataSet()
	myVocabList = createVocabList(listOfPosts)
	trainMat = []
	for postinDoc in listOfPosts:
		trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
	p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
	testEntry = ['love','my','dalmation']
	thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
	print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
	
def textParse(bigString):
	import re
	listOfTokens = re.split(r'\W*',bigString)
	return [tok.lower() for tok in listOfTokens if len(tok)>2]
	
def spamTest():
	docList = []; classList = []; fullText =[]
	for i in range(1,26):
		wordlist = textParse(open('email/spam/%d.txt' % i).read())
		docList.append(wordlist)
		fullText.extend(wordlist)
		classList.append(1)
		wordlist = textParse(open('email/ham/%d.txt' % i).read())
		docList.append(wordlist)
		fullText.extend(wordlist)
		classList.append(0)
	vocabList = createVocabList(docList)
	trainingSet = list(range(50)); testSet = []   #python3不返回list对象，返回range对象
	for i in range(10):
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMat = []; trainClass = []
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
		trainClass.append(classList[docIndex])
	p0V,p1V,pAb = trainNB0(array(trainMat),array(trainClass))
	err_count = 0
	for docIndex in testSet:
		wordVec = setOfWords2Vec(vocabList,docList[docIndex])
		if classifyNB(array(wordVec),p0V,p1V,pAb) != classList[docIndex]:
			err_count += 1
	print('the error rate is ',float(err_count)/len(testSet))