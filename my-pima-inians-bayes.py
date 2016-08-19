import numpy as np
import math as math
import csv as csv
import random as random

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg, 2) for x in numbers]) / float(len(numbers))
	return math.sqrt(variance)

def calGuassianProb(x, mean, stdev):
	exponent = math.exp( -pow(x-mean, 2) / float(2 * pow(stdev, 2)) )
	return exponent / float(stdev * math.sqrt(2 * math.pi))

def csvFileReader(filePath):
	csvReader = csv.reader(file(filePath, 'rb'))
	dataSet = []
	for line in csvReader:
		dataSet.append([float(elem) for elem in line])
	return dataSet

def splitDataSet(dataSet, splitRatio):
	trainDataSize = int(len(dataSet)) * splitRatio
	trainDataSet = dataSet
	testDataSet = []
	while int(len(trainDataSet)) >= trainDataSize:
		index = random.randrange(len(trainDataSet))
		testDataSet.append(trainDataSet.pop(index))
	return trainDataSet, testDataSet

def clusterByCategory(dataSet):
	categoryDict = {}
	for elem in dataSet:
		label = elem[-1]
		if(label not in categoryDict):
			categoryDict[label] = []
		categoryDict[label].append(elem)
	return categoryDict

def summarizeByCategory(dataSet):
	summarizes = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataSet)]
	del summarizes[-1]
	return summarizes

def trainning(trainDataSet):
	categoryDict = clusterByCategory(trainDataSet)
	summarizeCategoryDict = {}
	for category, instances in categoryDict.iteritems():
		summarizeCategoryDict[category] = summarizeByCategory(instances)
	return summarizeCategoryDict

def calProbByCategory(inputVector, summarizeCategoryDict):
	probReslut = {}
	for category, instances in summarizeCategoryDict.iteritems():
		probReslut[category] = 1
		for i in range(len(instances)):
			mean, stdev = instances[i]
			attribute = inputVector[i]
			probReslut[category] *= calGuassianProb(attribute, mean, stdev)
	return probReslut

def getBestCategory(inputVector, summarizeCategoryDict):
	bestCategory = None
	bestProb = -1.0
	probReslut = calProbByCategory(inputVector, summarizeCategoryDict)
	for category, probability in probReslut.iteritems():
		if(category == None or probability > bestProb):
			bestProb = probability
			bestCategory = category
	return bestCategory, bestProb

def predict(testDataSet, summarizeCategoryDict):
	predictResult = []
	for dataVector in testDataSet:
		bestCategory, bestProb = getBestCategory(dataVector, summarizeCategoryDict)
		predictResult.append((bestCategory, bestProb))
	return predictResult

def calAccuracy(predictResult, testDataSet):
	correctNum = 0
	for i in range(len(testDataSet)):
		if(predictResult[i][0] == testDataSet[i][-1]):
			correctNum += 1
	return 100.0 * correctNum/float(len(testDataSet))

def main():
	workspace = 'E:/MachineLearning/'
	filename = workspace + 'pima-indians-diabetes.data.csv'
	splitRatio = 0.67
	dataSet = csvFileReader(filename)
	trainDataSet, testDataSet = splitDataSet(dataSet, splitRatio)
	print('Split {0} rows into train={1} and test={2} rows').format(len(dataSet), len(trainDataSet), len(testDataSet))

	summarizeCategoryDict = trainning(trainDataSet)

	predictResult = predict(testDataSet, summarizeCategoryDict)

	accuracy = calAccuracy(predictResult, testDataSet)
	print('Accuracy: {0}%').format(accuracy)

main()