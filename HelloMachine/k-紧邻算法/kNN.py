# coding=utf-8
__author__ = 'Administrator'
from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 欧式距离公式
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 数组行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # tile(inX,(dataSetSize,1))复制4个inX
    sqDiffMat = diffMat ** 2  # 每个向量的平方
    sqDistances = sqDiffMat.sum(axis=1)  # 两个向量相加
    distances = sqDistances ** 0.5  # 开方
    sortedDistIndicies = distances.argsort()  # 升序，取出数组下标
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,
                                                0) + 1  # classCount.get(voteIlabel,0) 从字典中取值voteIlabel，存在则返回1，不存在返回0
    # sorted()排序 classCount.iteritems()排序的对象 operator.itemgetter(1)排序关键字,这里取第字典二个域的值,reverse=True降序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 读取文本文件数据
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    lable = {'didntLike': 1, 'smallDoses': 2, 'largeDoses': 3}
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        v = lable[listFromLine[-1]]
        classLabelVector.append(v)
        index += 1
    return returnMat, classLabelVector


# 数据归一下化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# 测试数据
def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    print normMat
    print (ranges)
    print (minVals)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "the classifier come back with:%d,the real anwser is:%d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1;
    print "the total error rate is:%d" % (errorCount / float(numTestVecs))


# 将图片转换为向量存储
def img2Vector(fileName):
    returnVect = zeros((1, 1024))
    fr = open(fileName)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = lineStr[j]
    return returnVect
