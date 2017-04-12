import sys
import os
import random
import math
import pylab as pl
import scipy.io as scio

def readData(filename):
    data_file = open(filename, 'r')
    data_str = []
    data = []
    while 1:
        line = data_file.readline()
        if not line:
            break
        data_str.append(line)
    for i in range(0, len(data_str)):
        data_tmp = data_str[i].split()
        for j in range(0, len(data_tmp)):
            data_tmp[j] = float(data_tmp[j])

        data.append(data_tmp)
    data_file.close()

    return data

def normallize(data_in):
    data = data_in
    dim0_max = data[0][0]
    dim1_max = data[0][1]
    for i in range(1, len(data)):
        if abs(data[i][0]) > dim0_max:
            dim0_max = abs(data[i][0])
        if abs(data[i][1]) > dim1_max:
            dim1_max = abs(data[i][1])
    
    for i in range(0, len(data)):
        data[i][0] /= dim0_max
        data[i][1] /= dim1_max
    return data

def generateW(num, dim):
    w = []
    for i in range(0, num):

        w_tmp = []
        for j in range(0, dim):
            w_tmp.append(random.random())
        w.append(w_tmp)

    return w

def EuclidDist(vec1, vec2):
    sum = 0
    for i in range(0, len(vec1)):
        sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i])
    return sum

def neighFun(dij_2):
    sigma = 5
    return math.exp(-dij_2 / (2 * sigma * sigma))

def getLearningRate(n):
    yita0 = 0.1
    tor0 = 1000
    return yita0 * math.exp(-n / tor0)

def drawGraph(data):
    for i in range(0, len(data)):
        pl.plot(data[i][0], data[i][1], 'o')
    pl.show()

def writeData(filename, data):
    datafile = open(filename, 'w')
    for i in range(0, len(data)):
        for j in range(0, len(data[i])):
            datafile.write(str(data[i][j]) + ' ')
        datafile.write('\n')
    datafile.close()


def SOM(data, weight, iternum):

    for i in range(0, iternum):
        print i
        for j in range(0, len(data)):
            min_dist = 1000
            min_index = 0
            for k in range(0, len(weight)):
                if EuclidDist(data[j], weight[k]) < min_dist:
                    min_index = k
                    min_dist = EuclidDist(data[j], weight[k])
            for k in range(0, len(weight[min_index])):
                weight[min_index][k] += getLearningRate(iternum) * neighFun(min_dist) * (data[j][k] - weight[min_index][k])

def test(data, weight):
    label = [0] * len(data)
    for i in range(0, len(data)):
        min_dist = 1000
        min_index = 0
        for j in range(0, len(weight)):
            if EuclidDist(data[i], weight[j]) < min_dist:
                min_index = j
                min_dist = EuclidDist(data[i], weight[j])
        label[i] = min_index 
    datafile = open('./testresult.txt', 'w')
    for i in range(0, len(data)):
        for j in range(0, len(data[i])):
            datafile.write(str(data[i][j]) + ' ')
        datafile.write(str(label[i]))
        datafile.write('\n')
    datafile.close()


if __name__ == '__main__':
    iternum = 1000
    train_data = readData('./hw4-data.txt')

    weight = generateW(25, 2)

    SOM(train_data, weight, 1000)
    test(train_data, weight)
    writeData('./weight.txt', weight)

    weight_EEG = generateW(294, 310)

#    data_EEG = scio.loadmat('./hw4-EEG.mat')
#   data_EEG = data_EEG['EEG_X']
 #   data_list_eeg = data_EEG.tolist()

#    SOM(data_list_eeg, weight_EEG, 1000)

#    writeData('./weight_EEG.txt', weight_EEG)
