from pybrain.structure import FullConnection, RecurrentNetwork, LinearLayer, SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import FullConnection, RecurrentNetwork, LinearLayer, SigmoidLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain.tools.shortcuts import buildNetwork
import pickle
import csv




def createRecurrent(inputSize,nHidden):
    n = RecurrentNetwork()
    n.addInputModule(LinearLayer(inputSize, name='in'))
    n.addModule(SigmoidLayer(nHidden, name='hidden'))
    n.addOutputModule(LinearLayer(1, name='out'))
    n.addConnection(FullConnection(n['in'], n['hidden'], name='c1'))
    n.addConnection(FullConnection(n['hidden'], n['out'], name='c2'))
    n.addRecurrentConnection(FullConnection(n['hidden'], n['hidden'], name='c3'))
    n.sortModules()
    return n

def candleGen():
    fileName = 'bitstampUSD.csv'
    global maxV
    global minV
    maxV = 0.0
    minV = 0.0
    interval = 3600
    istart = 0 
    iend= 0
    openP = 0
    closeP = 0
    hi = 0
    low = 0
    avg = 0
    list = []
#prices = []

    fileName = 'bitstampUSD.csv'
    with open(fileName,'rb') as data:
        reader = csv.reader(data)
        row = reader.next()
        istart = float(row[0])
        openP = float(row[1])
        hi = float(openP)
        low = float(openP)
        n = 0
        sum = openP
    #    i = 0
        for line in reader:
            n = n + 1
            iend = float(line[0])
            current = float(line[1])
            sum = sum + current 
            if maxV < current:
                maxV = current
            if minV > current:
                minV = current
            if current > hi:
                hi = current
            if current < low:
                low = current
            if iend-istart >= interval:
                prices = []
                closeP = current
                avg = sum/n
                prices.append(openP)
                prices.append(closeP)
                prices.append(hi)
                prices.append(low)
                prices.append(avg)
                list.append(prices)
                istart = iend
                openP = current
                low = current
                hi = current
                sum = 0
                n = 0
                
#    print list
    return list


def createDataset(nInputs,inputSize,nOutputs):
    index = 0 
    ds = SupervisedDataSet(inputSize,nOutputs)
    i = 0
    j = 0
    pList =candleGen()
    input = []
 
    for sub in pList:
        if nInputs == j:
            break
        if i < inputSize:
            input.append(sub[index])
        else:
            ds.appendLinked(input,sub[index])
            input = []
            input.append(sub[index])
            i = 0
            j = j + 1
        i = i + 1
    return ds

def createDataset2(nInputs,inputSize,nOutputs):
    index = 1
    ds = SupervisedDataSet(inputSize,nOutputs)
    i = 0
    j =  0
    pList =candleGen()
    print len(pList)
    input = []
    z = 0
    for sub in pList:
       

        if nInputs == j:
            break
        elif i < inputSize:
            input.append(sub[index])
            i = i+1
        elif i == inputSize:
            ds.appendLinked(input,sub[index])
            input.pop(0)
            input.append(sub[index])
            j = j + 1
            i = i + 1
        else:
            ds.appendLinked(input,sub[index])
            input.pop(0)
            input.append(sub[index])
            j = j + 1
        
        
    return ds

def createDataset3(nInputs,inputSize,nOutputs):
    index = 1
    ds = SupervisedDataSet(inputSize,nOutputs)
    i = 0
    j =  0
    
    pList =candleGen()
    print len(pList)
    input = []
    z = 0
    for sub in pList:
        if nInputs == j:
            break
        elif i < inputSize:
            val = (sub[index]-minV)/(maxV-minV)
            input.append(val)
            i = i+1
        else:
            val = (sub[index]-minV)/(maxV-minV)
            ds.appendLinked(input,val)
            input.pop(0)
            input.append(val)
            j = j + 1
    return ds
               
 
#,recurrent=True
#net = createRecurrent(6,12)
ds = createDataset3(10000,1,1)
net = buildNetwork(1,6,1,bias=True,recurrent=True)
#print ds
trainer = BackpropTrainer(net,ds,batchlearning=False,lrdecay=0.0,momentum=0.0,learningrate=0.01)
#trainer = RPropMinusTrainer(net,ds)

#x = trainer.trainUntilConvergence(maxEpochs=100)
ran = 50
for i in range(ran):
    x = trainer.train()
    print x 
