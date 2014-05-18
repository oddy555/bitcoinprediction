from pybrain.structure import FullConnection, RecurrentNetwork, LinearLayer, SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import FullConnection, RecurrentNetwork, LinearLayer, SigmoidLayer
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

def createDataset(nInputs,inputSize):
    fileName = 'bitstampUSD.csv'
    
    ds = SupervisedDataSet(inputSize,1)
    i = 0;
    input = []
    with open(fileName,'rb') as data:
        reader = csv.reader(data)
        for row in reader:
           
            if i < 6:
                input = input + [row[2]]
                i = i +1
            elif i == 6:
                i = 0
                ds.appendLinked(input,[row[2]])
                input = []
    return ds
                
        
#net = createRecurrent(6,6)
ds = createDataset(10,6)
print ds
