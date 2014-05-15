from pybrain.structure import FullConnection, RecurrentNetwork, LinearLayer, SigmoidLayer
from pybrain.datasets import SupervisedDataSet

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
    f = open(fileName,'r')
    
    for i in range(nInputs):
        input = ()
        for j in range(inputSize):
            line = f.readline()
            data = [line.strip().split(',')]
            input = input + tuple(data[2])
        line = f.readline()
        data = [line.strip().split(',')]
        output = tuple(data[2])
        ds.addSample(input,output)
    return ds
        
net = createRecurrent(6,6)
ds = createDataset(10,6)
print net
