from pybrain.datasets import SupervisedDataSet
import csv
import pickle

fileName = 'bitstampUSD.csv'

interval = 60
istart = 0 
iend= 0
openP = 0
closeP = 0
hi = 0
low = 0
avg = 0
list = []
prices = []

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
    for line in data:
        n = n + 1
        iend = float(line[0])
        current = float(line[1])
        sum = sum + current 
        if current > hi:
            hi = current
        if current < low:
            low = current
        if istart-iend >= 60:
            iend = current
            closeP = current
            avg = round(sum/n)
            prices.append(openP)
            prices.append(closeP)
            prices.append(hi)
            prices.append(low)
            prices.append(avg)
            list.append(prices)
            prices = []
           #i = i+1
pickle.dump = (list,open('list.p','wb'))
try:
    list = pickle.load(open('list.p','rb'))
except EOFError:
    pass
print list
