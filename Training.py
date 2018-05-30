import torch
import torch.nn as nn
from DataObject import DataObject
from NeuralFingerprint import RegressionPredictor
import matplotlib.pyplot as plt
import numpy as np

def test(data, model, criterion, verbose=False):

    with torch.no_grad():

        meanExp = 3.900528889018617
        testLostList = []
        model.eval()

        for row in data:
            prediction = model([row[1]])
            pce = float(row[0])
            target = torch.FloatTensor([[pce]]).cuda()
            loss = criterion(prediction, target)

            currentLoss = loss.data.cpu().numpy()
            refLoss = (meanExp - pce) ** 2
            testLostList += [(currentLoss,refLoss)]

        refMean = 0
        predMean = 0
        max = -99999999999
        min = 99999999999
        for predVal, refVal in testLostList:

            refMean += refVal
            predMean += predVal
            if (min > predVal):
                min = predVal
            if (max < predVal):
                max = predVal

        model.train()
        refMean = refMean / len(testLostList)
        predMean = predMean / len(testLostList)
        if (verbose):
            print("-------------- Test result --------------")
            print("Mean loss = " + str(predMean))
            print("Min loss = " + str(min))
            print("Max loss = " + str(max))
            print("Mean loss using target mean = " + str(refMean))
            print("-----------------------------------------")

        return predMean

print("Loading data")
print("...")

data = DataObject()

print("Starting training")

fingerPrintLength = 512
radius = 5
model = RegressionPredictor(data.featureSize, fingerPrintLength, radius).cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.1)

lossList = []
testLossList = []
currentLossMean = 0
count = 0
moduloPrint = 10
epoch = 10
batchSize = 100
totalIt = epoch * (len(data.trainData) / batchSize)

for k in range(epoch):

    testLossList += [test(data.testData, model, criterion, verbose=True)]
    data.shuffleTrainingData()

    for i in range(0, len(data.trainData) - batchSize, batchSize):

        prediction = model(data.trainData[i:i+batchSize,1])
        float_arr = np.array(data.trainData[i:i+batchSize,0]).astype(np.float)
        target = torch.FloatTensor(float_arr).cuda()
        target = target.unsqueeze(1)
        loss = criterion(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        currentLoss = loss.data.cpu().numpy()
        lossList += [currentLoss]
        currentLossMean += currentLoss
        count += 1

        if (count % moduloPrint == 0):
            lossMean = currentLossMean/moduloPrint
            currentLossMean = 0
            print("Step : " + str(count) + " / " + str(totalIt) + ", Mean of last " + str(moduloPrint) + " Loss : " + str(lossMean))

testLossList += [test(data.testData, model, criterion, verbose=True)]

plt.plot(lossList)
plt.show()

plt.plot(testLossList)
plt.show()