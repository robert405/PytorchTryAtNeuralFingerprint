import torch
import torch.nn as nn
from DataObject import DataObject
from NeuralFingerprint import Lipophilicity
import matplotlib.pyplot as plt
import numpy as np

def test(data, model, criterion):

    with torch.no_grad():

        meanExp = 2.1863357142857165
        testLostList = []

        for row in data.testData:
            prediction = model([row[1]])
            exp = float(row[0])
            target = torch.FloatTensor([[exp]]).cuda()
            loss = criterion(prediction, target)

            currentLoss = loss.data.cpu().numpy()
            refLoss = (meanExp - exp) ** 2
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

        refMean = refMean / len(testLostList)
        predMean = predMean / len(testLostList)
        print("-------------- Test result --------------")
        print("Mean loss = " + str(predMean))
        print("Min loss = " + str(min))
        print("Max loss = " + str(max))
        print("Mean loss using exp mean = " + str(refMean))
        print("-----------------------------------------")

print("Loading data")
print("...")

data = DataObject()

print("Starting training")

fingerPrintLength = 2048
radius = 3
model = Lipophilicity(data.featureSize, fingerPrintLength, radius).cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

lossList = []
currentLossMean = 0
count = 0
moduloPrint = 10
epoch = 10
batchSize = 50
totalIt = epoch * (len(data.trainData) / batchSize)

for k in range(epoch):

    test(data, model, criterion)
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

plt.plot(lossList)
plt.show()

test(data, model, criterion)