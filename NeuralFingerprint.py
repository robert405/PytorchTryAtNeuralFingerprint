import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeuralFingerprint(nn.Module):
    def __init__(self, featureSize, fingerPrintLength, radius):

        super(NeuralFingerprint, self).__init__()

        self.featureSize = featureSize
        self.fingerPrintLength = fingerPrintLength
        self.radius = radius
        self.lin1 = nn.Linear(featureSize, featureSize)
        self.lin2 = nn.Linear(featureSize, fingerPrintLength)

    def forward(self, molData):

        atomsFeatures = np.copy(molData['atomsFeatures'])
        graphList = molData['graphList']
        fingerPrint = torch.zeros(1,self.fingerPrintLength).cuda()
        nbAtoms = len(graphList)

        for i in range(self.radius):
            updatedAtomsFeatures = np.zeros_like(atomsFeatures)

            for i in range(nbAtoms):

                nodeEdge = graphList[i]['neighbor']
                neighborSum = atomsFeatures[i]
                if (nodeEdge.shape[0] > 0):
                    neighborFeatures = atomsFeatures[nodeEdge]
                    neighborSum += np.sum(neighborFeatures, axis=0)

                updatedAtomsFeatures[i] = neighborSum

            nodesHash = torch.FloatTensor(updatedAtomsFeatures).cuda()
            nodesHash = F.relu(self.lin1(nodesHash))

            idx = F.softmax(self.lin2(nodesHash), dim=1)
            fingerPrint += torch.sum(idx, dim=0)

            atomsFeatures = nodesHash.data.cpu().numpy()

        return fingerPrint

class RegressionPredictor(nn.Module):
    def __init__(self, featureSize, fingerPrintLength, radius):

        super(RegressionPredictor, self).__init__()

        self.fingerPrintLength = fingerPrintLength
        self.fingerPrintModule = NeuralFingerprint(featureSize, fingerPrintLength, radius).cuda()

        self.lin1 = nn.Linear(fingerPrintLength, 2048)
        self.drop1 = nn.Dropout(p=0.5)
        self.lin2 = nn.Linear(2048, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.lin3 = nn.Linear(1024, 512)
        self.drop3 = nn.Dropout(p=0.5)
        self.lin4 = nn.Linear(512, 256)
        self.drop4 = nn.Dropout(p=0.5)
        self.lin5 = nn.Linear(256, 1)

    def forward(self, molData):

        batch = torch.zeros(len(molData), self.fingerPrintLength).cuda()
        for i in range(len(molData)):
            batch[i] = self.fingerPrintModule(molData[i])

        h = F.relu(self.lin1(batch))
        h = self.drop1(h)
        h = F.relu(self.lin2(h))
        h = self.drop2(h)
        h = F.relu(self.lin3(h))
        h = self.drop3(h)
        h = F.relu(self.lin4(h))
        h = self.drop4(h)
        out = self.lin5(h)

        return out