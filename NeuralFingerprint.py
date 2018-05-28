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

        for i in range(self.radius):
            updatedAtomsFeatures = np.zeros_like(atomsFeatures)

            for nodeDict in graphList:

                nodeEdge = nodeDict['neighbor']
                neighborSum = atomsFeatures[nodeDict['idx']]
                if (nodeEdge.shape[0] > 0):
                    neighborFeatures = atomsFeatures[nodeEdge]
                    neighborSum += np.sum(neighborFeatures, axis=0)

                neighborSumTensor = torch.FloatTensor(neighborSum).cuda()
                neighborSumTensor = neighborSumTensor.unsqueeze(0)

                nodeHash = F.relu(self.lin1(neighborSumTensor))
                updatedAtomsFeatures[nodeDict['idx']] = nodeHash.data.cpu().numpy()[0]

                idx = F.softmax(self.lin2(nodeHash), dim=1)
                fingerPrint += idx

            atomsFeatures = updatedAtomsFeatures

        return fingerPrint

class Lipophilicity(nn.Module):
    def __init__(self, featureSize, fingerPrintLength, radius):

        super(Lipophilicity, self).__init__()

        self.fingerPrintLength = fingerPrintLength
        self.fingerPrintModule = NeuralFingerprint(featureSize, fingerPrintLength, radius).cuda()

        self.lin1 = nn.Linear(fingerPrintLength, 512)
        self.lin2 = nn.Linear(512, 512)
        self.lin3 = nn.Linear(512, 512)
        self.lin4 = nn.Linear(512, 512)
        self.lin5 = nn.Linear(512, 1)

    def forward(self, molData):

        batch = torch.zeros(len(molData), self.fingerPrintLength).cuda()
        for i in range(len(molData)):
            batch[i] = self.fingerPrintModule(molData[i])

        h = F.relu(self.lin1(batch))
        h = F.relu(self.lin2(h))
        h = F.relu(self.lin3(h))
        h = F.relu(self.lin4(h))
        out = self.lin5(h)

        return out