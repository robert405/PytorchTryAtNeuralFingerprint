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
        self.bn = nn.BatchNorm1d(featureSize)

        self.initialSoft = nn.Linear(featureSize, fingerPrintLength).cuda()

        self.layerList = []
        for i in range(radius):
            layer = {'hashW': nn.Linear(featureSize, featureSize).cuda(), 'softW':nn.Linear(featureSize, fingerPrintLength).cuda()}
            self.layerList += [layer]

    def forward(self, molData):

        atomsFeatures = np.copy(molData['atomsFeatures'])
        graphList = molData['graphList']
        fingerPrint = torch.zeros(1,self.fingerPrintLength).cuda()
        nbAtoms = len(graphList)

        initialHash = torch.FloatTensor(atomsFeatures).cuda()
        idx = F.softmax(self.initialSoft(initialHash), dim=1)
        fingerPrint += torch.sum(idx, dim=0)

        for l in range(self.radius):
            updatedAtomsFeatures = np.zeros_like(atomsFeatures)

            for i in range(nbAtoms):

                nodeEdge = graphList[i]['neighbor']
                neighborSum = atomsFeatures[i]
                if (nodeEdge.shape[0] > 0):
                    neighborFeatures = atomsFeatures[nodeEdge]
                    neighborSum += np.sum(neighborFeatures, axis=0)

                updatedAtomsFeatures[i] = neighborSum

            nodesHash = torch.FloatTensor(updatedAtomsFeatures).cuda()
            nodesHash = self.bn(F.relu(self.layerList[l]['hashW'](nodesHash)))

            idx = F.softmax(self.layerList[l]['softW'](nodesHash), dim=1)
            fingerPrint += torch.sum(idx, dim=0)

            atomsFeatures = nodesHash.data.cpu().numpy()

        return fingerPrint

class RegressionPredictor(nn.Module):
    def __init__(self, featureSize, fingerPrintLength, radius):

        super(RegressionPredictor, self).__init__()

        self.fingerPrintLength = fingerPrintLength
        self.fingerPrintModule = NeuralFingerprint(featureSize, fingerPrintLength, radius).cuda()

        self.lin1 = nn.Linear(fingerPrintLength, 512)
        self.lin2 = nn.Linear(512, 512)
        self.lin3 = nn.Linear(512, 512)
        self.lin4 = nn.Linear(512, 1)

    def forward(self, molData):

        batch = torch.zeros(len(molData), self.fingerPrintLength).cuda()
        for i in range(len(molData)):
            batch[i] = self.fingerPrintModule(molData[i])

        h = F.relu(self.lin1(batch))
        h = F.relu(self.lin2(h))
        h = F.relu(self.lin3(h))
        out = self.lin4(h)

        return out