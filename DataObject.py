import torch
import csv
from random import shuffle
from rdkit import Chem
import numpy as np

class DataObject:

    def __init__(self, embedSize=8):

        self.data = self.loadData()
        self.dataLength = len(self.data)

        self.vocab = self.calculateVocab(self.data)
        self.vocabSize = len(self.vocab)
        self.vocabMap = self.makeMappingForAtomicNum(self.vocab)

        self.embedSize = embedSize
        self.embedFn = torch.nn.Embedding(self.vocabSize, embedSize)

        self.featureSize = embedSize + 7

        self.chemData = self.convertDataToGraph(self.data)

        sep = int((self.dataLength/10)*8)
        self.trainData = self.chemData[0:sep]
        self.testData = self.chemData[sep:self.dataLength-1]

    def shuffleTrainingData(self):
        shuffle(self.trainData)

    def calculateVocab(self, data):
        vocab = []
        for row in data:
            mol = row['mol']
            atoms = mol.GetAtoms()
            for atom in atoms:
                if (not atom.GetAtomicNum() in vocab):
                    vocab += [atom.GetAtomicNum()]

        return vocab

    def makeMappingForAtomicNum(self, vocab):

        map = {}
        for i in range(len(vocab)):
            map[vocab[i]] = i

        return map

    def loadData(self):
        data = []
        with open('Lipophilicity.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data += [{'CMPD_CHEMBLID': row['CMPD_CHEMBLID'], 'exp': row['exp'], 'smiles': row['smiles'], 'mol': Chem.MolFromSmiles(row['smiles'])}]

        return data

    def convertDataToGraph(self, data):

        chemData = []
        for row in data:
            mol = row['mol']
            atoms = mol.GetAtoms()
            graphList = []
            atomsFeatures = np.zeros((len(atoms), self.featureSize))

            for atom in atoms:
                neighborList = []
                bonds = atom.GetBonds()

                for bond in bonds:
                    neighbor = bond.GetBeginAtom()
                    if (neighbor.GetIdx() == atom.GetIdx()):
                        neighbor = bond.GetEndAtom()

                    neighborList += [neighbor.GetIdx()]

                vocabNum = [self.vocabMap[atom.GetAtomicNum()]]
                atomicNumFeature = self.embedFn(torch.LongTensor(vocabNum)).data.numpy()[0]
                otherFeature = np.array(self.createFeatureList(atom))
                features = np.concatenate((atomicNumFeature, otherFeature), axis=0)
                atomsFeatures[atom.GetIdx()] = features
                graphList += [{'idx': atom.GetIdx(), 'neighbor': np.array(neighborList)}]


            chemData += [[float(row['exp']), {'graphList': graphList, 'atomsFeatures': atomsFeatures}]]

        chemData = np.array(chemData)

        return chemData

    def smilesToIdx(self, smiles):
        idxEmbed = []
        for char in smiles:
            idxEmbed += [self.charToIdx[char]]

        return idxEmbed

    def smilesToEmbed(self, smiles):
        idxMol = self.smilesToIdx(smiles)

        return self.embedFn(torch.LongTensor(idxMol))

    def embedAllData(self):

        embedData = []
        for row in self.data:
            embedData += [{'CMPD_CHEMBLID': row['CMPD_CHEMBLID'], 'exp': row['exp'], 'smiles': self.smilesToEmbed(row['smiles'])}]

        return embedData

    def createFeatureList(self, atom):

        feature = []
        feature += [atom.GetDegree()]
        #feature += [atom.GetExplicitValence()]
        feature += [atom.GetFormalCharge()]
        feature += [atom.GetImplicitValence()]
        feature += [1] if atom.GetIsAromatic() else [0]
        #feature += [atom.GetIsotope()]
        feature += [atom.GetMass()]
        #feature += [atom.GetNumExplicitHs()]
        feature += [atom.GetNumImplicitHs()]
        #feature += [atom.GetNumRadicalElectrons()]
        feature += [1] if atom.IsInRing() else [0]

        return feature