import csv
from random import shuffle
from rdkit import Chem
import numpy as np
from features import atom_features, bond_features

class DataObject:

    def __init__(self):

        self.data = self.loadData()
        self.dataLength = len(self.data)

        self.atomFeatureSize = 62
        self.bondFeatureSize = 6
        self.featureSize = self.atomFeatureSize + self.bondFeatureSize

        self.chemData = self.convertDataToGraph(self.data)

        sep = int((self.dataLength/10)*8)
        self.trainData = self.chemData[0:sep]
        self.testData = self.chemData[sep:self.dataLength-1]

    def shuffleTrainingData(self):
        shuffle(self.trainData)

    def loadData(self):
        data = []
        with open('cep-processed.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data += [{'pce': row['PCE'], 'smiles': row['smiles'], 'mol': Chem.MolFromSmiles(row['smiles'])}]

        return data

    def convertDataToGraph(self, data):

        chemData = []
        for row in data:
            mol = row['mol']
            atoms = mol.GetAtoms()
            graphList = []
            atomsFeatures = np.zeros((len(atoms), self.featureSize))

            for atom in atoms:

                atomFeature = atom_features(atom)
                bondFeature = np.zeros(6)

                neighborList = []
                bonds = atom.GetBonds()

                for bond in bonds:
                    neighbor = bond.GetBeginAtom()
                    if (neighbor.GetIdx() == atom.GetIdx()):
                        neighbor = bond.GetEndAtom()

                    neighborList += [neighbor.GetIdx()]
                    bondFeature += bond_features(bond)

                features = np.concatenate((atomFeature, bondFeature), axis=0)

                atomsFeatures[atom.GetIdx()] = features
                graphList += [{'idx': atom.GetIdx(), 'neighbor': np.array(neighborList)}]


            chemData += [[float(row['pce']), {'graphList': graphList, 'atomsFeatures': atomsFeatures}]]

        chemData = np.array(chemData)

        return chemData
