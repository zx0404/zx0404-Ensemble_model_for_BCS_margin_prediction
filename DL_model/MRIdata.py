'''
@article{perez-garcia_torchio_2021,
    title = {{TorchIO}: a {Python} library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning},
    journal = {Computer Methods and Programs in Biomedicine},
    pages = {106236},
    year = {2021},
    issn = {0169-2607},
    doi = {https://doi.org/10.1016/j.cmpb.2021.106236},
    url = {https://www.sciencedirect.com/science/article/pii/S0169260721003102},
    author = {P{\'e}rez-Garc{\'i}a, Fernando and Sparks, Rachel and Ourselin, S{\'e}bastien},
}
'''

import os, copy
import pandas as pd
import SimpleITK as sitk
import numpy as np
import torch
import torchio as tio
from torch.utils import data

class MRIdataset(data.Dataset):
	def __init__(self, rootDir, datasetType:str):
		super().__init__()
		self.labelFile = os.path.join(rootDir, f'all.csv')
		self.dataDir = os.path.join(rootDir, datasetType)
		self.type = datasetType
		self.prepared = False

		self.dataList, self.class0, self.class1 = self._readData()
		self.setDataWeight(1,2)

	def _readData(self):
		dataFrame = pd.read_csv(self.labelFile)
		dataFrame = dataFrame[dataFrame['type']==self.type]

		labelCount = np.bincount(dataFrame.label.values)
		class0, class1 = int(labelCount[0]), int(labelCount[1])

		dataList = []
		for d in dataFrame.to_dict('records'):
			img = sitk.ReadImage(os.path.join(self.dataDir, d['file']))
			img = sitk.GetArrayFromImage(img)
			img = torch.tensor(img).unsqueeze(0).float()
			dataList.append({
				'image':img,
				'margin':torch.tensor([d['Margin_ave_x'],d['Margin_ave_y'],d['Margin_ave_z'],d['Margin_ave_ave'],
									   d['MAD_x'],d['MAD_y'],d['MAD_z'],d['MAD_ave']]),
				'label':d['label'],
				'ID':d['file']
			})

		return dataList, class0, class1

	def setDataWeight(self, weight0, weight1):
		self.dataWeightList = [weight0]*self.class0 + [weight1]*self.class1

	def __getitem__(self, index):
		prepare = tio.Compose([
			# tio.RandomAffine(),
			# tio.RandomFlip(),
			tio.ZNormalization(),
		])
		result = copy.deepcopy( self.dataList[index] )
		if self.prepared:
			result['image'] = prepare(result['image'])

		return result
	
	def __len__(self):
		return len(self.dataList)