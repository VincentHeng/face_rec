import torch 
import os
from skimage import io, transform
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

class MyDataSet(Dataset):
	def __init__(self, csv_file, root_dir, transform=transforms.ToTensor()):
		self.faces = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.faces)
	def __getitem__(self, idx):
		# here we muist return the face and the label given the index 
		if torch.is_tensor(idx):
			idx = idx.tolist()
		img_name = os.path.join(self.root_dir,
                                self.faces.iloc[idx, 0])

		image = Image.open(img_name)
		im_array = np.array(image)
		im_array.astype(float)
		im_array = np.swapaxes(im_array, 0, 2)

		label = self.faces.iloc[idx, 1]
		sample = {}
		sample['image'] = im_array
		sample['label'] = label

		if self.transform:
			sample = self.transform(sample)

		return sample