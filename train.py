from torch import nn
from torch import optim
import torch
import numpy as np
from dataset import MyDataSet
import torchvision 
import torchvision.transforms as transforms
from PIL import Image

class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5) # inoput, output, stride, etc
		self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5) # inoput, output, stride, etc
		self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=4) # inoput, output, stride, etc
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(2,2)
		self.fc1 = nn.Linear(in_features=8 * 13 * 13 , out_features=8)
		self.fc2 = nn.Linear(in_features=8, out_features=4)

	def forward (self, x):
		# create  maxpool layers, norm, fully connected, and linear to output
		x = self.conv1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.conv2(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.conv3(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = x.view(-1, 8 * 13 * 13)

		x = self.fc1(x) # check linear dimensions
		x = self.fc2(x)
		return x

def main():
	dataset = MyDataSet('fam_face.csv', 'fam_face', transform=None)
	ds_size = len(dataset)
	train_set, val_set = torch.utils.data.random_split(dataset, [round(ds_size * 0.9), round(ds_size * 0.1)])

	train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	network = Network().to(device)
	criterion = nn.CrossEntropyLoss() # loss function
	optimizer = torch.optim.Adam(network.parameters(), lr=0.001)# momentum=0.9)
	n_epochs = 1

	for epoch in range(20):
		for i, data in enumerate(train_loader, 0):
			inputs = data['image'].type(torch.float32).to(device)
			labels = data['label'].to(device)
			optimizer.zero_grad()
			outputs = network(inputs)
			loss = criterion(outputs, labels)
			# print(loss)
			loss.backward()
			optimizer.step()
		print('Epoch: {}; loss: {}'.format(epoch, loss))
	print('Done')

	PATH = './face_rec.pth'
	torch.save(network.state_dict(), PATH)
		
	correct, total = 0, 0
	with torch.no_grad():
		for d in val_loader:
			inputs = data['image'].type(torch.float32).to(device)
			labels = data['label'].to(device)
			outputs = network(inputs)
			_, predicted = torch.max(outputs.data, 1)
			total += 1
			correct += (predicted == labels).sum().item()
	acc = correct / total
	print('Accuracy: {}'.format(acc))
	return acc

def pred(show_img = False): # change to infer
	dataset = MyDataSet('infer_set.csv', 'infer_set', transform=None)
	val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
	val_set = torch.utils.data

	classes = ('vincent',  'grandma',  'grandpa', 'mom')

	PATH = './face_rec.pth'
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = Network().to(device)
	model.load_state_dict(torch.load(PATH))

	correct, total = 0, 0
	with torch.no_grad():
		for d in val_loader:
			inputs = d['image'].type(torch.float32).to(device)
			labels = d['label'].to(device)
			outputs = model(inputs)
			_, predicted = torch.max(outputs.data, 1)
			total += 1
			if (predicted == labels):
				correct += 1
			npimg = np.array(d['image']).squeeze()
			npimg = np.swapaxes(npimg, 0 , 2)
			npimg = np.swapaxes(npimg, 0 , 1)
			img = Image.fromarray(npimg, 'RGB')
			if show_img:
				print('Prediction: {}'.format(classes[predicted]))
				img.show(title="test")
				break
	print('Accuracy: {}'.format(correct/total))

if __name__ == '__main__':
	# main()
	pred(show_img=True)