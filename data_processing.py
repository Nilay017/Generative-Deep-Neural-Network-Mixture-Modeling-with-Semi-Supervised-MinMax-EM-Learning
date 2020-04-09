from __future__ import print_function
import math
import os, time
import itertools
import pickle
import argparse
import math
from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import make_grid
import numpy as np
import cv2
import csv
import re
import pickle as pkl
from cleaned_Functions import *
from myMetrics import *
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as mplot
from scipy.stats import multivariate_normal
from CustomDatasets import *
from tqdm import tqdm
import time


import warnings
warnings.filterwarnings("ignore")



parser = argparse.ArgumentParser()
parser.add_argument("-shuffle", dest="shuffle", nargs='?', help="Set True if want to shuffle MNIST dataset", type=bool, default=True)
parser.add_argument("-use_cuda", dest="use_cuda", nargs='?', help="Set True if want to use GPU", type=bool, default=True)
parser.add_argument("-out", dest="filepathlocation", nargs='?', help="output file path", type=str, default="./Output_data_processing")
parser.add_argument("-in", dest="datafilepath", nargs='?', help="input file path", type=str, default="./MNIST_1000_121.pkl")
parser.add_argument("-g", "-–gpu", dest="gpu", default=0, type=int, help="GPU id to use")
parser.add_argument("-num_clusters", dest="num_clusters", default=-1, type=int, help="Number of clusters")
parser.add_argument("-seed", dest="seed", default=0, type=int, help="seed")
args = parser.parse_args()
"""
python3 data_processing.py -shuffle True -use_cuda True -in ./trial/MNIST_1000_10201.pkl -out ./Output_data_processing -g 7

python3 data_processing.py -shuffle True -use_cuda True -in ./MNIST_numperdigit_1000_numdigits_5_seed_0.pkl -out ./Output_data_processing -g 7 -num_clusters 5 -seed 0

python3 data_processing.py -shuffle True -use_cuda True -in ./CIFAR10_numperdigit_1000_numdigits_3_seed_0.pkl -out ./Output_data_processing -g 7 -num_clusters 3 -seed 0

python3 data_processing.py -shuffle True -use_cuda True -in ./CelebA_Data/CelebA_numperclass_1000_numclasses_5_images__seed_0.pkl -out ./Output_data_processing -g 7 -num_clusters 5 -seed 0

python3 data_processing.py -shuffle True -use_cuda True -in ./CelebA_Data/noisy_sigma_all_chan_0.2_CelebA_numperclass_1000_numclasses_5_images__seed_0.pkl -out ./Output_data_processing -g 7 -num_clusters 5 -seed 0

python3 data_processing.py -shuffle True -use_cuda True -in ./Diabetic_retinopathy_Data/Diabetic_Retinopathy_numperclass__0-1000_2-1000_4-353__images__seed_0.0.pkl -out ./Output_data_processing -g 7 -num_clusters 3 -seed 0
"""

class MyCropTransform:
	"""Custom Crop the PIL image according to (x, y, h, w) i.e img[:, x:(x+h), y:(y+w)]"""
	def __init__(self, x, y, h, w):
		self.x = x
		self.y = y
		self.h = h
		self.w = w
	def __call__(self, img):
		return transforms.functional.crop(img, self.x, self.y, self.h, self.w)

def Euclidean_squared_distance(x, y, dimensions=2):
	assert(x.shape == y.shape)
	distance = (x-y)**2
	assert(dimensions <= len(distance.shape))
	for i in range(dimensions):
		distance = torch.sum(distance, len(distance.shape) - 1)
	return distance

def RBF_Kernel(x, y, sigma=100, dimensions=2):
	distance = Euclidean_squared_distance(x, y, dimensions)
	distance = distance.to(torch.double)
	return torch.exp(-distance/(2.00*sigma))

def Euclidean_kernel(x, y, sigma=100, dimensions=2):
	return torch.sum(x*y)
	
def choose_cluster_images(data, labels, number, distance_metric=Euclidean_squared_distance, seed=0):
	unique_labels = np.unique(np.array(labels))
	assert(unique_labels.shape[0] >= 1)

	idx = labels == unique_labels[0]
	finaldata, finallabels = Kmeans_plus_plus(data[idx], labels[idx], number, distance_metric, seed)
	print("done")

	for i in range(1, unique_labels.shape[0]):
		idx = labels == unique_labels[i]
		tmp_data, tmp_labels = Kmeans_plus_plus(data[idx], labels[idx], number, distance_metric, seed)
		print("done")
		finaldata = torch.cat((finaldata, tmp_data), 0)
		finallabels = torch.cat((finallabels, tmp_labels), 0)

	return finaldata, finallabels

# def split_by_supervision_fraction(datafile_path, supervision_level=0.0):

def choose_cluster_images_3D(data, labels, number, distance_metric=Euclidean_squared_distance, seed=0):
	unique_labels = np.unique(np.array(labels))
	assert(unique_labels.shape[0] >= 1)

	label_number_map = {}
	if isinstance(number, type(1)):
		for unq_label in unique_labels:
			label_number_map[unq_label] = number
	else:
		# Else the number is dict object with mapping from label to num of images for that label
		assert(isinstance(number, type({})))
		label_number_map = number

	idx = labels == unique_labels[0]
	finaldata, finallabels = Kmeans_plus_plus_3D(data[idx], labels[idx], label_number_map[unique_labels[0]], distance_metric, seed)	
	print("done")

	for i in range(1, unique_labels.shape[0]):
		idx = labels == unique_labels[i]
		tmp_data, tmp_labels = Kmeans_plus_plus_3D(data[idx], labels[idx], label_number_map[unique_labels[i]], distance_metric, seed)
		print("done")
		finaldata = torch.cat((finaldata, tmp_data), 0)
		finallabels = torch.cat((finallabels, tmp_labels), 0)

	return finaldata, finallabels

def Kmeans_plus_plus(data, labels, cluster_num, sq_distance_function, seed):
	torch.manual_seed(seed)
	dtype = torch.double
	data = data.to(dtype)
	cumulative_prob = torch.cumsum(torch.ones(data.shape[0]) / data.shape[0], dim=0)
	cluster_centers = torch.zeros(cluster_num, data.shape[1], data.shape[2]).to(dtype)
	cluster_center_labels = torch.zeros(cluster_num)

	#first center
	index = binarysearch(cumulative_prob, torch.rand(1))
	cluster_centers[0, :, :] = data[index].to(dtype)
	cluster_center_labels[0] = labels[index]
	distance_square_array = sq_distance_function(data.to(dtype), (cluster_centers[0, :, :]).repeat(data.shape[0], 1, 1), 2).to(dtype)	
	
	#Kmeans++
	for i in range(1, cluster_num):
		#Next center
		cumulative_prob = torch.cumsum(distance_square_array / sum(distance_square_array), dim=0).to(dtype)
		index = binarysearch(cumulative_prob, torch.rand(1).to(dtype))
		cluster_centers[i, :, :] = data[index].to(dtype)
		cluster_center_labels[i] = labels[index]

		#update distance matrix
		torch.min(input = distance_square_array, other = sq_distance_function(data, (cluster_centers[i, :, :]).repeat(data.shape[0], 1, 1), 2).to(dtype), out = distance_square_array)

	return cluster_centers,  cluster_center_labels

def Kmeans_plus_plus_3D(data, labels, cluster_num, sq_distance_function, seed):
	torch.manual_seed(seed)
	dtype = torch.double
	data = data.to(dtype)
	cumulative_prob = torch.cumsum(torch.ones(data.shape[0]) / data.shape[0], dim=0)
	cluster_centers = torch.zeros(cluster_num, data.shape[1], data.shape[2], data.shape[3]).to(dtype)
	cluster_center_labels = torch.zeros(cluster_num)

	#first center
	index = binarysearch(cumulative_prob, torch.rand(1))
	cluster_centers[0] = data[index].to(dtype)
	cluster_center_labels[0] = labels[index]
	distance_square_array = sq_distance_function(data.to(dtype), (cluster_centers[0]).repeat(data.shape[0], 1, 1, 1), 3).to(dtype)	
	
	#Kmeans++
	for i in range(1, cluster_num):
		#Next center
		cumulative_prob = torch.cumsum(distance_square_array / sum(distance_square_array), dim=0).to(dtype)
		index = binarysearch(cumulative_prob, torch.rand(1).to(dtype))
		cluster_centers[i] = data[index].to(dtype)
		cluster_center_labels[i] = labels[index]

		#update distance matrix
		torch.min(input = distance_square_array, other = sq_distance_function(data, (cluster_centers[i, :, :]).repeat(data.shape[0], 1, 1, 1), 3).to(dtype), out = distance_square_array)

	return cluster_centers,  cluster_center_labels


def create_and_store_MNISTdataset(pathstr, digits, img_size=64, num_images_per_digit=100, seed=0):
	torch.manual_seed(seed)

	transform = transforms.Compose([
		transforms.Resize(img_size), #Used transforms.Resize() instead of transforms.Scale()
		transforms.ToTensor(),
		transforms.Normalize([0.5], [0.5])
	])

	dataMNIST = datasets.MNIST('data', train=True, download=True, transform=transform)

	# Initializing dataset points
	idx = dataMNIST.targets == digits[0]
	target = dataMNIST.targets[idx]
	data = dataMNIST.data[idx]

	for j in range(1, digits.shape[0]):
		idx = dataMNIST.targets == digits[j]
		target = torch.cat((target, dataMNIST.targets[idx]), 0)
		data = torch.cat((data, dataMNIST.data[idx]), 0)

	finaldata, finallabels = choose_cluster_images(data, target, num_images_per_digit, seed=seed)
	
	# with open(pathstr + '/MNIST_' + str(num_images_per_digit) + '_' + str(unique_index) + '_seed_' + str(seed) + '.pkl','wb') as f:
	# 	pkl.dump((finaldata, finallabels), f)

	with open(pathstr + '/MNIST_numperdigit_' + str(num_images_per_digit) + '_numdigits_' + str(digits.shape[0]) + '_seed_' + str(seed) + '.pkl','wb') as f:
		pkl.dump((finaldata, finallabels), f)

def create_and_store_CIFARdataset(pathstr, class_labels, img_size=32, num_images_per_digit=1000, seed=0, datapath=None):
	torch.manual_seed(seed)
	transform = transforms.Compose([
		transforms.Resize(img_size), #Used transforms.Resize() instead of transforms.Scale()
		transforms.ToTensor(),
		transforms.Normalize([0.5], [0.5])
	])

	dataCIFAR = None
	if datapath is None:
		dataCIFAR = datasets.CIFAR10('data', train=True, download=True, transform=transform)
	else:
		dataCIFAR = datasets.CIFAR10(datapath, train=True, download=False, transform=transform)

	originaltargets = torch.tensor(dataCIFAR.targets)
	originaldata = torch.tensor(dataCIFAR.data)

	# Initializing dataset points
	idx = originaltargets == class_labels[0]
	target = originaltargets[idx]
	data = originaldata[idx]

	for j in range(1, class_labels.shape[0]):
		idx = originaltargets == class_labels[j]
		target = torch.cat((target, originaltargets[idx]), 0)
		data = torch.cat((data, originaldata[idx]), 0)

	finaldata, finallabels = choose_cluster_images_3D(data, target, num_images_per_digit, seed=seed)

	with open(pathstr + '/CIFAR10_numperdigit_' + str(num_images_per_digit) + '_numdigits_' + str(class_labels.shape[0]) + '_seed_' + str(seed) + '.pkl','wb') as f:
		pkl.dump((finaldata, finallabels), f)


def create_and_store_CelebAdataset(pathstr, classes =\
 np.array(['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Bald']),\
  choose_indices = False, img_size=32, num_images_to_choose_from=3000,\
   num_images_per_class=1000, seed=0):
	torch.manual_seed(seed)
	dtype = torch.double
	num_classes = classes.shape[0]
	bit_mask_class = np.zeros([num_classes, 40])
	valid_indices = np.zeros([num_classes, num_images_to_choose_from])

	transform = transforms.Compose([
	MyCropTransform(40, 15, 148, 148),
	transforms.Resize((img_size, img_size)),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	train_dataCelebA = datasets.CelebA('/home/nilay/GANMM-master/data', split="all",  target_type=["attr", "bbox"], transform=transform, target_transform=None, download=False)
	attribute_names = np.array(train_dataCelebA.attr_names)

	class_id = 0
	for class_ in list(classes):
		bit_mask_class[class_id] = (attribute_names == class_).astype(np.long)
		class_id += 1

	list_sets_indices = []
	for class_id in range(num_classes):
		list_sets_indices.append(set(np.array((train_dataCelebA.attr[:, bit_mask_class[class_id] == 1] == 1).reshape(-1).to(torch.long).nonzero().reshape(-1))))
	
	for class_id in range(num_classes):
		myset = list_sets_indices[class_id]
	
		for class_id_2 in range(num_classes):
			if class_id == class_id_2:
				continue
			else:
				myset.difference_update(list_sets_indices[class_id_2])

		valid_indices[class_id] = np.array(list(myset))[:num_images_to_choose_from]

	data = torch.tensor([]).to(dtype)
	target = torch.tensor([]).to(dtype)
	indices = torch.tensor([]).to(dtype)

	t1 = time.time()
	for class_id in range(num_classes):
		t2 = time.time()
		for index in list(valid_indices[class_id].astype(np.long)):
			data = torch.cat((data, train_dataCelebA[index][0].reshape(1, 3, img_size, img_size).to(dtype)), 0)
			target = torch.cat((target, torch.tensor([class_id]).to(dtype)), 0)
		
		print(time.time() - t2, " secs elapsed")
		print("One class done")
		indices = torch.cat((indices, torch.tensor(valid_indices[class_id]).to(dtype)), 0)

	print(time.time() - t1, " secs elapsed total")
	print("All classes Done")

	with open(pathstr + '/CelebA_all' + '_seed_' + str(seed) + '.pkl','wb') as f:
		pkl.dump((data, (indices, target)), f)
	
	print("Starting Kmeans++")
	finaldata = None
	finallabels = None
	choose_indices_str = None

	if choose_indices:
		finaldata, finallabels = choose_cluster_image_indices(data, target, indices, num_images_per_class, seed=seed, dimension=3)
		choose_indices_str = '_indices_'
	else:
		finaldata, finallabels = choose_cluster_images_3D(data, target, num_images_per_class, seed=seed)
		choose_indices_str = '_images_'
	
	with open(pathstr + '/CelebA_numperclass_' + str(num_images_per_class) + '_numclasses_' +\
	 str(num_classes) + choose_indices_str + '_seed_' + str(seed) + '.pkl','wb') as f:
		pkl.dump((finaldata, finallabels), f)


def create_and_store_DiabeticRetinopathydataset(pathstr, inputdir, classes=[0, 2, [3, 4]],\
 choose_indices=False, img_size=256, num_images_to_choose_from=[2000, 2000, 778],\
  num_images_per_class=[1000, 1000, 778], left_eye=True, train=True, seed=0.0):
	torch.manual_seed(seed)
	dtype = torch.double
	imagedir = None
	labelpath = None
	if train:
		imagedir = inputdir + "/train"
		labelpath = inputdir + "/trainLabels.csv"
	else:
		imagedir = inputdir + "/test"
		labelpath = inputdir + "/retinopathy_solution.csv"

	label_csv_reader = csv.reader(open(labelpath), delimiter=",")
	label_file = []
	for line in label_csv_reader:
		if line[1] == 'level':
			continue
		label_file.append([line[0], line[1]])

	regex = None
	if left_eye:
		regex = re.compile('.*_left')
	else:
		regex = re.compile('.*_right')
	matcher = np.vectorize(lambda x: bool(regex.match(x)))
	
	label_file = np.array(label_file)
	all_image_names = []
	all_indices = torch.zeros(sum(num_images_to_choose_from))
	all_labels = np.zeros(sum(num_images_to_choose_from))
	
	start_index = 0
	for i in range(len(classes)):
		class_indices = None
		if isinstance(classes[i], type(1)):
			class_indices = label_file[:, 1] == str(classes[i])
		elif isinstance(classes[i], type([])):
			assert(len(classes[i]) > 0)
			class_indices = label_file[:, 1] == str(classes[i][0])
			for sub_class in classes[i][1:]:
				class_indices += (label_file[:, 1] == str(sub_class))
		else:
			print("Invalid classes")
			assert(1==0)

		class_images = label_file[class_indices][:, 0]
		matched_indices = matcher(class_images)
		selected_indices = matched_indices[matched_indices]
		selected_indices[num_images_to_choose_from[i]:] = False
		matched_indices[matched_indices] = selected_indices

		class_indices[class_indices] = matched_indices
		all_indices[start_index:(start_index+num_images_to_choose_from[i])] = torch.tensor(class_indices).nonzero()[:, 0]
		
		images_for_given_eye = class_images[matched_indices]
		all_image_names += list(images_for_given_eye)

		if isinstance(classes[i], type(1)):
			all_labels[start_index:(start_index+num_images_to_choose_from[i])] = classes[i]
		else:
			all_labels[start_index:(start_index+num_images_to_choose_from[i])] = classes[i][-1]

		start_index += num_images_to_choose_from[i]

	all_images = []
	for image_name in all_image_names:
		image_file_path = imagedir + "/" + image_name + '.jpeg'
		read_image = cv2.imread(image_file_path, 1)
		read_image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
		all_images.append(read_image)

	all_images = np.array(all_images)
	all_image_names = np.array(all_image_names)

	# all_images = torch.tensor(all_images).to(dtype)
	all_indices = all_indices.to(torch.int)
	all_labels = torch.tensor(all_labels).to(dtype)

	transform1 = transforms.Compose([
		transforms.CenterCrop((1600, 2400)),
		transforms.Resize((img_size, img_size)),
		transforms.ToTensor(),
	])
	transform2 = transforms.ToPILImage()
	
	# all_images  = all_images.to(torch.uint8).clone()
	all_images_transformed = torch.zeros([all_images.shape[0], 3, img_size, img_size])\
	.to(torch.uint8)
	for image_idx in range(all_images.shape[0]):
		print(image_idx)
		all_images_transformed[image_idx] = transform1(transform2(\
			torch.tensor(all_images[image_idx]).to(torch.uint8).transpose(2, 1).transpose(1, 0)\
			.clone()))

	all_label_number_map = {}
	for i in range(len(classes)):
		if isinstance(classes[i], type(1)):
			all_label_number_map[classes[i]] = num_images_per_class[i]
		else:
			all_label_number_map[classes[i][-1]] = num_images_per_class[i]

	with open(pathstr + '/Diabetic_Retinopathy_transformed_all' + '_seed_' + str(seed) + '.pkl','wb') as f:
		pkl.dump((all_images_transformed, (all_indices, (all_image_names, (all_labels, all_label_number_map)))), f)
	
	# print("Starting Kmeans++")
	# finaldata = None
	# finallabels = None
	# choose_indices_str = None

	# if choose_indices:
	# 	finaldata, finallabels = choose_cluster_image_indices(all_images_transformed, all_labels, all_indices, all_label_number_map, seed=seed, dimension=3)
	# 	choose_indices_str = '_indices_'
	# else:
	# 	finaldata, finallabels = choose_cluster_images_3D(all_images_transformed, all_labels, all_label_number_map, seed=seed)
	# 	choose_indices_str = '_images_'
	
	# all_label_number_map_str = "_"
	# for i in range(len(classes)):
	# 	if isinstance(classes[i], type(1)):
	# 		all_label_number_map_str += str(classes[i]) + "-" +\
	# 		 str(all_label_number_map[classes[i]]) + "_"
	# 	else:
	# 		all_label_number_map_str += str(classes[i]) + "-" +\
	# 		 str(all_label_number_map[classes[i][-1]]) + "_"
	
	# with open(pathstr + '/Diabetic_Retinopathy_transformed_numperclass_' + all_label_number_map_str\
	#  + choose_indices_str + '_seed_' + str(seed) + '.pkl','wb') as f:
	# 	pkl.dump((finaldata, finallabels), f)


def gen_kmeansplusplus_retinopathydataset(pathstr, inputfile, choose_indices = True, classes=[0, 2, [3, 4]],\
 seed=0):
	f = open(pathstr + '/' + inputfile, 'rb')
	all_images_transformed, (all_indices, (all_image_names, (all_labels, all_label_number_map))) = pkl.load(f)
	
	print("Starting Kmeans++")
	finaldata = None
	finallabels = None
	choose_indices_str = None

	if choose_indices:
		finaldata, finallabels = choose_cluster_image_indices(all_images_transformed, all_labels, all_indices, all_label_number_map, seed=seed, dimension=3)
		choose_indices_str = '_indices_'
	else:
		finaldata, finallabels = choose_cluster_images_3D(all_images_transformed, all_labels, all_label_number_map, seed=seed)
		choose_indices_str = '_images_'
	
	all_label_number_map_str = "_"
	for i in range(len(classes)):
		if isinstance(classes[i], type(1)):
			all_label_number_map_str += str(classes[i]) + "-" +\
			 str(all_label_number_map[classes[i]]) + "_"
		else:
			all_label_number_map_str += str(classes[i]) + "-" +\
			 str(all_label_number_map[classes[i][-1]]) + "_"
	
	with open(pathstr + '/Diabetic_Retinopathy_transformed_numperclass_' + all_label_number_map_str\
	 + choose_indices_str + '_seed_' + str(seed) + '.pkl','wb') as g:
		pkl.dump((finaldata, finallabels), g)

def gen_kmeansplusplus_CelebAdataset(pathstr, inputfile, choose_indices = True,\
 num_images_per_class=1000, num_classes=5, seed=0):
	f = open(pathstr + '/' + inputfile, 'rb')
	data, (indices, target) = pkl.load(f)

	print("Starting Kmeans++")
	finaldata = None
	finallabels = None
	choose_indices_str = None

	if choose_indices:
		finaldata, finallabels = choose_cluster_image_indices(data, target, indices, num_images_per_class, seed=seed, dimension=3)
		choose_indices_str = '_indices_'
	else:
		finaldata, finallabels = choose_cluster_images_3D(data, target, num_images_per_class, seed=seed)
		choose_indices_str = '_images_'
	
	with open(pathstr + '/CelebA_numperclass_' + str(num_images_per_class) + '_numclasses_' +\
	 str(num_classes) + choose_indices_str + '_seed_' + str(seed) + '.pkl','wb') as f:
		pkl.dump((finaldata, finallabels), f)

def transform_labels_2_zero_idx(inputfile):
	f = open(inputfile, 'rb')
	data, labels = pkl.load(f)
	unique_labels = np.unique(np.array(labels))
	map = {}
	for i in range(unique_labels.shape[0]):
		map[unique_labels[i]] = i	
	transformed_labels = torch.tensor([map[x] for x in np.array(labels)]).to(torch.float32)
	outputfile = inputfile[:-4] + "_zero_index_labels.pkl"
	g = open(outputfile, 'wb')
	pkl.dump((data, transformed_labels), g)
	g.close()
	f.close() 

def add_noise_and_save_again(pathstr, dataset, num_images_per_digit, num_digits, seed, mean, sigma, noise_to_all_channels=False):
	torch.manual_seed(seed)
	dtype = torch.double

	if dataset == 'CelebA':
		file_name = dataset + '_numperclass_' + str(num_images_per_digit) + '_numclasses_' +\
		 str(num_digits) + '_images_' '_seed_' + str(int(seed)) + '.pkl'
	else:	
		file_name = dataset + '_numperdigit_' + str(num_images_per_digit) + '_numdigits_' +\
		 str(num_digits) + '_seed_' + str(seed) + '.pkl'

	f = open(pathstr + '/' + file_name, 'rb')
	finaldata, finallabels = pkl.load(f)

	noise = np.zeros(finaldata.shape).astype('double')
	batch_size = finaldata.shape[0]
	num_channels = finaldata.shape[1]

	########
	# Imp!!!!
	# Note: cv2.randn(x, mean, sigma) only fills first 2 dimensions of x with noise and not the third
	# Hence, here were adding noise only in first 2 dimensions of the RGB image
	# Ideally loop over the first dimension of RGB image and add noise to the rest of the 2 dimensions
	#######

	if noise_to_all_channels:
		for batch_no in range(batch_size):
			for channel in range(num_channels):
				cv2.randn(noise[batch_no][channel], mean, sigma)
	else:
		for batch_no in range(batch_size):
			cv2.randn(noise[batch_no], mean, sigma)

	finalnoisydata = finaldata.clone() + torch.tensor(noise).to(dtype)

	if dataset == 'CelebA':
		finalnoisydata  = torch.clamp(finalnoisydata, -1, 1).clone()

	if noise_to_all_channels:
		with open(pathstr + '/noisy_sigma_all_chan_' +str(sigma) + '_' + file_name, 'wb') as newf:
			pkl.dump((finalnoisydata, finallabels), newf)
	else:
		with open(pathstr + '/noisy_sigma_' +str(sigma) + '_' + file_name, 'wb') as newf:
			pkl.dump((finalnoisydata, finallabels), newf)
		# with open(pathstr + '/noisy_sigma_' +str(sigma[0]) + '_' + file_name, 'wb') as newf:
		# 	pkl.dump((finalnoisydata, finallabels), newf)


def semisupervised_Kmeans(datafile_path, distance_metric, num_clusters, log_final_results, supervision_level=0, seed=0):
	dtype = torch.double
	with open(datafile_path, 'rb') as f:
		data, labels = pkl.load(f)

	torch.manual_seed(seed)
	unique_cluster_labels = np.unique(np.array(labels))

	labels = labels.to(dtype)
	num_images = data.shape[0]

	clustering = (torch.zeros(num_images) - 1).to(torch.double)
	cluster_centers = torch.zeros(num_clusters, data.shape[1], data.shape[2]).to(torch.double)
	fixed_indices = torch.zeros(num_images).to(dtype)

	if supervision_level > 0:
		assert(supervision_level <= 1.0)
		unique_cluster_labels = torch.tensor(np.unique(np.array(labels))).to(dtype)
		assert(num_clusters == unique_cluster_labels.shape[0])

		for i in range(unique_cluster_labels.shape[0]):
			idx = labels == unique_cluster_labels[i]

			temp_fixed_indices = torch.zeros(idx[idx == 1].shape[0])
			num_images_to_select = int(temp_fixed_indices.shape[0] * supervision_level)
			# temp_fixed_indices[torch.randperm(temp_fixed_indices.shape[0])[:num_images_to_select]] = 1
			temp_fixed_indices[:num_images_to_select] = 1
			idx[idx==1] = temp_fixed_indices.to(torch.bool)

			fixed_indices += idx.to(dtype)
			clustering[idx] = unique_cluster_labels[i]
			cluster_centers[i] = torch.mean(data[idx], 0)
	else:
		assert(supervision_level == 0)
		cluster_centers, dummy_cluster_center_labels = Kmeans_plus_plus(data, labels, num_clusters, distance_metric, seed)

	# Kmeans algorithm
	indices_to_update = (1 - fixed_indices).to(torch.long)
	indices_to_update = indices_to_update == 1
	
	finalvalues = data[indices_to_update]

	old_clustering = clustering.clone()

	temp_distance_arr = distance_metric(finalvalues.view(finalvalues.shape[0], 1, finalvalues.shape[1], finalvalues.shape[2]).repeat(1, num_clusters, 1, 1), cluster_centers.repeat(finalvalues.shape[0], 1, 1, 1), 2).to(dtype)
	label_indices = torch.min(temp_distance_arr, 1)[1]	
	clustering[indices_to_update] = torch.tensor([unique_cluster_labels[j] for j in label_indices]).to(dtype)
	
	while not torch.prod((old_clustering == clustering)):
		old_clustering = clustering.clone()
		# Update cluster centers
		for i in range(num_clusters):
			if torch.sum(clustering == unique_cluster_labels[i]) == 0:
				print("cluster", i, ": No point in this cluster!")
				continue

			cluster_centers[i] = torch.mean(data[clustering == unique_cluster_labels[i]], 0)

		# Update clustering
		label_indices = torch.min(distance_metric(finalvalues.view(finalvalues.shape[0], 1, finalvalues.shape[1], finalvalues.shape[2]).repeat(1, num_clusters, 1, 1), cluster_centers.repeat(finalvalues.shape[0], 1, 1, 1), 2).to(dtype), 1)[1]
		clustering[indices_to_update] = torch.tensor([unique_cluster_labels[j] for j in label_indices]).to(dtype)


	nmi = NMI(np.array(labels), np.array(clustering))
	ari = ARI(np.array(labels), np.array(clustering))
	acc = ACC(np.array(labels.to(torch.long)), np.array(clustering.to(torch.long)))
	print("NMI : ", nmi)
	print("ARI : ", ari)
	print("ACC : ", acc)

	log_final_results.write("kmeans_metrics_NMI: " + str(nmi) + " \n")
	log_final_results.write("kmeans_metrics_ARI: " + str(ari) + " \n")
	log_final_results.write("kmeans_metrics_ACC: " + str(acc[0]) + " \n")

	return indices_to_update, clustering.to(torch.long), nmi, ari, acc[0]

def semisupervised_Kmeans_3D(datafile_path, distance_metric, num_clusters, log_final_results, supervision_level=0, seed=0):
	dtype = torch.double
	with open(datafile_path, 'rb') as f:
		data, labels = pkl.load(f)

	torch.manual_seed(seed)
	unique_cluster_labels = np.unique(np.array(labels))

	labels = labels.to(dtype)
	num_images = data.shape[0]

	clustering = (torch.zeros(num_images) - 1).to(torch.double)
	cluster_centers = torch.zeros(num_clusters, data.shape[1], data.shape[2], data.shape[3]).to(torch.double)
	fixed_indices = torch.zeros(num_images).to(dtype)

	if supervision_level > 0:
		assert(supervision_level <= 1.0)
		unique_cluster_labels = torch.tensor(np.unique(np.array(labels))).to(dtype)
		assert(num_clusters == unique_cluster_labels.shape[0])

		for i in range(unique_cluster_labels.shape[0]):
			idx = labels == unique_cluster_labels[i]

			temp_fixed_indices = torch.zeros(idx[idx == 1].shape[0])
			num_images_to_select = int(temp_fixed_indices.shape[0] * supervision_level)
			# temp_fixed_indices[torch.randperm(temp_fixed_indices.shape[0])[:num_images_to_select]] = 1
			temp_fixed_indices[:num_images_to_select] = 1
			idx[idx==1] = temp_fixed_indices.to(torch.bool)

			fixed_indices += idx.to(dtype)
			clustering[idx] = unique_cluster_labels[i]
			cluster_centers[i] = torch.mean(data[idx], 0)
	else:
		assert(supervision_level == 0)
		cluster_centers, dummy_cluster_center_labels = Kmeans_plus_plus_3D(data, labels, num_clusters, distance_metric, seed)

	# Kmeans algorithm
	indices_to_update = (1 - fixed_indices).to(torch.long)
	indices_to_update = indices_to_update == 1
	
	finalvalues = data[indices_to_update]

	old_clustering = clustering.clone()

	temp_distance_arr = distance_metric(finalvalues.view(finalvalues.shape[0], 1, finalvalues.shape[1], finalvalues.shape[2], finalvalues.shape[3]).repeat(1, num_clusters, 1, 1, 1), cluster_centers.repeat(finalvalues.shape[0], 1, 1, 1, 1), 3).to(dtype)
	label_indices = torch.min(temp_distance_arr, 1)[1]	
	clustering[indices_to_update] = torch.tensor([unique_cluster_labels[j] for j in label_indices]).to(dtype)
	
	while not torch.prod((old_clustering == clustering)):
		old_clustering = clustering.clone()
		# Update cluster centers
		for i in range(num_clusters):
			if torch.sum(clustering == unique_cluster_labels[i]) == 0:
				print("cluster", i, ": No point in this cluster!")
				continue

			cluster_centers[i] = torch.mean(data[clustering == unique_cluster_labels[i]], 0)

		# Update clustering
		label_indices = torch.min(distance_metric(finalvalues.view(finalvalues.shape[0], 1, finalvalues.shape[1], finalvalues.shape[2], finalvalues.shape[3]).repeat(1, num_clusters, 1, 1, 1), cluster_centers.repeat(finalvalues.shape[0], 1, 1, 1, 1), 3).to(dtype), 1)[1]
		clustering[indices_to_update] = torch.tensor([unique_cluster_labels[j] for j in label_indices]).to(dtype)


	nmi = NMI(np.array(labels), np.array(clustering))
	ari = ARI(np.array(labels), np.array(clustering))
	acc = ACC(np.array(labels.to(torch.long)), np.array(clustering.to(torch.long)))
	print("NMI : ", nmi)
	print("ARI : ", ari)
	print("ACC : ", acc)

	log_final_results.write("kmeans_metrics_NMI: " + str(nmi) + " \n")
	log_final_results.write("kmeans_metrics_ARI: " + str(ari) + " \n")
	log_final_results.write("kmeans_metrics_ACC: " + str(acc[0]) + " \n")

	return indices_to_update, clustering.to(torch.long), nmi, ari, acc[0]

def Kernel_Kmeans_plus_plus(data, labels, cluster_num, Kernel_distance_array, seed):
	torch.manual_seed(seed)
	dtype = torch.double
	data = data.to(dtype)
	unique_cluster_labels = torch.tensor(np.unique(np.array(labels))).to(dtype)
	cumulative_prob = torch.cumsum(torch.ones(data.shape[0]) / data.shape[0], dim=0)
	cluster_centers = torch.zeros(cluster_num).to(dtype)
	cluster_center_labels = torch.zeros(cluster_num)

	#distance array
	self_distance = (torch.tensor([Kernel_distance_array[i][i] for i in range(data.shape[0])]).to(dtype)).view(data.shape[0], 1).repeat(1, data.shape[0])
	pair_wise_distance = self_distance + torch.transpose(self_distance, 0, 1) - (2*Kernel_distance_array)

	#first center
	index = binarysearch(cumulative_prob, torch.rand(1))
	cluster_centers[0] = index
	cluster_center_labels[0] = labels[index]

	distance_square_array = pair_wise_distance[index][:].clone().to(dtype)	

	#Kmeans++
	for i in range(1, cluster_num):
		#Next center
		cumulative_prob = torch.cumsum(distance_square_array / torch.sum(distance_square_array), dim=0).to(dtype)
		index = binarysearch(cumulative_prob, torch.rand(1).to(dtype))
		cluster_centers[i] = index
		cluster_center_labels[i] = labels[index]

		#update distance matrix
		torch.min(input = distance_square_array, other = pair_wise_distance[index][:].clone().to(dtype), out = distance_square_array)

	clustering = unique_cluster_labels[torch.min(pair_wise_distance[:, cluster_centers.to(torch.long)], 1)[1]]

	assert(clustering.shape[0]==data.shape[0])

	return clustering, cluster_centers, cluster_center_labels


def semisupervised_Kernel_Kmeans(datafile_path, num_clusters, log_final_results, Kernel=RBF_Kernel, sigma=100, supervision_level=0, seed=0):
	dtype = torch.double
	with open(datafile_path, 'rb') as f:
		data, labels = pkl.load(f)

	torch.manual_seed(seed)

	unique_cluster_labels = torch.tensor([i for i in range(num_clusters)]).to(dtype)
	labels = labels.to(dtype)
	num_images = data.shape[0]

	# Precompute pairwise kernel function
	Kernel_distance_array = torch.zeros([num_images, num_images]).to(dtype)
	for i in range(num_images):
		for j in range(num_images):
			Kernel_distance_array[i][j] = Kernel(data[i], data[j], sigma, 2)

	clustering = (torch.zeros(num_images) - 1).to(torch.double)
	fixed_indices = torch.zeros(num_images).to(dtype)
	indices_to_update = (1 - fixed_indices).to(torch.long)
	indices_to_update = indices_to_update == 1

	if supervision_level > 0:
		assert(supervision_level <= 1.0)
		unique_cluster_labels = torch.tensor(np.unique(np.array(labels))).to(dtype)
		assert(num_clusters == unique_cluster_labels.shape[0])

		for i in range(unique_cluster_labels.shape[0]):
			idx = labels == unique_cluster_labels[i]

			temp_fixed_indices = torch.zeros(idx[idx == 1].shape[0])
			num_images_to_select = int(temp_fixed_indices.shape[0] * supervision_level)
			# temp_fixed_indices[torch.randperm(temp_fixed_indices.shape[0])[:num_images_to_select]] = 1
			temp_fixed_indices[:num_images_to_select] = 1
			idx[idx==1] = temp_fixed_indices.to(torch.bool)

			fixed_indices += idx.to(dtype)
			clustering[idx] = unique_cluster_labels[i]

		clustering = clustering.to(dtype)
		old_clustering = clustering.clone()
		indices_to_update = (1 - fixed_indices).to(torch.long)
		indices_to_update = indices_to_update == 1

		for i in range(num_images):
			if clustering[i] != -1:
				assert(indices_to_update[i] == 0)
				continue

			correct_cluster_label = None
			min_distance_i_cluster_r = None
			for r in range(unique_cluster_labels.shape[0]):
				cluster_indices = old_clustering == unique_cluster_labels[r]
				size_of_cluster = torch.sum(cluster_indices).to(dtype)
				distance_i_cluster_r = Kernel_distance_array[i][i] + (-2.00 * torch.sum(Kernel_distance_array[i][cluster_indices]) / size_of_cluster) + (torch.sum(Kernel_distance_array[cluster_indices ,:][:, cluster_indices]) / (size_of_cluster**2))
				if r == 0:
					min_distance_i_cluster_r = distance_i_cluster_r
					correct_cluster_label = unique_cluster_labels[r]
				else:
					if min_distance_i_cluster_r > distance_i_cluster_r:
						min_distance_i_cluster_r = distance_i_cluster_r
						correct_cluster_label = unique_cluster_labels[r]
			clustering[i] = correct_cluster_label
	else:
		assert(supervision_level == 0)
		old_clustering = clustering.clone()
		clustering, _, _ = Kernel_Kmeans_plus_plus(data, labels, num_clusters, Kernel_distance_array, seed)
		clustering = clustering.to(dtype)
		unique_cluster_labels = np.unique(np.array(labels))	

	# Kmeans algorithm
	while not torch.prod((old_clustering == clustering)):
		old_clustering = clustering.clone()

		# Update clustering
		for i in range(num_images):
			if indices_to_update[i] == 0:
				continue

			correct_cluster_label = None
			min_distance_i_cluster_r = None
			for r in range(unique_cluster_labels.shape[0]):
				cluster_indices = old_clustering == unique_cluster_labels[r]
				size_of_cluster = torch.sum(cluster_indices).to(dtype)
				distance_i_cluster_r = Kernel_distance_array[i][i] + (-2.00 * torch.sum(Kernel_distance_array[i][cluster_indices]) / size_of_cluster) + (torch.sum(Kernel_distance_array[cluster_indices ,:][:, cluster_indices]) / (size_of_cluster**2))
				if r == 0:
					min_distance_i_cluster_r = distance_i_cluster_r
					correct_cluster_label = unique_cluster_labels[r]
				else:
					if min_distance_i_cluster_r > distance_i_cluster_r:
						min_distance_i_cluster_r = distance_i_cluster_r
						correct_cluster_label = unique_cluster_labels[r]
			clustering[i] = correct_cluster_label


	nmi = NMI(np.array(labels), np.array(clustering))
	ari = ARI(np.array(labels), np.array(clustering))
	acc = ACC(np.array(labels.to(torch.long)), np.array(clustering.to(torch.long)))
	print("NMI : ", nmi)
	print("ARI : ", ari)
	print("ACC : ", acc)

	log_final_results.write("kernel_kmeans_metrics_NMI: " + str(nmi) + " \n")
	log_final_results.write("kernel_kmeans_metrics_ARI: " + str(ari) + " \n")
	log_final_results.write("kernel_kmeans_metrics_ACC: " + str(acc[0]) + " \n")

	return clustering.to(torch.long), nmi, ari, acc[0]


# ################################################################
# ################################################################

def create_and_store_MNISTdataset_indexfile(pathstr, digits, img_size=64, num_images_per_digit=100, seed=0):
	torch.manual_seed(seed)

	transform = transforms.Compose([
		transforms.Resize(img_size), #Used transforms.Resize() instead of transforms.Scale()
		transforms.ToTensor(),
		transforms.Normalize([0.5], [0.5])
	])

	dataMNIST = datasets.MNIST('data', train=True, download=True, transform=transform)

	# Initializing dataset points
	idx = dataMNIST.targets == digits[0]
	target = dataMNIST.targets[idx]
	data = dataMNIST.data[idx]
	index = (idx.nonzero().reshape(-1))

	for j in range(1, digits.shape[0]):
		idx = dataMNIST.targets == digits[j]
		target = torch.cat((target, dataMNIST.targets[idx]), 0)
		data = torch.cat((data, dataMNIST.data[idx]), 0)
		index = torch.cat((index, (idx.nonzero().reshape(-1))), 0)

	finalindices, finallabels = choose_cluster_image_indices(data, target, index, num_images_per_digit, seed=seed, dimension=2)
	
	# with open(pathstr + '/MNIST_' + str(num_images_per_digit) + '_' + str(unique_index) + '_seed_' + str(seed) + '.pkl','wb') as f:
	# 	pkl.dump((finaldata, finallabels), f)

	with open(pathstr + '/MNIST_indices_numperdigit_' + str(num_images_per_digit) + '_numdigits_' + str(digits.shape[0]) + '_seed_' + str(seed) + '.pkl','wb') as f:
		pkl.dump((finalindices, finallabels), f)

def create_and_store_CIFARdataset_indexfile(pathstr, class_labels, img_size=32, num_images_per_digit=1000, seed=0, datapath=None):
	torch.manual_seed(seed)
	transform = transforms.Compose([
		transforms.Resize(img_size), #Used transforms.Resize() instead of transforms.Scale()
		transforms.ToTensor(),
		transforms.Normalize([0.5], [0.5])
	])

	dataCIFAR = None
	if datapath is None:
		dataCIFAR = datasets.CIFAR10('data', train=True, download=True, transform=transform)
	else:
		dataCIFAR = datasets.CIFAR10(datapath, train=True, download=False, transform=transform)
	
	originaltargets = torch.tensor(dataCIFAR.targets)
	originaldata = torch.tensor(dataCIFAR.data)

	# Initializing dataset points
	idx = originaltargets == class_labels[0]
	target = originaltargets[idx]
	data = originaldata[idx]
	index = (idx.nonzero().reshape(-1))

	for j in range(1, class_labels.shape[0]):
		idx = originaltargets == class_labels[j]
		target = torch.cat((target, originaltargets[idx]), 0)
		data = torch.cat((data, originaldata[idx]), 0)
		index = torch.cat((index, (idx.nonzero().reshape(-1))), 0)

	finalindices, finallabels = choose_cluster_image_indices(data, target, index, num_images_per_digit, seed=seed, dimension=3)
	
	with open(pathstr + '/CIFAR10_indices_numperdigit_' + str(num_images_per_digit) + '_numdigits_' + str(class_labels.shape[0]) + '_seed_' + str(seed) + '.pkl','wb') as f:
		pkl.dump((finalindices, finallabels), f)

def Kmeans_plus_plus_indices(data, labels, indices, cluster_num, sq_distance_function, seed, dimension):
	torch.manual_seed(seed)
	dtype = torch.double
	data = data.to(dtype)
	cumulative_prob = torch.cumsum(torch.ones(data.shape[0]) / data.shape[0], dim=0)
	cluster_center_labels = torch.zeros(cluster_num)
	cluster_center_indices = torch.zeros(cluster_num)


	#first center
	index = binarysearch(cumulative_prob, torch.rand(1))
	cluster_center_labels[0] = labels[index]
	cluster_center_indices[0] = indices[index]

	if dimension == 2:
		distance_square_array = sq_distance_function(data.to(dtype), (data[index].to(dtype)).repeat(data.shape[0], 1, 1), 2).to(dtype)
	elif dimension == 3:
		distance_square_array = sq_distance_function(data.to(dtype), (data[index].to(dtype)).repeat(data.shape[0], 1, 1, 1), 3).to(dtype)
	else:
		assert(1==0)	
	
	#Kmeans++
	for i in range(1, cluster_num):
		#Next center
		cumulative_prob = torch.cumsum(distance_square_array / sum(distance_square_array), dim=0).to(dtype)
		index = binarysearch(cumulative_prob, torch.rand(1).to(dtype))
		cluster_center_labels[i] = labels[index]
		cluster_center_indices[i] = indices[index]

		#update distance matrix
		if dimension == 2:
			torch.min(input = distance_square_array, other = sq_distance_function(data, (data[index].to(dtype)).repeat(data.shape[0], 1, 1), 2).to(dtype), out = distance_square_array)
		elif dimension == 3:
			torch.min(input = distance_square_array, other = sq_distance_function(data, (data[index].to(dtype)).repeat(data.shape[0], 1, 1, 1), 3).to(dtype), out = distance_square_array)
		else:
			assert(1==0)

	return cluster_center_indices,  cluster_center_labels


def choose_cluster_image_indices(data, labels, indices, number, distance_metric=Euclidean_squared_distance, seed=0, dimension=2):
	unique_labels = np.unique(np.array(labels))
	assert(unique_labels.shape[0] >= 1)

	label_number_map = {}
	if isinstance(number, type(1)):
		# Same number of images for all the unique labels
		for unq_label in unique_labels:
			label_number_map[unq_label] = number
	else:
		# Else the number is dict object with mapping from a unique label to num of images for that label
		assert(isinstance(number, type({})))
		label_number_map = number

	idx = labels == unique_labels[0]
	finalindices, finallabels = Kmeans_plus_plus_indices(data[idx], labels[idx], indices[idx], label_number_map[unique_labels[0]], distance_metric, seed, dimension)

	for i in range(1, unique_labels.shape[0]):
		idx = labels == unique_labels[i]
		tmp_indices, tmp_labels = Kmeans_plus_plus_indices(data[idx], labels[idx], indices[idx], label_number_map[unique_labels[i]], distance_metric, seed, dimension)
		finalindices = torch.cat((finalindices, tmp_indices), 0)
		finallabels = torch.cat((finallabels, tmp_labels), 0)

	return finalindices, finallabels

def verify_index_file(index_file_path, data_file_path, dataset_path=None, dataset_name='MNIST', img_size=32):
	transform = transforms.Compose([
		transforms.Resize(img_size), #Used transforms.Resize() instead of transforms.Scale()
		transforms.ToTensor(),
		transforms.Normalize([0.5], [0.5])
	])

	dtype = torch.double

	dataset = None
	originaldata = None

	with open(data_file_path, 'rb') as f:
		imagedata, _ = pkl.load(f)

	with open(index_file_path, 'rb') as f:
		imageindices, _ = pkl.load(f)

	if dataset_name == 'MNIST':
		if dataset_path is None:
			dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
		else:
			dataset = datasets.MNIST(dataset_path, train=True, download=False, transform=transform)
		originaldata = dataset.data
	elif dataset_name == 'CIFAR10':
		if dataset_path is None:
			dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
		else:
			dataset = datasets.CIFAR10(dataset_path, train=True, download=False, transform=transform)
		originaldata = torch.tensor(dataset.data)	
	elif dataset_name == 'CelebA':
		with open(dataset_path, 'rb') as f:
			allimagedata, (allindices, alltarget) = pkl.load(f)
		allindices = allindices.to(torch.long)
		imageindices = imageindices.to(torch.long)
		# localindices = torch.sum(allindices.repeat(imageindices.shape[0], 1).transpose(1, 0) == imageindices.repeat(allindices.shape[0], 1), 1).to(torch.bool)
		localindices = ((allindices.repeat(imageindices.shape[0], 1).transpose(1, 0) == imageindices.repeat(allindices.shape[0], 1)).to(torch.long)).transpose(1, 0).nonzero()[:, 1]
		originaldata = allimagedata[localindices]
	else:
		assert(1==0)
		
	imagedataclone = None
	if dataset_name == 'CelebA':
		imagedataclone = originaldata
	else:
		imagedataclone = originaldata[imageindices.to(torch.long)]
	
	if torch.sum((imagedataclone.to(dtype) - imagedata.to(dtype))**2) == 0:
		print('verified!!!')
		return 1
	else:
		print('Not equal!!!')
		return 0

# if __name__ == '__main__':
# 	seed = args.seed
# 	filepathlocation = args.filepathlocation
# 	use_cuda = args.use_cuda
# 	shuffle = args.shuffle
# 	datafile_path = args.datafilepath
# 	gpu = args.gpu
# 	given_num_clusters = args.num_clusters

# 	log_final_results = open("logfile_MNIST_Kmeans_" + str(given_num_clusters), "a+")
# 	log_final_results.write("filepathlocation: " + str(filepathlocation) + " \n")
# 	log_final_results.write("use_cuda: " + str(use_cuda) + " \n")
# 	log_final_results.write("shuffle: " + str(shuffle) + " \n")
# 	log_final_results.write("datafile_path: " + str(datafile_path) + " \n")
# 	log_final_results.write("num_clusters: " + str(given_num_clusters) + " \n")
# 	log_final_results.write("seed: " + str(seed) + " \n")

# 	print("filepathlocation: ", filepathlocation)
# 	print("use_cuda", use_cuda)
# 	print("datafilepath:", datafile_path)
# 	print("num_clusters: ", given_num_clusters)
# 	print("seed: ", seed)

# 	torch.manual_seed(seed)
# 	torch.set_default_dtype(torch.double)
# 	dtype = torch.double

# 	if use_cuda:
# 		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 		print('Using device:', device)
# 		print()
# 		torch.cuda.set_device(gpu)

# 	print("Yann Lecun dataset")

# 	dtype = torch.double
# 	with open(datafile_path, 'rb') as f:
# 		data, labels = pkl.load(f)

# 	unique_cluster_labels = np.unique(np.array(labels))
# 	num_clusters = unique_cluster_labels.shape[0]
# 	assert(num_clusters == given_num_clusters)
# 	supervision_array = np.array([0.0, 0.1 ,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])

# 	kmeans_metrics_ACC = np.zeros(len(supervision_array)).astype(float)
# 	kmeans_metrics_ARI = np.zeros(len(supervision_array)).astype(float)
# 	kmeans_metrics_NMI = np.zeros(len(supervision_array)).astype(float)

# 	for index in range(len(supervision_array)):
# 		indices_to_update, kmeans_clustering, kmeans_metrics_NMI[index], kmeans_metrics_ARI[index], kmeans_metrics_ACC[index] = semisupervised_Kmeans(datafile_path, Euclidean_squared_distance, num_clusters, log_final_results, supervision_array[index], seed)
# 		fixed_indices = (1 - indices_to_update.to(torch.long)).to(torch.long)
# 		fixed_indices = fixed_indices == 1
# 		log_final_results.write("Supervision: " + str(supervision_array[index]) + " \n")
# 		log_final_results.write("--------------------------------------------- \n")
# 		current_time_stamp = time.strftime('%H:%M:%S %d-%m-%Y', time.localtime(time.time()))

# 		preprocessed_data = {}
# 		preprocessed_data['data'] = data
# 		preprocessed_data['actual_labels'] = labels
# 		preprocessed_data['kmeans_clustering'] = kmeans_clustering
# 		preprocessed_data['num_clusters'] = num_clusters
# 		preprocessed_data['supervision_level'] = supervision_array[index]
# 		preprocessed_data['kmeans_accuracy'] = kmeans_metrics_ACC[index]
# 		preprocessed_data['fixed_indices'] = fixed_indices
# 		preprocessed_data['time_stamp'] = current_time_stamp
# 		preprocessed_data['unique_cluster_labels'] = unique_cluster_labels

# 		savefilepathlocation = filepathlocation + "/" +"kmeans_clustering_data" + "_" + str(num_clusters) + "_" + str(supervision_array[index]) + "_seed_" + str(seed) + ".pkl" 
# 		with open(savefilepathlocation,'wb') as f:
# 			pkl.dump(preprocessed_data, f)


# 	print(kmeans_metrics_NMI)
# 	print(kmeans_metrics_ARI)
# 	print(kmeans_metrics_ACC)
# 	print("---------------------------------------------")

# 	save_metrics_location = filepathlocation + "/" +"kmeans_clustering_metrics" + "_" + str(num_clusters) + "_seed_" + str(seed) + ".pkl" 
# 	metrics = {}
# 	metrics['supervision_array'] = supervision_array
# 	metrics['kmeans_metrics_ACC'] = kmeans_metrics_ACC
# 	metrics['kmeans_metrics_NMI'] = kmeans_metrics_NMI
# 	metrics['kmeans_metrics_ARI'] = kmeans_metrics_ARI

# 	with open(save_metrics_location,'wb') as f1:
# 		pkl.dump(metrics, f1)

# 	log_final_results.write("--------------------------------------------- \n")

'''
save_image(transform(transform2(ten_img_class_0_1.to(torch.uint8).clone())), "./ten_img_class_0.jpeg", nrow=1, normalize=True)
ten_img_class_0_1 = torch.tensor(img_class_0).clone()
img_class_0 = cv2.imread(inputdir + "/train/10_left.jpeg", 1)
ten_img_class_0 = cv2.cvtColor(img_class_0, cv2.COLOR_BGR2RGB)
ten_img_class_0_1 = torch.tensor(ten_img_class_0).clone()
save_image(transform(transform2(ten_img_class_0_1.to(torch.uint8).clone())), "./ten_img_class_0.jpeg", nrow=1, normalize=True)
ten_img_class_0_1 = torch.tensor(ten_img_class_0).clone().transpose(2, 1).transpose(1, 0)
save_image(transform(transform2(ten_img_class_0_1.to(torch.uint8).clone())), "./ten_img_class_0.jpeg", nrow=1, normalize=True)
cv2.COLOR_BGR2RGB --> tensor --> to(torch.uint8) --> clone() --> CustomMNISTdigit(*, *, transform)
--> save_image(*, *, *, normalize=True)
^^ No need to transpose(2, 1).transpose(1, 0) for a single img or transpose(3, 2).transpose(2, 1)
Need to transpose if saving directly or to fixed data then 
transform(transform2(img.transpose().transpose().to(torch.uint8).clone())) 
transform1 = transforms.Compose([
    transforms.CenterCrop((1600, 2400)),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
'''


if __name__ == '__main__':
	seed = args.seed
	filepathlocation = args.filepathlocation
	use_cuda = args.use_cuda
	shuffle = args.shuffle
	datafile_path = args.datafilepath
	gpu = args.gpu
	given_num_clusters = args.num_clusters

	# change
	log_final_results = open("logfile_Diabetic_Retinopathy_zero_idx_Kmeans_" + str(given_num_clusters), "a+")
	# log_final_results = open("noisy_all_chan_sigma_0.2_logfile_CelebA_Kmeans_" + str(given_num_clusters), "a+")
	# log_final_results = open("logfile_CelebA_Kmeans_" + str(given_num_clusters), "a+")
	
	log_final_results.write("filepathlocation: " + str(filepathlocation) + " \n")
	log_final_results.write("use_cuda: " + str(use_cuda) + " \n")
	log_final_results.write("shuffle: " + str(shuffle) + " \n")
	log_final_results.write("datafile_path: " + str(datafile_path) + " \n")
	log_final_results.write("num_clusters: " + str(given_num_clusters) + " \n")
	log_final_results.write("seed: " + str(seed) + " \n")

	print("filepathlocation: ", filepathlocation)
	print("use_cuda", use_cuda)
	print("datafilepath:", datafile_path)
	print("num_clusters: ", given_num_clusters)
	print("seed: ", seed)

	torch.manual_seed(seed)
	torch.set_default_dtype(torch.double)
	dtype = torch.double

	if use_cuda:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		print('Using device:', device)
		print()
		torch.cuda.set_device(gpu)

	# Change
	print("Zero indexed Diabetic Retinopathy dataset")

	dtype = torch.double
	with open(datafile_path, 'rb') as f:
		data, labels = pkl.load(f)

	unique_cluster_labels = np.unique(np.array(labels))
	num_clusters = unique_cluster_labels.shape[0]
	assert(num_clusters == given_num_clusters)
	supervision_array = np.array([0.0, 0.1 ,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])

	kmeans_metrics_ACC = np.zeros(len(supervision_array)).astype(float)
	kmeans_metrics_ARI = np.zeros(len(supervision_array)).astype(float)
	kmeans_metrics_NMI = np.zeros(len(supervision_array)).astype(float)

	for index in range(len(supervision_array)):
		indices_to_update, kmeans_clustering, kmeans_metrics_NMI[index], kmeans_metrics_ARI[index], kmeans_metrics_ACC[index] = semisupervised_Kmeans_3D(datafile_path, Euclidean_squared_distance, num_clusters, log_final_results, supervision_array[index], seed)
		fixed_indices = (1 - indices_to_update.to(torch.long)).to(torch.long)
		fixed_indices = fixed_indices == 1
		log_final_results.write("Supervision: " + str(supervision_array[index]) + " \n")
		log_final_results.write("--------------------------------------------- \n")
		current_time_stamp = time.strftime('%H:%M:%S %d-%m-%Y', time.localtime(time.time()))

		preprocessed_data = {}
		preprocessed_data['data'] = data
		preprocessed_data['actual_labels'] = labels
		preprocessed_data['kmeans_clustering'] = kmeans_clustering
		preprocessed_data['num_clusters'] = num_clusters
		preprocessed_data['supervision_level'] = supervision_array[index]
		preprocessed_data['kmeans_accuracy'] = kmeans_metrics_ACC[index]
		preprocessed_data['fixed_indices'] = fixed_indices
		preprocessed_data['time_stamp'] = current_time_stamp
		preprocessed_data['unique_cluster_labels'] = unique_cluster_labels

		# change
		savefilepathlocation = filepathlocation + "/" +"Diabetic_Retinopathy_zero_idx_kmeans_clustering_data" + "_" + str(num_clusters) + "_" + str(supervision_array[index]) + "_seed_" + str(seed) + ".pkl" 
		with open(savefilepathlocation,'wb') as f:
			pkl.dump(preprocessed_data, f)


	print(kmeans_metrics_NMI)
	print(kmeans_metrics_ARI)
	print(kmeans_metrics_ACC)
	print("---------------------------------------------")

	# change
	save_metrics_location = filepathlocation + "/" +"Diabetic_Retinopathy_zero_idx_kmeans_clustering_metrics" + "_" + str(num_clusters) + "_seed_" + str(seed) + ".pkl" 
	metrics = {}
	metrics['supervision_array'] = supervision_array
	metrics['kmeans_metrics_ACC'] = kmeans_metrics_ACC
	metrics['kmeans_metrics_NMI'] = kmeans_metrics_NMI
	metrics['kmeans_metrics_ARI'] = kmeans_metrics_ARI

	with open(save_metrics_location,'wb') as f1:
		pkl.dump(metrics, f1)

	log_final_results.write("--------------------------------------------- \n")
	