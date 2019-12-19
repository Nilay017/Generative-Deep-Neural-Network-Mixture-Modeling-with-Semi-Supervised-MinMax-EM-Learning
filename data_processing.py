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
import pickle as pkl
from cleaned_Functions import *
from myMetrics import *
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as mplot
from scipy.stats import multivariate_normal
from CustomDatasets import *
from tqdm import tqdm


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
"""

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

	idx = labels == unique_labels[0]
	finaldata, finallabels = Kmeans_plus_plus_3D(data[idx], labels[idx], number, distance_metric, seed)
	print("done")

	for i in range(1, unique_labels.shape[0]):
		idx = labels == unique_labels[i]
		tmp_data, tmp_labels = Kmeans_plus_plus_3D(data[idx], labels[idx], number, distance_metric, seed)
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


if __name__ == '__main__':
	seed = args.seed
	filepathlocation = args.filepathlocation
	use_cuda = args.use_cuda
	shuffle = args.shuffle
	datafile_path = args.datafilepath
	gpu = args.gpu
	given_num_clusters = args.num_clusters

	log_final_results = open("logfile_CIFAR10_Kmeans_" + str(given_num_clusters), "a+")
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

	print("CIFAR10 dataset")

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

		savefilepathlocation = filepathlocation + "/" +"CIFAR10_kmeans_clustering_data" + "_" + str(num_clusters) + "_" + str(supervision_array[index]) + "_seed_" + str(seed) + ".pkl" 
		with open(savefilepathlocation,'wb') as f:
			pkl.dump(preprocessed_data, f)


	print(kmeans_metrics_NMI)
	print(kmeans_metrics_ARI)
	print(kmeans_metrics_ACC)
	print("---------------------------------------------")

	save_metrics_location = filepathlocation + "/" +"CIFAR10_kmeans_clustering_metrics" + "_" + str(num_clusters) + "_seed_" + str(seed) + ".pkl" 
	metrics = {}
	metrics['supervision_array'] = supervision_array
	metrics['kmeans_metrics_ACC'] = kmeans_metrics_ACC
	metrics['kmeans_metrics_NMI'] = kmeans_metrics_NMI
	metrics['kmeans_metrics_ARI'] = kmeans_metrics_ARI

	with open(save_metrics_location,'wb') as f1:
		pkl.dump(metrics, f1)

	log_final_results.write("--------------------------------------------- \n")
	