from __future__ import print_function
import os, time
import itertools
import pickle
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as mplot
from torchvision.utils import make_grid
from torchvision.utils import save_image
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs

def normal_init(m, mean, std):
	if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
		m.weight.data.normal_(mean, std)
		m.bias.data.zero_()

log_norm_constant = -0.5 * np.log(2 * np.pi)
def log_gaussian(x, mean, sigma=1):
	a = (x - mean)**2
	log_p = (-0.5*a)/(sigma**2)
	log_p = log_p + log_norm_constant-math.log(sigma)
	return log_p

def binarysearch(array,  val):
	start = 0
	end = array.shape[0] - 1
	mid = math.floor((start / 2.0) + (end / 2.0))
	old_mid = -1

	while(mid  != old_mid):
		if val == array[mid]:
			break
		elif val > array[mid]:
			start = mid
			old_mid = mid
			mid = math.floor((start / 2.0) + (end / 2.0))
		else:
			end = mid
			old_mid = mid
			mid = math.floor((start / 2.0) + (end / 2.0))

	return mid

def show_gen_data(model_gen, batch_size=1, filename="First", num_gen_batches_to_show=5, nc=1):
	imagelst=torch.zeros([num_gen_batches_to_show, nc, 64, 64])
	for k in range(len(model_gen)):
		model_gen[k].eval()
		for i in range(num_gen_batches_to_show):
			z_ = torch.randn((batch_size, 100)).view(batch_size, 100, 1, 1).cuda()
			zz = model_gen[k](z_)
			zz=  zz.data
			zz = zz.cpu().numpy()
			zz = np.reshape(zz,[batch_size, nc, 64, 64])
			ll=zz[0]
			ll = torch.from_numpy(ll)
			ll = ll.squeeze()
			imagelst[i, :, :, :] = ll
		model_gen[k].train()
		loc = "./Results_CIFAR10/"+filename+"_"+str(k)+"_Cluster.png"    
		save_image(imagelst, loc, nrow=5)

# x - imgsize X imgsize
# samples -  K X samplespercluster X imgsize X imgsize
# prior is P(z| theta) - K X 1
# discriminato_probability - K X 1
def ExpectationMNIST_KNN(x, samples, discriminator_probability, prior, sigma=2, dtype=torch.float32, knn=4):
	torch.set_default_dtype(dtype)
	K = len(prior)
	total = 0
	total1 = 0
	posterior = torch.zeros([K])
	logProb = torch.zeros([K])
	epsilon = 1e-45

	totlist = torch.zeros([samples.shape[0], samples.shape[1]])

	mean = (samples.view(samples.shape[0], samples.shape[1], samples.shape[2]*samples.shape[3])).cuda()
	xnew = (x.view(-1)).repeat(samples.shape[0], samples.shape[1], 1)

	# totlist is K X samplespercluster
	totlist = (torch.sum(log_gaussian(xnew, mean, sigma), 2)).clone()

	if sum(sum(torch.isnan(totlist))) == 1:
		print("----------")
		print(totlist)
		print(torch.isnan(totlist))
		print(xnew)
		print(sigma)
		print("----------")
		assert 1==0

	############################new#################################
	totlist = (torch.sort(totlist, 1, True))[0].clone()
	totlist = totlist[:, :knn].clone()

	# To prevent inf or nan values due to values originally in totlist
	# Can still occur if prior or discriminator probability is very low or zero
	max_totlist_per_row = torch.max(totlist, 1)[0].detach()
	newtotlist = totlist - max_totlist_per_row.view(totlist.shape[0], 1)
	# newtotlist2 = (torch.mean(torch.exp(newtotlist), 1)*discriminator_probability*prior)
	# logProb = torch.log(newtotlist2) + max_totlist_per_row.detach()
	newtotlist2 = (torch.mean(torch.exp(newtotlist), 1)*discriminator_probability)

	if 0.0 in prior:
		prior = prior + epsilon
		prior = prior / torch.sum(prior)
	
	logProb = torch.log(newtotlist2) + max_totlist_per_row.detach() + torch.log(prior)

	# Now compute Posterior using logProb
	logProb2 = logProb.detach()
	posterior = F.softmax(logProb2, -1)
	############################new#################################
		
	############################old################################
	# # Subtracting the minimum value to prevent negative overflow
	# # in computation
	# acc = (torch.min(totlist)).clone().detach()
	# totlist = totlist - acc
	# totlist = totlist.exp()

	# #select knn along dimension 1 (samplespercluster)
	# totlist = (torch.sort(totlist, 1, True))[0].clone()
	# totlist = totlist[:, :knn].clone()
	
	# # torch.mean(totlist, 1) is a tensor of size K X 1 where in second dimension
	# # we have mean of guassian probablities of generated knn in that cluster 
	# # totlist becomes P(xi | zi = k, theta)*P(zi = k | theta)*(e^(-acc)) -> P(xi, zi = k | theta)*(e^(-acc))
	# totlist = ((torch.mean(totlist, 1))*discriminator_probability*prior)
	# # totlist = ((torch.mean(totlist, 1))*prior)

	# # log(P(xi, zi = k| theta))-acc + acc, here z=k
	# logProb = (torch.log(totlist)+acc).clone()

	# # P(xi, zi = k | theta_n)*(e^(-acc))
	# posterior =  totlist.detach()

	# # P(xi | theta_n)*(e^(-acc))  
	# total = sum(posterior)

	# if total == 0:
	# 	print(posterior)
	# 	print(logProb)
	# 	assert 1==0
	
	# # P(zi = k | xi, theta) = P(xi, zi = k | theta_n)*e^(-acc) / (P(xi | theta_n)*e^(-acc))
	# # Calculating gamma_i_k
	# posterior = posterior/total
	############################old################################ 

	return logProb.clone(), posterior


# x - 3 X imgsize X imgsize
# samples -  K X samplespercluster 3 X imgsize X imgsize
# prior is P(z| theta) - K X 1
# discriminato_probability - K X 1
def Expectation3D_KNN(x, samples, discriminator_probability, prior, sigma=2, dtype=torch.float32, knn=4):
	torch.set_default_dtype(dtype)
	K = len(prior)
	total = 0
	total1 = 0
	posterior = torch.zeros([K])
	logProb = torch.zeros([K])

	totlist = torch.zeros([samples.shape[0], samples.shape[1]])

	mean = (samples.view(samples.shape[0], samples.shape[1], samples.shape[2]*samples.shape[3]*samples.shape[4])).cuda()
	xnew = (x.view(-1)).repeat(samples.shape[0], samples.shape[1], 1)

	# totlist is K X samplespercluster
	totlist = (torch.sum(log_gaussian(xnew, mean, sigma), 2)).clone()

	if sum(sum(torch.isnan(totlist))) == 1:
		print("----------")
		print(totlist)
		print(torch.isnan(totlist))
		print(xnew)
		print(sigma)
		print("----------")
		assert 1==0


	############################new#################################
	totlist = (torch.sort(totlist, 1, True))[0].clone()
	totlist = totlist[:, :knn].clone()

	# To prevent inf or nan values due to values originally in totlist
	# Can still occur if prior or discriminator probability is very low or zero
	max_totlist_per_row = torch.max(totlist, 1)[0].detach()
	newtotlist = totlist - max_totlist_per_row.view(totlist.shape[0], 1)
	# newtotlist2 = (torch.mean(torch.exp(newtotlist), 1)*discriminator_probability*prior)
	# logProb = torch.log(newtotlist2) + max_totlist_per_row.detach()
	newtotlist2 = (torch.mean(torch.exp(newtotlist), 1)*discriminator_probability)
	logProb = torch.log(newtotlist2) + max_totlist_per_row.detach() + torch.log(prior)

	# Now compute Posterior using logProb
	logProb2 = logProb.detach()
	posterior = F.softmax(logProb2, -1)
	############################new#################################
	
	############################old################################
	# # Subtracting the minimum value to prevent negative overflow
	# # in computation
	# acc = (torch.min(totlist)).clone().detach()
	# # Subtracting the average value to prevent negative or positive overflow
	# acc = ((torch.min(totlist)).clone().detach() + (torch.max(totlist)).clone().detach()) / 2.00
	# totlist = totlist - acc

	# # select knn along dimension 1 (samplespercluster)
	# totlist = (torch.sort(totlist, 1, True))[0].clone()
	# totlist = totlist[:, :knn].clone()

	# totlist = totlist.exp()
	
	# # torch.mean(totlist, 1) is a tensor of size K X 1 where in second dimension
	# # we have mean of guassian probablities of generated knn in that cluster 
	# # totlist becomes P(xi | zi = k, theta)*P(zi = k | theta)*(e^(-acc)) -> P(xi, zi = k | theta)*(e^(-acc))
	# totlist = ((torch.mean(totlist, 1))*discriminator_probability*prior)

	# # log(P(xi, zi = k, theta))-acc + acc, here z=k
	# logProb = (torch.log(totlist)+acc).clone()
	
	# # P(xi, zi = k, theta_n)*(e^(-acc))
	# posterior =  totlist.detach()

	# # P(xi, theta_n)*(e^(-acc))  
	# total = sum(posterior)

	# if total == 0:
	# 	print(posterior)
	# 	print(logProb)
	# 	assert 1==0
	
	# # P(zi = k | xi, theta) = P(xi, zi = k, theta_n)*e^(-acc) / (P(xi, theta_n)*e^(-acc))
	# # Calculating gamma_i_k
	# posterior = posterior/total 
	############################old################################
	return logProb.clone(), posterior


# Calls ExpectationMNIST_KNN for each image in X
# Samples  - clusterno X samplespercluster X img_size X img_size
# X - B X img_size X img_size 
# Discriminator_probability - B X K
# Posterior -> B X K -> P(zi | xi, theta_n)
# logProb -> B X K -> log(P(xi, zi | theta))
# torch.set_default_dtype(dtype)
def ExpectationBatchMNIST_KNN(X, Samples, Discriminator_probability, Prior, sigma=2, dtype=torch.float32, use_cuda = False, knn=4):
	B = X.size()[0]
	if use_cuda:
		Posterior = torch.zeros([B, len(Prior)])
		logProb = torch.zeros([B, len(Prior)])
	else:
		Posterior = torch.zeros([B, len(Prior)]).cuda()
		logProb = torch.zeros([B, len(Prior)]).cuda()
	for i in range(B):
		logProb[i, :], Posterior[i, :] = ExpectationMNIST_KNN(X[i], Samples, Discriminator_probability[i, :], Prior, sigma, knn=knn)

	return logProb.clone(), Posterior

# Calls Expectation3D_KNN for each image in X
# Samples  - clusterno X samplespercluster X 3 X img_size X img_size
# X - B X 3 X img_size X img_size 
# Discriminator_probability - B X K
# Posterior -> B X K -> P(zi | xi, theta_n)
# logProb -> B X K -> log(P(xi, zi | theta))
# torch.set_default_dtype(dtype)
def ExpectationBatch3D_KNN(X, Samples, Discriminator_probability, Prior, sigma=2, dtype=torch.float32, use_cuda = False, knn=4):
	B = X.size()[0]
	if use_cuda:
		Posterior = torch.zeros([B, len(Prior)])
		logProb = torch.zeros([B, len(Prior)])
	else:
		Posterior = torch.zeros([B, len(Prior)]).cuda()
		logProb = torch.zeros([B, len(Prior)]).cuda()
	for i in range(B):
		logProb[i, :], Posterior[i, :] = Expectation3D_KNN(X[i], Samples, Discriminator_probability[i, :], Prior, sigma, knn=knn)

	return logProb.clone(), Posterior