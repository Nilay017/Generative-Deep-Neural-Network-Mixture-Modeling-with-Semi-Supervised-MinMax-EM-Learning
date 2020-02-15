from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np
    import math
    import time
    import datetime
    import cv2

    import matplotlib
    import matplotlib.pyplot as plt

    import pandas as pd
    import pickle as pkl
    
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    
    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    
    from itertools import chain as ichain

    from clusgan.definitions import DATASETS_DIR, RUNS_DIR
    from clusgan.models import Generator_CNN, Encoder_CNN, Discriminator_CNN
    from clusgan.utils import save_model, calc_gradient_penalty, sample_z, cross_entropy
    from clusgan.datasets import get_dataloader, dataset_list
    from clusgan.plots import plot_train_loss
    from myMetrics import *
    from CustomDatasets import *

except ImportError as e:
    print(e)
    raise ImportError

parser = argparse.ArgumentParser(description="My Training Script")
#Change
parser.add_argument("-model_loadpath", "--model_loadpath", dest="model_loadpath", default="", type=str, help="Directory of model to be evaluated")
parser.add_argument("-w", "--wass_metric", dest="wass_metric", action='store_true', help="Flag for Wasserstein metric")
parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=500, type=int, help="Number of epochs the model was trained on")
parser.add_argument("-noise", "--noise", dest="noise", default="None", type=str, help="Type of reproducable noise to be added to images")
parser.add_argument("-g", "--gpu", dest="gpu", default=0, type=int, help="GPU id to use")
parser.add_argument("-k", "--num_workers", dest="num_workers", default=1, type=int, help="Number of dataset workers")
parser.add_argument("-sup", "--supervision_level", dest="supervision_level", default=0.2, type=float, help="supervision_level")
parser.add_argument("-seed", "--seed", dest="seed", default=0.0, type=float, help="seed for experiments")
parser.add_argument("-num_clusters", "--num_clusters", dest="num_clusters", default=3, type=int, help="Number of clusters data is clustered into")
args = parser.parse_args()

#Temporary center function
def get_centers(num_clusters, latent_dim, sigma, dtype=torch.float32):
    centers = torch.zeros([num_clusters, latent_dim]).to(dtype)

    # Inter center distance is now 5 sigma
    radius = float(5.00*float(sigma) / (math.sqrt(2.00)))
    for i in range(num_clusters):
        centers[i][i] = radius
    
    return centers

def get_model_path(n_epochs, noise, supervision_level, seed, num_clusters, mtype):
    model_loadpath = ""
    model_dir_prefix = "training_supervision"

    if noise != "None":
        model_dir_prefix = "dec_noisy_" + model_dir_prefix
    else:
        model_dir_prefix =  "dec_" + model_dir_prefix

    if supervision_level == 0.99:
        model_dir_prefix += "_" + str(supervision_level) + '0000'
    else:
        model_dir_prefix += "_" + str(supervision_level) + '00000'

    model_dir_prefix += "__epoch_" + str(n_epochs) + "____" + str(mtype)\
     + "_____seed_" + str(int(seed)) 

    if noise != "None":
        model_dir_prefix += "__noisy_cifar10_train_clusgan_" + str(num_clusters) + "_all"
    else:
        model_dir_prefix += "__cifar10_train_clusgan_" + str(num_clusters) + "_all"

    model_loadpath = "./runs/cifar10/" + model_dir_prefix

    return model_loadpath

def get_random_state_preserving_reproducible_noise(shape, seed=0.0, mean=0.0, sigma=10.0):
	random_num_gen_state = np.random.get_state()
	np.random.seed(seed)
	random_noise = torch.tensor(np.random.normal(mean, sigma, shape))
	np.random.set_state(random_num_gen_state)
	return random_noise


def get_cv2_random_noise(shape, seed=0, mean=0.0, sigma=10.0):
	cv2.setRNGSeed(int(seed))
	dtype = torch.double
	noise = np.zeros(shape).astype('double')
	batch_size = shape[0]
	########
	# Imp!!!!
	# Note: cv2.randn(x, mean, sigma) only fills first 2 dimensions of x with noise and not the third
	# Hence, here were adding noise only in first 2 dimensions of the RGB image
	# Ideally loop over the first dimension of RGB image and add noise to the rest of the 2 dimensions
	#######
	for batch_no in range(batch_size):
		cv2.randn(noise[batch_no], mean, sigma)
	return torch.tensor(noise).to(dtype)


def run_main(seed):
    # python3 cifar10_eval_em_clusgan.py -n 600 -noise "Other" -g 1 -sup 0.0 -seed 0.0 -num_clusters 5
    global args
    model_loadpath = args.model_loadpath
    device_id = args.gpu
    num_workers = args.num_workers
    wass_metric = args.wass_metric
    n_epochs = args.n_epochs
    noise = args.noise
    supervision_level = args.supervision_level
    num_clusters  = args.num_clusters

    final_accuracy = None
    final_NMI = None
    final_ARI = None

    torch.manual_seed(seed)
    dtype = torch.float32

    torch.set_printoptions(threshold=5000)

    img_size = 32
    channels = 3
   
    # Latent space info
    # Change
    datafile_path = "./CIFAR10_Data/CIFAR10_indices_numperdigit_1000_numdigits_" + str(num_clusters) + "_seed_" + str(seed) + ".pkl"
    with open(datafile_path, 'rb') as f:    
        finalindices, _ = pkl.load(f)

    unique_cluster_labels = np.array(range(num_clusters))

    latent_dim = 30
    sigma = float(5.00)
    cluster_centers = get_centers(num_clusters, latent_dim, sigma, dtype)
    print(cluster_centers)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])


    dataCIFAR = datasets.CIFAR10('/home/nilay/GANMM-master/data', train=True, download=False, transform=transform)
    originaltargets = torch.tensor(dataCIFAR.targets)
    originaldata = torch.tensor(dataCIFAR.data)

    finalindices = finalindices.to(torch.long)

    eval_index = torch.ones(originaldata.shape[0]).to(torch.long)
    eval_index[finalindices] = 0
    eval_index = (eval_index == 1).nonzero().reshape(-1).to(torch.long)
    
    eval_data_all = originaldata[eval_index].clone()
    eval_targets_all = originaltargets[eval_index].clone()

    # Extracting relevant data from eval_data_all
    
    # Change
    digits = np.array(range(num_clusters))
    
    idx = eval_targets_all == digits[0]
    eval_target = eval_targets_all[idx]
    eval_data = eval_data_all[idx]

    for j in range(1, digits.shape[0]):
        idx = eval_targets_all == digits[j]
        eval_target = torch.cat((eval_target, eval_targets_all[idx]), 0)
        eval_data = torch.cat((eval_data, eval_data_all[idx]), 0)

    batch_size = eval_data.shape[0]
    num_images = eval_data.shape[0]

    eval_data = eval_data.to(torch.double)

    if noise == "cv2":
        eval_data = eval_data.clone() + get_cv2_random_noise(eval_data.shape, seed, mean=0.0, sigma=10.0)
    elif noise == "numpy":
        eval_data = eval_data.clone() + get_random_state_preserving_reproducible_noise(eval_data.shape, seed, mean=0.0, sigma=10.0)
    else:
        assert((noise == "None") or (noise == "Other"))

    eval_data = eval_data.to(torch.uint8)

    dataCIFAR10_eval = CustomMNISTdigit(eval_data, eval_target, transform)
    dataloader_eval = DataLoader(dataCIFAR10_eval, num_workers=num_workers,  batch_size=batch_size,  shuffle=True)

    # Wasserstein metric flag
    mtype = 'van'
    if (wass_metric):
        mtype = 'wass'
    
    
    x_shape = (channels, img_size, img_size)
    
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device_id)

    #Change
    if model_loadpath == "":
        model_loadpath = get_model_path(n_epochs, noise, supervision_level, seed, num_clusters, mtype)

    model_loadpath += "/models/"
    
    print(model_loadpath)
    print("!!!!!!!!!!!!!!!!")

    # Initialize generator and discriminator
    generator = Generator_CNN(latent_dim, x_shape)
    encoder = Encoder_CNN(latent_dim, channels=channels, x_shape=x_shape)
    discriminator = Discriminator_CNN(wass_metric=wass_metric, channels=channels, x_shape=x_shape)

    generator.load_state_dict(torch.load(model_loadpath + 'generator.pth.tar'))
    encoder.load_state_dict(torch.load(model_loadpath + 'encoder.pth.tar'))
    discriminator.load_state_dict(torch.load(model_loadpath + 'discriminator.pth.tar'))
    
    if cuda:
        generator.cuda()
        encoder.cuda()
        discriminator.cuda()
        
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    #Change
    log_final_results = open("eval_noisy_clusgan_CIFAR10_3_digits_semi_sup_" + str(supervision_level) + "_seed_" + str(seed), "a+")

    ## Load saved cluster weights ideally
    ## Forgot to save them earlier so using heursitic of 1/num_clusters 
    cluster_weights = torch.ones([num_clusters]).to(torch.float).cuda()
    cluster_weights = cluster_weights / torch.sum(cluster_weights)

    print('\nBegin eval session...\n')


    myImages = torch.tensor([])
    GTlabel = torch.tensor([])
    finallabelling = torch.tensor([])

    for i, (imgs, itruth_label) in enumerate(dataloader_eval):

        # Ensure generator/encoder are in eval mode
        generator.eval()
        encoder.eval()
        discriminator.eval()

        # Zero gradients for models
        generator.zero_grad()
        encoder.zero_grad()
        discriminator.zero_grad()
        
        with torch.no_grad():
            real_imgs = Variable(imgs.type(Tensor))
            all_imgs = real_imgs
            z_real_encoded = encoder(all_imgs)
            z_real_encoded = z_real_encoded.repeat(cluster_centers.shape[0], 1, 1)
            repeated_cluster_centers = cluster_centers.view(cluster_centers.shape[0], 1, cluster_centers.shape[1]).repeat(1, z_real_encoded.shape[1], 1).cuda()
            totlist = -((z_real_encoded - repeated_cluster_centers)**2) / (2*sigma*sigma)
            totlist = torch.sum(totlist, 2) 

            if 0.0 in cluster_weights:
                cluster_weights = cluster_weights + epsilon
                cluster_weights = cluster_weights / torch.sum(cluster_weights)

            logProb = totlist + torch.log(cluster_weights.view(cluster_centers.shape[0], 1).repeat(1, z_real_encoded.shape[1]))
            logProb = torch.transpose(logProb, 0, 1)

            logProb2 = logProb.detach()
            Posterior = F.softmax(logProb2, -1)

            if myImages.shape[0] == 0:
                myImages = imgs.type(Tensor).clone()
                GTlabel = itruth_label
                finallabelling = torch.max(Posterior, 1)[1]
            else:
                myImages = torch.cat((myImages, imgs.type(Tensor).clone()), 0)
                GTlabel = torch.cat((GTlabel, itruth_label), 0)
                finallabelling = torch.cat((finallabelling, torch.max(Posterior, 1)[1]), 0)
            

    Local_NMI = NMI(np.array(GTlabel.cpu().to(torch.long)), np.array(finallabelling.cpu().to(torch.long)))
    Local_ARI = ARI(np.array(GTlabel.cpu().to(torch.long)), np.array(finallabelling.cpu().to(torch.long)))
    Local_ACC = ACC(np.array(GTlabel.cpu().to(torch.long)), np.array(finallabelling.cpu().to(torch.long)))

    final_NMI = Local_NMI
    final_accuracy = Local_ACC[0]
    final_ARI = Local_ARI

    log_final_results.write("NMI : " + str(Local_NMI) + " \n")
    log_final_results.write("ARI : " + str(Local_ARI) + " \n")
    log_final_results.write("ACC : " + str(Local_ACC[0]) + " \n")
    log_final_results.write("------------------------------------------\n")
    
    print("NMI : ", Local_NMI)
    print("ARI : ", Local_ARI)
    print("ACC : ", Local_ACC)

    return final_accuracy, final_ARI, final_NMI


if __name__ == "__main__":
    final_dict = {}
    
    # global args
    seed = args.seed
    print(seed)
    seeds = [seed]

    #Change
    log_final_results_metrics = open("metrics_eval_noisy_clusgan_CIFAR10_3_digits_semi_sup_" + str(args.supervision_level), "a+")
    log_final_results_metrics.write('Supervision : ' + str(args.supervision_level) + '\n')
    final_dict = {}
    for seed in seeds:
        final_dict[seed] = {}
        acc, ari, nmi = run_main(seed)
        final_dict[seed]['acc'] = acc
        final_dict[seed]['nmi'] = nmi
        final_dict[seed]['ari'] = ari

        log_final_results_metrics.write('Seed : ' + str(seed) + '\n')
        log_final_results_metrics.write('Accuracy : ' + str(acc) + '\n')
        log_final_results_metrics.write('NMI : ' + str(nmi) + '\n')
        log_final_results_metrics.write('ARI : ' + str(ari) + '\n')
    

    #Change
    pkl.dump(final_dict, open( './new_eval_noisy_cifar10_final_dict_3_' + str(args.supervision_level) + '_seed_' + str(seed) + '.pkl', "wb"), protocol=pkl.HIGHEST_PROTOCOL)
    print("Dumping into  ./new_eval_noisy_cifar10_final_dict_3_" + str(args.supervision_level) + '_seed_' + str(seed) + ".pkl")