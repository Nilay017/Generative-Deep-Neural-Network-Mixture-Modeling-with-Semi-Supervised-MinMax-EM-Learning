from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np
    import math
    import time
    import datetime

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

parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
#Change
parser.add_argument("-r", "--run_name", dest="run_name", default='noisy_all_chan_sigma_0.2_celebA_train_clusgan_5_all', help="Name of training run")
parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=500, type=int, help="Number of epochs")
parser.add_argument("-b", "--batch_size", dest="batch_size", default=200, type=int, help="Batch size")
parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='celebA', choices=dataset_list,  help="Dataset name")
parser.add_argument("-w", "--wass_metric", dest="wass_metric", action='store_true', help="Flag for Wasserstein metric")
parser.add_argument("-g", "-–gpu", dest="gpu", default=0, type=int, help="GPU id to use")
parser.add_argument("-k", "-–num_workers", dest="num_workers", default=0, type=int, help="Number of dataset workers")
parser.add_argument("-sup", "--supervision_level", dest="supervision_level", default=0.2, type=float, help="supervision_level")
parser.add_argument("-seed", "--seed", dest="seed", default=0., type=float, help="seed for experiments")
parser.add_argument("-gamma", "--gamma", dest="gamma", default=1.0, type=float, help="Gamma")
parser.add_argument("-dgamma", "--dgamma", dest="dgamma", default=1.0, type=float, help="DGamma")
args = parser.parse_args()

#Temporary center function
def get_centers(num_clusters, latent_dim, sigma, dtype=torch.float32):
	centers = torch.zeros([num_clusters, latent_dim]).to(dtype)

	# Inter center distance is now 5 sigma
	radius = float(5.00*float(sigma) / (math.sqrt(2.00)))
	for i in range(num_clusters):
		centers[i][i] = radius
	
	return centers

def run_main(seed):
    # python3 celebA_em_clusgan.py -n 600 -g 0 -sup 0.0 -seed 0.0
    global args
    run_name = args.run_name
    dataset_name = args.dataset_name
    device_id = args.gpu
    num_workers = args.num_workers
    gamma = args.gamma
    d_gamma = args.dgamma
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    wass_metric = args.wass_metric
    given_supervision_level = args.supervision_level

    final_accuracy = None
    final_NMI = None
    final_ARI = None

    torch.manual_seed(seed)
    dtype = torch.float32

    torch.set_printoptions(threshold=5000)

    # Training details
    lr = 1e-4
    b1 = 0.5
    b2 = 0.9 #99
    decay = 2.5*1e-5
    n_skip_iter = 1 #5

    img_size = 32
    channels = 3
   
    
    # Change
    datafile_path = "./Output_data_processing/Noisy_all_chan_CelebA_sigma_0.2_kmeans_clustering_data_5_" + str(given_supervision_level) + "_seed_" + str(int(seed)) + ".pkl"

    with open(datafile_path, 'rb') as f:	
        preprocessed_data = pkl.load(f)

    unique_cluster_labels = preprocessed_data['unique_cluster_labels']
    num_clusters = preprocessed_data['num_clusters']
    data = preprocessed_data['data']
    supervision_level = preprocessed_data['supervision_level']
    assert(supervision_level == given_supervision_level)
    initial_clustering = preprocessed_data['kmeans_clustering']
    actual_labels = preprocessed_data['actual_labels']

    latent_dim = 30
    sigma = float(5.00)
    cluster_centers = get_centers(num_clusters, latent_dim, sigma, dtype)
    print(cluster_centers)
    epsilon = 1e-45
    betan = 3
    betac = 3
    beta = 1
   
    # Change
    finaldata = data.clone().to(dtype)
    finaldata = (finaldata*0.50000) + 0.500000
    myfinaldata = finaldata.clone()
    finallabels = actual_labels.clone()

    num_images = finaldata.shape[0]

    # Wasserstein metric flag
    mtype = 'van'
    if (wass_metric):
        mtype = 'wass'
    
    # Make directory structure for this run
    # Change
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d')
    sep_und = '_'
    run_name_comps = ['training_supervision_%f_'%supervision_level,'epoch_%i_'%n_epochs, '_', mtype, '_', time_stamp, '_', 'seed_%i_'%seed, run_name]
    run_name = sep_und.join(run_name_comps)

    run_dir = os.path.join(RUNS_DIR, dataset_name, run_name)
    data_dir = os.path.join(DATASETS_DIR, dataset_name)
    imgs_dir = os.path.join(run_dir, 'images')
    models_dir = os.path.join(run_dir, 'models')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    print('\nResults to be saved in directory %s\n'%(run_dir))
    
    x_shape = (channels, img_size, img_size)
    
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device_id)

    # Loss function
    bce_loss = torch.nn.BCELoss()
    xe_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    
    # Initialize generator and discriminator
    model_loadpath = '/home/nilay/clusterGAN-master_2/runs/celebA/celebA_pretraining_supervision_'
    if supervision_level == 0.99:
        model_loadpath += str(supervision_level) + '0000'
    else:
        model_loadpath += str(supervision_level) + '00000'
    
    # model_loadpath += str(0.6) + '00000'
    # assert(supervision_level == 0.0)
    #Change
    model_loadpath += "__epoch_400____van___01-29___seed_" + str(int(seed)) + "__noisy_all_chan_sigma_0.2_celebA_pretrain_clusgan_5_all/models/"

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
        bce_loss.cuda()
        xe_loss.cuda()
        mse_loss.cuda()
        
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    #Change
    log_final_results = open("train_clusgan_noisy_all_chan_sigma_0.2_CelebA_5_digits_semi_sup_" + str(supervision_level) + "_seed_" + str(seed), "a+")
     
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay)
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))


    ##############################################################################
    # Pretraining Done ###########################################################
    ##############################################################################

    print("Pretraining Done!!!!!!!!!")
    print("Begin Training")


    # ----------
    #  Training
    # ----------
    ge_l_tr = []
    d_l_tr = []
    
    c_z_tr = []
    # c_zc = []
    c_i_tr = []

    c_sup_loss_tr = []
    
    log_final_results.write("Gamma: " + str(gamma) + " \n")
    log_final_results.write("DGamma: " + str(d_gamma) + " \n")

    cluster_weights = torch.ones([num_clusters]).to(torch.float).cuda()
    cluster_weights = cluster_weights / torch.sum(cluster_weights)

    print('\nBegin training session with %i epochs...\n'%(n_epochs))


    fixed_indices = None
    fixed_data = None
    fixed_gt_idx = None
    fixed_clustering = None
    dataMNIST_only_unlabelled = None
    dataloader_only_unlabelled = None

    if supervision_level > 0.0:
        fixed_indices = preprocessed_data['fixed_indices']
        fixed_data_0 = myfinaldata[fixed_indices]
        fixed_gt_idx = finallabels[fixed_indices].to(torch.long).clone() - 1
        fixed_clustering = finallabels[fixed_indices].to(torch.long).clone()
        
        assert(fixed_data_0.shape[1] == channels)
        assert(fixed_data_0.shape[2] == img_size)
        assert(fixed_data_0.shape[3] == img_size)

        fixed_data = torch.zeros([fixed_data_0.shape[0], channels, img_size, img_size]).type(Tensor)
        fixed_data = fixed_data_0.clone().cuda()
        
        indices_to_update = (1  - fixed_indices.to(torch.long)).to(torch.long)
        indices_to_update = indices_to_update == 1
        indices_to_update = indices_to_update.to(torch.bool)
        # Change
        dataMNIST_only_unlabelled = CustomMNISTdigit(finaldata[indices_to_update], finallabels[indices_to_update], None)
        dataloader_only_unlabelled = DataLoader(dataMNIST_only_unlabelled, num_workers=num_workers,  batch_size=batch_size,  shuffle=True)
    else:
        dataMNIST_only_unlabelled = CustomMNISTdigit(finaldata, finallabels, None)
        dataloader_only_unlabelled = DataLoader(dataMNIST_only_unlabelled, num_workers=num_workers,  batch_size=batch_size,  shuffle=True)

    t_imgs, t_label = myfinaldata.clone().cuda(), finallabels.clone()
    
    assert(t_imgs.shape[1] == channels)
    assert(t_imgs.shape[2] == img_size)
    assert(t_imgs.shape[3] == img_size)

    # n_epochs = 250
    for epoch in range(n_epochs):

        myImages = torch.tensor([])
        GTlabel = torch.tensor([])

        per_epoch_ge_loss = 0.00
        per_epoch_d_loss = 0.00
        per_epoch_img_loss = 0.00
        per_epoch_latent_loss = 0.00
        per_epoch_latent_sup_loss = 0.00

        for i, (imgs, itruth_label) in enumerate(dataloader_only_unlabelled):

            # Ensure generator/encoder are trainable
            generator.train()
            encoder.train()
            discriminator.train()

            # Zero gradients for models
            generator.zero_grad()
            encoder.zero_grad()
            discriminator.zero_grad()
            
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            if myImages.shape[0] == 0:
                myImages = imgs.type(Tensor).clone()
                GTlabel = itruth_label
            else:
                myImages = torch.cat((myImages, imgs.type(Tensor).clone()), 0)
                GTlabel = torch.cat((GTlabel, itruth_label), 0)

            # ---------------------------
            #  Train Generator + Encoder
            # ---------------------------
            
            optimizer_G.zero_grad()
            optimizer_E.zero_grad()
            
            # Sample random latent variables
            z = sample_z(shape=imgs.shape[0], latent_dim=latent_dim, n_c=num_clusters, fix_class=-1, cluster_centers=cluster_centers, req_grad=False)
    
            # Generate a batch of images
            gen_imgs = generator(z)
            
            # Discriminator output from real and generated samples
            D_gen = discriminator(gen_imgs)
            D_real = discriminator(real_imgs)

            D_fixed = None
            if supervision_level > 0.0:
                D_fixed = discriminator(fixed_data)

            # -------------------------------
            # EM
            # -------------------------------
            all_imgs = None
            if supervision_level > 0.0:
                all_imgs = torch.cat((real_imgs, fixed_data), 0).type(Tensor)
            else:
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
            if supervision_level > 0.0:
                Posterior[real_imgs.shape[0]:] = torch.eye(cluster_centers.shape[0]).type(Tensor)[fixed_gt_idx]
            MLE_loss = -torch.sum(logProb*Posterior)

            ge_loss = 0.00
            mle_loss_lambda = 1.00

            cluster_weights = torch.sum(Posterior, 0)
            Norm = torch.sum(cluster_weights)

            #Normalize the Prior
            cluster_weights = cluster_weights/Norm
            cluster_weights = cluster_weights.cuda()

            # Step for Generator & Encoder, n_skip_iter times less than for discriminator
            if (i % n_skip_iter == 0):
                # Encode the generated images
                enc_gen_z = encoder(gen_imgs)

                # Calculate losses for z
                z_loss = mse_loss(enc_gen_z, z)

                sup_z_loss = 0.00
                if supervision_level > 0.0:
                    sup_z = encoder(fixed_data)
                    sup_z_loss = mse_loss(sup_z, cluster_centers[fixed_gt_idx.to(torch.long)].cuda())

    
                # Check requested metric
                if wass_metric:
                    assert(1==0)
                    # Wasserstein GAN loss
                    # ge_loss = torch.mean(D_gen) + betan * zn_loss + betac * zc_loss
                    ge_loss = torch.mean(D_gen) + beta * z_loss + gamma * sup_z_loss
                else:
                    # Vanilla GAN loss
                    valid = Variable(Tensor(gen_imgs.size(0), 1).fill_(1.0), requires_grad=False)
                    v_loss = bce_loss(D_gen, valid)
                    ge_loss = v_loss + beta * z_loss + gamma * sup_z_loss
    
            ge_loss = ge_loss + (mle_loss_lambda * MLE_loss)
            ge_loss.backward(retain_graph=True)
            optimizer_E.step()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
    
            discriminator.zero_grad()
            optimizer_D.zero_grad()
    
            # Measure discriminator's ability to classify real from generated samples
            if wass_metric:
                # Gradient penalty term
                grad_penalty = calc_gradient_penalty(discriminator, real_imgs, gen_imgs)

                # Wasserstein GAN loss w/gradient penalty
                if supervision_level > 0.0:
                    d_loss = torch.mean(D_real) - torch.mean(D_gen) + grad_penalty + d_gamma * torch.mean(D_fixed)
                else:
                    d_loss = torch.mean(D_real) - torch.mean(D_gen) + grad_penalty
   
            else:
                # Vanilla GAN loss
                fake = Variable(Tensor(gen_imgs.size(0), 1).fill_(0.0), requires_grad=False)
                if supervision_level > 0.0:
                    fixed = Variable(Tensor(fixed_data.size(0), 1).fill_(1.0), requires_grad=False)

                real_loss = bce_loss(D_real, valid)
                fake_loss = bce_loss(D_gen, fake)
                fixed_loss= 0.00
                
                if supervision_level > 0.0:
                    fixed_loss= bce_loss(D_fixed, fixed)

                d_loss = 3*((real_loss + fake_loss) / 2) + (d_gamma * fixed_loss / 2)
    
            d_loss.backward()
            optimizer_D.step()

            # Save training losses
            per_epoch_d_loss += d_loss.item()
            per_epoch_ge_loss += ge_loss.item()
 

            # Generator in eval mode
            generator.eval()
            encoder.eval()
            discriminator.eval()

            # Set number of examples for cycle calcs
            n_sqrt_samp = 5
            n_samp = n_sqrt_samp * n_sqrt_samp


            ## Cycle through test real -> enc -> gen
            
            #r_imgs, i_label = real_imgs.data[:n_samp], itruth_label[:n_samp]
            # Encode sample real instances
            e_tz = encoder(t_imgs)
            # Generate sample instances from encoding
            teg_imgs = generator(e_tz)
            # Calculate cycle reconstruction loss
            img_mse_loss = mse_loss(t_imgs, teg_imgs)
            # Save img reco cycle loss
            per_epoch_img_loss += img_mse_loss.item()
           

            ## Cycle through randomly sampled encoding -> generator -> encoder
            z_samp = sample_z(shape=imgs.shape[0], latent_dim=latent_dim, n_c=num_clusters, fix_class=-1, cluster_centers=cluster_centers, req_grad=False)
            # Generate sample instances
            gen_imgs_samp = generator(z_samp)
            # Encode sample instances
            z_e = encoder(gen_imgs_samp)
            # Calculate cycle latent losses
            lat_mse_loss = mse_loss(z_e, z_samp)
            per_epoch_latent_loss += lat_mse_loss.item()

            lat_sup_z_loss = 0.00
            if supervision_level > 0.0:
                lat_sup_z = encoder(fixed_data)
                lat_sup_z_loss = mse_loss(lat_sup_z, cluster_centers[fixed_gt_idx.cuda().to(torch.long)].cuda())
                per_epoch_latent_sup_loss += lat_sup_z_loss.item()
                # Check for supervision thoroughly before implementing
            else:
                per_epoch_latent_sup_loss += lat_sup_z_loss


        # Save latent space cycle losses
        c_z_tr.append(per_epoch_latent_loss)
        c_i_tr.append(per_epoch_img_loss)
        c_sup_loss_tr.append(per_epoch_latent_sup_loss)
        d_l_tr.append(per_epoch_d_loss)
        ge_l_tr.append(per_epoch_ge_loss)

        # Save cycled and generated examples!
        r_imgs, i_label = real_imgs.data[:n_samp], itruth_label[:n_samp]
        e_z = encoder(r_imgs)
        reg_imgs = generator(e_z)
        # save_image(r_imgs.data[:n_samp],
        #            '%s/actual_train_real_%06i.png' %(imgs_dir, epoch), 
        #            nrow=n_sqrt_samp, normalize=True)
        # save_image(reg_imgs.data[:n_samp],
        #            '%s/actual_train_reg_%06i.png' %(imgs_dir, epoch), 
        #            nrow=n_sqrt_samp, normalize=True)
        # save_image(gen_imgs_samp.data[:n_samp],
        #            '%s/actual_train_gen_%06i.png' %(imgs_dir, epoch), 
        #            nrow=n_sqrt_samp, normalize=True)
        
        ## Generate samples for specified classes
        stack_imgs = []
        for idx in range(cluster_centers.shape[0]):
            # Sample specific class
            z_samp = sample_z(shape=5, latent_dim=latent_dim, n_c=num_clusters, fix_class=idx, cluster_centers=cluster_centers)

            # Generate sample instances
            gen_imgs_samp = generator(z_samp)

            if (len(stack_imgs) == 0):
                stack_imgs = gen_imgs_samp
            else:
                stack_imgs = torch.cat((stack_imgs, gen_imgs_samp), 0)

        # Save class-specified generated examples!
        # save_image(stack_imgs,
        #            '%s/actual_gen_classes_%06i.png' %(imgs_dir, epoch), 
        #            nrow=num_clusters, normalize=True)
     

        print ("[Epoch %d/%d] \n"\
               "\tModel Losses: [D: %f] [GE: %f]" % (epoch, 
                                                     n_epochs, 
                                                     per_epoch_d_loss,
                                                     per_epoch_ge_loss)
              )
        

        
        print("\tCycle Losses: [x: %f] [z: %f] [sup_zc: %f]"%(per_epoch_img_loss, 
                                                             per_epoch_latent_loss,
                                                             per_epoch_latent_sup_loss)
             )


        
        if epoch % 2 == 1:
            all_imgs = None
            if supervision_level > 0.0:
                all_imgs = torch.cat((myImages, fixed_data), 0).type(Tensor)
            else:
                all_imgs = myImages
            z_myimg = encoder(all_imgs)
            z_myimg = z_myimg.repeat(cluster_centers.shape[0], 1, 1)
            my_repeated_cluster_centers = cluster_centers.view(cluster_centers.shape[0], 1, cluster_centers.shape[1]).repeat(1, z_myimg.shape[1], 1).cuda()
            my_totlist = -((z_myimg - my_repeated_cluster_centers)**2) / (2*sigma*sigma)
            my_totlist = torch.sum(my_totlist, 2)   
            my_cluster_weights = cluster_weights.clone().detach()

            if 0.0 in my_cluster_weights:
                my_cluster_weights = my_cluster_weights + epsilon
                my_cluster_weights = my_cluster_weights / torch.sum(my_cluster_weights)

            my_logProb = my_totlist + torch.log(my_cluster_weights.view(cluster_centers.shape[0], 1).repeat(1, z_myimg.shape[1]))
            my_logProb = torch.transpose(my_logProb, 0, 1)

            my_logProb2 = my_logProb.detach()
            my_Posterior = F.softmax(my_logProb2, -1)
            if supervision_level > 0.0:
                my_Posterior[myImages.shape[0]:] = torch.eye(cluster_centers.shape[0]).type(Tensor)[fixed_gt_idx]
            finallabelling = torch.max(my_Posterior, 1)[1]

            # print(finallabelling)
            # print(my_cluster_weights)
            # sleep(0.1)
            # print(GTlabel)

            if supervision_level > 0.0:
                GTlabel = torch.cat((GTlabel.cpu().to(torch.long), fixed_clustering.cpu().to(torch.long)), 0)
            
            Local_NMI = NMI(np.array(GTlabel.cpu().to(torch.long)), np.array(finallabelling.cpu().to(torch.long)))
            Local_ARI = ARI(np.array(GTlabel.cpu().to(torch.long)), np.array(finallabelling.cpu().to(torch.long)))
            Local_ACC = ACC(np.array(GTlabel.cpu().to(torch.long)), np.array(finallabelling.cpu().to(torch.long)))

            final_NMI = Local_NMI
            final_accuracy = Local_ACC[0]
            final_ARI = Local_ARI

            log_final_results.write("Epoch : " + str(epoch) + " \n")
            log_final_results.write("NMI : " + str(Local_NMI) + " \n")
            log_final_results.write("ARI : " + str(Local_ARI) + " \n")
            log_final_results.write("ACC : " + str(Local_ACC[0]) + " \n")
            log_final_results.write("------------------------------------------\n")
            
            print("NMI : ", Local_NMI)
            print("ARI : ", Local_ARI)
            print("ACC : ", Local_ACC)

    # Save training results
    train_df = pd.DataFrame({
                             'n_epochs' : n_epochs,
                             'learning_rate' : lr,
                             'beta_1' : b1,
                             'beta_2' : b2,
                             'weight_decay' : decay,
                             'n_skip_iter' : n_skip_iter,
                             'latent_dim' : latent_dim,
                             'n_classes' : cluster_centers.shape[0],
                             'beta_n' : betan,
                             'beta_c' : betac,
                             'wass_metric' : wass_metric,
                             'gen_enc_loss' : ['G+E', ge_l_tr],
                             'disc_loss' : ['D', d_l_tr],
                             'z_cycle_loss' : ['$||Z-E(G(x))||$', c_z_tr],
                             # 'zc_cycle_loss' : ['$||Z_c-E(G(x))_c||$', c_zc],
                             'img_cycle_loss' : ['$||X-G(E(x))||$', c_i_tr]
                            })

    train_df.to_csv('%s/actual_tr_training_details.csv'%(run_dir))


    # Plot some training results
    plot_train_loss(df=train_df,
                    arr_list=['gen_enc_loss', 'disc_loss'],
                    figname='%s/actual_tr_training_model_losses.png'%(run_dir)
                    )

    plot_train_loss(df=train_df,
                    arr_list=['z_cycle_loss', 'img_cycle_loss'],
                    figname='%s/actual_tr_training_cycle_loss.png'%(run_dir)
                    )


    # Save current state of trained models
    model_list = [discriminator, encoder, generator]
    save_model(models=model_list, out_dir=models_dir)

    return final_accuracy, final_ARI, final_NMI, cluster_weights


if __name__ == "__main__":
    final_dict = {}
    # global args
    seed = args.seed
    seeds = [seed]
    #Change
    log_final_results_metrics = open("metrics_train_clusgan_noisy_all_chan_sigma_0.2_CelebA_5_digits_semi_sup_" + str(args.supervision_level), "a+")
    log_final_results_metrics.write('Supervision : ' + str(args.supervision_level) + '\n')
    final_dict = {}
    for seed in seeds:
        final_dict[seed] = {}
        acc, ari, nmi, cluster_weights = run_main(seed)
        final_dict[seed]['acc'] = acc
        final_dict[seed]['nmi'] = nmi
        final_dict[seed]['ari'] = ari
        final_dict[seed]['cluster_weights'] = cluster_weights

        log_final_results_metrics.write('Seed : ' + str(seed) + '\n')
        log_final_results_metrics.write('Accuracy : ' + str(acc) + '\n')
        log_final_results_metrics.write('NMI : ' + str(nmi) + '\n')
        log_final_results_metrics.write('ARI : ' + str(ari) + '\n')
    

    #Change
    pkl.dump(final_dict, open( './new_noisy_all_chan_sigma_0.2_celebA_final_dict_5_' + str(args.supervision_level) + '_seed_' + str(seed) + '.pkl', "wb"), protocol=pkl.HIGHEST_PROTOCOL)
    print("Dumping into  ./new_noisy_all_chan_sigma_0.2_celebA_final_dict_5_" + str(args.supervision_level) + '_seed_' + str(seed) + ".pkl")
