from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np
    import math

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


#Temporary center function
def get_centers(num_clusters, latent_dim, sigma, dtype=torch.float32):
	centers = torch.zeros([num_clusters, latent_dim]).to(dtype)

	# Inter center distance is now 5 sigma
	radius = float(5.00*float(sigma) / (math.sqrt(2.00)))
	for i in range(num_clusters):
		centers[i][i] = radius
	
	return centers

def main():
    global args
    torch.manual_seed(12111)
    dtype = torch.float32
    

    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default='clusgan_3_123_new', help="Name of training run")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=200, type=int, help="Batch size")
    parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='mnist', choices=dataset_list,  help="Dataset name")
    parser.add_argument("-w", "--wass_metric", dest="wass_metric", action='store_true', help="Flag for Wasserstein metric")
    parser.add_argument("-g", "-–gpu", dest="gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("-k", "-–num_workers", dest="num_workers", default=1, type=int, help="Number of dataset workers")
    parser.add_argument("-sup", "--supervision", dest="supervision", default=0.0, type=float, help="Fraction of data supervised")
    parser.add_argument("-gamma", "--gamma", dest="gamma", default=1.0, type=float, help="Gamma")
    parser.add_argument("-dgamma", "--dgamma", dest="dgamma", default=1.0, type=float, help="DGamma")
    args = parser.parse_args()

    torch.manual_seed(12111)
    torch.set_printoptions(threshold=5000)


    run_name = args.run_name
    dataset_name = args.dataset_name
    device_id = args.gpu
    num_workers = args.num_workers
    supervision_level = args.supervision

    gamma = args.gamma
    d_gamma = args.dgamma

    # Training details
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    test_batch_size = 5000
    lr = 1e-4
    b1 = 0.5
    b2 = 0.9 #99
    decay = 2.5*1e-5
    n_skip_iter = 1 #5

    img_size = 28
    channels = 1
   
    # Latent space info
    datafile_path = "./MNIST_1000_121.pkl"
    with open(datafile_path, 'rb') as f:	
        finalvalues, labels = pkl.load(f)

    unique_cluster_labels = np.unique(np.array(labels))
    num_clusters = unique_cluster_labels.shape[0]

    latent_dim = 30
    sigma = float(5.00)
    cluster_centers = get_centers(num_clusters, latent_dim, sigma, dtype)
    print(cluster_centers)
    # n_c = 10
    # betan = 10
    # betac = 10
    # n_c = 3
    epsilon = 1e-45
    betan = 3
    betac = 3
    beta = 1
    mle_loss_lambda = 0 
    

    finaldata = finalvalues.to(torch.uint8).clone()
    finaltarget = labels.clone()

    num_images = finaldata.shape[0]
    images_per_class = int(num_images / num_clusters)
    num_images_to_select = int(images_per_class * supervision_level)

    fixed_data_per_cluster = None
    fixed_true_labelling = None
    fixed_clustering = None
    fixed_gt_idx = None

    if supervision_level != 0:
        fixed_data_per_cluster = torch.zeros(num_clusters, num_images_to_select, img_size, img_size).to(dtype)
        fixed_gt_idx = torch.zeros(num_clusters, num_images_to_select, num_clusters).to(dtype)
        fixed_true_labelling  = torch.zeros(num_clusters*num_images_to_select).to(dtype)
        fixed_clustering  = torch.zeros(num_clusters*num_images_to_select).to(torch.long) 

    fixed_indices = torch.zeros(num_images).to(dtype)
    indices_to_update = (1 - fixed_indices).to(torch.long)
    indices_to_update = indices_to_update == 1

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform2 = transforms.ToPILImage()

    # import pdb
    # pdb.set_trace()

    if supervision_level > 0:
        assert(supervision_level <= 1.0)
        unique_cluster_labels = torch.tensor(np.unique(np.array(labels)))
        assert(num_clusters == unique_cluster_labels.shape[0])

        for i in range(num_clusters):
            idx = labels == unique_cluster_labels[i]

            if fixed_true_labelling is not None:
                fixed_true_labelling[i*num_images_to_select:((i+1)*num_images_to_select)] = unique_cluster_labels[i]
                fixed_clustering[i*num_images_to_select:((i+1)*num_images_to_select)] = i

            temp_fixed_indices = torch.zeros(idx[idx == 1].shape[0])
            assert(temp_fixed_indices.shape[0] == images_per_class)

            temp_fixed_indices[torch.randperm(temp_fixed_indices.shape[0])[:num_images_to_select]] = 1
            # temp_fixed_indices[:num_images_to_select] = 1
            idx[idx==1] = temp_fixed_indices.to(torch.bool)

            if fixed_data_per_cluster is not None:
                for j in range(num_images_to_select):
                    fixed_data_per_cluster[i, j, :, :] = transform(transform2(finaldata[idx][j]))
                    fixed_gt_idx[i, j, i] = 1.00

            fixed_indices += idx.to(dtype)

        indices_to_update = (1 - fixed_indices).to(torch.long)
        indices_to_update = indices_to_update == 1

    newfinaldata = finaldata[indices_to_update].clone()
    newfinaltarget = finaltarget[indices_to_update].clone()

    if fixed_data_per_cluster is not None:
   		fixed_data_per_cluster = fixed_data_per_cluster.reshape([num_clusters*num_images_to_select, 1, img_size, img_size])
   		fixed_gt_idx = fixed_gt_idx.reshape([num_clusters*num_images_to_select, num_clusters])


    # newfinaldata = finaldata.clone()
    # newfinaltarget = finaltarget.clone()

    dataMNIST = CustomMNISTdigit(newfinaldata, newfinaltarget, transform)
    dataloader = DataLoader(dataMNIST, num_workers=num_workers,  batch_size=batch_size,  shuffle=True)

    testdataMNIST = CustomMNISTdigit(newfinaldata.clone(), newfinaltarget.clone(), transform)
    testdata = DataLoader(testdataMNIST, num_workers=num_workers,  batch_size=batch_size,  shuffle=True)    
   
    # Wasserstein metric flag
    # Wasserstein metric flag
    wass_metric = args.wass_metric
    mtype = 'van'
    if (wass_metric):
        mtype = 'wass'
    
    # Make directory structure for this run
    sep_und = '_'
    run_name_comps = ['%iepoch'%n_epochs, 'z%s'%str(latent_dim), mtype, 'bs%i'%batch_size, run_name]
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device_id)

    # Loss function
    bce_loss = torch.nn.BCELoss()
    xe_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    
    # Initialize generator and discriminator
    generator = Generator_CNN(latent_dim, x_shape)
    encoder = Encoder_CNN(latent_dim)
    discriminator = Discriminator_CNN(wass_metric=wass_metric)
    
    if cuda:
        generator.cuda()
        encoder.cuda()
        discriminator.cuda()
        bce_loss.cuda()
        xe_loss.cuda()
        mse_loss.cuda()
        
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    test_imgs, test_labels = next(iter(testdata))
    test_imgs = Variable(test_imgs.type(Tensor))
   
    ge_chain = ichain(generator.parameters(),
                      encoder.parameters())

    optimizer_GE = torch.optim.Adam(ge_chain, lr=lr, betas=(b1, b2), weight_decay=decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    

    # ----------
    #  Training
    # ----------
    ge_l = []
    d_l = []
    
    c_z = []
    # c_zc = []
    c_i = []

    c_sup_loss = []
    
    # Training loop 
    log_final_results = open("clustergan_our_method_MNIST_3_digits_semi_sup", "a+")
    log_final_results.write("Gamma: " + str(gamma) + " \n")
    log_final_results.write("DGamma: " + str(d_gamma) + " \n")

    cluster_weights = torch.ones([num_clusters]).to(torch.float).cuda()
    cluster_weights = cluster_weights / torch.sum(cluster_weights)

    print('\nBegin training session with %i epochs...\n'%(n_epochs))

    for epoch in range(n_epochs):

        myImages = torch.tensor([])
        GTlabel = torch.tensor([])

        for i, (imgs, itruth_label) in enumerate(dataloader):

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
            
            optimizer_GE.zero_grad()
            
            # Sample random latent variables
            z = sample_z(shape=imgs.shape[0], latent_dim=latent_dim, n_c=3, fix_class=-1, cluster_centers=cluster_centers, req_grad=False)
    
            # Generate a batch of images
            gen_imgs = generator(z)
            
            # Discriminator output from real and generated samples
            D_gen = discriminator(gen_imgs)
            D_real = discriminator(real_imgs)

            D_fixed = None
            if fixed_data_per_cluster is not None:
                D_fixed = discriminator(fixed_data_per_cluster.cuda())

            # -------------------------------
            # EM
            # -------------------------------

            z_real_encoded = encoder(real_imgs)
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
            MLE_loss = -torch.sum(logProb*Posterior)

            ge_loss = 0.00

            cluster_weights = torch.sum(Posterior, 0)
            Norm = torch.sum(cluster_weights)

            #Normalize the Prior
            cluster_weights = cluster_weights/Norm
            cluster_weights = cluster_weights.cuda()

            # Step for Generator & Encoder, n_skip_iter times less than for discriminator
            if (i % n_skip_iter == 0):
                # Encode the generated images
                enc_gen_z = encoder(gen_imgs)

                # Calculate losses for z_n, z_c
                z_loss = mse_loss(enc_gen_z, z)

                sup_z_loss = 0.00
                if fixed_data_per_cluster is not None:
                    sup_z = encoder(fixed_data_per_cluster.cuda())
                    # sup_z_loss = bce_loss(sup_zc, fixed_gt_idx.cuda())
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
            optimizer_GE.step()

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
                if fixed_data_per_cluster is not None:
                    d_loss = torch.mean(D_real) - torch.mean(D_gen) + grad_penalty + d_gamma * torch.mean(D_fixed)
                else:
                    d_loss = torch.mean(D_real) - torch.mean(D_gen) + grad_penalty
   
            else:
                # Vanilla GAN loss
                fake = Variable(Tensor(gen_imgs.size(0), 1).fill_(0.0), requires_grad=False)
                if fixed_data_per_cluster is not None:
                    fixed = Variable(Tensor(fixed_data_per_cluster.size(0), 1).fill_(1.0), requires_grad=False)

                real_loss = bce_loss(D_real, valid)
                fake_loss = bce_loss(D_gen, fake)
                fixed_loss= 0.00
                
                if fixed_data_per_cluster is not None:
                    fixed_loss= bce_loss(D_fixed, fixed)

                d_loss = 3*((real_loss + fake_loss) / 2) + (d_gamma * fixed_loss / 2)
    
            d_loss.backward()
            optimizer_D.step()

        # Save training losses
        d_l.append(d_loss.item())
        ge_l.append(ge_loss.item())
   

        # Generator in eval mode
        generator.eval()
        encoder.eval()
        discriminator.eval()

        # Set number of examples for cycle calcs
        n_sqrt_samp = 5
        n_samp = n_sqrt_samp * n_sqrt_samp


        ## Cycle through test real -> enc -> gen
        t_imgs, t_label = test_imgs.data, test_labels
        #r_imgs, i_label = real_imgs.data[:n_samp], itruth_label[:n_samp]
        # Encode sample real instances
        e_tz = encoder(t_imgs)
        # Generate sample instances from encoding
        teg_imgs = generator(e_tz)
        # Calculate cycle reconstruction loss
        img_mse_loss = mse_loss(t_imgs, teg_imgs)
        # Save img reco cycle loss
        c_i.append(img_mse_loss.item())
       

        ## Cycle through randomly sampled encoding -> generator -> encoder
        z_samp = sample_z(shape=imgs.shape[0], latent_dim=latent_dim, n_c=num_clusters, fix_class=-1, cluster_centers=cluster_centers, req_grad=False)
        # Generate sample instances
        gen_imgs_samp = generator(z_samp)
        # Encode sample instances
        z_e = encoder(gen_imgs_samp)
        # Calculate cycle latent losses
        lat_mse_loss = mse_loss(z_e, z_samp)

        lat_sup_z_loss = 0.00
        if fixed_data_per_cluster is not None:
            lat_sup_z = encoder(fixed_data_per_cluster.cuda())
            lat_sup_z_loss = mse_loss(lat_sup_z, cluster_centers[fixed_gt_idx.cuda().to(torch.long)])
            assert(1==0)
            # Check for supervision thoroughly before implementing

        # Save latent space cycle losses
        c_z.append(lat_mse_loss.item())
        # c_zc.append(lat_xe_loss.item())

        if fixed_data_per_cluster is not None:
            c_sup_loss.append(lat_sup_z_loss.item())
        else:
            c_sup_loss.append(lat_sup_z_loss)
      
        # Save cycled and generated examples!
        r_imgs, i_label = real_imgs.data[:n_samp], itruth_label[:n_samp]
        e_z = encoder(r_imgs)
        reg_imgs = generator(e_z)
        save_image(r_imgs.data[:n_samp],
                   '%s/real_%06i.png' %(imgs_dir, epoch), 
                   nrow=n_sqrt_samp, normalize=True)
        save_image(reg_imgs.data[:n_samp],
                   '%s/reg_%06i.png' %(imgs_dir, epoch), 
                   nrow=n_sqrt_samp, normalize=True)
        save_image(gen_imgs_samp.data[:n_samp],
                   '%s/gen_%06i.png' %(imgs_dir, epoch), 
                   nrow=n_sqrt_samp, normalize=True)
        
        ## Generate samples for specified classes
        stack_imgs = []
        for idx in range(cluster_centers.shape[0]):
            # Sample specific class
            z_samp = sample_z(shape=3, latent_dim=latent_dim, n_c=num_clusters, fix_class=idx, cluster_centers=cluster_centers)

            # Generate sample instances
            gen_imgs_samp = generator(z_samp)

            if (len(stack_imgs) == 0):
                stack_imgs = gen_imgs_samp
            else:
                stack_imgs = torch.cat((stack_imgs, gen_imgs_samp), 0)

        # Save class-specified generated examples!
        save_image(stack_imgs,
                   '%s/gen_classes_%06i.png' %(imgs_dir, epoch), 
                   nrow=3, normalize=True)
     

        print ("[Epoch %d/%d] \n"\
               "\tModel Losses: [D: %f] [GE: %f]" % (epoch, 
                                                     n_epochs, 
                                                     d_loss.item(),
                                                     ge_loss.item())
              )
        

        if fixed_data_per_cluster is not None:
            print("\tCycle Losses: [x: %f] [z: %f] [sup_zc: %f]"%(img_mse_loss.item(), 
                                                                 lat_mse_loss.item(), 
                                                                 lat_sup_z_loss.item())
                 )
        else:
            print("\tCycle Losses: [x: %f] [z: %f] [sup_zc: %f]"%(img_mse_loss.item(), 
                                                                 lat_mse_loss.item(), 
                                                                 lat_sup_z_loss)
                 )


        
        if epoch % 2 == 1:
            z_myimg = encoder(myImages)
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
            finallabelling = torch.max(my_Posterior, 1)[1]

            # print(finallabelling)
            print(my_cluster_weights)
            # sleep(0.1)
            # print(GTlabel)

            if fixed_data_per_cluster is not None:
            	GTlabel = torch.cat((GTlabel.cpu().to(torch.long), fixed_true_labelling.cpu().to(torch.long)), 0)
            	finallabelling = torch.cat((finallabelling.cpu().to(torch.long), fixed_clustering.cpu().to(torch.long)), 0)
            
            Local_NMI = NMI(np.array(GTlabel.cpu().to(torch.long)), np.array(finallabelling.cpu().to(torch.long)))
            Local_ARI = ARI(np.array(GTlabel.cpu().to(torch.long)), np.array(finallabelling.cpu().to(torch.long)))
            Local_ACC = ACC(np.array(GTlabel.cpu().to(torch.long)), np.array(finallabelling.cpu().to(torch.long)))

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
                             'gen_enc_loss' : ['G+E', ge_l],
                             'disc_loss' : ['D', d_l],
                             'z_cycle_loss' : ['$||Z-E(G(x))||$', c_z],
                             # 'zc_cycle_loss' : ['$||Z_c-E(G(x))_c||$', c_zc],
                             'img_cycle_loss' : ['$||X-G(E(x))||$', c_i]
                            })

    train_df.to_csv('%s/training_details.csv'%(run_dir))


    # Plot some training results
    plot_train_loss(df=train_df,
                    arr_list=['gen_enc_loss', 'disc_loss'],
                    figname='%s/training_model_losses.png'%(run_dir)
                    )

    plot_train_loss(df=train_df,
                    arr_list=['z_cycle_loss', 'img_cycle_loss'],
                    figname='%s/training_cycle_loss.png'%(run_dir)
                    )


    # Save current state of trained models
    model_list = [discriminator, encoder, generator]
    save_model(models=model_list, out_dir=models_dir)


if __name__ == "__main__":
    main()
