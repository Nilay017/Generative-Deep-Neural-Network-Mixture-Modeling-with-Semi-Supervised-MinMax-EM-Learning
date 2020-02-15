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
        
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    #change
    parser.add_argument("-r", "--run_name", dest="run_name", default='noisy_all_chan_sigma_0.2_celebA_pretrain_clusgan_5_all', help="Name of training run")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=500, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=200, type=int, help="Batch size")
    parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='celebA', choices=dataset_list,  help="Dataset name")
    parser.add_argument("-w", "--wass_metric", dest="wass_metric", action='store_true', help="Flag for Wasserstein metric")
    parser.add_argument("-g", "-–gpu", dest="gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("-k", "-–num_workers", dest="num_workers", default=0, type=int, help="Number of dataset workers")
    parser.add_argument("-sup", "--supervision_level", dest="supervision_level", default=0.2, type=float, help="supervision_level")
    parser.add_argument("-seed", "--seed", dest="seed", default=0, type=float, help="seed for experiments")
    args = parser.parse_args()

    # python3 celebA_initialize_em_clusgan.py -n 300 -g 0 -sup 0.5 -seed 0.0

    run_name = args.run_name
    dataset_name = args.dataset_name
    device_id = args.gpu
    num_workers = args.num_workers
    given_supervision_level = args.supervision_level
    seed = args.seed


    torch.manual_seed(seed)
    dtype = torch.float32

    torch.set_printoptions(threshold=5000)

    # Training details
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = 1e-4
    b1 = 0.5
    b2 = 0.9 #99
    decay = 2.5*1e-5
    n_skip_iter = 1 #5

    img_size = 32
    channels = 3
   
    # Latent space info
    # datafile_path = "./MNIST_1000_121.pkl"
    #change
    datafile_path = "./Output_data_processing/Noisy_all_chan_CelebA_sigma_0.2_kmeans_clustering_data_5_" + str(given_supervision_level) + "_seed_" + str(int(seed)) + ".pkl"
    with open(datafile_path, 'rb') as f:
        preprocessed_data = pkl.load(f)

    unique_cluster_labels = preprocessed_data['unique_cluster_labels']
    num_clusters = preprocessed_data['num_clusters']
    data = preprocessed_data['data']
    supervision_level = preprocessed_data['supervision_level']
    initial_clustering = preprocessed_data['kmeans_clustering']
    actual_labels = preprocessed_data['actual_labels']

    assert(supervision_level == given_supervision_level)

    latent_dim = 30
    sigma = float(5.00)
    cluster_centers = get_centers(num_clusters, latent_dim, sigma, dtype)
    epsilon = 1e-45
    betan = 3
    betac = 3
    beta = 1
   
    # Change
    finaldata = data.clone()
    finaldata = (finaldata*0.50000) + 0.500000
    finaltarget = initial_clustering.clone()
    finallabels = actual_labels.clone()

    num_images = finaldata.shape[0]

    dataMNIST = CustomMNISTdigit(finaldata, finaltarget, None)
    dataloader = DataLoader(dataMNIST, num_workers=num_workers,  batch_size=batch_size,  shuffle=True)
 
    # Wasserstein metric flag
    wass_metric = args.wass_metric
    mtype = 'van'
    if (wass_metric):
        mtype = 'wass'
    
    # Make directory structure for this run
    # time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-%H-%M')
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d')
    # time_stamp = '11-18-01-27'
    sep_und = '_'
    run_name_comps = ['celebA_pretraining_supervision_%f_'%supervision_level,'epoch_%i_'%n_epochs, '_', mtype, '_', time_stamp, '_', 'seed_%i_'%seed, run_name]
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
    generator = Generator_CNN(latent_dim, x_shape)
    encoder = Encoder_CNN(latent_dim, channels=channels, x_shape=x_shape)
    discriminator = Discriminator_CNN(wass_metric=wass_metric, channels=channels, x_shape=x_shape)
    
    if cuda:
        generator.cuda()
        encoder.cuda()
        discriminator.cuda()
        bce_loss.cuda()
        xe_loss.cuda()
        mse_loss.cuda()
        
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
     
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay)
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # ----------
    #  Training
    # ----------
    ge_l = []
    d_l = []
    
    c_z = []
    c_i = []

    c_sup_loss = []
    
    # Training loop 
    #change
    log_final_results = open("noisy_all_chan_sigma_0.2_pretrain_clusgan_CelebA_5_all_digits_semi_sup_" + str(supervision_level) + "_seed_" + str(seed), "a+")

    cluster_weights = torch.ones([num_clusters]).to(torch.float).cuda()
    cluster_weights = cluster_weights / torch.sum(cluster_weights)

    print('\nBegin training session with %i epochs...\n'%(n_epochs))
 
    # -----------
    # Pretrain encoder
    # -----------

    print("Pretraining encoder")
    encoder_loss = []
    encoder_pretraining_acc = []

    n_enc_epochs = 250
    for epoch in range(n_enc_epochs):

        myImages = torch.tensor([])
        GTlabel = torch.tensor([])
        
        per_epoch_z_loss = 0.00
        for i, (imgs, itruth_label) in enumerate(dataloader):
            encoder.train()
            encoder.zero_grad()
            optimizer_E.zero_grad()

            real_imgs = Variable(imgs.type(Tensor))

            if myImages.shape[0] == 0:
                myImages = imgs.type(Tensor).clone()
                GTlabel = itruth_label
            else:
                myImages = torch.cat((myImages, imgs.type(Tensor).clone()), 0)
                GTlabel = torch.cat((GTlabel, itruth_label), 0)

            enc_real_z = encoder(real_imgs)

            real_z = cluster_centers[itruth_label.to(torch.long) - 1, :].cuda()
            z_loss = mse_loss(enc_real_z, real_z)

            z_loss.backward()
            optimizer_E.step()
            per_epoch_z_loss += z_loss

        encoder_loss.append(per_epoch_z_loss)

        if epoch % 1 == 0:
            encoder.eval()
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
            
            Local_NMI = NMI(np.array(GTlabel.cpu().to(torch.long)), np.array(finallabelling.cpu().to(torch.long)))
            Local_ARI = ARI(np.array(GTlabel.cpu().to(torch.long)), np.array(finallabelling.cpu().to(torch.long)))
            Local_ACC = ACC(np.array(GTlabel.cpu().to(torch.long)), np.array(finallabelling.cpu().to(torch.long)))

            print("NMI : ", Local_NMI)
            print("ARI : ", Local_ARI)
            print("ACC : ", Local_ACC)

            encoder_pretraining_acc.append(Local_ACC[0])
            encoder.train()

        print ("[Epoch %d/%d] \n"\
               "\tModel Losses: [encoder loss: %f]" % (epoch, 
                                                     n_epochs, 
                                                     per_epoch_z_loss)
              )

    # Save training results
    train_df = pd.DataFrame({
                             'n_epochs' : n_epochs,
                             'learning_rate' : lr,
                             'beta_1' : b1,
                             'beta_2' : b2,
                             'weight_decay' : decay,
                             'latent_dim' : latent_dim,
                             'n_classes' : cluster_centers.shape[0],
                             'encoder_loss' : ['enc_loss', encoder_loss],
                             'encoder_pretraining_acc' : ['enc_pretrain_acc', encoder_pretraining_acc]
                            })

    # Plot some training results
    plot_train_loss(df=train_df,
                    arr_list=['encoder_loss'],
                    figname='%s/pretraining_encoder_model_losses.png'%(run_dir)
                    )

    plot_train_loss(df=train_df,
                    arr_list=['encoder_pretraining_acc'],
                    figname='%s/pretraining_encoder_acc.png'%(run_dir)
                    )


    # ------------------
    # Pretrain Generator
    # ------------------

    print("Pretraining Generator")
    lambda_1 = 1.00
    lambda_2 = 4.00
    generator_loss = []
    img_loss = []
    latent_loss = []


    for epoch in range(n_epochs):
        per_epoch_latent_loss = 0.00
        per_epoch_img_loss = 0.00
        per_epoch_total_loss = 0.00        
        for i, (imgs, itruth_label) in enumerate(dataloader):
            generator.train()
            encoder.eval()
            encoder.zero_grad()
            generator.zero_grad()
            optimizer_G.zero_grad()

            real_imgs = Variable(imgs.type(Tensor))

            enc_real_z = encoder(real_imgs)
            reconstructed_imgs = generator(enc_real_z)
            img_loss_item = mse_loss(reconstructed_imgs, real_imgs)

            # Sample random latent variables
            sampled_z = sample_z(shape=imgs.shape[0], latent_dim=latent_dim, n_c=num_clusters, fix_class=-1, cluster_centers=cluster_centers, req_grad=False)
    
            # Generate a batch of images
            gen_imgs = generator(sampled_z)
            enc_gen_z = encoder(gen_imgs)
            latent_loss_item = mse_loss(enc_gen_z, sampled_z)

            total_loss = (lambda_1*img_loss_item) + (lambda_2*latent_loss_item) 
            total_loss.backward()
            optimizer_G.step()

            per_epoch_latent_loss += latent_loss_item
            per_epoch_img_loss += img_loss_item
            per_epoch_total_loss += total_loss
            

        generator_loss.append(per_epoch_total_loss)
        latent_loss.append(per_epoch_latent_loss)
        img_loss.append(per_epoch_img_loss)

        print ("[Epoch %d/%d] \n"\
               "\tModel Losses: [Total loss: %f] [latent loss: %f] [img loss: %f]" % (epoch, 
                                                     n_epochs, 
                                                     per_epoch_total_loss,
                                                     per_epoch_latent_loss,
                                                     per_epoch_img_loss)
              )


        generator.eval()

        if epoch % 20 == 9:
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
            save_image(stack_imgs,
                       '%s/before_GAN_game_gen_classes_%06i.png' %(imgs_dir, epoch), 
                       nrow=num_clusters, normalize=True)

    # Save training results
    train_df = pd.DataFrame({
                             'n_epochs' : n_epochs,
                             'learning_rate' : lr,
                             'beta_1' : b1,
                             'beta_2' : b2,
                             'weight_decay' : decay,
                             'latent_dim' : latent_dim,
                             'n_classes' : cluster_centers.shape[0],
                             'generator_loss' : ['gen_loss', generator_loss],
                             'generator_latent_loss' : ['gen_lat_loss', latent_loss],
                             'generator_image_loss' : ['gen_img_loss', img_loss]
                            })

    # Plot some training results
    plot_train_loss(df=train_df,
                    arr_list=['generator_loss', 'generator_image_loss', 'generator_latent_loss'],
                    figname='%s/pretraining_generator_model_losses.png'%(run_dir)
                    )


    # -----------------------------------------------
    # Pretraining Discriminator + Generator + Encoder
    # -----------------------------------------------

    print("Pretraining Generator, Discriminator, Encoder")
    n_skip_iter_gen = 1
    n_skip_iter_enc_gen = 8
    dataMNIST_correct = CustomMNISTdigit(finaldata, finallabels, None)
    dataloader_correct = DataLoader(dataMNIST_correct, num_workers=num_workers,  batch_size=batch_size,  shuffle=True)
    for epoch in range(n_epochs):

        myImages = torch.tensor([])
        GTlabel = torch.tensor([])

        per_epoch_ge_loss = 0.00
        per_epoch_d_loss = 0.00

        for i, (imgs, itruth_label) in enumerate(dataloader_correct):

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
            sampled_z_new = sample_z(shape=imgs.shape[0], latent_dim=latent_dim, n_c=num_clusters, fix_class=-1, cluster_centers=cluster_centers, req_grad=False)
    
            # Generate a batch of images
            gen_imgs_new = generator(sampled_z_new)
            
            # Discriminator output from real and generated samples
            D_gen = discriminator(gen_imgs_new)
            D_real = discriminator(real_imgs)

            g_loss = 0.00
            z_loss = 0.00
            ge_total_loss = 0.00
            valid = Variable(Tensor(gen_imgs_new.size(0), 1).fill_(1.0), requires_grad=False)

            # Step for Generator & Encoder, n_skip_iter times less than for discriminator
            if ((i % n_skip_iter_enc_gen == 1) and (i >= n_skip_iter_enc_gen)):
                enc_gen_z = encoder(gen_imgs_new)
                z_loss = mse_loss(enc_gen_z, sampled_z_new)
    
            if ((i % n_skip_iter_gen == 0) and (i>=n_skip_iter_gen)):
                # Check requested metric
                if wass_metric:
                    assert(1==0)
                    # Wasserstein GAN loss
                    g_loss = torch.mean(D_gen)
                else:
                    # Vanilla GAN loss
                    v_loss = bce_loss(D_gen, valid)
                    g_loss = v_loss
    
            ge_total_loss = g_loss + (beta * z_loss)

            if ge_total_loss == 0.00:
                pass
            else:
                # print("dooing a pass for GE")
                # print(ge_total_loss)
                ge_total_loss.backward(retain_graph=True)
                optimizer_G.step()
                optimizer_E.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
    
            discriminator.zero_grad()
            optimizer_D.zero_grad()
    
            # Measure discriminator's ability to classify real from generated samples
            if wass_metric:
                # Gradient penalty term
                grad_penalty = calc_gradient_penalty(discriminator, real_imgs, gen_imgs_new)
                d_loss = torch.mean(D_real) - torch.mean(D_gen) + grad_penalty 
            else:
                # Vanilla GAN loss
                fake = Variable(Tensor(gen_imgs_new.size(0), 1).fill_(0.0), requires_grad=False)
                real_loss = bce_loss(D_real, valid)
                fake_loss = bce_loss(D_gen, fake)
                d_loss = ((real_loss + fake_loss) / 2) 
    
            d_loss.backward()
            optimizer_D.step()

            per_epoch_d_loss += d_loss.item()
            if ge_total_loss == 0.00:
                pass
            else:
                per_epoch_ge_loss += ge_total_loss.item()

        # Save training losses
        d_l.append(per_epoch_d_loss)
        ge_l.append(per_epoch_ge_loss)

        
        print ("[Epoch %d/%d] \n"\
           "\tModel Losses: [D: %f] [GE: %f]" % (epoch, 
                                                 n_epochs, 
                                                 per_epoch_d_loss,
                                                 per_epoch_ge_loss)
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

            print(my_cluster_weights)
            
            Local_NMI = NMI(np.array(GTlabel.cpu().to(torch.long)), np.array(finallabelling.cpu().to(torch.long)))
            Local_ARI = ARI(np.array(GTlabel.cpu().to(torch.long)), np.array(finallabelling.cpu().to(torch.long)))
            Local_ACC = ACC(np.array(GTlabel.cpu().to(torch.long)), np.array(finallabelling.cpu().to(torch.long)))

            log_final_results.write("Epoch : " + str(epoch) + " \n")
            # log_final_results.write("NMI : " + str(Local_NMI) + " \n")
            # log_final_results.write("ARI : " + str(Local_ARI) + " \n")
            log_final_results.write("ACC : " + str(Local_ACC[0]) + " \n")
            log_final_results.write("supervision_level : " + str(supervision_level) + "\n")
            log_final_results.write("------------------------------------------\n")
            
            print("NMI : ", Local_NMI)
            print("ARI : ", Local_ARI)
            print("ACC : ", Local_ACC)

        generator.eval()

        if epoch % 20 == 9:
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
            save_image(stack_imgs,
                       '%s/after_GAN_game_gen_classes_%06i.png' %(imgs_dir, epoch), 
                       nrow=num_clusters, normalize=True)

        generator.train()


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
                            })

    train_df.to_csv('%s/training_details.csv'%(run_dir))


    # Plot some training results
    plot_train_loss(df=train_df,
                    arr_list=['gen_enc_loss', 'disc_loss'],
                    figname='%s/pretraining_total_model_losses.png'%(run_dir)
                    )

    # Save current state of trained models
    model_list = [discriminator, encoder, generator]
    save_model(models=model_list, out_dir=models_dir)


if __name__ == "__main__":
    main()
