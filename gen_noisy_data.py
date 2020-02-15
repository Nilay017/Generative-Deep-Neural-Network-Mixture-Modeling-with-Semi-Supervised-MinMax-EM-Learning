from data_processing import *
pathstr = './'
dataset = "CIFAR10"
num_images_per_digit = 1000
num_digits = 5
seeds = [0.0, 1.0, 2.0, 3.0, 4.0]
mean = (0., 0., 0.)
sigma = (20., 20., 20.)

for seed in seeds:
	add_noise_and_save_again(pathstr, dataset, num_images_per_digit, num_digits, seed, mean, sigma)